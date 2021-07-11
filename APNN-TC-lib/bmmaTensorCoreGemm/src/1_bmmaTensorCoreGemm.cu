/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// CUDA sample demonstrating a integer GEMM computation using the Warp Matrix
// Multiply and Accumulate API.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
// matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the
//   A and B data from shared memory, thus reducing the number of data copies
//   from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
// additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
// warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory
//   contents to global memory, again avoiding redundant random global memory
//   accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
// utilization,
//   but carefully enough to avoid local memory use.

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Externally configurable parameters.

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 1024
#define N_TILES 1024
#define K_TILES 1024

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 uint8_t-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 8
#else
#define CHUNK_K 16
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(uint8_t))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 32
// one-byte "uint8_t" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_UINT8 32

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B,
    const int *C, int *D, int alpha, int beta) {
  extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp computes.
  int *shmem_warp_tile_ptr = (int *)&shmem[0][0] +
                             (warpId / 2) * SHMEM_STRIDE * K * 2 +
                             (warpId % 2) * SHMEM_OFFSET; // Will be used only when writing back D. TODO.

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * K; // Will be used only when writing back D. TODO.

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx =
        (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;


    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                     [WARP_ROW_TILES];

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const uint8_t *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                                              M * K_GLOBAL * (warpId % 4) * 2)
                                           : (&B[block_tile_j * N * K_GLOBAL] +
                                              N * K_GLOBAL * (warpId % 4) * 2);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                       (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
           i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (int4 *)((uint8_t *)lane_ptr +
                            K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major>
            b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
          const uint8_t *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const uint8_t *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

      // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

        int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
          *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}

int main(int argc, char **argv) {
  printf("Initializing...\n");

  int dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  // Tensor cores require a GPU of Volta (SM72) architecture or higher.
  if (deviceProp.major < 7 || (deviceProp.major <= 7 && deviceProp.minor < 2)) {
    printf(
        "immaTensorCoreGemm requires SM 7.2 or higher to use Tensor Cores.  "
        "Exiting...\n");
    exit(EXIT_WAIVED);
  }

  printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

  uint8_t *A_h = NULL;
  uint8_t *B_h = NULL;
  int *C_h = NULL;

  A_h = (uint8_t *)malloc(sizeof(uint8_t) * M_GLOBAL * K_GLOBAL);
  B_h = (uint8_t *)malloc(sizeof(uint8_t) * K_GLOBAL * N_GLOBAL);
  C_h = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);

  uint8_t *A = NULL;
  uint8_t *B = NULL;
  int *C = NULL;
  int *D = NULL;

  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&A), sizeof(uint8_t) * M_GLOBAL * K_GLOBAL));
  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&B), sizeof(uint8_t) * N_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C), sizeof(int) * M_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&D), sizeof(int) * M_GLOBAL * N_GLOBAL));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(uint8_t) * M_GLOBAL * K_GLOBAL,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(uint8_t) * N_GLOBAL * K_GLOBAL,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(int) * M_GLOBAL * N_GLOBAL,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(D, 0, sizeof(int) * M_GLOBAL * N_GLOBAL));

  printf("Preparing data for GPU...\n");

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M) *
                       (CHUNK_K * K + SKEW_UINT8) * 2,
                   M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                       (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int))
  };

  printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  int alpha = 1;
  int beta = 1;

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  // If enough shared memory available on the GPU use high performant kernel
  printf("Computing... using high performance kernel compute_gemm_imma \n");

  checkCudaErrors(cudaFuncSetAttribute(
      compute_gemm_imma, cudaFuncAttributeMaxDynamicSharedMemorySize,
      SHMEM_SZ));
  checkKernelErrors(
      (compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                            SHMEM_SZ>>>(A, B, C, D, alpha, beta)));


  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float milliseconds = 0;

  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Time: %f ms\n", milliseconds);
    printf("TOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2)/(milliseconds/1000.)) / 1e12);

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(D)));

  return EXIT_SUCCESS;
}
