#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "0_bmmaTensorCoreGemm.h"

// Externally configurable parameters.
#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

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
  #if CPU_DEBUG
    int *result_hD = NULL;
    int *result_host = NULL;
  #endif
  
    A_h = (uint8_t *)malloc(sizeof(uint8_t) * M_GLOBAL * K_GLOBAL);
    B_h = (uint8_t *)malloc(sizeof(uint8_t) * K_GLOBAL * N_GLOBAL);
    C_h = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
  #if CPU_DEBUG
    result_hD = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
    result_host = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
  #endif
  
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
  
    init_host_matrices(A_h, B_h, C_h);
  
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
    if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) {
      printf("Computing... using high performance kernel compute_gemm_imma \n");
  
      checkCudaErrors(cudaFuncSetAttribute(
          compute_gemm_imma, cudaFuncAttributeMaxDynamicSharedMemorySize,
          SHMEM_SZ));
      checkKernelErrors(
          (compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                               SHMEM_SZ>>>(A, B, C, D, alpha, beta)));
  #if CPU_DEBUG
      checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(int) * M_GLOBAL * N_GLOBAL,
                                 cudaMemcpyDeviceToHost));
  #endif
    } else {
      dim3 gridDim;
      dim3 blockDim;
  
      // blockDim.x must be a multiple of warpSize
      // 128x4 means we have 16 warps and a block computes a 64x64 output tile
      blockDim.x = 128;
      blockDim.y = 4;
  
      gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
                  (WMMA_M * blockDim.x / 32);
      gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
  
      printf("Computing... using simple_wmma_gemm_imma kernel\n");
      simple_wmma_gemm_imma<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL,
                                                   K_GLOBAL, alpha, beta);
  #if CPU_DEBUG
      checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(int) * M_GLOBAL * N_GLOBAL,
                                 cudaMemcpyDeviceToHost));
  #endif
    }
  
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
  
  #if CPU_DEBUG
    printf("Verifying correctness of the computations...\n");
  
    memcpy(result_host, C_h, sizeof(int) * M_GLOBAL * N_GLOBAL);
  
    matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
                      K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);
  
    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
      if (abs(result_hD[i] - result_host[i]) > 0) {
        printf("mismatch i=%d result_hD=%d result_host=%d\n", i, result_hD[i],
               result_host[i]);
      }
    }
    free(result_host);
    free(result_hD);
  #endif
  
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
  