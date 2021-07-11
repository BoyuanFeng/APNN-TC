/*
  1-bit BMMA code.
  Runs at 500TOPS for matrix size of 4096x4096x8192.
  Borrows largely from CUDA-SDK.

  By Boyuan
*/

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 8
#define N 8
#define K 128

// GEMM configuration.

#define M_TILES 128
#define N_TILES 128
#define K_TILES 16

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define CHUNK_K 8

#define CHUNK_LINE_BYTES (CHUNK_K * sizeof(int4))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 2
#define WARP_COL_TILES 1

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
#define SKEW 2 // Updated for int4

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

__global__ void compute_gemm_imma(const int4 *A, const int4 *B, int *D) {
  extern __shared__ int4 shmem[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i = block_pos / (N_TILES/4) * 4;
    const unsigned int block_tile_j = block_pos % (N_TILES/4) * 4;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                     [WARP_ROW_TILES];

    for(int i=0; i < WARP_COL_TILES; i++)
      for(int j = 0; j < WARP_ROW_TILES; j++)
        wmma::fill_fragment(c[i][j], 0);
    
    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const int4 *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * (K_GLOBAL/128)] +
                                              M * (K_GLOBAL/128) * (warpId % 4))
                                           : (&B[block_tile_j * N * (K_GLOBAL/128)] +
                                              N * (K_GLOBAL/128) * (warpId % 4));


    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Offset in shared memory from which the B matrix is stored.
      const size_t shmem_idx_b_off = BLOCK_COL_TILES * M; // TODO: This BLOCK_COL_TILES may be selected to improve performance. Maybe moved outside the for loop.

      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)))
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * (K/128) +
                                (laneId / CHUNK_COPY_LINE_LANES) * (K_GLOBAL/128)) +
                       (laneId % CHUNK_COPY_LINE_LANES); // (K/128), since K=128 in bit. int4 is 128 bit.
                       
      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < (8 / CHUNK_COPY_LINES_PER_WARP); i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (int4 *)(lane_ptr +
                            (K_GLOBAL/128) * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // if (warpId==0 and laneId == 0) {
      //   for(int row_idx = 0; row_idx < 32; row_idx++) {
      //     for(int check_i = 0; check_i < 32; check_i++) {
      //       printf("shmem[%d][%d]: %x, shmem[%d][%d]: %x\n", row_idx, check_i, *((int*)&shmem[row_idx][0]+check_i), row_idx+32, check_i, *((int*)&shmem[shmem_idx_b_off+row_idx][0]+check_i));
      //     }    
      //   }
      // }

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, precision::b1, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, precision::b1, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M + (i * M);
          const int4 *tile_ptr = &shmem[shmem_idx_a][k_step];

          wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const int4 *tile_ptr = &shmem[shmem_idx_b][k_step * (K/128)];

              wmma::load_matrix_sync(b[j], tile_ptr, (CHUNK_K + SKEW)*128);
            }
            // printf("ckpt4\n");

            wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }
      __syncthreads();
    }
    
    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int*)&shmem[0][0] +
                              (warpId / 2) * SHMEM_STRIDE * M +
                              (warpId % 2) * SHMEM_OFFSET; // Will be used only when writing back D. May be moved outside the for loop. TODO.

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * M + j * N;
        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // if (warpId==0 and laneId == 0) {
    //   for(int row_idx = 0; row_idx < 32; row_idx++) {
    //     for(int col_idx = 0; col_idx < 32; col_idx++) {
    //       printf("shmem[%d][%d]: %d\n", row_idx, col_idx, *((int*)&shmem[0][0]+row_idx*32+col_idx));
    //     }
    //   }
    // }

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    // int *shmem_warp_stream_ptr = (int*)&shmem[0][0] + warpId * SHMEM_STRIDE * M; // Will be used only when writing back D. Maybe moved outside the for loop. TODO.
    const size_t idx = warpId * SHMEM_STRIDE * 4 + laneId*4;
    int *shmem_warp_stream_ptr = (int*)&shmem[0][0]+idx;

    // This warp's pointer to the C matrix data to copy memory from to shared memory. 
    // TODO: May be moved outside the for loop.
    size_t gmem_idx = block_tile_i * M * GLOBAL_MEM_STRIDE + block_tile_j*N + warpId*4*GLOBAL_MEM_STRIDE + laneId/8 * GLOBAL_MEM_STRIDE + laneId%8 * 4;

    // Now that shared memory contains all the D tiles, stream them to global memory.
    int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

    *((int4 *)dst_gmem_warp_stream_ptr) = *((int4 *)shmem_warp_stream_ptr);

    __syncthreads();
  }
}

void init_matrices(int4 *A, int4 *B){
  int *A_int = (int*) A;
  int *B_int = (int*) B;
  for(int i = 0; i < M_GLOBAL; i++) {
    for(int j = 0; j < K_GLOBAL/32; j++) {
      A_int[i*K_GLOBAL/32+j] = rand();
      // A_int[i*K_GLOBAL/32+j] = j;
    }
  }

  for(int i = 0; i < N_GLOBAL; i++) {
    for(int j = 0; j < K_GLOBAL/32; j++) {
      // B_int[i*K_GLOBAL/32+j] = 0xFFFFFFFF;
      B_int[i*K_GLOBAL/32+j] = rand();
    }
  }
}

int popcnt(int i) {
     // Java: use int, and use >>> instead of >>
     // C or C++: use int
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

void compute_ref(int4 *A, int4 *B, int *ref_C) {
  int *A_int = (int*) A;
  int *B_int = (int*) B;

  for (int m = 0; m < M_GLOBAL; m++) {
    for (int n = 0; n < N_GLOBAL; n++) {
      int tmp = 0;
      for (int k = 0; k < K_GLOBAL; k += 32) {
        // bit vector from row A and column B, accumulation and addition.
        tmp += popcnt(A_int[(m*K_GLOBAL + k)/32] ^ B_int[(n*K_GLOBAL + k)/32]);
      }
      // ref_C[m * K + n]= K - 2 * tmp;
      ref_C[m * N_GLOBAL + n]= tmp;
    }
  }
}


void validate_results(int *C, int* ref_C, int M_, int N_) {
  printf("Checking computed result for correctness: ");
  bool correct = true;
  double eps = 1.e-6;  // machine zero

  for(int i = 0; i < M_; i++) {
    for(int j = 0; j < N_; j++) {
      int idx = i*N_+j;
      double dst = fabs(C[idx] - ref_C[idx]);
      double abs = fabs(C[idx]) * fabs(ref_C[idx]);
      double ref_err = dst / abs;
      if (ref_err > eps) {
        // printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",, eps);
        printf("i: %d, j: %d, C: %d, ref_C: %d\n", i, j, C[idx], ref_C[idx]);
        // printf("non equal\n");
        correct = false;
      }
    }
  }
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}

#define verify_output

int main(int argc, char **argv) {
  printf("Initializing...\n");

  int dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

  int4 *A_h = NULL;
  int4 *B_h = NULL;
  int *C_h = NULL;

  A_h = (int4 *)malloc(sizeof(int4) * M_GLOBAL * (K_GLOBAL/128));
  B_h = (int4 *)malloc(sizeof(int4) * (K_GLOBAL/128) * N_GLOBAL);
  C_h = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);

  int4 *A = NULL;
  int4 *B = NULL;
  int *C = NULL;

  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&A), sizeof(int4) * M_GLOBAL * (K_GLOBAL/128)));
  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&B), sizeof(int4) * N_GLOBAL * (K_GLOBAL/128)));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C), sizeof(int) * M_GLOBAL * N_GLOBAL));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);

  enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = MAX(sizeof(int4) * (BLOCK_COL_TILES * M) *
                       (CHUNK_K * (K/128) + SKEW) * 2,
                   M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                       (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int))
  };

  printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

#ifdef verify_output
  printf("Preparing validation data for GPU...\n");
  init_matrices(A_h, B_h);
  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(int4) * M_GLOBAL * (K_GLOBAL/128), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(int4) * N_GLOBAL * (K_GLOBAL/128), cudaMemcpyHostToDevice));
#endif

  checkCudaErrors(cudaFuncSetAttribute(
    compute_gemm_imma, cudaFuncAttributeMaxDynamicSharedMemorySize,
    SHMEM_SZ));

  // Run ours NUM_PROFILES times and record time.
  int NUM_PROFILES = 1;
  float bmma_ms_avg = 0.0f;
  for(int iter=0; iter<NUM_PROFILES; ++iter){
          float bmma_ms = 0.0f;
          cudaEvent_t bmma_start;
          cudaEvent_t bmma_end;
          cudaEventCreate(&bmma_start);
          cudaEventCreate(&bmma_end);
          cudaEventRecord(bmma_start);
          checkKernelErrors(
            (compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                                  SHMEM_SZ>>>(A, B, C)));
                cudaEventRecord(bmma_end);
          cudaEventSynchronize(bmma_end);
          cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
          cudaEventDestroy(bmma_start);
          cudaEventDestroy(bmma_end);
          bmma_ms_avg += bmma_ms;
  }

  bmma_ms_avg = bmma_ms_avg/((float)NUM_PROFILES);

  printf("Time: %f ms\n", bmma_ms_avg);

  printf("TOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2)/(bmma_ms_avg/1000.)) / 1e12);


#ifdef verify_output
  printf("Validating results...\n");
  checkCudaErrors(cudaMemcpy(C_h, C, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

  int *C_ref = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);

  /* Copmpute reference matrix on CPU */
  compute_ref(A_h, B_h, C_ref);

  /* validation results */
  validate_results(C_h, C_ref, M_GLOBAL, N_GLOBAL);
#endif

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));

  return EXIT_SUCCESS;
}
