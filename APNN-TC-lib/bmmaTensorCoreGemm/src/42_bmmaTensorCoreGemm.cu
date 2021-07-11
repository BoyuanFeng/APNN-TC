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

__global__ void apmm_w6a2(const int4 *A, const int4 *B, int *D, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL, int xb, int wb) {
  // GEMM configuration.
  M_GLOBAL = M_GLOBAL * wb;
  N_GLOBAL = N_GLOBAL * xb;
  // printf("M_GLOBAL: %d, N_GLOBAL: %d\n", M_GLOBAL, N_GLOBAL);

  int M_TILES = M_GLOBAL / M;
  int N_TILES = N_GLOBAL / N;
  int K_TILES = K_GLOBAL / K;  extern __shared__ int4 shmem[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i = block_pos / (N_TILES/8) * 8;
    const unsigned int block_tile_j = block_pos % (N_TILES/8) * 8;

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
                                              M * (K_GLOBAL/128) * (warpId % 4) * 2)
                                           : (&B[block_tile_j * N * (K_GLOBAL/128)] +
                                              N * (K_GLOBAL/128) * (warpId % 4) * 2);


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
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * (K/128) +
                                (laneId / CHUNK_COPY_LINE_LANES) * (K_GLOBAL/128)) +
                       (laneId % CHUNK_COPY_LINE_LANES); // (K/128), since K=128 in bit. int4 is 128 bit.
                       
      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP); i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (int4 *)(lane_ptr +
                            (K_GLOBAL/128) * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, precision::b1, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, precision::b1, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
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

            wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j], bmmaBitOpAND);
          }
        }
      }
      __syncthreads();
    }
    
    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int*)&shmem[0][0] +
                              (warpId / 2) * SHMEM_STRIDE * M * 2 +
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

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    // int *shmem_warp_stream_ptr = (int*)&shmem[0][0] + warpId * SHMEM_STRIDE * M; // Will be used only when writing back D. Maybe moved outside the for loop. TODO.
    size_t idx = warpId * 12 * 64 + (laneId%16) * 4 + (laneId/16)*6*64;

    int *shmem_warp_stream_ptr = (int*)&shmem[0][0]+idx;
    int val[2];

    if (warpId < 5) {

      typedef union {
        int4 vec;
        int a[4];
      } U4;
      U4 tmp0;
      U4 tmp1;
      U4 tmp2;
      U4 tmp3;
      U4 tmp4;
      U4 tmp5;

      tmp0.vec = *((int4*)shmem_warp_stream_ptr);
      tmp1.vec = *((int4*)shmem_warp_stream_ptr+32);
      tmp2.vec = *((int4*)shmem_warp_stream_ptr+64);
      tmp3.vec = *((int4*)shmem_warp_stream_ptr+96);
      tmp4.vec = *((int4*)shmem_warp_stream_ptr+128);
      tmp5.vec = *((int4*)shmem_warp_stream_ptr+160);

      // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
      //   for(int i = 0; i < 4; i++) {
      //     printf("tmp0.a[%d], %d ", 3-i, tmp0.a[3-i]);
      //   }
      //   printf("\n");
      //   for(int i = 0; i < 4; i++) {
      //     printf("tmp1.a[%d], %d ", 3-i, tmp1.a[3-i]);
      //   }
      //   printf("\n");
      // }

      val[0] = tmp0.a[0] + 2*tmp1.a[0] + 4*tmp2.a[0] + 8*tmp3.a[0] + 16*tmp4.a[0] + 32*tmp5.a[0] + 2*(tmp0.a[1] + 2*tmp1.a[1] + 4*tmp2.a[1] + 8*tmp3.a[1] + 16*tmp4.a[1] + 32*tmp5.a[1]);
      val[1] = tmp0.a[2] + 2*tmp1.a[2] + 4*tmp2.a[2] + 8*tmp3.a[2] + 16*tmp4.a[2] + 32*tmp5.a[2] + 2*(tmp0.a[3] + 2*tmp1.a[3] + 4*tmp2.a[3] + 8*tmp3.a[3] + 16*tmp4.a[3] + 32*tmp5.a[3]);

    }

    __syncthreads();

    if (warpId < 5) {
      idx = threadIdx.x*2;
      shmem_warp_stream_ptr = (int*)&shmem[0][0]+idx;
      *(shmem_warp_stream_ptr+0) = val[0];
      *(shmem_warp_stream_ptr+1) = val[1];  
    }
    __syncthreads();

    // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
    //   for(int i = 0; i < 2; i++) {
    //     for(int j = 0; j < 2; j++) {
    //       printf("%d ", *((int*)&shmem[0][0]+i*64+j));
    //     }
    //     printf("\n");
    //   }
    // }

    if (threadIdx.x < 80) {
      shmem_warp_stream_ptr = (int*)&shmem[0][0]+ threadIdx.x*4;

      // This warp's pointer to the C matrix data to copy memory from to shared memory. 
      // TODO: May be moved outside the for loop.
      size_t gmem_idx = block_tile_i*M/6*N_GLOBAL + block_tile_j*N + (threadIdx.x%8) * 4 + (threadIdx.x/8)*N_GLOBAL;
      
      // Now that shared memory contains all the D tiles, stream them to global memory.
      int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

      *((int4 *)(dst_gmem_warp_stream_ptr)) = *((int4 *)(shmem_warp_stream_ptr));
    }
    __syncthreads();
  }
}

// #define verify_output

int main(int argc, char **argv) {

  int dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  int X_BIT = 2;
  int W_BIT = 6;

  for (int M_GLOBAL=128; M_GLOBAL<=1024; M_GLOBAL += 128 ) {
    int N_GLOBAL = M_GLOBAL;
    int K_GLOBAL = M_GLOBAL;
  
    int4 *X = NULL;
    int4 *W = NULL;
    int *Output = NULL;
  
    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void **>(&W), sizeof(int4) * M_GLOBAL * (K_GLOBAL/128) * W_BIT));
    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void **>(&X), sizeof(int4) * N_GLOBAL * (K_GLOBAL/128)* X_BIT));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&Output), sizeof(int) * M_GLOBAL * N_GLOBAL* W_BIT));
    
    // printf("W size: %d, X size: %d, Output size: %d\n", (int)sizeof(int4) * M_GLOBAL * (K_GLOBAL/128) * W_BIT, (int)sizeof(int4) * N_GLOBAL * (K_GLOBAL/128)* X_BIT, (int)sizeof(int) * M_GLOBAL * N_GLOBAL);
    
// #ifdef verify_output
//     int4 *X_h = NULL;
//     int4 *W_h = NULL;
//     int *Output_h = NULL;
  
//     X_h = (int4 *)malloc(sizeof(int4) * M_GLOBAL * (K_GLOBAL/128) * X_BIT);
//     W_h = (int4 *)malloc(sizeof(int4) * (K_GLOBAL/128) * N_GLOBAL * W_BIT);
//     Output_h = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
//     printf("Preparing validation data for GPU...\n");
//     init_matrices(A_h, B_h);
//     checkCudaErrors(cudaMemcpy(A, A_h, sizeof(int4) * M_GLOBAL * (K_GLOBAL/128), cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaMemcpy(B, B_h, sizeof(int4) * N_GLOBAL * (K_GLOBAL/128), cudaMemcpyHostToDevice));
// #endif
  
    int SHMEM_SZ = 65536;
    checkCudaErrors(cudaFuncSetAttribute(
      apmm_w6a2, cudaFuncAttributeMaxDynamicSharedMemorySize,
      SHMEM_SZ));
  
    // Run ours NUM_PROFILES times and record time.
    float bmma_ms_avg = 0.0f;
    int NUM_PROFILES = 200;
    for(int iter=0; iter<NUM_PROFILES; ++iter){
            float bmma_ms = 0.0f;
            cudaEvent_t bmma_start;
            cudaEvent_t bmma_end;
            cudaEventCreate(&bmma_start);
            cudaEventCreate(&bmma_end);
            cudaEventRecord(bmma_start);
            checkKernelErrors(
              (apmm_w6a2<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                                    SHMEM_SZ>>>(X, W, Output, M_GLOBAL, N_GLOBAL, K_GLOBAL, X_BIT, W_BIT)));
                  cudaEventRecord(bmma_end);
            cudaEventSynchronize(bmma_end);
            cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
            cudaEventDestroy(bmma_start);
            cudaEventDestroy(bmma_end);
            bmma_ms_avg += bmma_ms;
    }
  
    bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;

    printf("V42, 64x64. M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d, X_BIT: %d, W_BIT: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL, X_BIT, W_BIT);
    printf("Time: %f ms\n", bmma_ms_avg);  
    printf("TOPS: %.2f\n", (((double)(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)/(bmma_ms_avg/1000.)) / 1e12);
  
  
// #ifdef verify_output
//   printf("Validating results...\n");
//   checkCudaErrors(cudaMemcpy(C_h, C, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

//   int *C_ref = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);

//   /* Copmpute reference matrix on CPU */
//   compute_ref_w1a2(A_h, B_h, C_ref);

//   /* validation results */
//   validate_results(C_h, C_ref, M_GLOBAL, N_GLOBAL/2);
//   free(A_h);
//   free(B_h);
//   free(C_h);
// #endif
  
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(X)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(W)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(Output)));
  
  }

  return EXIT_SUCCESS;
}
