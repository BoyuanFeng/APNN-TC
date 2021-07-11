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

#define CHUNK_K 4
#define SKEW 1
#define WARPS_PER_BLOCK 8
#define WARP_SIZE 32
#define THREADS_PER_BLOCK WARP_SIZE * WARPS_PER_BLOCK
#define CHUNK_LINE_BYTES CHUNK_K * sizeof(int4)
#define WARP_COPY_BYTES WARP_SIZE * sizeof(int4)
#define CHUNK_COPY_LINES_PER_WARP WARP_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES WARP_SIZE / CHUNK_COPY_LINES_PER_WARP
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define BLOCK_ROW_TILES WARP_ROW_TILES * BLOCK_ROW_WARPS
#define BLOCK_COL_TILES WARP_COL_TILES * BLOCK_COL_WARPS
#define M 8
#define N 8
#define K 128

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

typedef union {
  int4 vec;
  int a[4];
} U4;


// Assume that Kernel size is 3x3.
// Assume CIN is 128.
__global__ void compute_conv_imma(const int4 *W, const int4 *X, int *Output, int Height, int Width, int CIN, int COUT) {
  // GEMM Configuration
  int X_bit_offset = Height * Width * CIN/128;

  extern __shared__ int4 shmem[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.
  wmma::fragment<wmma::accumulator, 8, 8, 128, int> c[WARP_COL_TILES]
    [WARP_ROW_TILES];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_i = (block_pos/(COUT/64)) / (Width/8) * 4;
    const unsigned int block_j = (block_pos/(COUT/64)) % (Width/8) * 8;
    const unsigned int block_z = block_pos % (COUT/64) * 64;

    if (block_i >= Height) {
      break;
    }

    int image_starting_idx = block_i * 4 * Width * CIN/32 + block_j * 8 * CIN/32;

    for(int i=0; i < WARP_COL_TILES; i++)
      for(int j=0; j < WARP_ROW_TILES; j++)
        wmma::fill_fragment(c[i][j], 0);
    
    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < int(9*CIN/128/4); tile_k += CHUNK_K) {

      int SHMEM_i = threadIdx.x/4;
      int SHMEM_part = SHMEM_i / 32;
      int SHMEM_offset = SHMEM_i % 32;
      int row = SHMEM_offset / 8;
      int col = SHMEM_offset % 8;
      int t = threadIdx.x % 4;

      int GL_idx = image_starting_idx + SHMEM_part*X_bit_offset + row*Width*CIN/128 + col*CIN/18 + tile_k*4 + t;

      shmem[SHMEM_i][t] = X[GL_idx];

      SHMEM_i += 64;
      int weight_load_idx = SHMEM_part * 9 * CIN * COUT / 128 + (block_z + SHMEM_offset) * 9 * CIN/128;
      int thread_load_idx = weight_load_idx + (tile_k*4 + t) * CIN/128;
      shmem[SHMEM_i][t] = W[thread_load_idx];

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, precision::b1, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, precision::b1, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 4 + (i * M);
          const int4 *tile_ptr = &shmem[shmem_idx_a][k_step];

          wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = 64 +
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

    // Needs special handle for the remaining K.

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        int *tile_ptr = (int*)&shmem[0][0] + warpId*8*64 + (i*4+j) * 64;
        wmma::store_matrix_sync(tile_ptr, c[i][j], 8,  wmma::mem_row_major);
      }
    }

    __syncthreads();



    U4 tmp0;
    U4 tmp1;
    U4 tmp2;
    U4 tmp3;
    U4 val;

    int *shmem_warp_stream_ptr = (int*)&shmem[0][0]+threadIdx.x/8*64 + (threadIdx.x%8)*4;
    tmp0.vec = *((int4*)shmem_warp_stream_ptr);
    tmp1.vec = *((int4*)shmem_warp_stream_ptr+8);
    tmp2.vec = *((int4*)shmem_warp_stream_ptr+32*16);
    tmp3.vec = *((int4*)shmem_warp_stream_ptr+32*16+8);
    val.a[0] = tmp0.a[0] + 2*tmp1.a[0] + 2*tmp2.a[0] + 4*tmp3.a[0];
    val.a[1] = tmp0.a[1] + 2*tmp1.a[1] + 2*tmp2.a[1] + 4*tmp3.a[1];
    val.a[2] = tmp0.a[2] + 2*tmp1.a[2] + 2*tmp2.a[2] + 4*tmp3.a[2];
    val.a[3] = tmp0.a[3] + 2*tmp1.a[3] + 2*tmp2.a[3] + 4*tmp3.a[3];

    int shmem_row = threadIdx.x/8;
    int row = shmem_row / 8;
    int col = shmem_row % 8;
    int* dst_gmem_warp_stream_ptr = Output + block_i * 4 * Width * COUT + block_j*8*COUT 
              + row*Width*COUT + col*4;
    *(int4*)dst_gmem_warp_stream_ptr = val.vec;
    __syncthreads();
  }
}

// void init_matrices(int4 *A, int4 *B){
//   int *A_int = (int*) A;
//   int *B_int = (int*) B;
//   for(int i = 0; i < M_GLOBAL; i++) {
//     for(int j = 0; j < K_GLOBAL/32; j++) {
//       A_int[i*K_GLOBAL/32+j] = rand();
//     }
//   }

//   for(int i = 0; i < N_GLOBAL; i++) {
//     for(int j = 0; j < K_GLOBAL/32; j++) {
//       B_int[i*K_GLOBAL/32+j] = 0xFFFFFFFF;
//       B_int[i*K_GLOBAL/32+j] = rand();
//     }
//   }
// }

// int popcnt(int i) {
//      // Java: use int, and use >>> instead of >>
//      // C or C++: use int
//      i = i - ((i >> 1) & 0x55555555);
//      i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
//      return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
// }

// void compute_ref(int4 *A, int4 *B, int *ref_C) {
//   int *A_int = (int*) A;
//   int *B_int = (int*) B;

//   for (int m = 0; m < M_GLOBAL; m++) {
//     for (int n = 0; n < N_GLOBAL; n++) {
//       int tmp = 0;
//       for (int k = 0; k < K_GLOBAL; k += 32) {
//         // bit vector from row A and column B, accumulation and addition.
//         tmp += popcnt(A_int[(m*K_GLOBAL + k)/32] ^ B_int[(n*K_GLOBAL + k)/32]);
//       }
//       // ref_C[m * K + n]= K - 2 * tmp;
//       ref_C[m * N_GLOBAL + n]= tmp;
//     }
//   }
// }


// void validate_results(int *C, int* ref_C, int M_, int N_) {
//   printf("Checking computed result for correctness: ");
//   bool correct = true;
//   double eps = 1.e-6;  // machine zero

//   for(int i = 0; i < M_; i++) {
//     for(int j = 0; j < N_; j++) {
//       int idx = i*N_+j;
//       double dst = fabs(C[idx] - ref_C[idx]);
//       double abs = fabs(C[idx]) * fabs(ref_C[idx]);
//       double ref_err = dst / abs;
//       if (ref_err > eps) {
//         // printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",, eps);
//         printf("i: %d, j: %d, C: %d, ref_C: %d\n", i, j, C[idx], ref_C[idx]);
//         // printf("non equal\n");
//         correct = false;
//       }
//     }
//   }
//   printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
// }

// #define verify_output

int main(int argc, char **argv) {
  printf("Initializing...\n");

  int dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  int Height = 128;
  int Width = 16;
  int CIN = 128;
  int COUT = 256;
  int bit = 2;

  int4 *X = NULL;
  int4 *W = NULL;
  int *Output = NULL;

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&X), sizeof(int4) * Height * Width * (CIN/128) * bit));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&W), sizeof(int4) * 9 * (CIN/128) * COUT * bit));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&Output), sizeof(int4) * Height * Width * COUT ));

// #ifdef verify_output
//   printf("Preparing validation data for GPU...\n");
// int4 *W_h = NULL;
// int4 *X_h = NULL;
// int *Output_h = NULL;

// X_h = (int4 *)malloc(sizeof(int4) * H * W * (CIN/128) * X_bit);
// W_h = (int4 *)malloc(sizeof(int4) * 9 * (CIN/128) * COUT * W_bit);
// Output_h = (int *)malloc(sizeof(int4) * H * W * (COUT/128) * X_bit);
//   init_matrices(A_h, B_h);
//   checkCudaErrors(cudaMemcpy(A, A_h, sizeof(int4) * M_GLOBAL * (K_GLOBAL/128), cudaMemcpyHostToDevice));
//   checkCudaErrors(cudaMemcpy(B, B_h, sizeof(int4) * N_GLOBAL * (K_GLOBAL/128), cudaMemcpyHostToDevice));
// #endif

  int SHMEM_SZ = 65536;
  checkCudaErrors(cudaFuncSetAttribute(
    compute_conv_imma, cudaFuncAttributeMaxDynamicSharedMemorySize,
    SHMEM_SZ));

  // Run ours NUM_PROFILES times and record time.
  float bmma_ms_avg = 0.0f;
  int NUM_PROFILES = 1000;
  for(int iter=0; iter<NUM_PROFILES; ++iter){
          float bmma_ms = 0.0f;
          cudaEvent_t bmma_start;
          cudaEvent_t bmma_end;
          cudaEventCreate(&bmma_start);
          cudaEventCreate(&bmma_end);
          cudaEventRecord(bmma_start);
          checkKernelErrors(
            (compute_conv_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                                  SHMEM_SZ>>>(W, X, Output, Height, Width, CIN, COUT)));
                cudaEventRecord(bmma_end);
          cudaEventSynchronize(bmma_end);
          cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
          cudaEventDestroy(bmma_start);
          cudaEventDestroy(bmma_end);
          bmma_ms_avg += bmma_ms;
  }

  bmma_ms_avg = bmma_ms_avg/(double)NUM_PROFILES;

  printf("Time: %f ms\n", bmma_ms_avg);

  printf("TOPS: %.2f\n", (((double)9 * CIN * Height * Width * COUT * 2)/(bmma_ms_avg/1000.)) / 1e12);


// #ifdef verify_output
//   printf("Validating results...\n");
//   checkCudaErrors(cudaMemcpy(C_h, C, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

//   int *C_ref = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);

//   /* Copmpute reference matrix on CPU */
//   // compute_ref(A_h, B_h, C_ref);

//   /* validation results */
//   // validate_results(C_h, C_ref, M_GLOBAL, N_GLOBAL);
// #endif

  // free(A_h);
  // free(B_h);
  // free(C_h);
  // checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
  // checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
  // checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));

  return EXIT_SUCCESS;
}
