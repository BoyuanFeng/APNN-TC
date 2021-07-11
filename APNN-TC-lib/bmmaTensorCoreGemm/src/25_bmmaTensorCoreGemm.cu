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
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 8
#define N 8
#define K 128

// GEMM configuration.

#define M_TILES 512
#define N_TILES 512
#define K_TILES 32

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

#define WARP_ROW_TILES 8
#define WARP_COL_TILES 4

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


// Takes 0.10735 ms
// __global__ void compute_gemm_imma(const int4 *global1, const int4 *global2, size_t subset_count) {
//   extern __shared__ int4 shared[];
//   auto group = cooperative_groups::this_thread_block();
 
//   for (size_t subset = 0; subset < subset_count; ++subset) {
//       // copy 512 uint4 in total.
//       shared[group.thread_rank()               ] = global1[subset * group.size() + group.thread_rank()];
//       shared[group.size() + group.thread_rank()] = global2[subset * group.size() + group.thread_rank()];

//       group.sync(); // Wait for all copies to complete

//       // compute(shared);

//       group.sync();
//   }
// }

// Takes 0.112201 ms
__global__ void compute_gemm_imma(const int4 *global1, const int4 *global2, size_t subset_count) {
  extern __shared__ int4 shared[];
  auto group = cooperative_groups::this_thread_block();
 
  for (size_t subset = 0; subset < subset_count; ++subset) {
    cooperative_groups::memcpy_async(group, shared,
                                     &global1[subset * group.size()], sizeof(int4) * group.size());
    cooperative_groups::memcpy_async(group, shared + group.size(),
                                     &global2[subset * group.size()], sizeof(int4) * group.size());

    cooperative_groups::wait(group); // Wait for all copies to complete

    // compute(shared);

    group.sync();
}
}


void init_matrices(int4 *A, int4 *B){
  int *A_int = (int*) A;
  int *B_int = (int*) B;
  for(int i = 0; i < M_GLOBAL; i++) {
    for(int j = 0; j < K_GLOBAL/32; j++) {
      A_int[i*K_GLOBAL/32+j] = rand();
    }
  }

  for(int i = 0; i < N_GLOBAL; i++) {
    for(int j = 0; j < K_GLOBAL/32; j++) {
      B_int[i*K_GLOBAL/32+j] = 0xFFFFFFFF;
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
  float bmma_ms_avg = 0.0f;
  for(int iter=0; iter<200; ++iter){
          float bmma_ms = 0.0f;
          cudaEvent_t bmma_start;
          cudaEvent_t bmma_end;
          cudaEventCreate(&bmma_start);
          cudaEventCreate(&bmma_end);
          cudaEventRecord(bmma_start);
          checkKernelErrors(
            (compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                                  SHMEM_SZ>>>(A, B, M_GLOBAL * (K_GLOBAL/128)/512)));
                cudaEventRecord(bmma_end);
          cudaEventSynchronize(bmma_end);
          cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
          cudaEventDestroy(bmma_start);
          cudaEventDestroy(bmma_end);
          bmma_ms_avg += bmma_ms;
  }

  bmma_ms_avg = bmma_ms_avg/200.0f;

  printf("Time: %f ms\n", bmma_ms_avg);

  printf("TOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2)/(bmma_ms_avg/1000.)) / 1e12);


#ifdef verify_output
  printf("Validating results...\n");
  checkCudaErrors(cudaMemcpy(C_h, C, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

  int *C_ref = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);

  /* Copmpute reference matrix on CPU */
  // compute_ref(A_h, B_h, C_ref);

  /* validation results */
  // validate_results(C_h, C_ref, M_GLOBAL, N_GLOBAL);
#endif

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));

  return EXIT_SUCCESS;
}
