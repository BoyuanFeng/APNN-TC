#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;

#define NUM_PROFILES 200

#define BIT_WIDTH 8

#if BIT_WIDTH == 16
    typedef half input_t;
    typedef float output_t;
#elif BIT_WIDTH == 8
    typedef int8_t input_t;
//     typedef float output_t;
    typedef int output_t;
#endif 


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}


void run_test(int m, int n, int k){

        int alpha = 1;
        int beta = 0;
        
        input_t* A;
        input_t* B;
        output_t* C;

        int size_A = m*k;
        int size_B = n*k;
        int size_C = m*n;

        if (cudaMalloc((void **)&A, size_A * sizeof(input_t)) !=
                cudaSuccess) {
                fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
                exit(cudaSuccess);
        }

        if (cudaMalloc((void **)&B, size_B * sizeof(input_t)) !=
                cudaSuccess) {
                fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
                exit(cudaSuccess);
        }

        if (cudaMalloc((void **)&C, size_C * sizeof(output_t)) !=
                cudaSuccess) {
                fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
                exit(cudaSuccess);
        }

        vector<float> cublas_times;
     

        cublasHandle_t cublasHandle;
        cublasStatus_t status;

        cublasCreate(&cublasHandle);

        // cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
        for(int iter=0; iter<NUM_PROFILES; ++iter){
                float cublas_ms = 0.0f;
                cudaEvent_t cublas_start;
                cudaEvent_t cublas_end;
                cudaEventCreate(&cublas_start);
                cudaEventCreate(&cublas_end);
                cudaEventRecord(cublas_start);

                #if BIT_WIDTH == 16
                    cublasGemmEx(cublasHandle, 
                                    CUBLAS_OP_T, 
                                    CUBLAS_OP_N,
                                    m, n, k,
                                    &alpha,
                                    A, CUDA_R_16F, k,
                                    B, CUDA_R_16F, k,
                                    &beta,
                                    C, CUDA_R_32F, m,
                                    CUBLAS_COMPUTE_32F, 
                                    CUBLAS_GEMM_DFALT_TENSOR_OP);
                    
                #elif BIT_WIDTH == 8
                //     status = cublasGemmEx(cublasHandle, 
                //             CUBLAS_OP_T, 
                //             CUBLAS_OP_N,
                //             m, n, k,
                //             &alpha,
                //             A, CUDA_R_8I, k,
                //             B, CUDA_R_8I, k,
                //             &beta,
                //             C, CUDA_R_32F, m,
                //             CUBLAS_COMPUTE_32F, 
                //             CUBLAS_GEMM_DFALT_TENSOR_OP);

                status = cublasGemmEx(cublasHandle, 
                        CUBLAS_OP_T, 
                        CUBLAS_OP_N,
                        m, n, k,
                        &alpha,
                        A, CUDA_R_8I, k,
                        B, CUDA_R_8I, k,
                        &beta,
                        C, CUDA_R_32I, m,
                        CUBLAS_COMPUTE_32I, 
                        CUBLAS_GEMM_DFALT_TENSOR_OP);
                #endif

                if(status !=CUBLAS_STATUS_SUCCESS)
                {
                    fprintf(stderr, "Error in cublasGemmEx()\n");
                    fprintf(stderr, "CUBLAS error code: ");
                    fprintf(stderr,  cublasGetErrorString(status));
                    fprintf(stderr,"\n");
                    exit(EXIT_FAILURE);
                }

                cudaEventRecord(cublas_end);
                cudaEventSynchronize(cublas_end);
                cudaEventElapsedTime(&cublas_ms, cublas_start, cublas_end);
                cudaEventDestroy(cublas_start);
                cudaEventDestroy(cublas_end);
                cublas_times.push_back(cublas_ms);
        }

        float cublas_ms_avg = 0.0f;
        for(int iter=0; iter<NUM_PROFILES; ++iter){
                cublas_ms_avg += cublas_times[iter];
        }
        cublas_ms_avg = cublas_ms_avg/(float)NUM_PROFILES;

        float cublas_tflops = 0.0f;
        float gflops = (float)m*(float)n*(float)k*2.0f/1024.0f/1024.0f/1024.0f;

        cublas_tflops = gflops/cublas_ms_avg;
        printf("cuBLASGemmEX (%d-bit), m: %6d, n: %6d, k:%6d, TFLOPS: %.2f\n", BIT_WIDTH, m, n, k, cublas_tflops);

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        return;
}

int main(int argc, char* argv[]){

        if (argc < 2 ){
                printf("Arguement Error: Usage ./prog test_dim \n");
                return -1;
        }

        int N = atoi(argv[1]);
        run_test(N, N, N);
        // printf("----------------\n");

        return 0;
}