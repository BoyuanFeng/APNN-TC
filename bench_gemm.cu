/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
Please check example 07 and 08 for the basics of tensor op gemm kernels.  On NVIDIA Ampere
architecture, most concept still holds.  The two main differences are

1. NVIDIA Ampere architecture introduces a new series of tensor core instructions (see 
   include/cutlass/arch/mma_sm80.h) which are more efficient on Ampere.

2. NVIDIA Ampere architecture uses cp_async() to build multistage software pipeline to better hide
   latency (see include/cutlass/gemm/threadblock/mma_multistage.h)

Moreover, NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"
#include <cutlass/numeric_types.h>


#define PAD32(x)                \
(                               \
  (((x + 32 - 1)/32)*32)        \ 
)


// #define BIT_WIDTH 32
// #define BIT_WIDTH 16
#define BIT_WIDTH 8
// #define BIT_WIDTH 4
// #define BIT_WIDTH 1

#if BIT_WIDTH == 32
  typedef float input_t;
  typedef float output_t;
#elif BIT_WIDTH == 16
  typedef cutlass::half_t input_t;
  typedef cutlass::half_t output_t;
#elif BIT_WIDTH == 8
  typedef int8_t input_t;
  // typedef int32_t output_t;
  typedef int8_t output_t;
#elif BIT_WIDTH == 4
  typedef cutlass::int4b_t input_t;
  typedef int32_t output_t;
#elif BIT_WIDTH == 1
  typedef cutlass::uint1b_t input_t;
  typedef int32_t output_t;
#endif


// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = output_t;                   // <- data type of accumulator
using ElementComputeEpilogue = output_t;               // <- data type of epilogue operations
using ElementInputA = input_t;                        // <- data type of elements in input matrix A
using ElementInputB = input_t;                        // <- data type of elements in input matrix B
using ElementOutput = output_t;                       // <- data type of elements in output matrix D


// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

//-------------full precision CUDA core (PASS) --------------------
#if BIT_WIDTH == 32

using Element = float;

using Gemm = cutlass::gemm::device::Gemm<
  Element, 
  cutlass::layout::RowMajor,
  Element, 
  cutlass::layout::ColumnMajor,
  Element,
  cutlass::layout::RowMajor, 
  Element,
  cutlass::arch::OpClassSimt, 
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<32, 64, 8>,
  cutlass::gemm::GemmShape<32, 64, 8>, 
  cutlass::gemm::GemmShape<1, 1, 1>,
  cutlass::epilogue::thread::LinearCombination<
      Element, 
      1,
      Element, 
      Element>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
  4
>;


//-------------half precision Tensor core (PASS) --------------------
#elif BIT_WIDTH == 16

using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;

using Gemm = cutlass::gemm::device::Gemm<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 64>,
  cutlass::gemm::GemmShape<64, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 8>,
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    64 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2
>;

//-------------INT-8 Tensor core (PASS) --------------------
#elif BIT_WIDTH == 8

// using ElementOutput = int32_t;
// using ElementAccumulator = int32_t;
// using ElementCompute = int32_t;

// using Gemm = cutlass::gemm::device::Gemm<
//     int8_t, cutlass::layout::RowMajor, 
//     int8_t, cutlass::layout::ColumnMajor, 
//     ElementOutput, cutlass::layout::RowMajor,
//     ElementAccumulator, 
//     cutlass::arch::OpClassTensorOp, 
//     cutlass::arch::Sm80,
//     cutlass::gemm::GemmShape<64, 64, 64>,
//     cutlass::gemm::GemmShape<32, 32, 64>, 
//     cutlass::gemm::GemmShape<16, 8, 32>,
//     cutlass::epilogue::thread::LinearCombinationClamp<
//         ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
//         ElementAccumulator, ElementCompute>,
//     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

using ElementOutput = int8_t;
// using ElementAccumulator = int32_t;
using ElementCompute = float;

using Gemm = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor, 
    int8_t, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, 
    int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>, 
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::FastLinearCombinationClamp<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;


//-------------INT-4 Tensor core (PASS) --------------------
#elif BIT_WIDTH == 4

using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;

using Gemm = cutlass::gemm::device::Gemm<
  cutlass::int4b_t,
  cutlass::layout::RowMajor,
  cutlass::int4b_t,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 256, 128>,
  cutlass::gemm::GemmShape<64, 64, 128>,
  cutlass::gemm::GemmShape<8, 8, 32>,
  cutlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2
>;

//-------------INT-1 Tensor core (PASS)--------------------
#elif BIT_WIDTH == 1
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;
    const int pipe_stages = 4;

    using Gemm = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor, 
    cutlass::uint1b_t, cutlass::layout::ColumnMajor, 
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    // RTX3090 setting for block, warp, and mma shape
    cutlass::gemm::GemmShape<128, 256, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>, 
    cutlass::gemm::GemmShape<8, 8, 128>,
    // A100 setting for block, warp, and mma shape
    // cutlass::gemm::GemmShape<256, 128, 1024>,
    // cutlass::gemm::GemmShape<64, 64, 1024>, 
    // cutlass::gemm::GemmShape<16, 8, 256>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, pipe_stages, 128, 128,
    false, cutlass::arch::OpXorPopc>;
    
#endif



template <
  /// Data type of element stored within tensor (concept: NumericType)
  typename out_Element_,
  /// Defines a mapping from logical coordinate to linear memory (concept: Layout)
  typename out_Layout_
>
cutlass::TensorRef<out_Element_, out_Layout_> 
MLP_input_layer(int M, int N, int K) {

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,           // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Launch initialized CUTLASS kernel
  #define NUM_PROFILE 1 
  for(int trial = 0; trial < NUM_PROFILE; trial++) {
    status = gemm_op();
    CUTLASS_CHECK(status);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("CUTLASS-GEMM (%d-bit). M: %6d, N: %6d, K: %6d,\t Time (ms): %.2f, TOPS: %4.2f\t\n", BIT_WIDTH, M, N, K, milliseconds/NUM_PROFILE,
                                                static_cast<double>(NUM_PROFILE*(static_cast<double>(M)*N*K*2) /
                                               (milliseconds / 1000.)) / 1e12);
  return tensor_d.device_ref();
}


template <
  /// Data type of element stored within tensor (concept: NumericType)
  typename out_Element_,
  /// Defines a mapping from logical coordinate to linear memory (concept: Layout)
  typename out_Layout_
>
cutlass::TensorRef<out_Element_, out_Layout_> 
MLP_hidden_layer(int M, int N, int K, cutlass::TensorRef<out_Element_, out_Layout_>& last_tensor_ref) {

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  ElementInputA* d_a;
  ElementInputB* d_b; 
  out_Element_* d_c; 
  out_Element_* d_d; 

  cudaMalloc(&d_a, M*K*sizeof(ElementInputA)); 
  cudaMalloc(&d_b, K*N*sizeof(ElementInputB)); 
  cudaMalloc(&d_c, M*N*sizeof(out_Element_)); 
  cudaMalloc(&d_d, M*N*sizeof(out_Element_)); 
  
  auto a_tensor_ref = cutlass::TensorRef<ElementInputA,LayoutInputA>(d_a); 
  auto b_tensor_ref = cutlass::TensorRef<ElementInputB,LayoutInputB>(d_b); 
  auto c_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput>(d_c); 
  auto d_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput>(d_d); 

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     a_tensor_ref,
                                     b_tensor_ref,
                                     c_tensor_ref,
                                     d_tensor_ref,
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Launch initialized CUTLASS kernel
  #define NUM_PROFILE 1 
  for(int trial = 0; trial < NUM_PROFILE; trial++) {
    status = gemm_op();
    CUTLASS_CHECK(status);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("CUTLASS-GEMM (%d-bit). M: %6d, N: %6d, K: %6d,\t Time (ms): %.2f, TOPS: %4.2f\t\n", BIT_WIDTH, M, N, K, milliseconds/NUM_PROFILE,
                                                static_cast<double>(NUM_PROFILE*(static_cast<double>(M)*N*K*2) /
                                               (milliseconds / 1000.)) / 1e12);

  /*
  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue> gemm_device;

  // Launch device reference gemm kernel
  gemm_device(problem_size,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_c.device_ref(),
              tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << (passed ? "Passed" : "Failed") << std::endl;

  // return (passed ? 0  : -1);*/
  // return 0;
  return d_tensor_ref;
}

int main(int argc, char* argv[]) {

  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.0. 
  //
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Turing Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  const int batch_size = 32;
  std::vector<std::vector<int>> MLP_layers_config = 
    {
     {768,  1024},
     {1024, 1024},
     {1024, 1024},
     {1024, 10  }
    };

  auto out = MLP_input_layer<ElementOutput, LayoutOutput>(batch_size, PAD32(MLP_layers_config[0][1]), PAD32(MLP_layers_config[0][0]));
  for (int i = 1; i < MLP_layers_config.size(); i++){
      out = MLP_hidden_layer<ElementOutput, LayoutOutput>(batch_size, PAD32(MLP_layers_config[i][1]), PAD32(MLP_layers_config[i][0]), out);
  }

  return 0;
}
