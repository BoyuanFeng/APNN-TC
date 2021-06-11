
#ifndef GEMM_CUH
#define GEMM_CUH

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include <cutlass/numeric_types.h>

#include "helper.h"
#include "config.h"

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
  cutlass::HostTensor<ElementInputA, LayoutInputA_gemm> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB_gemm> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput_gemm> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput_gemm> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput_gemm> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue_gemm alpha = ElementComputeEpilogue_gemm(1);
  ElementComputeEpilogue_gemm beta = ElementComputeEpilogue_gemm(0);

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
  
  // auto a_tensor_ref = cutlass::TensorRef<ElementInputA,LayoutInputA_gemm>(d_a); 
  auto b_tensor_ref = cutlass::TensorRef<ElementInputB,LayoutInputB_gemm>(d_b); 
  auto c_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput_gemm>(d_c); 
  auto d_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput_gemm>(d_d); 

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue_gemm alpha = ElementComputeEpilogue_gemm(1);
  ElementComputeEpilogue_gemm beta = ElementComputeEpilogue_gemm(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     last_tensor_ref,
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
                                   LayoutInputA_gemm,
                                   ElementInputB,
                                   LayoutInputB_gemm,
                                   ElementOutput,
                                   LayoutOutput_gemm,
                                   ElementComputeEpilogue_gemm,
                                   ElementComputeEpilogue_gemm> gemm_device;

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

#endif // GEMM_CUH