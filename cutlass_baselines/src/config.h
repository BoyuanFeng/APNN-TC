#ifndef config_H
#define config_H

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#define BIT_WIDTH 32
// #define BIT_WIDTH 16
// #define BIT_WIDTH 8

#if BIT_WIDTH == 32
  typedef float input_t;
  typedef float output_t;
  #define CUDNN_DTYPE CUDNN_DATA_FLOAT
  typedef float cuDNNtype;
#elif BIT_WIDTH == 16
  typedef cutlass::half_t input_t;
  typedef cutlass::half_t output_t;
  #define CUDNN_DTYPE CUDNN_DATA_HALF
  // typedef __half cuDNNtype;
  typedef cutlass::half_t cuDNNtype;
#elif BIT_WIDTH == 8
  typedef int8_t input_t;
  typedef int8_t output_t;
  #define CUDNN_DTYPE CUDNN_DATA_INT8
  typedef int8_t cuDNNtype;
#endif

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator_gemm = output_t;                   // <- data type of accumulator
using ElementComputeEpilogue_gemm = output_t;               // <- data type of epilogue operations
using ElementInputA = input_t;                        // <- data type of elements in input matrix A
using ElementInputB = input_t;                        // <- data type of elements in input matrix B
using ElementOutput = output_t;                       // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA_gemm = cutlass::layout::RowMajor;
using LayoutInputB_gemm = cutlass::layout::ColumnMajor;
using LayoutOutput_gemm = cutlass::layout::RowMajor;

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
using ElementAccumulator_gemm = cutlass::half_t;

using Gemm = cutlass::gemm::device::Gemm<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  ElementAccumulator_gemm,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 64>,
  cutlass::gemm::GemmShape<64, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 8>,
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    64 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator_gemm,
    ElementAccumulator_gemm
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2
>;

//-------------INT-8 Tensor core (PASS) --------------------
#elif BIT_WIDTH == 8

using ElementOutput = int8_t;
// using ElementAccumulator_gemm = int32_t;
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

using ElementCompute = float;

using Gemm = cutlass::gemm::device::Gemm<
  cutlass::int4b_t,
  cutlass::layout::RowMajor,
  cutlass::int4b_t,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  int32_t,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 128, 128>,
  cutlass::gemm::GemmShape<32, 64, 128>,
  cutlass::gemm::GemmShape<8, 8, 32>,
  cutlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput,
    64 / cutlass::sizeof_bits<ElementOutput>::value,
    int32_t,
    ElementCompute
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2
>;

//-------------INT-1 Tensor core (PASS)--------------------
#elif BIT_WIDTH == 1
    using ElementOutput = int32_t;
    using ElementAccumulator_gemm = int32_t;
    using ElementCompute = int32_t;
    const int pipe_stages = 4;

    using Gemm = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor, 
    cutlass::uint1b_t, cutlass::layout::ColumnMajor, 
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator_gemm, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    // RTX3090 setting for block, warp, and mma shape
    cutlass::gemm::GemmShape<128, 256, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>, 
    cutlass::gemm::GemmShape<8, 8, 128>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator_gemm, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, pipe_stages, 128, 128,
    false, cutlass::arch::OpXorPopc>;
#endif


#if BIT_WIDTH == 32
using ElementInputA           = float;
using ElementInputB           = float;
// using ElementOutput           = float;
using ElementAccumulator      = float;
using ElementCompute          = float;
using ElementComputeEpilogue = ElementAccumulator;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, 
    cutlass::layout::TensorNHWC,
    ElementInputB, 
    cutlass::layout::TensorNHWC,
    ElementOutput, 
    cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;
#endif 


#if BIT_WIDTH == 16
using ElementInputA           = cutlass::half_t;
using ElementInputB           = cutlass::half_t;
using ElementAccumulator      = float;
using ElementCompute          = float;
using ElementComputeEpilogue = ElementAccumulator;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;  // Threadblock tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;          // Warp tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // TensorCore instruction shape

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 3;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kAnalytic;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<ElementOutput,
                    128 / cutlass::sizeof_bits<ElementOutput>::value, float, float>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;
#endif

#if BIT_WIDTH == 8

using ElementInputA           = int8_t;
using ElementInputB           = int8_t;
using ElementAccumulator = int32_t;                   // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;    // Data type of epilogue computation (alpha, beta)

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;  // Threadblock tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;          // Warp tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;    // TensorCore instruction shape
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 3;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kAnalytic;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<ElementOutput,
                                                64 / cutlass::sizeof_bits<ElementOutput>::value,
                                                int32_t,
                                                float>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;
#endif // END if BIT_WIDTH == 8

#if BIT_WIDTH == 4
using ElementInputA = cutlass::int4b_t;              // Data type of elements in input tensor
using ElementInputB = cutlass::int4b_t;              // Data type of elements in input tensor
using ElementOutput = cutlass::int4b_t;              // Data type of elements in output tensor
using ElementAccumulator = int32_t;                   // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;    // Data type of epilogue computation (alpha, beta)

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;    // Threadblock tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;             // Warp tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;       // TensorCore instruction shape
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 3;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kAnalytic;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<ElementOutput,
                                                64 / cutlass::sizeof_bits<ElementOutput>::value,
                                                int32_t,
                                                float>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;
#endif  // END if BIT_WIDTH == 4


#if BIT_WIDTH == 1
using ElementInputA = cutlass::uint1b_t;              // Data type of elements in input tensor
using ElementInputB = cutlass::uint1b_t;              // Data type of elements in input tensor
using ElementOutput = int32_t;                        // Data type of elements in output tensor
using ElementAccumulator = int32_t;                   // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;    // Data type of epilogue computation (alpha, beta)

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 1024>;  // Threadblock tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 1024>;         // Warp tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;    // TensorCore instruction shape
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 2;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kAnalytic;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<ElementOutput,
                                                128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                int32_t,
                                                float>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;
#endif  // END if BIT_WIDTH == 1


using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

#define PAD32(x)                \
(                               \
  (((x + 32 - 1)/32)*32)        \ 
)

#define checkCUDNN(exp) \
  { \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error on line " << __LINE__ << ": " \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } 
  

// #define IN_DATA_BYTES (IN_SIZE*sizeof(dtype))
// #define OUT_DATA_BYTES (OUT_SIZE*sizeof(dtype))

// #define IN_SIZE (2*2*10*10)
// #define OUT_SIZE (2*2*8*8)
// #define TOL (0.000005)




#endif // config_H