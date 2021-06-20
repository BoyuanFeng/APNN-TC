#ifndef layer_h
#define layer_h

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include <cutlass/numeric_types.h>

#include <ctime>
#include <cudnn.h>
#include "helper.h"
#include "gemm.cuh"
#include "config.h"

class CONV{
public:
    CONV(int batch_size, int input_height, int input_width, 
        int in_channels, int out_channels,
        int filter_height, int filter_width, int stride, int padding){
        
        _batch_size = batch_size;
        _input_height = input_height;
        _input_width = input_width;     
        
        _in_channels = in_channels;
        _out_channels = out_channels;

        _filter_height = filter_height;
        _filter_width = filter_width;

        _stride = stride;
        _padding_h = padding;
        _padding_w = padding;

        // compute the output shape.
        _output_height = (_input_height + 2*_padding_h - filter_height)/_stride + 1;
        _output_width = (_input_width + 2*_padding_w - filter_width)/_stride + 1;
        
        init();

        printf("Init CONV (n,c,h,w): %d,%d,%d,%d\n", _batch_size, _in_channels, _input_height, _input_width);
        printf("CONV output (h, w): %d,%d\n", _output_height, _output_width);
    }

    ~CONV(){
        cudaFree(filter);
        cudaFree(output);
    }

    void init()
    {
        mode = cutlass::conv::Mode::kCrossCorrelation;
        // problem_size(_batch_size, _in_channels, _out_channels);
        // allocate memory for weight.
        cudaMalloc(&filter, _in_channels*_out_channels*_filter_height*_filter_width*sizeof(ElementInputB)); 
        // allocate memory for output.
        cudaMalloc(&output, _batch_size*_out_channels*_output_height*_output_width*sizeof(ElementOutput)); 
        // update with tensor reference.
        filter_tensor_ref = cutlass::TensorRef<ElementInputB,LayoutInputB>(filter); 
        output_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput>(output); 
        // Initialize alpha and beta for dot product computation.
        alpha = ElementComputeEpilogue(1);
        beta = ElementComputeEpilogue(0);

        input_size = cutlass::Tensor4DCoord(_batch_size, _input_height, _input_width, _in_channels); // n, h, w, c
        filter_size = cutlass::Tensor4DCoord(_out_channels, _filter_height, _filter_width, _in_channels); // o, k, k, c
        padding = cutlass::Tensor4DCoord(1, _padding_h, _padding_w, 1); // n, h, w, c
        conv_stride = cutlass::MatrixCoord(_stride, _stride);
        dilation = cutlass::MatrixCoord(1,1);
        output_size = cutlass::Tensor4DCoord(_batch_size, _output_height, _output_width, _out_channels);
    }

    ElementInputA* forward(ElementInputA* input){

        input_tensor_ref = cutlass::TensorRef<ElementInputA, LayoutInputA>(input); 

        // runnking kernel.
        typename ImplicitGemm::Arguments arguments{
            {
              input_size,
              filter_size,
              padding,
              conv_stride,
              dilation,
              output_size,
              mode,
              split_k_slices 
            },
            input_tensor_ref,
            filter_tensor_ref,
            output_tensor_ref,
            output_tensor_ref,
            {alpha, beta},    
          };
        //
        // Initialize CUTLASS Convolution
        //
        // implicit_gemm_op;
        workspace_size = implicit_gemm_op.get_workspace_size(arguments);

        // Allocate workspace memory
        workspace = cutlass::device_memory::allocation<uint8_t>(workspace_size);
        CUTLASS_CHECK(implicit_gemm_op.initialize(arguments, workspace.get()));

        #define N 1
        //
        // Launch initialized CUTLASS kernel
        //
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int i = 0; i < N; i++)
            implicit_gemm_op();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Forward CONV (ms): %.3f, TFLOPs: %.3f\n", 
                milliseconds/N, 
                2.0f*_batch_size*_out_channels*_output_height* _output_width*_filter_height*_filter_width*_in_channels/(milliseconds/N/1e3)/1e12);

        return output;
    }

    int get_output_height(){ return _output_height;}
    int get_output_width(){ return _output_width; }
    int get_out_channels(){ return _out_channels; }

private:
    int _in_channels;
    int _out_channels;
    int _input_height;
    int _input_width;
    int _filter_height;
    int _filter_width;
    int _stride;
    int _batch_size;

    int _output_height;
    int _output_width;

    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord output_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;

    ImplicitGemm implicit_gemm_op;
    size_t workspace_size;
    cutlass::device_memory::allocation<uint8_t> workspace;
    cutlass::conv::Mode mode; 
    cutlass::TensorRef<ElementInputA, LayoutInputA> input_tensor_ref;
    cutlass::TensorRef<ElementInputB, LayoutInputB> filter_tensor_ref;
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_tensor_ref;
    ElementComputeEpilogue alpha, beta;

    int split_k_slices = 1;     //-> Split K dimension into 1 partitions
    int _padding_w = 1;
    int _padding_h = 1;

    ElementInputA* input;
    ElementInputB* filter;
    ElementOutput* output;
};

class FC{
public:
    FC(int batch_size, int in_channels, int out_channels){

        _batch_size = batch_size;
        _in_channels = in_channels;
        _out_channels = out_channels;

        init();

        printf("Init FC (n,in,out): %d,%d,%d\n", _batch_size, _in_channels, _out_channels);
    }

    ~FC(){
        cudaFree(weight);
        cudaFree(output);
    }

    void init()
    {   
        problem_size = cutlass::gemm::GemmCoord(_batch_size, _in_channels, _out_channels);
        // allocate memory for weight.
        cudaMalloc(&weight, _in_channels*_out_channels*sizeof(ElementInputB)); 
        // allocate memory for output.
        cudaMalloc(&output, _batch_size*_out_channels*sizeof(ElementOutput)); 
        // update with tensor reference.
        weight_tensor_ref = cutlass::TensorRef<ElementInputB,LayoutInputB_gemm>(weight); 
        output_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput_gemm>(output); 
        // Initialize alpha and beta for dot product computation.
        alpha = ElementComputeEpilogue_gemm(1);
        beta = ElementComputeEpilogue_gemm(0);
    }


    ElementInputA* forward(ElementInputA* input){

        input_tensor_ref = cutlass::TensorRef<ElementInputA, LayoutInputA_gemm>(input); 
        // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
        // instantiated CUTLASS kernel
        typename Gemm::Arguments arguments{
            problem_size,  // <- problem size of matrix multiplication
            input_tensor_ref,
            weight_tensor_ref,
            output_tensor_ref,
            output_tensor_ref,
            {alpha, beta},          // <- tuple of alpha and beta
            split_k_slices};        // <- k-dimension split factor

        // Using the arguments, query for extra workspace required for matrix multiplication computation
        workspace_size = Gemm::get_workspace_size(arguments);
        // Allocate workspace memory
        workspace = cutlass::device_memory::allocation<uint8_t>(workspace_size);
        // Instantiate CUTLASS kernel depending on templates
        // cutlass::Status status = 
        CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        CUTLASS_CHECK(gemm_op());
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Forward FC (ms): %.3f\n", milliseconds);

        return output;
    }

    int get_out_channels(){ return _out_channels; }
private:
    int _batch_size;
    int _in_channels;
    int _out_channels;
    
    int split_k_slices = 1;         // <-- Split K dimension into 1 partitions

    cutlass::gemm::GemmCoord problem_size;
    cutlass::TensorRef<ElementInputA, LayoutInputA_gemm> input_tensor_ref;
    cutlass::TensorRef<ElementInputB, LayoutInputB_gemm> weight_tensor_ref;
    cutlass::TensorRef<ElementOutput, LayoutOutput_gemm> output_tensor_ref;
    ElementComputeEpilogue_gemm alpha;
    ElementComputeEpilogue_gemm beta;
    cutlass::device_memory::allocation<uint8_t> workspace;
    size_t workspace_size;
    Gemm gemm_op;

    ElementInputA* input;
    ElementInputB* weight;
    ElementOutput* output;
};


//
// Pooling Layer kernel_size = 3, stride = 2, padding = 0.
//
class POOL{

    public:
        POOL(int batch_size, int in_channels, int input_height, 
            int input_width, cudnnHandle_t* cuDNN_handler, 
            int kernel=3, int stride=2, int padding=0)
    {

        cudnn = cuDNN_handler;
        _kernel_size = kernel;
        _padding = padding;
        _stride = stride;
        _batch_size = batch_size;
        _in_channels = in_channels;
        _input_height = input_height;
        _input_width = input_width;     

        _output_height = (_input_height - _kernel_size)/stride+1;
        _output_width = (_input_width - _kernel_size)/stride+1;

        output_bytes = _batch_size*_in_channels*_output_height*_output_width*sizeof(cuDNNtype);

        _init();

        printf("Init POOL (n,c,h,w): %d,%d,%d,%d\n",_batch_size, _in_channels, _input_height, _input_width);
        printf("POOL output (h, w): %d,%d\n", _output_height, _output_width);
    }

    ~POOL(){
        cudaFree(output);
        cudaFree(filter);

    }

    void _init() {

        checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));        
        checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                                CUDNN_POOLING_MAX,       //mode - max pooling
                                                CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                                _kernel_size,                       //window height
                                                _kernel_size,                       //window width
                                                _padding,                              //vertical padding
                                                _padding,                               //horizontal padding
                                                _stride,                       //vertical stride
                                                _stride));                     //horizontal stride
                                                
        checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));

        checkCUDNN(cudnnSetTensor4dDescriptor(
                    input_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width));            
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    output_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _output_height,                
                    _output_width));  

        cudaMalloc(&output, output_bytes);
    }

    cuDNNtype* forward(cuDNNtype* input){

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        checkCUDNN(cudnnPoolingForward(*cudnn,         
                                        pooling_desc,  
                                        &alpha,       
                                        input_desc,
                                        input,
                                        &beta,        
                                        output_desc,   
                                        output));   

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Forward Pooling (ms): %.3f\n", milliseconds);

        return output;
    }

    int get_output_height(){ return _output_height;}
    int get_output_width(){ return _output_width; }
    int get_out_channels(){ return _in_channels; }
private:
    cudnnHandle_t* cudnn;
    cudnnPoolingDescriptor_t pooling_desc;
    cudnnTensorDescriptor_t input_desc, output_desc;

    float alpha = 1.0f;
    float beta = 0.0f;

    int _batch_size;
    int _input_height, _input_width;
    int _output_height, _output_width;
    int _in_channels;
    int _kernel_size;
    int _stride;
    int _padding;

    int output_bytes;

    cuDNNtype* input;
    cuDNNtype* filter;
    cuDNNtype* output;
};

//
// RELU Layer
//
class RELU{
public:
    RELU(int batch_size, int in_channels, int input_height, int input_width, cudnnHandle_t* cuDNN_handler)
    {

        cudnn = cuDNN_handler;
        _batch_size = batch_size;
        _in_channels = in_channels;
        _input_height = input_height;
        _input_width = input_width;     

        output_bytes = _batch_size*_in_channels*_input_height*_input_width*sizeof(cuDNNtype);

        _init();
        printf("Init ReLU (n,c,h,w): %d,%d,%d,%d\n",_batch_size, _in_channels, _input_height, _input_width);
    }

    ~RELU(){
        cudaFree(output);
    }

    void _init() {

        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc));
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                    CUDNN_ACTIVATION_RELU,
                    CUDNN_PROPAGATE_NAN,
                    0.0) );
        
        checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));

        checkCUDNN(cudnnSetTensor4dDescriptor(
                    input_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width));            
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    output_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width));  

        cudaMalloc(&output, output_bytes);
    }

    cuDNNtype* forward(cuDNNtype* input){

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        checkCUDNN( cudnnActivationForward(
                    *cudnn,
                    activDesc,
                    &alpha,
                    input_desc,
                    input,
                    &beta,
                    output_desc,
                    output) );   


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
      
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Forward ReLU (ms): %.3f\n", milliseconds);
        
        return output;
    }

    int get_output_height(){ return _input_height;}
    int get_output_width(){ return _input_width; }
    int get_out_channels(){ return _in_channels; }

private:
    cudnnHandle_t* cudnn;

    cudnnActivationDescriptor_t  activDesc;
    cudnnTensorDescriptor_t input_desc, output_desc;

    float alpha = 1.0f;
    float beta = 0.0f;

    int _batch_size;

    int _input_height;
    int _input_width;

    int _in_channels;
    int _out_channels;

    int output_bytes;

    cuDNNtype* input;
    cuDNNtype* output;

};


// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormalizationForwardInference

//
// BN Layer
//
class BN{
public:
    BN(int batch_size, int in_channels, int input_height, int input_width, 
        cudnnHandle_t* cuDNN_handler, bool residual=false, cuDNNtype*residual_tensor=NULL)
    {

        cudnn = cuDNN_handler;
        _batch_size = batch_size;
        _in_channels = in_channels;
        _input_height = input_height;
        _input_width = input_width;    
        _residual = residual;
        _residual_tensor = residual_tensor;

        output_bytes = _batch_size*_in_channels*_input_height*_input_width*sizeof(cuDNNtype);

        _init();
        printf("Init BN (n,c,h,w): %d,%d,%d,%d\n",_batch_size, _in_channels, _input_height, _input_width);
    }

    ~BN(){
        cudaFree(output);
    }

    void _init() {

        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc));
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );
        
        checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(
                    input_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width));            
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    output_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width));  

        checkCUDNN(cudnnSetTensor4dDescriptor(
                    bnScaleBiasMeanVarDesc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    1,                       
                    _in_channels,                      
                    1,                
                    1)); 
                    
                    
        cudaMalloc(&bnScale,            _in_channels*sizeof(float));
        cudaMalloc(&bnBias,             _in_channels*sizeof(float));
        cudaMalloc(&estimatedMean,      _in_channels*sizeof(float));
        cudaMalloc(&estimatedVariance,  _in_channels*sizeof(float));

        cudaMalloc(&output, output_bytes);
    }

    cuDNNtype* forward(cuDNNtype* input){

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // checkCUDNN( cudnnBatchNormalizationForwardInference(
        //                                     *cudnn,
        //                                     CUDNN_BATCHNORM_SPATIAL,
        //                                     &alpha,
        //                                     &beta,
        //                                     input_desc,
        //                                     input,
        //                                     output_desc,
        //                                     output,
        //                                     bnScaleBiasMeanVarDesc,
        //                                     bnScale,
        //                                     bnBias,
        //                                     estimatedMean,
        //                                     estimatedVariance,
        //                                     epsilon));   

        if (_residual == false)
            checkCUDNN( cudnnNormalizationForwardInference(
                                                *cudnn,
                                                CUDNN_NORM_PER_CHANNEL,
                                                CUDNN_NORM_OPS_NORM, //CUDNN_NORM_OPS_NORM_ADD_ACTIVATION
                                                CUDNN_NORM_ALGO_STANDARD,
                                                &alpha,
                                                &beta, 
                                                input_desc,
                                                input,
                                                bnScaleBiasMeanVarDesc,
                                                bnScale,
                                                bnBias,
                                                bnScaleBiasMeanVarDesc,
                                                estimatedMean,
                                                estimatedVariance,
                                                NULL,
                                                NULL,
                                                NULL,
                                                output_desc,
                                                output,
                                                epsilon,
                                                1));

            if  (_residual)
            checkCUDNN( cudnnNormalizationForwardInference(
                                                        *cudnn,
                                                        CUDNN_NORM_PER_CHANNEL,
                                                        CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, //CUDNN_NORM_OPS_NORM_ADD_ACTIVATION
                                                        CUDNN_NORM_ALGO_STANDARD,
                                                        &alpha,
                                                        &beta, 
                                                        input_desc,
                                                        input,
                                                        bnScaleBiasMeanVarDesc,
                                                        bnScale,
                                                        bnBias,
                                                        bnScaleBiasMeanVarDesc,
                                                        estimatedMean,
                                                        estimatedVariance,
                                                        output_desc,
                                                         _residual_tensor,
                                                        activDesc,
                                                        output_desc,
                                                        output,
                                                        epsilon,
                                                        1));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
      
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Forward BN (ms): %.3f\n", milliseconds);
        
        return output;
    }

    int get_output_height(){ return _input_height;}
    int get_output_width(){ return _input_width; }
    int get_out_channels(){ return _in_channels; }

private:
    cudnnHandle_t* cudnn;
    cudnnTensorDescriptor_t input_desc, output_desc, bnScaleBiasMeanVarDesc;
    cudnnActivationDescriptor_t  activDesc;

    float alpha = 1.0f;
    float beta = 0.0f;
    double epsilon = 0.001;
    float *bnScale;
    float *bnBias;
    float *estimatedMean;
    float *estimatedVariance;
    bool _residual;

    int _batch_size;
    int _in_channels;
    int _input_height;
    int _input_width;

    int output_bytes;

    cuDNNtype* input;
    cuDNNtype* output;
    cuDNNtype* _residual_tensor;
};
#endif