#ifndef layer_h
#define layer_h


#include <cudnn.h>

class CONV{
public:
    CONV(int batch_size, int input_height, int input_height, 
        int in_channels, int out_channels,
        int filter_height, int filter_width){

        _input_height = input_height;
        _input_width = input_height;     
        _in_channels = in_channels;
        _out_channels = out_channels;
        _filter_height = filter_height;
        _filter_width = filter_width;

        // compute the output shape.
        _output_height = ;
        _output_width = ;
    }

    ~CONV(){

    }

    void init(float* input_gpu)
    {
        this->input = input_gpu;
        // allocate memory for filter.
        // allocate memory for output
    }

    float* forward(){
        // runnking kernel.
        // kernel(output, this->input);
        return output;
    }


private:
    int _in_channels;
    int _out_channels;
    int _input_height;
    int _input_width;
    int _filter_height;
    int _filter_width;
    
    int _output_height;
    int _output_width;

    float* output;
    float* filter;
    float* input;
};

class FC{
public:
    FC(int batch_size, int in_channels, int out_channels){

        _batch_size = batch_size;
        _in_channels = in_channels;
        _out_channels = out_channels;
    }

    ~FC(){
        cudaFree(weight);
        cudaFree(output);
    }

    void init(float*input_gpu)
    {   
        problem_size(_batch_size, _in_channels, _out_channels);
        // allocate memory for weight.
        cudaMalloc(&weight, in_channels*out_channels*sizeof(float)); 
        // allocate memory for output
        cudaMalloc(&output, _batch_size*out_channels*sizeof(float)); 

        weight_tensor_ref = cutlass::TensorRef<ElementInputB,LayoutInputB_gemm>(weight); 
        output_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput_gemm>(output); 

        // Initialize alpha and beta for dot product computation
        alpha = ElementComputeEpilogue_gemm(1);
        beta = ElementComputeEpilogue_gemm(0);
    }


    float* forward(float* input){

        input_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput_gemm>(input); 

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
        size_t workspace_size = Gemm::get_workspace_size(arguments);
        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        // Instantiate CUTLASS kernel depending on templates
        Gemm gemm_op;

        // Initialize CUTLASS kernel with arguments and workspace pointer
        cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);

        status = gemm_op();
        CUTLASS_CHECK(status);

        return output;
    }

private:
    int batch_size;
    int _in_channels;
    int _out_channels;
    
    int split_k_slices = 1;         // <-- Split K dimension into 1 partitions

    cutlass::gemm::GemmCoord problem_size;

    cutlass::TensorRef<ElementInputB,LayoutInputB_gemm> input_tensor_ref;
    cutlass::TensorRef<ElementInputB,LayoutInputB_gemm> weight_tensor_ref;
    cutlass::TensorRef<ElementOutput,LayoutOutput_gemm> output_tensor_ref;
    
    ElementComputeEpilogue_gemm alpha;
    ElementComputeEpilogue_gemm beta;

    float* output;
    float* weight;
    float* input;
};


class POOL{
public:
    POOL(int batch_size, int input_height, int input_height, 
        int in_channels, cudnnHandle_t* cuDNN_handler){

        cudnn = cuDNN_handler;
        _batch_size = batch_size;

        _input_height = input_height;
        _input_width = input_height;     
        
        _in_channels = in_channels;

        _output_height = (_input_height - 3)/2+1;
        _output_width = (_input_width - 3)/2+1;

        output_bytes = _batch_size*_in_channels*_output_height*_output_width*sizeof(float);

        _init();
    }

    ~POOL(){
        cudaFree(cudaMalloc);
    }

    void _init() {

        checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));        
        checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                                CUDNN_POOLING_MAX,       //mode - max pooling
                                                CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                                3,                       //window height
                                                3,                       //window width
                                                0,                       //vertical padding
                                                0,                       //horizontal padding
                                                2,                       //vertical stride
                                                2));                     //horizontal stride
                                                
        checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));

        checkCUDNN(cudnnSetTensor4dDescriptor(
                    input_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width);            
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    output_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width);  

        cudaMalloc(&output, output_bytes);
    }

    float* forward(float* input){
        //Call pooling operator
        checkCUDNN(cudnnPoolingForward(*cudnn,         
                                        pooling_desc,  
                                        &alpha,       
                                        input_desc,
                                        input,
                                        &beta,        
                                        output_desc,   
                                        output));   
        return output;
    }


private:
    cudnnHandle_t* cudnn;
    // checkCUDNN(cudnnCreate(&cudnn));

    cudnnPoolingDescriptor_t pooling_desc;
    cudnnTensorDescriptor_t input_desc, output_desc;

    float alpha = 1.0f;
    float beta = 0.0f;

    int _batch_size;

    int _input_height;
    int _input_width;

    int _in_channels;
    int _out_channels;

    int output_bytes;

    float* input;
    float* filter;
    float* output;

};


class RELU{
public:
    RELU(int batch_size, int input_height, int input_height, 
        int in_channels, cudnnHandle_t* cuDNN_handler){

        cudnn = cuDNN_handler;
        _batch_size = batch_size;

        _input_height = input_height;
        _input_width = input_height;     
        
        _in_channels = in_channels;

        output_bytes = _batch_size*_in_channels*_input_height*_input_width;

        _init();
    }

    ~POOL(){
        cudaFree(cudaMalloc);
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
                    _input_width);            
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    output_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width);  

        cudaMalloc(&output, output_bytes);
    }

    float* forward(float* input){
        // Runnking kernel.
        checkCUDNN( cudnnActivationForward(
                    *cudnn,
                    activDesc,
                    &alpha,
                    input_desc,
                    input,
                    &beta,
                    output_desc,
                    output) );   
        return output;
    }


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

    float* input;
    float* filter;
    float* output;

};

#endif