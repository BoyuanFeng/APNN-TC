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
        // _output_height = ;
        // _output_width = ;
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
        // compute the output shape.
        // _output_height = ;
        // _output_width = ;
    }

    void init(float*input_gpu)
    {
        this->input = input_gpu;
        // allocate memory for weight.
        // allocate memory for output

    }

    float* forward(){

        // runnking kernel.
        // kernel(output, this->input);
        return output;
    }

    ~FC(){

    }
private:
    int batch_size;
    int _in_channels;
    int _out_channels;

    float* output;
    float* filter;
    float* input;
};


class RELU{
public:
    RELU(int batch_size, int input_height, int input_height, int in_channels){
        _batch_size = batch_size;

        _input_height = input_height;
        _input_width = input_height;     
        
        _in_channels = in_channels;

        output_bytes = _batch_size*_in_channels*_input_height*_input_width;

        _init();
    }

    ~POOL(){

    }

    void _init() {

        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc));
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                    CUDNN_ACTIVATION_RELU,
                    CUDNN_PROPAGATE_NAN,
                    0.0) );
        
        checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));

        checkCUDNN(cudnnSetTensor4dDescriptor(input_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width);            
        checkCUDNN(cudnnSetTensor4dDescriptor(output_desc,             
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
        checkCUDNN( cudnnActivationForward(cudnn,
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