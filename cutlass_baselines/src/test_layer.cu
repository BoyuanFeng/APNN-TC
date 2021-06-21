
#include <stdio.h>
#include <ctime>
#include "layer.cuh"
#include "config.h"

int main(int argc, char*argv[]){

    ElementInputA* in_data;
    ElementOutput* out;

    int batch_size = 8;
    int in_channels = 32;
    int input_height = 224;
    int input_width = 224;

    int out_channels = 16;
    int filter_height = 3;
    int filter_width = 3;
    int stride = 1;

    // for CUTLASS test
    // int in_bytes = batch_size * in_channels * input_height * input_width * sizeof(ElementInputA);
    // cudaMalloc(&in_data, in_bytes);
    // auto conv  = new CONV(batch_size, input_height, input_width, in_channels, out_channels, filter_height, filter_width, stride, 1);
    // out = conv->forward(in_data);
    // auto fc  = new FC(batch_size, in_channels*input_height*input_width, out_channels);
    // out = fc->forward(in_data);


    // for cuDNN test.
    // set the pooling layer for evaluation.
    
    int in_bytes = batch_size * in_channels * input_height * input_width * sizeof(cuDNNtype);
    cudaMalloc(&in_data, in_bytes);
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    auto bn = new BN(batch_size, in_channels, input_height, input_width, &cudnn);
    // auto pool = new POOL(batch_size, in_channels, input_height, input_width, &cudnn);
    // auto relu = new RELU(batch_size, in_channels, pool->get_output_height(), pool->get_output_width(), &cudnn);

    out = bn->forward(in_data);
    // out = pool->forward(in_data);
    // out = relu->forward(in_data);

    return 0;
}