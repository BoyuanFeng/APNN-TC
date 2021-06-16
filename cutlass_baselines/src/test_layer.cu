
#include <stdio.h>
#include "layer.cuh"
#include "config.h"

int main(int argc, char*argv[]){

    float *in_data, *out;
    int batch_size = 10;
    int in_channels = 16;
    int out_channels = 16;
    int input_height = 9;
    int input_width = 9;
    int filter_height = 3;
    int filter_width = 3;
    int stride = 1;

    int in_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
    cudaMalloc(&in_data, in_bytes);

    // auto layer_conv  = new CONV(batch_size, input_height, input_width, in_channels, out_channels, filter_height, filter_width, stride);
    // out = layer_conv->forward(in_data);

    auto layer_fc  = new FC(batch_size, in_channels*input_height*input_width, out_channels);
    out = layer_fc->forward(in_data);

    // set the pooling layer for evaluation.
    // cudnnHandle_t cudnn;
    // checkCUDNN(cudnnCreate(&cudnn));

    // auto layer_pool = new POOL(batch_size, in_channels, input_height, input_width, &cudnn);
    // auto layer_relu = new RELU(batch_size, in_channels, layer_pool->get_output_height(), layer_pool->get_output_width(), &cudnn);

    // out = layer_pool->forward(in_data);
    // out = layer_relu->forward(in_data);


    return 0;
}