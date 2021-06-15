
#include <stdio.h>
#include "layer.cuh"
#include "config.h"

int main(int argc, char*argv[]){

    float *in_data, *out;
    int bathsize = 10;
    int channel = 16;
    int input_height = 9;
    int input_width = 9;

    int in_bytes = bathsize * channel * input_height * input_width * sizeof(float);
    cudaMalloc(&in_data, in_bytes);


    // set the pooling layer for evaluation.
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    auto layer_pool = new POOL(bathsize, channel, input_height, input_width, &cudnn);
    // auto layer_relu = new RELU(bathsize, channel, layer_pool->get_output_height(), layer_pool->get_output_width(), &cudnn);

    out = layer_pool->forward(in_data);
    // out = layer_relu->forward(in_data);


    return 0;
}