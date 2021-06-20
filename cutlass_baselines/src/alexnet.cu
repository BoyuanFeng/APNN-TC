
#include <stdio.h>
#include <ctime>
#include "layer.cuh"
#include "config.h"


int main(int argc, char*argv[]){

        ElementInputA* in_data;
        cuDNNtype* out;
    
        int batch_size = 8;
        int in_channels = 32;
        int input_height = 224;
        int input_width = 224;
        int num_classes = 1000;
    
        cudnnHandle_t cudnn;
        checkCUDNN(cudnnCreate(&cudnn));

        // for CUTLASS test
        int in_bytes = batch_size * in_channels * input_height * input_width * sizeof(ElementInputA);
        cudaMalloc(&in_data, in_bytes);
    
        auto conv_1  = new CONV(batch_size, input_height, input_width, in_channels, 64, 11, 11, 4, 2);
        auto relu_1  = new RELU(batch_size, conv_1->get_out_channels(), conv_1->get_output_height(), conv_1->get_output_width(), &cudnn);
        auto pool_1  = new POOL(batch_size, relu_1->get_out_channels(), relu_1->get_output_height(), relu_1->get_output_width(), &cudnn);

        auto conv_2  = new CONV(batch_size, pool_1->get_output_height(), pool_1->get_output_width(), 64, 192, 5, 5, 1, 2);
        auto relu_2  = new RELU(batch_size, conv_2->get_out_channels(), conv_2->get_output_height(), conv_2->get_output_width(), &cudnn);
        auto pool_2  = new POOL(batch_size, relu_2->get_out_channels(), relu_2->get_output_height(), relu_2->get_output_width(), &cudnn);

        auto conv_3  = new CONV(batch_size, pool_2->get_output_height(), pool_2->get_output_width(), 192, 384, 3, 3, 1, 1);
        auto relu_3  = new RELU(batch_size, conv_3->get_out_channels(), conv_3->get_output_height(), conv_3->get_output_width(), &cudnn);

        auto conv_4  = new CONV(batch_size, relu_3->get_output_height(), relu_3->get_output_width(), 384, 256, 3, 3, 1, 1);
        auto relu_4  = new RELU(batch_size, conv_4->get_out_channels(), conv_4->get_output_height(), conv_4->get_output_width(), &cudnn);
        
        auto conv_5  = new CONV(batch_size, relu_4->get_output_height(), relu_4->get_output_width(), 256, 256, 3, 3, 1, 1);
        auto relu_5  = new RELU(batch_size, conv_5->get_out_channels(), conv_5->get_output_height(), conv_5->get_output_width(), &cudnn);
        auto pool_5  = new POOL(batch_size, relu_5->get_out_channels(), relu_5->get_output_height(), relu_5->get_output_width(), &cudnn);

        auto fc_1  = new FC(batch_size, relu_5->get_out_channels()*relu_5->get_output_height()*relu_5->get_output_width(), 4096);
        auto fc_2  = new FC(batch_size, 4096, 4096);
        auto fc_3  = new FC(batch_size, 4096, num_classes);


        printf("=============================\n");
        // std::clock_t c_start = std::clock();

        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start);

        out = conv_1->forward(in_data);
        out = relu_1->forward(out);
        out = pool_1->forward(out);

        out = conv_2->forward(out);
        out = relu_2->forward(out);
        out = pool_2->forward(out);
        
        out = conv_3->forward(out);
        out = relu_3->forward(out);
        
        out = conv_4->forward(out);
        out = relu_4->forward(out);
        
        out = conv_5->forward(out);
        out = relu_5->forward(out);
        out = pool_5->forward(out);

        out = fc_1->forward(out);  
        out = fc_2->forward(out);  
        out = fc_3->forward(out);  

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
      
        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);

        // std::clock_t c_end = std::clock();
        // float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;
        // printf("\n---------\nCPU (ms): %.3f, CUDA (ms): %.3f\n", time_elapsed_ms, milliseconds);

        return 0;
}



// int main(int argc, char*argv[]){

//     // float *in_data, *out;
//     ElementInputA* in_data;
//     ElementOutput* out;

//     int batch_size = 8;
//     int in_channels = 32;
    
//     int input_height = 224;
//     int input_width = 224;

//     int out_channels = 16;
//     int filter_height = 3;
//     int filter_width = 3;
//     int stride = 1;


//     // for CUTLASS test
//     // int in_bytes = batch_size * in_channels * input_height * input_width * sizeof(ElementInputA);
//     // cudaMalloc(&in_data, in_bytes);
//     // auto conv  = new CONV(batch_size, input_height, input_width, in_channels, out_channels, filter_height, filter_width, stride);
//     // out = conv->forward(in_data);
//     // auto fc  = new FC(batch_size, in_channels*input_height*input_width, out_channels);
//     // out = fc->forward(in_data);


//     // for cuDNN test.
//     // set the pooling layer for evaluation.
//     // int in_bytes = batch_size * in_channels * input_height * input_width * sizeof(cuDNNtype);
//     // cudaMalloc(&in_data, in_bytes);
//     // cudnnHandle_t cudnn;
//     // checkCUDNN(cudnnCreate(&cudnn));

//     // auto pool = new POOL(batch_size, in_channels, input_height, input_width, &cudnn);
//     // auto relu = new RELU(batch_size, in_channels, pool->get_output_height(), pool->get_output_width(), &cudnn);

//     // out = pool->forward(in_data);
//     // out = relu->forward(in_data);

//     return 0;
// }