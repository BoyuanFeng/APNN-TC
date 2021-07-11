
#include <stdio.h>
#include <ctime>
#include "layer.cuh"
#include "config.h"


int main(int argc, char*argv[]){

        // float *in_data, *out;
        ElementInputA* in_data;
        ElementInputA* out;
    
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
    
        auto conv_1  = new CONV(batch_size, input_height, input_width, in_channels, 96, 7, 7, 2, 3);
        auto relu_1  = new RELU(batch_size, conv_1->get_out_channels(), conv_1->get_output_height(), conv_1->get_output_width(), &cudnn);
        auto pool_1  = new POOL(batch_size, relu_1->get_out_channels(), relu_1->get_output_height(), relu_1->get_output_width(), &cudnn, 2, 2);

        // CONV-2
        auto conv_2_1  = new CONV(batch_size, pool_1->get_output_height(), pool_1->get_output_width(), 96, 256, 3, 3, 1, 1);
        auto relu_2_1  = new RELU(batch_size, conv_2_1->get_out_channels(), conv_2_1->get_output_height(), conv_2_1->get_output_width(), &cudnn);

        auto conv_2_2  = new CONV(batch_size, relu_2_1->get_output_height(), relu_2_1->get_output_width(), 256, 256, 3, 3, 1, 1);
        auto relu_2_2  = new RELU(batch_size, conv_2_2->get_out_channels(), conv_2_2->get_output_height(), conv_2_2->get_output_width(), &cudnn);

        auto conv_2_3  = new CONV(batch_size, relu_2_2->get_output_height(), relu_2_2->get_output_width(), 256, 256, 3, 3, 1, 1);
        auto relu_2_3  = new RELU(batch_size, conv_2_3->get_out_channels(), conv_2_3->get_output_height(), conv_2_3->get_output_width(), &cudnn);
        auto pool_2_3  = new POOL(batch_size, relu_2_3->get_out_channels(), relu_2_3->get_output_height(), relu_2_3->get_output_width(), &cudnn, 2, 2);

        // CONV-3
        auto conv_3_1  = new CONV(batch_size, pool_2_3->get_output_height(), pool_2_3->get_output_width(), 256, 512, 3, 3, 1, 1);
        auto relu_3_1  = new RELU(batch_size, conv_3_1->get_out_channels(), conv_3_1->get_output_height(), conv_3_1->get_output_width(), &cudnn);

        auto conv_3_2  = new CONV(batch_size, relu_3_1->get_output_height(), relu_3_1->get_output_width(), 512, 512, 3, 3, 1, 1);
        auto relu_3_2  = new RELU(batch_size, conv_3_2->get_out_channels(), conv_3_2->get_output_height(), conv_3_2->get_output_width(), &cudnn);

        auto conv_3_3  = new CONV(batch_size, relu_3_2->get_output_height(), relu_3_2->get_output_width(), 512, 512, 3, 3, 1, 1);
        auto relu_3_3  = new RELU(batch_size, conv_3_3->get_out_channels(), conv_3_3->get_output_height(), conv_3_3->get_output_width(), &cudnn);
        auto pool_3_3  = new POOL(batch_size, relu_3_3->get_out_channels(), relu_3_3->get_output_height(), relu_3_3->get_output_width(), &cudnn, 2, 2);

        // CONV-4
        auto conv_4_1  = new CONV(batch_size, pool_3_3->get_output_height(), pool_3_3->get_output_width(), 512, 512, 3, 3, 1, 1);
        auto relu_4_1  = new RELU(batch_size, conv_4_1->get_out_channels(), conv_4_1->get_output_height(), conv_4_1->get_output_width(), &cudnn);

        auto conv_4_2  = new CONV(batch_size, relu_4_1->get_output_height(), relu_4_1->get_output_width(), 512, 512, 3, 3, 1, 1);
        auto relu_4_2  = new RELU(batch_size, conv_4_2->get_out_channels(), conv_4_2->get_output_height(), conv_4_2->get_output_width(), &cudnn);

        auto conv_4_3  = new CONV(batch_size, relu_4_2->get_output_height(), relu_4_2->get_output_width(), 512, 512, 3, 3, 1, 1);
        auto relu_4_3  = new RELU(batch_size, conv_4_3->get_out_channels(), conv_4_3->get_output_height(), conv_4_3->get_output_width(), &cudnn);
        auto pool_4_3  = new POOL(batch_size, relu_4_3->get_out_channels(), relu_4_3->get_output_height(), relu_4_3->get_output_width(), &cudnn, 2, 2);
        
        // FC-layer
        printf("pool_4_3->get_out_channels(): %d,\n pool_4_3->get_output_height(): %d,\npool_4_3->get_output_width():%d\n",
        pool_4_3->get_out_channels(), pool_4_3->get_output_height(), pool_4_3->get_output_width());
        
        auto fc_1  = new FC(batch_size, pool_4_3->get_out_channels()*pool_4_3->get_output_height()*pool_4_3->get_output_width(), 4096);
        auto relu_fc_1  = new RELU(batch_size, fc_1->get_out_channels(), 1, 1, &cudnn);

        auto fc_2  = new FC(batch_size, 4096, 4096);
        auto relu_fc_2  = new RELU(batch_size, fc_2->get_out_channels(), 1, 1, &cudnn);

        auto fc_3  = new FC(batch_size, 4096, num_classes);

        // printf("=> \n\n");
        // printf("=> Start DNN forward!!\n");

        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start);

        printf("=============================\n");
        std::clock_t c_start = std::clock();

        out = conv_1->forward(in_data);
        out = relu_1->forward(out); 
        out = pool_1->forward(out); 

        out = conv_2_1->forward(out); 
        out = relu_2_1->forward(out); 
        out = conv_2_2->forward(out); 
        out = relu_2_2->forward(out); 
        out = conv_2_3->forward(out); 
        out = relu_2_3->forward(out); 
        out = pool_2_3->forward(out); 

        out = conv_3_1->forward(out); 
        out = relu_3_1->forward(out); 
        out = conv_3_2->forward(out); 
        out = relu_3_2->forward(out); 
        out = conv_3_3->forward(out); 
        out = relu_3_3->forward(out); 
        out = pool_3_3->forward(out); 

        out = conv_4_1->forward(out);  
        out = relu_4_1->forward(out);  
        out = conv_4_2->forward(out);  
        out = relu_4_2->forward(out);  
        out = conv_4_3->forward(out);  
        out = relu_4_3->forward(out);  
        out = pool_4_3->forward(out);  

        out = fc_1->forward(out);
        out = relu_fc_1->forward(out);
        out = fc_2->forward(out);
        out = relu_fc_2->forward(out);
        out = fc_3->forward(out);

        cudaDeviceSynchronize(); 
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        std::clock_t c_end = std::clock();
        float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;
        printf("\n---------\nTime (ms): %.3f\n", time_elapsed_ms);

        return 0;
}