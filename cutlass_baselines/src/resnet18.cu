
#include <stdio.h>
#include <ctime>
#include "layer.cuh"
#include "config.h"


int main(int argc, char*argv[]){

        ElementInputA* in_data;
        ElementInputA* out;
        // cuDNNtype* out;
    
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
    
        auto conv_1  = new CONV(batch_size, input_height, input_width, in_channels, 64, 7, 7, 2, 3);
        auto bn_1  = new BN(batch_size, conv_1->get_out_channels(), conv_1->get_output_height(), conv_1->get_output_width(), &cudnn);
        auto relu_1  = new RELU(batch_size, conv_1->get_out_channels(), conv_1->get_output_height(), conv_1->get_output_width(), &cudnn);
        auto pool_1  = new POOL(batch_size, relu_1->get_out_channels(), relu_1->get_output_height(), relu_1->get_output_width(), &cudnn, 3, 2, 1);

        // Layer_0
        // BB_0
        auto conv_2_1  = new CONV(batch_size, pool_1->get_output_height(), pool_1->get_output_width(), 64, 64, 3, 3, 1, 1);
        auto bn_2_1  = new BN(batch_size, conv_2_1->get_out_channels(), conv_2_1->get_output_height(), conv_2_1->get_output_width(), &cudnn);
        auto relu_2_1  = new RELU(batch_size, conv_2_1->get_out_channels(), conv_2_1->get_output_height(), conv_2_1->get_output_width(), &cudnn);
        auto conv_2_2  = new CONV(batch_size, relu_2_1->get_output_height(), relu_2_1->get_output_width(), 64, 64, 3, 3, 1, 1);
        auto bn_2_2  = new BN(batch_size, conv_2_2->get_out_channels(), conv_2_2->get_output_height(), conv_2_2->get_output_width(), &cudnn);

        // BB_1
        auto conv_3_1  = new CONV(batch_size, conv_2_2->get_output_height(), conv_2_2->get_output_width(), 64, 64, 3, 3, 1, 1);
        auto bn_3_1  = new BN(batch_size, conv_3_1->get_out_channels(), conv_3_1->get_output_height(), conv_3_1->get_output_width(), &cudnn);
        auto relu_3_1  = new RELU(batch_size, conv_3_1->get_out_channels(), conv_3_1->get_output_height(), conv_3_1->get_output_width(), &cudnn);
        auto conv_3_2  = new CONV(batch_size, relu_3_1->get_output_height(), relu_3_1->get_output_width(), 64, 64, 3, 3, 1, 1);
        auto bn_3_2  = new BN(batch_size, conv_3_2->get_out_channels(), conv_3_2->get_output_height(), conv_3_2->get_output_width(), &cudnn);

        // Layer_1
        // BB_2
        auto conv_4_1  = new CONV(batch_size, conv_3_2->get_output_height(), conv_3_2->get_output_width(), 64, 128, 3, 3, 1, 1);
        auto bn_4_1  = new BN(batch_size, conv_4_1->get_out_channels(), conv_4_1->get_output_height(), conv_4_1->get_output_width(), &cudnn);
        auto relu_4_1  = new RELU(batch_size, conv_4_1->get_out_channels(), conv_4_1->get_output_height(), conv_4_1->get_output_width(), &cudnn);
        auto conv_4_2  = new CONV(batch_size, relu_4_1->get_output_height(), relu_4_1->get_output_width(), 128, 128, 3, 3, 1, 1);
        auto bn_4_2  = new BN(batch_size, conv_4_2->get_out_channels(), conv_4_2->get_output_height(), conv_4_2->get_output_width(), &cudnn);


        // BB_3
        auto conv_5_1  = new CONV(batch_size, conv_4_2->get_output_height(), conv_4_2->get_output_width(), 128, 128, 3, 3, 1, 1);
        auto bn_5_1  = new BN(batch_size, conv_5_1->get_out_channels(), conv_5_1->get_output_height(), conv_5_1->get_output_width(), &cudnn);
        auto relu_5_1  = new RELU(batch_size, conv_5_1->get_out_channels(), conv_5_1->get_output_height(), conv_5_1->get_output_width(), &cudnn);
        auto conv_5_2  = new CONV(batch_size, relu_5_1->get_output_height(), relu_5_1->get_output_width(), 128, 128, 3, 3, 1, 1);
        auto bn_5_2  = new BN(batch_size, conv_5_2->get_out_channels(), conv_5_2->get_output_height(), conv_5_2->get_output_width(), &cudnn);

        // Layer_2
        // BB_0
        auto conv_6_1  = new CONV(batch_size, conv_5_2->get_output_height(), conv_5_2->get_output_width(), 128, 256, 3, 3, 1, 1);
        auto bn_6_1  = new BN(batch_size, conv_6_1->get_out_channels(), conv_6_1->get_output_height(), conv_6_1->get_output_width(), &cudnn);
        auto relu_6_1  = new RELU(batch_size, conv_6_1->get_out_channels(), conv_6_1->get_output_height(), conv_6_1->get_output_width(), &cudnn);
        auto conv_6_2  = new CONV(batch_size, relu_6_1->get_output_height(), relu_6_1->get_output_width(), 256, 256, 3, 3, 1, 1);
        auto bn_6_2  = new BN(batch_size, conv_6_2->get_out_channels(), conv_6_2->get_output_height(), conv_6_2->get_output_width(), &cudnn);

        // BB_1
        auto conv_7_1  = new CONV(batch_size, conv_6_2->get_output_height(), conv_6_2->get_output_width(), 256, 256, 3, 3, 1, 1);
        auto bn_7_1  = new BN(batch_size, conv_7_1->get_out_channels(), conv_7_1->get_output_height(), conv_7_1->get_output_width(), &cudnn);
        auto relu_7_1  = new RELU(batch_size, conv_7_1->get_out_channels(), conv_7_1->get_output_height(), conv_7_1->get_output_width(), &cudnn);
        auto conv_7_2  = new CONV(batch_size, relu_7_1->get_output_height(), relu_7_1->get_output_width(), 256, 256, 3, 3, 1, 1);
        auto bn_7_2  = new BN(batch_size, conv_7_2->get_out_channels(), conv_7_2->get_output_height(), conv_7_2->get_output_width(), &cudnn);

        // Layer_3
        // BB_2
        auto conv_8_1  = new CONV(batch_size, conv_7_2->get_output_height(), conv_7_2->get_output_width(), 256, 512, 3, 3, 1, 1);
        auto bn_8_1  = new BN(batch_size, conv_8_1->get_out_channels(), conv_8_1->get_output_height(), conv_8_1->get_output_width(), &cudnn);
        auto relu_8_1  = new RELU(batch_size, conv_8_1->get_out_channels(), conv_8_1->get_output_height(), conv_8_1->get_output_width(), &cudnn);
        auto conv_8_2  = new CONV(batch_size, relu_8_1->get_output_height(), relu_8_1->get_output_width(), 512, 512, 3, 3, 1, 1);
        auto bn_8_2  = new BN(batch_size, conv_8_2->get_out_channels(), conv_8_2->get_output_height(), conv_8_2->get_output_width(), &cudnn);


        // BB_3
        auto conv_9_1  = new CONV(batch_size, conv_8_2->get_output_height(), conv_8_2->get_output_width(), 512, 512, 3, 3, 1, 1);
        auto bn_9_1  = new BN(batch_size, conv_9_1->get_out_channels(), conv_9_1->get_output_height(), conv_9_1->get_output_width(), &cudnn);
        auto relu_9_1  = new RELU(batch_size, conv_9_1->get_out_channels(), conv_9_1->get_output_height(), conv_9_1->get_output_width(), &cudnn);
        auto conv_9_2  = new CONV(batch_size, relu_9_1->get_output_height(), relu_9_1->get_output_width(), 512, 512, 3, 3, 1, 1);
        auto bn_9_2  = new BN(batch_size, conv_9_2->get_out_channels(), conv_9_2->get_output_height(), conv_9_2->get_output_width(), &cudnn);

        auto pool_last  = new POOL(batch_size, conv_9_2->get_out_channels(), conv_9_2->get_output_height(), conv_9_2->get_output_width(), &cudnn, 2, 2);
        
        // FC-layer
        printf("pool_last->get_out_channels(): %d,\n pool_last->get_output_height(): %d,\npool_last->get_output_width():%d\n",
        pool_last->get_out_channels(), pool_last->get_output_height(), pool_last->get_output_width());

        auto fc_1  = new FC(batch_size, pool_last->get_out_channels()*pool_last->get_output_height()*pool_last->get_output_width(), num_classes);

        printf("=> \n\n");
        printf("=> Start DNN forward!!\n");
        std::clock_t c_start = std::clock();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        out = conv_1->forward(in_data);  
        out = bn_1->forward(out);
        out = relu_1->forward(out); 
        out = pool_1->forward(out);  

        out = conv_2_1->forward(out);  
        out = bn_2_1->forward(out);
        out = relu_2_1->forward(out);  
        out = conv_2_2->forward(out);  
        out = bn_2_2->forward(out);

        out = conv_3_1->forward(out);  
        out = bn_3_1->forward(out);
        out = relu_3_1->forward(out);  
        out = conv_3_2->forward(out);  
        out = bn_3_2->forward(out);

        out = conv_4_1->forward(out);  
        out = bn_4_1->forward(out);
        out = relu_4_1->forward(out);  
        out = conv_4_2->forward(out);  
        out = bn_4_2->forward(out);

        out = conv_5_1->forward(out);  
        out = bn_5_1->forward(out);
        out = relu_5_1->forward(out);  
        out = conv_5_2->forward(out);  
        out = bn_5_2->forward(out);

        out = conv_6_1->forward(out);  
        out = bn_6_1->forward(out);
        out = relu_6_1->forward(out);  
        out = conv_6_2->forward(out);  
        out = bn_6_2->forward(out);

        out = conv_7_1->forward(out);  
        out = bn_7_1->forward(out);
        out = relu_7_1->forward(out);  
        out = conv_7_2->forward(out);  
        out = bn_7_2->forward(out);

        out = conv_8_1->forward(out); 
        out = bn_8_1->forward(out);
        out = relu_8_1->forward(out);  
        out = conv_8_2->forward(out);  
        out = bn_8_2->forward(out);

        out = conv_9_1->forward(out);  
        out = bn_9_1->forward(out);
        out = relu_9_1->forward(out);  
        out = conv_9_2->forward(out);  
        out = bn_9_2->forward(out);

        out = pool_last->forward(out);  
        out = fc_1->forward(out);  

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
      
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::clock_t c_end = std::clock();
        float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;
        printf("\n---------\nCPU (ms): %.3f, CUDA (ms): %.3f\n", time_elapsed_ms, milliseconds);

        return 0;
}