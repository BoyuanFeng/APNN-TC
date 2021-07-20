#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <ctime>
#include <iostream>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "utility.h"
#include "param.h"
#include "kernel.cuh"
#include "data.h"

using namespace std;

int main()
{
    int dev = 0;
    cudaSetDevice(dev);

    const unsigned batch = 8;
    const unsigned output_size = 1000;
    const unsigned image_height = 224;
    const unsigned image_width = 224;
    const unsigned image_channel = 3;
    const unsigned n_hidden = 4096;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
//     read_ImageNet_normalized("./imagenet_files.txt", images, image_labels, batch);

    //================ Get Weight =================
    FILE* config_file = fopen("./alexnet_imagenet.csv","r");

    //================ Set Network =================
    // Bconv1 Layer
    uin32* lowBit_image_gpu = images_quantization(images, batch, image_height, image_width, image_channel);
    Conv128LayerParam* bconv1 = new Conv128LayerParam("Conv1", image_height, image_width, 
            11, 11, 3, 64, batch, 4, 4, true, 2, 2); 
    Conv128LayerParam* bconv1_gpu = bconv1->initialize(config_file, lowBit_image_gpu);

    //Bconv2 Layer
    Conv128LayerParam* bconv2 = new Conv128LayerParam("Conv2", bconv1->output_height, 
            bconv1->output_width, 5, 5, 64, 192, batch, 1, 1,
            true, 2, 2, false, false, false, 0, false, a_bit, w_bit);
    Conv128LayerParam* bconv2_gpu = bconv2->initialize(config_file, bconv1->get_output_gpu());

    //Bconv3 Layer
    Conv128LayerParam* bconv3 = new Conv128LayerParam("Conv3", bconv2->output_height, 
            bconv2->output_width, 3, 3, 192, 384, batch,
            1, 1, true, 1, 1, false, false, false, 0, false, a_bit, w_bit);
    Conv128LayerParam* bconv3_gpu = bconv3->initialize(config_file, bconv2->get_output_gpu());

    //Bconv4 Layer
    Conv128LayerParam* bconv4 = new Conv128LayerParam("Conv4", bconv3->output_height, 
            bconv3->output_width, 3, 3, 384, 256, batch,
            1, 1, true, 1, 1, false, false, false, 0, false, a_bit, w_bit);
    Conv128LayerParam* bconv4_gpu = bconv4->initialize(config_file, bconv3->get_output_gpu());

    //Bconv5 Layer
    Conv128LayerParam* bconv5 = new Conv128LayerParam("Conv5", bconv4->output_height, 
            bconv4->output_width, 3, 3, 256, 256, batch, 1, 1, true, 2, 2, true,
            false, false, 0, false, a_bit, w_bit);
    Conv128LayerParam* bconv5_gpu = bconv5->initialize(config_file, bconv4->get_output_gpu());

    //Fc1 Layer
    Fc128LayerParam* bfc1 = new Fc128LayerParam("Fc1", batch, (bconv5->output_height)
            *(bconv5->output_width)*256, n_hidden, a_bit, w_bit); 
    Fc128LayerParam* bfc1_gpu = bfc1->initialize(config_file, bconv5->get_output_gpu());
    
    //Fc2 Layer
    Fc128LayerParam* bfc2 = new Fc128LayerParam("Fc2", batch, n_hidden, n_hidden, a_bit, w_bit); 
    Fc128LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    //Out Layer
    Out128LayerParam* bout = new Out128LayerParam("Fout", batch, n_hidden, output_size, a_bit, w_bit);
    Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc2->get_output_gpu());  

    //================ Setup Kernel =================
    int numThreads = 512;
    int numBlocks = 8;
    int shared_memory = 65536; // 64KB

    std::clock_t c_start = std::clock();

    cudaFuncSetAttribute(Conv_new_global, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaFuncSetAttribute(FC_new_global, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaFuncSetAttribute(Output_new_global, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);

    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(bconv1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(bconv2_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(bconv3_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(bconv4_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(bconv5_gpu);
    cudaDeviceSynchronize(); 
    FC_new_global<<<numBlocks, numThreads, shared_memory>>>(bfc1_gpu);
    cudaDeviceSynchronize(); 
    FC_new_global<<<numBlocks, numThreads, shared_memory>>>(bfc2_gpu);
    cudaDeviceSynchronize(); 
    Output_new_global<<<numBlocks, numThreads, shared_memory>>>(bout_gpu);
    cudaDeviceSynchronize(); 

    cudaError_t err = cudaGetLastError();

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;
    printf("\n==============\nAlexNet (ms): %.3f\n", time_elapsed_ms);


    if ( err != cudaSuccess ){
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
       exit(-1);
    }

    delete bconv1;
    delete bconv2;
    delete bconv3;
    delete bconv4;
    delete bconv5;
    delete bfc1;
    delete bfc2;
    delete bout;

    return 0;
}
