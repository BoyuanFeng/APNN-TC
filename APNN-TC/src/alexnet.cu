#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <ctime>
#include <iostream>
#include <string>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "utility.h"
#include "param.h"
#include "kernel.cuh"
#include "data.h"

using namespace cooperative_groups;
using namespace std;

__global__ void alexnet128(
        // InConv128LayerParam* bconv1, 
        Conv128LayerParam* bconv1, 
        Conv128LayerParam* bconv2, 
        Conv128LayerParam* bconv3,
        Conv128LayerParam* bconv4, 
        Conv128LayerParam* bconv5, 
        Fc128LayerParam* bfc1, 
        Fc128LayerParam* bfc2, 
        Out128LayerParam* bout)
{
    grid_group grid = this_grid();
    //========= Conv1 ============
    // InConv128Layer(bconv1);
    Conv_new(bconv1);
    grid.sync();
    //========= Conv2 ============
    // Conv128Layer(bconv2);
    Conv_new(bconv2);
    grid.sync();
    //========= Conv3 ============
    // Conv128Layer(bconv3);
    Conv_new(bconv3);
    grid.sync();
    //========= Conv4 ============
    // Conv128Layer(bconv4);
    Conv_new(bconv4);
    grid.sync();
    //========= Conv5 ============
    // Conv128Layer(bconv5);
    Conv_new(bconv5);
    grid.sync();
    //========= Fc1 ============
//     Fc128Layer(bfc1);
    FC_new(bfc1);
    grid.sync();
    //========= Fc2 ============
//     Fc128Layer(bfc2);
    FC_new(bfc2);
    grid.sync();
    //========== Output ===========
//     Out128Layer(bout);
    Output_new(bout);
}
// #endif

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
    //Bconv1 Layer
    // InConv128LayerParam* bconv1 = new InConv128LayerParam("Conv1", image_height, image_width, 
    //         11, 11, 3, 64, batch, 4, 4, true, 2, 2); 
    // InConv128LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);

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
    int numThreads = 256;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 65536; // 64KB
    
    cudaFuncSetAttribute(alexnet128, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, alexnet128, numThreads, shared_memory);

    void* args[] = {&bconv1_gpu, &bconv2_gpu, &bconv3_gpu, &bconv4_gpu, &bconv5_gpu, 
        &bfc1_gpu, &bfc2_gpu, &bout_gpu};

    printf("numBlocks: %d, shared_memory (KB): %.3f\n", numBlocksPerSm, 1.0f*shared_memory/1e3);

    std::clock_t c_start = std::clock();
    // START_TIMER;

    cudaLaunchCooperativeKernel((void*)alexnet128, deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);

    // STOP_TIMER;
    cudaDeviceSynchronize(); 
    cudaError_t err = cudaGetLastError();

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;
    printf("\n==============\n AlexNet (ms): %.3f\n", time_elapsed_ms);


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
