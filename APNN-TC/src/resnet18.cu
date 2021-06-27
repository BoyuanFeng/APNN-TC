#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
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

// __global__ void resnet128(
//         InConv128LayerParam* bconv1, 
//         Conv128LayerParam* l1b1c1, 
//         Conv128LayerParam* l1b1c2,
//         Conv128LayerParam* l1b2c1, 
//         Conv128LayerParam* l1b2c2,
//         Conv128LayerParam* l2b1c1, 
//         Conv128LayerParam* l2b1c2,
//         Conv128LayerParam* l2b2c1, 
//         Conv128LayerParam* l2b2c2,
//         Conv128LayerParam* l3b1c1, 
//         Conv128LayerParam* l3b1c2,
//         Conv128LayerParam* l3b2c1, 
//         Conv128LayerParam* l3b2c2,
//         Conv128LayerParam* l4b1c1, 
//         Conv128LayerParam* l4b1c2,
//         Conv128LayerParam* l4b2c1, 
//         Conv128LayerParam* l4b2c2,
//         Fc128LayerParam* bfc1, 
//         Out128LayerParam* bout)
// {
//     grid_group grid = this_grid();
//     //========= Conv1 ============
//     InConv128Layer(bconv1);
//     grid.sync();
//     //========= L1B1 ============
// //     Conv128Layer(l1b1c1);
//     Conv_new(l1b1c1);
//     grid.sync();
// //     Conv128Layer(l1b1c2);
//     Conv_new(l1b1c2);
//     grid.sync();
//     //========= L1B2 ============
// //     Conv128Layer(l1b2c1);
//     Conv_new(l1b2c1);
//     grid.sync();
// //     Conv128Layer(l1b2c2);
//     Conv_new(l1b2c2);
//     grid.sync();
//     //========= L2B1 ============
// //     Conv128Layer(l2b1c1);
//     Conv_new(l2b1c1);
//     grid.sync();
// //     Conv128Layer(l2b1c2);
//     Conv_new(l2b1c2);
//     grid.sync();
//     //========= L2B2 ============
// //     Conv128Layer(l2b2c1);
//     Conv_new(l2b2c1);
//     grid.sync();
// //     Conv128Layer(l2b2c2);
//     Conv_new(l2b2c2);
//     grid.sync();
//     //========= L3B1 ============
// //     Conv128Layer(l3b1c1);
//     Conv_new(l3b1c1);
//     grid.sync();
// //     Conv128Layer(l3b1c2);
//     Conv_new(l3b1c2);
//     grid.sync();

//     //========= L3B2 ============
// //     Conv128Layer(l3b2c1);
//     Conv_new(l3b2c1);
//     grid.sync();
// //     Conv128Layer(l3b2c2);
//     Conv_new(l3b2c2);
//     grid.sync();
//     //========= L4B1 ============
// //     Conv128Layer(l4b1c1);
//     Conv_new(l4b1c1);
//     grid.sync();
// //     Conv128Layer(l4b1c2);
//     Conv_new(l4b1c2);
//     grid.sync();
// //     ========= L4B2 ============
// //     Conv128Layer(l4b2c1);
//     Conv_new(l4b2c1);
//     grid.sync();
// //     Conv128Layer(l4b2c2);
//     Conv_new(l4b2c2);
//     grid.sync();
//     //========= Fc1 ============
// //     Fc128Layer(bfc1);
//     FC_new(bfc1);
//     grid.sync();
//     //========== Output ===========
// //     Out128Layer(bout); //** failed with illegal memory access... check here with memory size
//     Output_new(bout);
// }

int main()
{
    int dev = 0;
    cudaSetDevice(dev);
    
    const unsigned batch = 8;
    const unsigned output_size = 1000;
    const unsigned image_height = 224;
    const unsigned image_width = 224;
    const unsigned image_channel = 3;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
//     read_ImageNet_normalized("./imagenet_files.txt", images, image_labels, batch);
    uin32* lowBit_image_gpu = images_quantization(images, batch, image_height, image_width, image_channel);

    //================ Get Weight =================
    FILE* config_file = fopen("./resnet_imagenet.csv","r");

    //================ Set Network =================
    //Layer-0
    Conv128LayerParam* bconv1 = new Conv128LayerParam("Conv1", image_height, image_width, 
            7, 7, 3, 64, batch,4,4,true,1,1,true); //save residual 
    Conv128LayerParam* bconv1_gpu = bconv1->initialize(config_file, lowBit_image_gpu);

    //Layer-1, basic-block-1, conv1
    Conv128LayerParam* l1b1c1 = new Conv128LayerParam("L1B1C1", bconv1->output_height, 
            bconv1->output_width, 3, 3, 64, 64, batch);
    Conv128LayerParam* l1b1c1_gpu = l1b1c1->initialize(config_file, bconv1->get_output_gpu());

    //Layer-1, basic-block-1, conv2
    Conv128LayerParam* l1b1c2 = new Conv128LayerParam("L1B1C2", l1b1c1->output_height, 
            l1b1c1->output_width, 3, 3, 64, 64, batch,1,1,true,1,1,false,true,true,64);
    Conv128LayerParam* l1b1c2_gpu = l1b1c2->initialize(config_file, l1b1c1->get_output_gpu(),
            bconv1->get_output_residual_gpu());

    //Layer-1, basic-block-2, conv1
    Conv128LayerParam* l1b2c1 = new Conv128LayerParam("L1B2C1", l1b1c2->output_height, 
            l1b1c2->output_width, 3, 3, 64, 64, batch);
    Conv128LayerParam* l1b2c1_gpu = l1b2c1->initialize(config_file, l1b1c2->get_output_gpu());

    //Layer-1, basic-block-2, conv2
    Conv128LayerParam* l1b2c2 = new Conv128LayerParam("L1B2C2", l1b2c1->output_height, 
            l1b2c1->output_width, 3, 3, 64, 64, batch,1,1,true,1,1,false,true,true,128);
    Conv128LayerParam* l1b2c2_gpu = l1b2c2->initialize(config_file, l1b2c1->get_output_gpu(),
            l1b1c2->get_output_residual_gpu());

    //=============
    //Layer-2, basic-block-1, conv1
    Conv128LayerParam* l2b1c1 = new Conv128LayerParam("L2B1C1", l1b2c2->output_height, 
            l1b2c2->output_width, 3, 3, 64, 128, batch, 2, 2);
    Conv128LayerParam* l2b1c1_gpu = l2b1c1->initialize(config_file, l1b2c2->get_output_gpu());

    //Layer-2, basic-block-1, conv2
    Conv128LayerParam* l2b1c2 = new Conv128LayerParam("L2B1C2", l2b1c1->output_height, 
            l2b1c1->output_width, 3, 3, 128, 128, batch,1,1,true,1,1,false,true,true,128,true);
    Conv128LayerParam* l2b1c2_gpu = l2b1c2->initialize(config_file, l2b1c1->get_output_gpu(),
            l1b2c2->get_output_residual_gpu());

    //Layer-2, basic-block-2, conv1
    Conv128LayerParam* l2b2c1 = new Conv128LayerParam("L2B2C1", l2b1c2->output_height, 
            l2b1c2->output_width, 3, 3, 128, 128, batch, 1, 1);
    Conv128LayerParam* l2b2c1_gpu = l2b2c1->initialize(config_file, l2b1c2->get_output_gpu());

    //Layer-2, basic-block-2, conv2
    Conv128LayerParam* l2b2c2 = new Conv128LayerParam("L2B2C2", l2b2c1->output_height, 
            l2b2c1->output_width, 3, 3, 128, 128, batch,1,1,true,1,1,false,true,true,128);
    Conv128LayerParam* l2b2c2_gpu = l2b2c2->initialize(config_file, l2b2c1->get_output_gpu(),
            l2b1c2->get_output_residual_gpu());

    //=============
    //Layer-3, basic-block-1, conv1
    Conv128LayerParam* l3b1c1 = new Conv128LayerParam("L3B1C1", l2b2c2->output_height, 
            l2b2c2->output_width, 3, 3, 128, 256, batch, 2, 2);
    Conv128LayerParam* l3b1c1_gpu = l3b1c1->initialize(config_file, l2b2c2->get_output_gpu());

    //Layer-3, basic-block-1, conv2
    Conv128LayerParam* l3b1c2 = new Conv128LayerParam("L3B1C2", l3b1c1->output_height, 
            l3b1c1->output_width, 3, 3, 256, 256, batch,1,1,true,1,1,false,true,true,128,true);
    Conv128LayerParam* l3b1c2_gpu = l3b1c2->initialize(config_file, l3b1c1->get_output_gpu(),
            l2b2c2->get_output_residual_gpu());

    //Layer-3, basic-block-2, conv1
    Conv128LayerParam* l3b2c1 = new Conv128LayerParam("L3B2C1", l3b1c2->output_height, 
            l3b1c2->output_width, 3, 3, 256, 256, batch, 1, 1);
    Conv128LayerParam* l3b2c1_gpu = l3b2c1->initialize(config_file, l3b1c2->get_output_gpu());

    //Layer-3, basic-block-2, conv2
    Conv128LayerParam* l3b2c2 = new Conv128LayerParam("L3B2C2", l3b2c1->output_height, 
            l3b2c1->output_width, 3, 3, 256, 256, batch,1,1,true,1,1,false,true,true,256);
    Conv128LayerParam* l3b2c2_gpu = l3b2c2->initialize(config_file, l3b2c1->get_output_gpu(),
            l3b1c2->get_output_residual_gpu());

    //=============
    //Layer-4, basic-block-1, conv1
    Conv128LayerParam* l4b1c1 = new Conv128LayerParam("L4B1C1", l3b2c2->output_height, 
            l3b2c2->output_width, 3, 3, 256, 512, batch, 2, 2);
    Conv128LayerParam* l4b1c1_gpu = l4b1c1->initialize(config_file, l3b2c2->get_output_gpu());

    //Layer-4, basic-block-1, conv2
    Conv128LayerParam* l4b1c2 = new Conv128LayerParam("L4B1C2", l4b1c1->output_height, 
            l4b1c1->output_width, 3, 3, 512, 512, batch,1,1,true,1,1,false,true,true,256,true);
    Conv128LayerParam* l4b1c2_gpu = l4b1c2->initialize(config_file, l4b1c1->get_output_gpu(),
            l3b2c2->get_output_residual_gpu());

    //Layer-4, basic-block-2, conv1
    Conv128LayerParam* l4b2c1 = new Conv128LayerParam("L4B2C1", l4b1c2->output_height, 
            l4b1c2->output_width, 3, 3, 512, 512, batch, 1, 1);
    Conv128LayerParam* l4b2c1_gpu = l4b2c1->initialize(config_file, l4b1c2->get_output_gpu());

    //Layer-4, basic-block-2, conv2
    Conv128LayerParam* l4b2c2 = new Conv128LayerParam("L4B2C2", l4b2c1->output_height, 
            l4b2c1->output_width, 3, 3, 512, 512, batch,1,1,true,1,1,true,false,true,512);
    Conv128LayerParam* l4b2c2_gpu = l4b2c2->initialize(config_file, l4b2c1->get_output_gpu(),
            l4b1c2->get_output_residual_gpu());

    //=============
    //Layer-5
    Fc128LayerParam* bfc1 = new Fc128LayerParam("Fc1", batch, (l4b2c2->output_height)
            *(l4b2c2->output_width)*512, 512); 
    Fc128LayerParam* bfc1_gpu = bfc1->initialize(config_file, l4b2c2->get_output_gpu());
    //Out Layer
    Out128LayerParam* bout = new Out128LayerParam("Fout", batch, 512, output_size);
    Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc1->get_output_gpu());  

    //================ Setup Kernel =================
    int numThreads = 512;
    int numBlocks = 16;
    int shared_memory = 65536; // 64KB

    cudaFuncSetAttribute(Conv_new_global, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaFuncSetAttribute(FC_new_global, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaFuncSetAttribute(Output_new_global, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);

    std::clock_t c_start = std::clock();

    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(bconv1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l1b1c1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l1b1c2_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l1b2c1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l1b2c2_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l2b1c1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l2b1c2_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l2b2c1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l2b2c2_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l3b1c1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l3b1c2_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l3b2c1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l3b2c2_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l4b1c1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l4b1c2_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l4b2c1_gpu);
    cudaDeviceSynchronize(); 
    Conv_new_global<<<numBlocks, numThreads, shared_memory>>>(l4b2c2_gpu);
    cudaDeviceSynchronize(); 
    FC_new_global<<<numBlocks, numThreads, shared_memory>>>(bfc1_gpu);
    cudaDeviceSynchronize(); 
    Output_new_global<<<numBlocks, numThreads, shared_memory>>>(bout_gpu);
    cudaDeviceSynchronize(); 
    cudaError_t err = cudaGetLastError();

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;
    printf("\n==============\nResNet (ms): %.3f\n", time_elapsed_ms);



    delete bconv1;
    delete l1b1c1;
    delete l1b1c2;
    delete l1b2c1;
    delete l1b2c2;

    delete l2b1c1;
    delete l2b1c2;
    delete l2b2c1;
    delete l2b2c2;

    delete l3b1c1;
    delete l3b1c2;
    delete l3b2c1;
    delete l3b2c2;

    delete l4b1c1;
    delete l4b1c2;
    delete l4b2c1;
    delete l4b2c2;

    delete bfc1;
    delete bout;

    return 0;
}























