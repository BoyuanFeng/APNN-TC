#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>

#include "param.h"

#define w_bit 1
#define a_bit 2
#include "67_bmmaTensorCoreGemm.cuh" // for w1a2 CONV
#include "66_bmmaTensorCoreGemm.cuh" // for w1a2 GEMM

#define max_v 10
#define min_v -10

__inline__ __device__ 
float clip(float x, float lb, float ub){
    if (x < lb) return lb;
    if (x > ub) return ub;
    return x;
}

// input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
/////////////////////////////////////////////////////
__global__ 
void Quantize_val(uin32* input_qnt_gpu, float* input_gpu, int num_elements, int bitwidth){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // for all available threads.
    // for (int tid = start; tid < num_elements; tid += blockDim.x * gridDim.x) {
    if (tid < num_elements){
        /*
        * Quant_val  - 0            2^{bitwidth}    
        *-------------------- = ------------------
        * Actual_val - min_val  max_val - min_val
        */
        float input_val = clip(input_gpu[tid], min_v, max_v);
        float qnt_float = (input_val - min_v) * (1 << bitwidth) * 1.0f / (max_v - min_v);
        // input_qnt_gpu[tid]  =  __float2uint_rn(qnt_float);
    }
}  

__global__ 
void  Bit_compression(uin32* bit_images_gpu, uin32* qnt_images_gpu, \
        int batch, int image_width, int image_height, int image_channel, \
        int bit=2){
        /*
        N, H, W, C:
        C = PAD(128).
        total warps = (N, H, W)
        each warp manage = (C/128) compression --> uint4 (4 * uint32).
        output:  
        in32 or int4: (N, H, W, C/128).
        */
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x)/32; // global warp id.
        int lane_id = threadIdx.x % 32;             // get the laneid.
        int warp_dist =  blockDim.x * gridDim.x/32; // total number of warps.
        int bit_offset = batch*image_width*image_height*PAD128(image_channel)/32;

        for (int wid = warp_id; wid < batch*image_width*image_height; wid += warp_dist){
            int s_pos = wid*image_channel;
            int e_pos = (wid+1)*image_channel;
            for (int i = s_pos + lane_id; i < e_pos; i += 32){
                uin32 tmp = qnt_images_gpu[i];
                for (int bIdx = 0; bIdx < bit; bIdx++){
                    uin32 pack_32 =  __brev(__ballot_sync(0xFFFFFFFF, ((tmp >> bIdx) & 0x01) > 0?1:0));
                    //output to target position in (N, H, W, C/128);
                    int dst_pos = wid * PAD128(image_channel)/32 + (i - s_pos)/32;
                    bit_images_gpu[bit_offset*bIdx+dst_pos] = pack_32;
                    // bit_images_gpu[1] = pack_32;
                }            
            }
        }
}


__global__ 
void  Bit_compression_fc_hidden(uin32* bit_images_gpu, uin32* qnt_images_gpu, \
        int batch, int image_width, int image_height, int image_channel, \
        int bit=2){
        /*
        N, H, W, C:
        C = PAD(128).
        total warps = (N, H, W)
        each warp manage = (C/128) compression --> uint4 (4 * uint32).
        output:  
        in32 or int4: (N, H, W, C/128).
        */
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x)/32; // global warp id.
        int lane_id = threadIdx.x % 32;             // get the laneid.
        int warp_dist =  blockDim.x * gridDim.x/32; // total number of warps.
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // if (tid == 0)
        //     printf("batch: %d, warp_dist: %d, in_channel: %d\n", batch, warp_dist, image_channel);
        int bit_offset = batch*image_width*image_height*PAD128(image_channel)/32;

        for (int wid = warp_id; wid < batch*image_width*image_height; wid += warp_dist){
            // if (wid >= batch){
            //     printf("wid: %d\n", wid);
            // }
            int s_pos = wid*image_channel;
            int e_pos = (wid+1)*image_channel;
            for (int i = s_pos + lane_id; i < e_pos; i += 32){
                uin32 tmp = qnt_images_gpu[i];
                for (int bIdx = 0; bIdx < bit; bIdx++){
                    uin32 pack_32 =  __brev(__ballot_sync(0xFFFFFFFF, ((tmp >> bIdx) & 0x01) > 0?1:0));
                    //output to target position in (N, H, W, C/128);
                    int dst_pos = wid * PAD128(image_channel)/32 + (i - s_pos)/32;
                    // if (tid == 0)
                    //     printf("dst_pos: %d\n", dst_pos);
                    // bit_images_gpu[bit_offset*bIdx+dst_pos] = pack_32;
                    // if ( dst_pos >= (PAD128(image_channel) * batch)/32){
                    //     printf("batch: %d, dst_pos: %d -- %d\n", wid, dst_pos, (PAD128(image_channel) * batch)/32);
                    // }
                    // else{
                    bit_images_gpu[dst_pos] = pack_32;
                    // }
                    // bit_images_gpu[1] = pack_32;
                }            
            }
        }
}


uin32* images_quantization(float* images, int batch, int image_width, 
                            int image_height, int image_channel, int bit=2){
    /*
    *  quantize a float-based images to low-bit input.
    */
    uin32* quant_images_gpu;
    uin32* bit_images_gpu; 
    float* images_gpu;

    int total_size = batch*image_height*image_width*image_channel;

    SAFE_ALOC_GPU(images_gpu, total_size*sizeof(float));
    SAFE_ALOC_GPU(quant_images_gpu, total_size*sizeof(uin32));
    CUDA_SAFE_CALL( cudaMemcpy(images_gpu, images, total_size*sizeof(float), cudaMemcpyHostToDevice) );

    // do quantization.
    int block_size = 1024;
    int grid_size = (total_size + block_size - 1)/block_size;
    Quantize_val<<<grid_size, block_size>>>(quant_images_gpu, images_gpu, total_size, bit);

    // Low-bit compression.
    SAFE_ALOC_GPU(bit_images_gpu, bit*batch*image_height*image_width*PAD128(image_channel)*sizeof(uin32)/32);
    block_size = 1024;
    grid_size = (batch*image_height*image_width*32 + block_size - 1)/block_size;
    Bit_compression<<<grid_size, block_size>>>(bit_images_gpu, quant_images_gpu, \
                                        batch, image_height, image_width, image_channel,  \
                                        bit);

    SAFE_FREE_GPU(images_gpu);
    SAFE_FREE_GPU(quant_images_gpu);

    return bit_images_gpu;
}


void filter_quantization(uin32* bit_filter_gpu, float* filter_gpu,
                            int out_channel, int filter_width, int filter_height, 
                            int in_channel, int bit){
    /*
    *  quantize a float-based CONV filter to low-bit filter.
    */
    uin32* quant_filter_gpu;
    int total_size = out_channel*filter_height*filter_width*PAD128(in_channel);

    SAFE_ALOC_GPU(quant_filter_gpu, total_size*sizeof(uin32));


    // do quantization.
    int block_size = 1024;
    int grid_size = (total_size + block_size - 1)/block_size;
    
    Quantize_val<<<grid_size, block_size>>>(quant_filter_gpu, filter_gpu, total_size, bit);

    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // Low-bit compression.
    SAFE_ALOC_GPU(bit_filter_gpu, bit*total_size*sizeof(uin32)/32);
    block_size = 1024;
    grid_size = (out_channel*filter_height*filter_width*32 + block_size - 1)/block_size;

    Bit_compression<<<grid_size, block_size>>>(bit_filter_gpu, quant_filter_gpu, \
                                        out_channel, filter_height, filter_width, in_channel,  \
                                        bit);
                                        
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    SAFE_FREE_GPU(quant_filter_gpu);
}


void weight_quantization(uin32*bit_weight_gpu, float* weight_gpu, 
                        int in_channel, int out_channel, int bit)
    {
    /*
    *  quantize a float weight weight to low-bit weigth.
    */
    uin32* quant_weight_gpu;
    // uin32* bit_weight_gpu; 
    // float* weight_gpu;
    int total_size = out_channel*in_channel;
    SAFE_ALOC_GPU(quant_weight_gpu, total_size*sizeof(uin32));

    // SAFE_ALOC_GPU(weight_gpu, total_size*sizeof(float));
    // CUDA_SAFE_CALL( cudaMemcpy(weight_gpu, weight, total_size*sizeof(float), cudaMemcpyHostToDevice) );

    // do quantization.
    int block_size = 1024;
    int grid_size = (total_size + block_size - 1)/block_size;

    Quantize_val<<<grid_size, block_size>>>(quant_weight_gpu, weight_gpu, total_size, bit);
    
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // Low-bit compression.
    // SAFE_ALOC_GPU(bit_weight_gpu, bit*total_size*PAD128(in_channel)*sizeof(uin32)/32);
    block_size = 1024;
    grid_size = (out_channel*32 + block_size - 1)/block_size;

    Bit_compression<<<grid_size, block_size>>>(bit_weight_gpu, quant_weight_gpu, 
                                                out_channel, 1, 1, in_channel, 
                                                bit);

    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    SAFE_FREE_GPU(quant_weight_gpu);
    // return bit_weight_gpu;
}

void weight_quantization_fc_hidden(uin32* bit_weight_gpu, float* weight_gpu, 
                                    int in_channel, int out_channel, int bit)
{
    /*
    *  quantize a float weight weight to low-bit weigth.
    */
    uin32* quant_weight_gpu;
    // uin32* bit_weight_gpu; 
    // SAFE_FREE_GPU(bit_weight_gpu);

    // float* weight_gpu;
    int total_size = out_channel*in_channel;
    SAFE_ALOC_GPU(quant_weight_gpu, total_size*sizeof(uin32));

    // SAFE_ALOC_GPU(weight_gpu, total_size*sizeof(float));
    // CUDA_SAFE_CALL( cudaMemcpy(weight_gpu, weight, total_size*sizeof(float), cudaMemcpyHostToDevice) );

    // do quantization.
    int block_size = 1024;
    int grid_size = (total_size + block_size - 1)/block_size;

    Quantize_val<<<grid_size, block_size>>>(quant_weight_gpu, weight_gpu, total_size, bit);
    
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // Low-bit compression.
    // SAFE_ALOC_GPU(bit_weight_gpu, bit*PAD128(in_channel)*out_channel/32);
    // printf("bit: %d, out_channel:%d, in_channel: %d\n", bit, out_channel, in_channel);
    block_size = 1024;
    grid_size = (out_channel*32 + block_size - 1)/block_size;
    // printf("grid_size: %d, block_size:%d\n", grid_size, block_size);

    Bit_compression_fc_hidden<<<grid_size, block_size>>>(bit_weight_gpu, quant_weight_gpu, 
                                                        out_channel, 1, 1, in_channel, 
                                                        bit);

    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    SAFE_FREE_GPU(weight_gpu);
    SAFE_FREE_GPU(quant_weight_gpu);
    // return bit_weight_gpu;
}


__global__ 
void Conv_new_global(Conv128LayerParam* p){
    APConv_w1a2_pack((const int4*) p->filter_gpu, (const int4*) p->input_gpu, (int*) p->output_gpu, \
                    PAD8(p->batch)*p->input_height, p->input_width, p->input_channels, p->output_channels); 
}


__device__ __inline__ 
void Conv_new(Conv128LayerParam* p){
    APConv_w1a2_pack((const int4*) p->filter_gpu, (const int4*) p->input_gpu, (int*) p->output_gpu, \
                    PAD8(p->batch)*p->input_height, p->input_width, p->input_channels, p->output_channels); 
}


__global__ 
void Output_new_global(Out128LayerParam* p){
    apmm_w1a2_decompose((const int4*) p->input_gpu, (const int4*) p->weight_gpu, (int*) p->output_gpu, 
                        8*STEP8(p->input_height), STEP8(p->weight_width), STEP128(p->input_width), 1, 2);
}


__device__ __inline__ 
void FC_new(Fc128LayerParam* p){
    apmm_w1a2_decompose((const int4*) p->input_gpu, (const int4*) p->weight_gpu, (int*) p->output_gpu, 
    8*STEP8(p->input_height), STEP8(p->weight_width), STEP128(p->input_width), 1, 2);
}


__global__ 
void FC_new_global(Fc128LayerParam* p){

    apmm_w1a2_decompose((const int4*) p->input_gpu, (const int4*) p->weight_gpu, (int*) p->output_gpu, 
    8*STEP8(p->input_height), STEP8(p->weight_width), STEP128(p->input_width), 1, 2);
}


__device__ __inline__ void In128Layer(In128LayerParam* p)
{
    GET_LANEID;
    GET_WARPID;
    const int gdx = STEP8(p->input_height);
    const int gdy = STEP128(p->input_width);
    const int lx = (warpid>>2);
    const int ly = (warpid&0x3);
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid / gdy;
        const int by = bid % gdy;
        float f0 = ( (by*128+ly*32+laneid<(p->input_width)) 
                &&   (bx*8+lx<(p->input_height)) )?
            p->input_gpu[(bx*8+lx)*(p->input_width)+by*128+ly*32+laneid]:-1.0f;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0));
        if (laneid==0)
            p->output_gpu[(bx*8+lx)*gdy*4+by*4+ly] = r0;
    }
}


__device__ __inline__ void Fc128Layer(Fc128LayerParam* p)
{
    using namespace nvcuda;
    GET_LANEID;
    GET_WARPID;
    using namespace nvcuda::wmma::experimental;
    //__shared__ int Cs[32][64];
    extern __shared__ int Cs[];
    const int gdx = STEP8(p->input_height); //vertical
    const int gdy = STEP8(p->weight_width); //horizontal
    const int gdk = STEP128(p->input_width);
    const int gdm = STEP128(p->weight_width);

    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
        wmma::fill_fragment(c_frag, 0);
        const int bx = bid / gdy;
        const int by = bid % gdy;

        for (int i=0; i<gdk; i++)
        {
            load_matrix_sync(a_frag, p->input_gpu + bx*8*gdk*4 + i*128/32, gdk*128);
            load_matrix_sync(b_frag, p->weight_gpu + by*8*gdk*4 + i*128/32, gdk*128);
            bmma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);
        uin8* Cb = (uin8*)(&(p->output_gpu[0])); //?

        const int gy = (laneid%8);
        const int gx = (laneid/8);
        bool v0_in = ((by*8+gy)<(p->output_width)) && ((bx*8+gx)<(p->output_height));
        bool v1_in = ((by*8+gy)<(p->output_width)) && ((bx*8+gx+4)<(p->output_height)); 
        bool v0 = v0_in && ((((float)p->input_width)-2*(float)Cs[warpid*64+laneid])
                        >=(p->bn_gpu[by*8+gy]));
        bool v1 = v1_in && ((((float)p->input_width)-2*(float)Cs[warpid*64+32+laneid])
                        >=(p->bn_gpu[by*8+gy]));

        union{ uin32 data; uin8 elements[4];} p0, p1;
        p0.data = __brev(__ballot_sync(0xFFFFFFFF, v0));
        p1.data = __brev(__ballot_sync(0xFFFFFFFF, v1));
        __syncthreads();

        if (laneid < 4)
        {
            Cb[(bx*8+laneid)*gdm*16+FLIPBITS(by,2)] = p0.elements[3-laneid]; 
            Cb[(bx*8+4+laneid)*gdm*16+FLIPBITS(by,2)] = p1.elements[3-laneid]; 
        }
        //end
    }
}

__device__ __inline__ void Out128Layer(Out128LayerParam* p)
{
    using namespace nvcuda;
    GET_LANEID;
    GET_WARPID;
    using namespace nvcuda::wmma::experimental;
    extern __shared__ int Cs[];
    
    const int gdx = STEP8(p->input_height); //vertical
    const int gdy = STEP8(p->weight_width); //horizontal
    const int gdk = STEP128(p->input_width);

    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;

        wmma::fill_fragment(c_frag, 0);
        const int bx = bid / gdy;
        const int by = bid % gdy;

        for (int i=0; i<gdk; i++)
        {
            load_matrix_sync(a_frag, p->input_gpu + bx*8*gdk*4 + i*128/32, gdk*128);
            load_matrix_sync(b_frag, p->weight_gpu + by*8*gdk*4 + i*128/32, gdk*128);

            bmma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);
        float* output_sub = &(p->output_gpu[bx*(p->weight_width)*8+by*8]);

        if (laneid < 8)
        {
            for (int j=0; j<8; j++)
            {
                if ((bx*8+j)<(p->input_height))
                {
                    if (by*8+laneid<(p->weight_width))
                    {
                        float val = ((float)(p->input_width) 
                                - ((float)Cs[warpid*64+j*8+laneid])*2.0f)*
                                    (p->bn_scale_gpu[by*8+laneid]) 
                                + (p->bn_bias_gpu[by*8+laneid]);
                        output_sub[j*(p->weight_width)+laneid] = val;
                    }
                }
            }
        } //end
    }
}

//=================================== FMT ================================
__device__ __inline__ void In128LayerFMT(In128LayerParam* p)
{
    GET_LANEID;
    GET_WARPID;
    const int gdx = STEP8(p->input_height);
    const int gdy = STEP128(p->input_width);
    const int lx = (warpid>>2);
    const int ly = (warpid&0x3);
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid / gdy;
        const int by = bid % gdy;
        float f0 = ( (by*128+ly*32+laneid<(p->input_width)) 
                &&   (bx*8+lx<(p->input_height)) )?
            p->input_gpu[(bx*8+lx)*(p->input_width)+by*128+ly*32+laneid]:-1.0f;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0));
        //New format
        if (laneid==0) p->output_gpu[(bx*gdy+by)*32+warpid] = r0;
    }
}

__device__ __inline__ void Fc128LayerFMT(Fc128LayerParam* p)
{
    using namespace nvcuda;
    GET_LANEID;
    GET_WARPID;
    using namespace nvcuda::wmma::experimental;
    extern __shared__ int Cs[];

    const int gdx = STEP8(p->input_height); //vertical
    const int gdy = STEP8(p->weight_width); //horizontal
    const int gdk = STEP128(p->input_width);
    const int gdm = STEP128(p->weight_width);

    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;

        wmma::fill_fragment(c_frag, 0);
        const int bx = bid / gdy;
        const int by = bid % gdy;

        for (int i=0; i<gdk; i++)
        {
            load_matrix_sync(a_frag, p->input_gpu + bx*8*gdk*4 + i*128*8/32, 128);
            load_matrix_sync(b_frag, p->weight_gpu + by*8*gdk*4 + i*128*8/32, 128);
            bmma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);
        uin8* Cb = (uin8*)(&(p->output_gpu[0])); //?

        const int gy = (laneid%8);
        const int gx = (laneid/8);
        bool v0_in = ((by*8+gy)<(p->output_width)) && ((bx*8+gx)<(p->output_height));
        bool v1_in = ((by*8+gy)<(p->output_width)) && ((bx*8+gx+4)<(p->output_height)); 
        bool v0 = v0_in && ((((float)p->input_width)-2*(float)Cs[warpid*64+laneid])
                        >=(p->bn_gpu[by*8+gy]));
        bool v1 = v1_in && ((((float)p->input_width)-2*(float)Cs[warpid*64+32+laneid])
                        >=(p->bn_gpu[by*8+gy]));

        union{ uin32 data; uin8 elements[4];} p0, p1;
        p0.data = __brev(__ballot_sync(0xFFFFFFFF, v0 ));
        p1.data = __brev(__ballot_sync(0xFFFFFFFF, v1 ));
        __syncthreads();

        if (laneid < 4)
        {
            Cb[(bx*gdm + by/16)*128+ laneid*16 + FLIPBITS((by%16),2)] = p0.elements[3-laneid];
            Cb[(bx*gdm + by/16)*128+ (laneid+4)*16 + FLIPBITS((by%16),2)] = p1.elements[3-laneid];
        }
    }
}



__device__ __inline__ void Out128LayerFMT(Out128LayerParam* p)
{
    using namespace nvcuda;
    GET_LANEID;
    GET_WARPID;
    using namespace nvcuda::wmma::experimental;
    extern __shared__ int Cs[];
    
    const int gdx = STEP8(p->input_height); //vertical
    const int gdy = STEP8(p->weight_width); //horizontal
    const int gdk = STEP128(p->input_width);

    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;

        wmma::fill_fragment(c_frag, 0);
        const int bx = bid / gdy;
        const int by = bid % gdy;

        for (int i=0; i<gdk; i++)
        {
            load_matrix_sync(a_frag, p->input_gpu + bx*8*gdk*4 + i*128*8/32, 128);
            load_matrix_sync(b_frag, p->weight_gpu + by*8*gdk*4 + i*128*8/32, 128);

            bmma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);
        float* output_sub = &(p->output_gpu[bx*(p->weight_width)*8+by*8]);

        if (laneid < 8)
        {
            for (int j=0; j<8; j++)
            {
                if ((bx*8+j)<(p->input_height))
                {
                    if (by*8+laneid<(p->weight_width))
                    {
                        output_sub[j*(p->weight_width)+laneid] = ((float)(p->input_width) 
                                - (float)Cs[warpid*64+j*8+laneid]*2.0f)*
                                    (p->bn_scale_gpu[by*8+laneid]) 
                                + (p->bn_bias_gpu[by*8+laneid]);
                    }
                }
            }
        }
        //end
    }
}


//================================ Convolution ====================================


__device__ __inline__ void InConv128Layer(InConv128LayerParam* p)
{
    GET_LANEID;
    GET_WARPID;
    extern __shared__ int Cs[];
    const int ots = STEP32(p->output_channels); //number of steps in K: output_channels
    const int otm = STEP128(p->output_channels);
    volatile float* Csub = (float*)&Cs[warpid*(p->output_channels)];
    volatile uin32* s_filter = (uin32*)&Cs[32*(p->output_channels)]; 
    const int src_output_height = (p->pool_height)*(p->output_height);
    const int src_output_width = (p->pool_width)*(p->output_width);

    for (int i=threadIdx.x; i<(p->filter_height)*(p->filter_width)
            *(p->input_channels)*ots; i+=32*32) 
        s_filter[i] = p->filter_gpu[i];
    __syncthreads();

    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*(p->batch);
            bid += gridDim.x*32)
    {
        const int bz = bid/(src_output_width*src_output_height); //over N:batch
        const int by = (bid%(src_output_width*src_output_height))
            /(src_output_width);//over P:out_height
        const int bx = (bid%(src_output_width*src_output_height))
            %(src_output_width);//over Q:out_width 

        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);

        for (int i=laneid; i<(p->output_channels); i+=32) Csub[i] = 0;

        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            if ((ay>=0) && (ay<(p->input_height))) 
            {
                for (int s=0; s<(p->filter_width); s++)
                {
                    const int ax = ax0 + s; //x-coord in Input
                    //within Input frame
                    if ((ax>=0) && (ax<(p->input_width)) )
                    {
                        float f0 = p->input_gpu[(bz*3+0)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//R
                        float f1 = p->input_gpu[(bz*3+1)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//G
                        float f2 = p->input_gpu[(bz*3+2)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//B

                        for (int k=0; k<ots; k++)
                        {
                            uin32 l0 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 0*ots+k];
                            uin32 l1 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 1*ots+k];
                            uin32 l2 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 2*ots+k];

                            Csub[32*k+laneid] += (((l0>>(31-laneid))&0x1)?f0:-f0)
                                + (((l1>>(31-laneid))&0x1)?f1:-f1)
                                + (((l2>>(31-laneid))&0x1)?f2:-f2);
                        }
                    }
                }
            }
        }

        // To shape[batch, input_height, input_width, in_channels/32]
        const int dst_y = by/(p->pool_height);
        const int dst_x = bx/(p->pool_width);

        //const int idx = (bz*(p->output_height)*(p->output_width) //N
        //+dst_y*(p->output_width) //P
        //+dst_x)*ots; //Q

        // To shape[input_height, input_width, batch, in_channels/32]
        const int idx = (dst_y*(p->output_width)*PAD8(p->batch) //P
                +dst_x*PAD8(p->batch) //Q
                +bz)*otm*4;
        for (int k=0; k<ots; k++)
        {
            // save shape[batch, output_height, output_width, out_channels/64]
            bool bin = (float)(Csub[k*32+laneid])<(p->bn_gpu)[k*32+laneid]?0:1;
            unsigned C = __brev(__ballot_sync( 0xFFFFFFFF, bin));
            if (laneid==0) atomicOr(&p->output_gpu[idx+k], C); //Q
        }
        if (p->save_residual)
        {
            for (int k=0; k<ots; k++)
            {
                p->output_residual_gpu[(by*(p->output_width)+bx)*PAD8(p->batch)
                    *PAD128(p->output_channels) + bz*PAD128(p->output_channels)
                    + k*32 + laneid] = Csub[k*32+laneid];
            }
        }

    }
}

__device__ __inline__ void Conv128Layer(Conv128LayerParam* p)
{
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;
    GET_LANEID;
    GET_WARPID;
    const int ins = STEP128(p->input_channels); //1
    const int ots = STEP32(p->output_channels); //4
    const int bas = STEP8(p->batch);//4
    const int src_output_height = (p->pool_height)*(p->output_height);//32
    const int src_output_width = (p->pool_width)*(p->output_width);//32
    extern __shared__ int Cs[];
    volatile int* Csub = (int*)Cs[warpid];

    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*ots*bas;
            bid += gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b1_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b2_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b3_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c1_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c2_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c3_frag;

        const int by = bid/(src_output_width*ots*bas); //P: output_height
        const int bx = (bid%(src_output_width*ots*bas))/(ots*bas); //Q:output_width
        const int bz = (bid%(src_output_width*ots*bas))%(ots*bas); //output_channel/32*batch/8
        const int bn = bz / ots; //N:batch (8)
        const int bo = bz % ots; //O:out_channel (4*8)

        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);
        //track the number of filter entries that are masked off
        int exclude = 0;
        wmma::fill_fragment(c0_frag, 0);
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c2_frag, 0);
        wmma::fill_fragment(c3_frag, 0);

        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            for (int s=0; s<(p->filter_width); s++)
            {
                const int ax = ax0 + s; //x-coord in Input
                if ((ay>=0)&&(ay<(p->input_height))&&(ax>=0)&&(ax<(p->input_width)))
                {
                    for (int c=0; c<ins; c++)
                    {
                        //input: [H,W,N,C], filter: [K,K,O,C]
                        load_matrix_sync(a_frag, 
                            &(p->input_gpu[(ay*(p->input_width)+ax)*bas*8*ins*4+8*bn*ins*4+c*4]),
                            ins*128);
                        load_matrix_sync(b0_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+0)*ins*4+c*4]), ins*128);
                        bmma_sync(c0_frag, a_frag, b0_frag, c0_frag);
                        load_matrix_sync(b1_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+8)*ins*4+c*4]), ins*128);
                        bmma_sync(c1_frag, a_frag, b1_frag, c1_frag);
                        load_matrix_sync(b2_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+16)*ins*4+c*4]), ins*128);
                        bmma_sync(c2_frag, a_frag, b2_frag, c2_frag);
                        load_matrix_sync(b3_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+24)*ins*4+c*4]), ins*128);
                        bmma_sync(c3_frag, a_frag, b3_frag, c3_frag);
                    }
                }
                else //not in frame
                {
                    exclude++; //accumulate
                }
            }
        }
        store_matrix_sync(&Cs[warpid*256+0], c0_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+8], c1_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+16], c2_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+24], c3_frag, 32, wmma::mem_row_major);
        __syncthreads();

        for (int b=0; b<8; b++)
        {
            int res = (int)(p->input_channels)*(p->filter_width)*(p->filter_height) //C*R*S
                - (int)exclude*(p->input_channels) //eliminate padding distoration 
                - (int)(2*Cs[warpid*256+b*32+laneid]);//n-2acc(a^b) for 0/1 to sim +1/-1

            
            if (p->inject_residual && ((bo*32+laneid)<(p->residual_channels)))
            {
                int residual = 0;
                if (p->residual_pool)
                {
                /*

                    //if((bn*8+b)<(p->batch) && (bo*32+laneid)<(p->residual_channels))
                    {
                    int pl0 = p->input_residual_gpu[(by*2+0)*2*(p->output_width)*
                            bas*8*PAD128(p->residual_channels)
                            +(bx*2+0)*bas*8*PAD128(p->residual_channels)
                            +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid];

                    int pl1 = p->input_residual_gpu[(by*2+0)*2*(p->output_width)*
                            bas*8*PAD128(p->residual_channels)
                            +(bx*2+1)*bas*8*PAD128(p->residual_channels)
                            +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid];

                    int pl2 = p->input_residual_gpu[(by*2+1)*2*(p->output_width)*
                            bas*8*PAD128(p->residual_channels)
                            +(bx*2+0)*bas*8*PAD128(p->residual_channels)
                            +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid];

                    int pl3 = p->input_residual_gpu[(by*2+1)*2*(p->output_width)*
                            bas*8*PAD128(p->residual_channels)
                            +(bx*2+1)*bas*8*PAD128(p->residual_channels)
                            +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid];
                    residual = max(pl3,max(pl2,max(pl0,pl1)));
                    }
                    */

                    residual = INT_MIN;
                    for (int i=0; i<2; i++)
                        for( int j=0; j<2; j++)
                            residual = max(residual, p->input_residual_gpu[
                                    (by*2+i)*2*(p->output_width)*
                                    bas*8*PAD128(p->residual_channels)
                                    +(bx*2+j)*bas*8*PAD128(p->residual_channels)
                                    +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid]);

                }
                else
                {
                    //residual = p->input_residual_gpu[by*(p->output_width)
                    residual = p->input_residual_gpu[by*src_output_width
                        *bas*8*PAD128(p->residual_channels)
                        +bx*bas*8*PAD128(p->residual_channels)
                        +(bn*8+b)*PAD128(p->residual_channels)
                        +bo*32+laneid];
                }
                res += residual;
            }

            unsigned C = __brev(__ballot_sync( 0xFFFFFFFF, 
                        (float)res<(p->bn_gpu[bo*32+laneid])?0:1));


            if (p->ahead_fc)
            {
                if (laneid==0) //For FC layer BHWC
                    atomicOr(&(p->output_gpu[(bn*8+b)*(p->output_height)*(p->output_width)
                        *STEP128(p->output_channels)*4
                        + ((by/(p->pool_height))*(p->output_width)*STEP128(p->output_channels)*4)
                        + ((bx/(p->pool_width))*STEP128(p->output_channels)*4)
                        + bo]),C);

                    //atomicOr(&p->output_gpu[((by/(p->pool_height))*(p->output_width)
                    //*bas*8*STEP128(p->output_channels)*4) //P
                    //+ ((bx/(p->pool_width))*bas*8*STEP128(p->output_channels)*4) //Q
                    //+ bo*bas*8 + (bn*8+b)],C);


            }
            else
            {
                if (laneid==0) //For normal convolution layer HWBC
                    atomicOr(&p->output_gpu[((by/(p->pool_height))*(p->output_width)
                                *bas*8*STEP128(p->output_channels)*4) //P
                            + ((bx/(p->pool_width))*bas*8*STEP128(p->output_channels)*4) //Q
                            + (bn*8+b)*STEP128(p->output_channels)*4 + bo],C);
            }

            if (p->save_residual)
            {
                p->output_residual_gpu[by*(p->output_width)*bas*8*PAD128(p->output_channels)
                    + bx*bas*8*PAD128(p->output_channels)
                    + (bn*8+b)*PAD128(p->output_channels) 
                    + bo*32 + laneid] = res;
            }
        }
    }
}

//================================ Convolution FMT ====================================

__device__ __inline__ void InConv128LayerFMT(InConv128LayerParam* p)
{
    GET_LANEID;
    GET_WARPID;
    extern __shared__ int Cs[];
    const int ots = STEP32(p->output_channels); //number of steps in K: output_channels
    const int otm = STEP128(p->output_channels);
    volatile float* Csub = (float*)&Cs[warpid*(p->output_channels)];
    volatile uin32* s_filter = (uin32*)&Cs[32*(p->output_channels)]; 
    const int src_output_height = (p->pool_height)*(p->output_height);
    const int src_output_width = (p->pool_width)*(p->output_width);

    for (int i=threadIdx.x; i<(p->filter_height)*(p->filter_width)
            *(p->input_channels)*ots; i+=32*32) 
        s_filter[i] = p->filter_gpu[i];
    __syncthreads();

    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*(p->batch);
            bid += gridDim.x*32)
    {
        const int bz = bid/(src_output_width*src_output_height); //over N:batch
        const int by = (bid%(src_output_width*src_output_height))
            /(src_output_width);//over P:out_height
        const int bx = (bid%(src_output_width*src_output_height))
            %(src_output_width);//over Q:out_width 
        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);
        for (int i=laneid; i<(p->output_channels); i+=32) Csub[i] = 0;

        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            if ((ay>=0) && (ay<(p->input_height))) 
            {
                for (int s=0; s<(p->filter_width); s++)
                {
                    const int ax = ax0 + s; //x-coord in Input
                    //within Input frame
                    if ((ax>=0) && (ax<(p->input_width)) )
                    {
                        float f0 = p->input_gpu[(bz*3+0)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//R
                        float f1 = p->input_gpu[(bz*3+1)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//G
                        float f2 = p->input_gpu[(bz*3+2)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//B

                        for (int k=0; k<ots; k++)
                        {
                            uin32 l0 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 0*ots+k];
                            uin32 l1 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 1*ots+k];
                            uin32 l2 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 2*ots+k];

                            Csub[32*k+laneid] += (((l0>>(31-laneid))&0x1)?f0:-f0)
                                + (((l1>>(31-laneid))&0x1)?f1:-f1)
                                + (((l2>>(31-laneid))&0x1)?f2:-f2);
                        }
                    }
                }
            }
        }

        // To shape[batch, input_height, input_width, in_channels/32]
        const int dst_y = by/(p->pool_height);
        const int dst_x = bx/(p->pool_width);

        // To shape[input_height, input_width, batch/8*in_channels/128, batch8*in_channels128/32]
        const int idx = dst_y*(p->output_width)*PAD8(p->batch)*otm*4 //P
                +dst_x*PAD8(p->batch)*otm*4; //Q
        for (int k=0; k<ots; k++)
        {
            // save shape[batch, output_height, output_width, out_channels/64]
            bool bin = (float)(Csub[k*32+laneid])<(p->bn_gpu)[k*32+laneid]?0:1;
            unsigned C = __brev(__ballot_sync(0xFFFFFFFF, bin));

            //if (laneid==0) atomicOr(&p->output_gpu[idx+bz*otm*4+k], C); //Q
            if (laneid==0) atomicOr(&p->output_gpu[idx
                    +((bz/8)*otm+k/4)*32+((bz%8)*4+k%4)], C); //Q
        }
        if (p->save_residual)
        {
            for (int k=0; k<ots; k++)
            {
                p->output_residual_gpu[(by*(p->output_width)+bx)*PAD8(p->batch)
                    *PAD128(p->output_channels) + bz*PAD128(p->output_channels)
                    + k*32 + laneid] = Csub[k*32+laneid];
            }
        }
    }
}



__device__ __inline__ void Conv128LayerFMT(Conv128LayerParam* p)
{
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;
    GET_LANEID;
    GET_WARPID;
    const int ins = STEP128(p->input_channels); //1
    const int ots = STEP32(p->output_channels); //4
    const int bas = STEP8(p->batch);//4
    const int src_output_height = (p->pool_height)*(p->output_height);//32
    const int src_output_width = (p->pool_width)*(p->output_width);//32
    extern __shared__ int Cs[];
    volatile int* Csub = (int*)Cs[warpid];

    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*ots*bas;
            bid += gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b1_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b2_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b3_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c1_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c2_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c3_frag;

        const int by = bid/(src_output_width*ots*bas); //P: output_height
        const int bx = (bid%(src_output_width*ots*bas))/(ots*bas); //Q:output_width
        const int bz = (bid%(src_output_width*ots*bas))%(ots*bas); //output_channel/32*batch/8
        const int bn = bz / ots; //N:batch (8)
        const int bo = bz % ots; //O:out_channel (4*8)

        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);
        //track the number of filter entries that are masked off
        int exclude = 0;
        wmma::fill_fragment(c0_frag, 0);
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c2_frag, 0);
        wmma::fill_fragment(c3_frag, 0);

        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            for (int s=0; s<(p->filter_width); s++)
            {
                const int ax = ax0 + s; //x-coord in Input
                if ((ay>=0)&&(ay<(p->input_height))&&(ax>=0)&&(ax<(p->input_width)))
                {
                    for (int c=0; c<ins; c++)
                    {
                        //input: [H,W,N,C], filter: [K,K,O,C]
                        load_matrix_sync(a_frag, 
                            &(p->input_gpu[(ay*(p->input_width)+ax)*bas*8*ins*4
                            +8*bn*ins*4+c*4*8]), 128);
                        load_matrix_sync(b0_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+0)*ins*4+c*4*8]), 128);
                        bmma_sync(c0_frag, a_frag, b0_frag, c0_frag);
                        load_matrix_sync(b1_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+8)*ins*4+c*4*8]), 128);
                        bmma_sync(c1_frag, a_frag, b1_frag, c1_frag);
                        load_matrix_sync(b2_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+16)*ins*4+c*4*8]), 128);
                        bmma_sync(c2_frag, a_frag, b2_frag, c2_frag);
                        load_matrix_sync(b3_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+24)*ins*4+c*4*8]), 128);
                        bmma_sync(c3_frag, a_frag, b3_frag, c3_frag);
                    }
                }
                else //not in frame
                {
                    exclude++; //accumulate
                }
            }
        }
        store_matrix_sync(&Cs[warpid*256+0], c0_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+8], c1_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+16], c2_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+24], c3_frag, 32, wmma::mem_row_major);
        __syncthreads();

        for (int b=0; b<8; b++)
        {
            int res = (int)(p->input_channels)*(p->filter_width)*(p->filter_height) //C*R*S
                - (int)exclude*(p->input_channels) //eliminate padding distoration 
                - (int)(2*Cs[warpid*256+b*32+laneid]);//n-2acc(a^b) for 0/1 to sim +1/-1
            
            if (p->inject_residual && ((bo*32+laneid)<(p->residual_channels)))
            {
                int residual = 0;
                if (p->residual_pool)
                {
                    residual = INT_MIN;
                    for (int i=0; i<2; i++)
                        for( int j=0; j<2; j++)
                            residual = max(residual, p->input_residual_gpu[
                                    (by*2+i)*2*(p->output_width)*
                                    bas*8*PAD128(p->residual_channels)
                                    +(bx*2+j)*bas*8*PAD128(p->residual_channels)
                                    +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid]);
                }
                else
                {
                    //residual = p->input_residual_gpu[by*(p->output_width)
                    residual = p->input_residual_gpu[by*src_output_width
                        *bas*8*PAD128(p->residual_channels)
                        +bx*bas*8*PAD128(p->residual_channels)
                        +(bn*8+b)*PAD128(p->residual_channels)
                        +bo*32+laneid];
                }
                res += residual;
            }
            unsigned C = __brev(__ballot_sync(0xFFFFFFFF,
                        (float)res<(p->bn_gpu[bo*32+laneid])?0:1));

            if (p->ahead_fc)
            {
                if (laneid==0)
                {
                    int otm = (p->output_height)*(p->output_width)*STEP128(p->output_channels);
                    int k = ((by/(p->pool_height))*(p->output_width)
                            *STEP128(p->output_channels)*4)
                        + ((bx/(p->pool_width))*STEP128(p->output_channels)*4) + bo;
                    atomicOr(&(p->output_gpu[(bn*otm+(k/4))*32+b*4+(k%4)]),C);
                }
            }
            else
            {
                if (laneid==0) //For normal convolution layer HWBC
                    atomicOr(&p->output_gpu[((by/(p->pool_height))*(p->output_width)
                                *bas*8*STEP128(p->output_channels)*4) //P
                            + ((bx/(p->pool_width))*bas*8*STEP128(p->output_channels)*4) //Q
                            + (bn*STEP128(p->output_channels)+(bo/4))*32 + b*4+(bo%4)],C);
            }

            if (p->save_residual)
            {
                p->output_residual_gpu[by*(p->output_width)*bas*8*PAD128(p->output_channels)
                    + bx*bas*8*PAD128(p->output_channels)
                    + (bn*8+b)*PAD128(p->output_channels) 
                    + bo*32 + laneid] = res;
            }
        }
    }
}










#endif
