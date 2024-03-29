# APNN-TC: Accelerating Arbitrary Precision Neural Networks on Ampere GPU Tensor Cores
[**[Paper on arXiv]**](https://arxiv.org/abs/2106.12169)
```
@inproceedings{APNN-TC,
  title={APNN-TC: Accelerating Arbitrary Precision Neural Networks on Ampere GPU Tensor Cores},
  author={Boyuan Feng, Yuke Wang, Tong Geng, Ang Li, Yufei Ding.},
  booktitle={The International Conference for High Performance Computing, Networking, Storage, and Analysis. (SC'21)},
  year={2021}
}
```

## Clone this project.
```
git clone --recursive git@github.com:BoyuanFeng/APNN-TC.git
cd APNN-TC-kernel && git checkout main
```
in case of missing `--recursive` during the clone
```
git submodule init
git submodule update
```

## OS & Compiler:
+ `Ubuntu 16.04+`
+ `gcc >= 7.5`
+ `make >= 4.2.1`
+ `CUDA >= 11.0`
+ `libjpeg`
+ `cuDNN == 8.2`

## Files & Directory
+ `APNN-TC-kernel/`: our APNN-TC GEMM and CONV kernels with different bit combinations.
+ `APNN-TC/`: our APNN-TC NN low-bit model (AlextNet, VGG-variant, ResNet18) with `w1a2` for demonstration.
+ `cutlass/`: CUTLASS header and source files.
+ `cutlass_kernel/`: CUTLASS baselines GEMM and CONV kernels, including INT4 and INT1.
+ `cutlass_nn/`: CUTLASS baselines NN models, including FP32, FP16, and INT8.


## Setup Environment.
+ Install NVIDIA Docker.
```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

+ Build and Launch Docker.
```
cd Docker/
build.sh
launch.sh
```
or pull docker image from docker hub and launch.
```
docker pull happy233/apnn-tc:main
docker run -it --rm --gpus all -v $(PWD):/apnn-tc happy233/apnn-tc:main /bin/bash
```

# Experiments
## APNN-TC -- GEMM and CONV kernel
+ `cd APNN-TC-kernel && make`
+ Run `./gemm-w1a2.out`, `./gemm-w1a3.out`,`./gemm-w1a4.out`,`./gemm-w2a2.out`. Note that `w1a2` means 1-bit weight and 2-bit activation.
+ Run `./conv-w1a2.out`, `./conv-w1a3.out`, `./conv-w1a4_small.out`, `./conv-w1a4_large.out`,`./conv-w2a2_small.out`, `./conv-w2a2_large.out`
> Note that 
> + for GEMM kernel, we profile the GEMM shape as `[M, N, K]` as `[64, N, K]`, where `N=K=[128,256,384,...,1024]`.
> + for CONV kernel, we profile the CONV shape with on feature map with `[H, W] = [16,16]` and the kernel size is `[O, C, K, K] = [O, C, 3, 3]`, where `O=C=[128,256,384,...,1024]`. 
> + `conv-w1a4_small.out` is for `w1a4` in `IN=COUT=[128,..., 640]`, 
> + `conv-w1a4_large.out` is for `w1a4` in `IN=COUT>=640`
> + `conv-w2a2_small.out` is for `w2a2` in `IN=COUT=[128,..., 640]`
> + `conv-w2a2_large.out` is for `w2a2` in `IN=COUT>=640`

## CUTLASS -- GEMM kernel
+ `cd bench_cutlass/`
+ `make all`
+ `./run-gemm.py`
+  Select the precision (`INT4` and `INT1`) of CUTLASS, open **`cutlass_kernel/bench_gemm.cu`** and comment out other unused bitwidth (default 4-bit).
```
// #define BIT_WIDTH 1
#define BIT_WIDTH 4
```

## CUTLASS -- CONV kernel
+ `cd bench_cutlass/`
+ `make all`
+ `./run-conv.py`
+  Select the precision (`INT4` and `INT1`) of CUTLASS, open **`cutlass_kernel/bench_conv.cu`** and comment out other unused bitwidth (default 4-bit).
```
// #define BIT_WIDTH 1
#define BIT_WIDTH 4
```

## APNN-TC -- NN model  
+ Build and run the network with `w1a2` APNN (Table-2). 
+ `cd APNN-TC && make`
+ `./alexnet` to run alexnet with `w1a2`.
+ `./vgg` to run VGG-variant in `w1a2`. 
+ `./resnet18` to run ResNet18 in `w1a2`.

## CUTLASS -- NN model  
+ Build and run CUTLASS baseline.
+ Run `cd cutlass_baselines && make`
+  Select the precision (FP32, FP16, INT8) of CUTLASS, `cd cutlass_nn/src/config.h` and comment out other two unused bitwidth.
```
#define BIT_WIDTH 32
// #define BIT_WIDTH 16
// #define BIT_WIDTH 8
```

+ Run `./alexnet` to run AlexNet in (FP32, FP16, INT8).
+ Run `./vgg_variant` to run VGG-variant in (FP32, FP16, INT8).
+ Run `./resnet18` to run ResNet18 in (FP32, FP16, INT8).

# Expected Result.
## APNN-TC vs CUTLASS on GEMM kernel. 
+ cutlass-GEMM-int4
```
CUTLASS-GEMM (4-bit). M:     64, N:    128, K:    128,   Time (ms): 0.01, TOPS: 0.35
CUTLASS-GEMM (4-bit). M:     64, N:    256, K:    256,   Time (ms): 0.01, TOPS: 1.14
CUTLASS-GEMM (4-bit). M:     64, N:    384, K:    384,   Time (ms): 0.01, TOPS: 2.21
CUTLASS-GEMM (4-bit). M:     64, N:    512, K:    512,   Time (ms): 0.01, TOPS: 3.43
CUTLASS-GEMM (4-bit). M:     64, N:    640, K:    640,   Time (ms): 0.01, TOPS: 4.77
CUTLASS-GEMM (4-bit). M:     64, N:    768, K:    768,   Time (ms): 0.01, TOPS: 6.20
CUTLASS-GEMM (4-bit). M:     64, N:    896, K:    896,   Time (ms): 0.01, TOPS: 7.67
CUTLASS-GEMM (4-bit). M:     64, N:   1024, K:   1024,   Time (ms): 0.01, TOPS: 9.18
```
+ APNN-TC-GEMM-w1a2
```
V30, 64x64. M_GLOBAL: 64, N_GLOBAL: 128, K_GLOBAL: 128, X_BIT: 2, W_BIT: 1, Time: 0.004708 ms, TOPS: 0.45
V30, 64x64. M_GLOBAL: 64, N_GLOBAL: 256, K_GLOBAL: 256, X_BIT: 2, W_BIT: 1, Time: 0.004964 ms, TOPS: 1.69
V30, 64x64. M_GLOBAL: 64, N_GLOBAL: 384, K_GLOBAL: 384, X_BIT: 2, W_BIT: 1, Time: 0.005370 ms, TOPS: 3.52
V30, 64x64. M_GLOBAL: 64, N_GLOBAL: 512, K_GLOBAL: 512, X_BIT: 2, W_BIT: 1, Time: 0.005512 ms, TOPS: 6.09
V30, 64x64. M_GLOBAL: 64, N_GLOBAL: 640, K_GLOBAL: 640, X_BIT: 2, W_BIT: 1, Time: 0.006140 ms, TOPS: 8.54
V30, 64x64. M_GLOBAL: 64, N_GLOBAL: 768, K_GLOBAL: 768, X_BIT: 2, W_BIT: 1, Time: 0.006171 ms, TOPS: 12.23
V30, 64x64. M_GLOBAL: 64, N_GLOBAL: 896, K_GLOBAL: 896, X_BIT: 2, W_BIT: 1, Time: 0.006805 ms, TOPS: 15.10
V30, 64x64. M_GLOBAL: 64, N_GLOBAL: 1024, K_GLOBAL: 1024, X_BIT: 2, W_BIT: 1, Time: 0.007194 ms, TOPS: 18.66
```

## APNN-TC vs CUTLASS on CONV kernel. 
+ cutlass-CONV-int4 
```
Precision,      Layer,  N,      H,      W,      C,      K,      R,      S,      Runtime,        TFLOPs
BIT_WIDTH-4,    conv_1, 1,      16,     16,     128,    128,    3,      3,      0.0144896,      5.21046
BIT_WIDTH-4,    conv_2, 1,      16,     16,     256,    256,    3,      3,      0.02304,        13.1072
BIT_WIDTH-4,    conv_3, 1,      16,     16,     384,    384,    3,      3,      0.031592,       21.5079
BIT_WIDTH-4,    conv_4, 1,      16,     16,     512,    512,    3,      3,      0.0401408,      30.0931
BIT_WIDTH-4,    conv_5, 1,      16,     16,     640,    640,    3,      3,      0.04864,        38.8042
BIT_WIDTH-4,    conv_6, 1,      16,     16,     768,    768,    3,      3,      0.0572416,      47.4814
BIT_WIDTH-4,    conv_7, 1,      16,     16,     896,    896,    3,      3,      0.065792,       56.2284
BIT_WIDTH-4,    conv_8, 1,      16,     16,     1024,   1024,   3,      3,      0.0743424,      64.9944
```
+ APNN-TC-CONV-w1a2
```
H: 16, W: 16, CIN: 128, COUT: 128, W_BIT: 1, X_BIT: 2, Time: 0.006213 ms, TOPS: 12.15
H: 16, W: 16, CIN: 256, COUT: 256, W_BIT: 1, X_BIT: 2, Time: 0.008126 ms, TOPS: 37.16
H: 16, W: 16, CIN: 384, COUT: 384, W_BIT: 1, X_BIT: 2, Time: 0.010251 ms, TOPS: 66.29
H: 16, W: 16, CIN: 512, COUT: 512, W_BIT: 1, X_BIT: 2, Time: 0.010370 ms, TOPS: 116.48
H: 16, W: 16, CIN: 640, COUT: 640, W_BIT: 1, X_BIT: 2, Time: 0.013166 ms, TOPS: 143.35
H: 16, W: 16, CIN: 768, COUT: 768, W_BIT: 1, X_BIT: 2, Time: 0.024899 ms, TOPS: 109.16
H: 16, W: 16, CIN: 896, COUT: 896, W_BIT: 1, X_BIT: 2, Time: 0.028499 ms, TOPS: 129.81
H: 16, W: 16, CIN: 1024, COUT: 1024, W_BIT: 1, X_BIT: 2, Time: 0.025389 ms, TOPS: 190.31
```

## APNN-TC vs CUTLASS on NN model. 
+ Here we demonstrate an example with `APNN-w1a2` and `cutlass-FP32` and `cutlass-fp16` on `AlexNet` and `VGG_variant`.

|                | AlexNet(ms) | VGG(ms)    |
|:----------------|---------:|--------:|
| cutlass-32     |    4.26 |  25.22 |
| cutlass-16     |    3.79 |  24.19 |
| APNN-TC-w1a2   |    0.36 |   1.66 |
| Speedup (FP32) |  11.71x | 15.24x |
| Speedup (FP16) |  10.40x | 14.62x |

## Observations.
+ In the CUTLASS NN model with small batch (e.g,, 8), INT8 is not as fast as FP32 and FP16. This is because of small overall computation under the small batch cases. While for larger batch (e.g., 256) with more computations, INT8 would demonstrate its advantage for high throughput.
> 
> |      | CUTLASS-VGG-variant-b256 (ms) |
> |------|-----------------:|
> | FP32 |          628.254 |
> | FP16 |          540.707 |
> | INT8 |          368.626 |
+ Compared with the results in our paper (at the time of submission), we found that both the CUTLASS and APNN-TC performance has improved significantly, while the overall speedup trend is similar. **We will revise our paper with the improved design latency performance in the final version of our paper**.

## [*Updated*] BNN for NN model.
+ `cd bnn_baseline`
+ `make`
+ `./alexnet.bin`
+ `./vgg.bin`
+ `./resnet.bin`

|         | Current | Table-2 |
|--------:|--------:|--------:|
| AlexNet |   0.631 |    0.69 |
|     VGG |   2.233 |    2.17 |
|  ResNet |   0.733 |    0.68 |

Note that for the BNN-based NN model we use in our paper submission, we adopt the design from this [TCBNN](https://github.com/pnnl/TCBNN) (from TPDS-20) for the state-of-the-art BNN implementation on GPU tensor core, which can match the number in the Table-2.


## [*Updated*] APNN-TC NN model layer-wise latency breakdown.
+  We update our NN model source and enable the layer-wise latency breakdown.
+ `cd APNN-TC-nn/`
+ `make`
+ `./alexnet.bin`
+ `./vgg_variant.bin`
+ `./resnet.bin`
+ Example output for `AlexNet`
```
Conv1, 224, 224, 3, 64, 11, 11
Conv2, 28, 28, 64, 192, 5, 5
Conv3, 14, 14, 192, 384, 3, 3
Conv4, 14, 14, 384, 256, 3, 3
Conv5, 14, 14, 256, 256, 3, 3
Fc1, 12544, 4096
Fc2, 4096, 4096
Fout, 4096, 1000

==============
AlexNet (ms): 0.372
AlexNet Layer-0 (ms): 0.241
AlexNet Layer-1 (ms): 0.018
AlexNet Layer-2 (ms): 0.003
AlexNet Layer-3 (ms): 0.046
AlexNet Layer-4 (ms): 0.023
AlexNet Layer-5 (ms): 0.010
AlexNet Layer-6 (ms): 0.007
AlexNet Layer-7 (ms): 0.009
```