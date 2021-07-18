# APNN-TC for SC21


## Clone this project.
```
git clone --recursive git@github.com:BoyuanFeng/APNN-TC.git
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
+ `cutlass_baselines/`: CUTLASS baselines NN models, including FP21, FP16, and INT8.


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
+ Run `./gemm-w1a2`, `./gemm-w1a3`,`./gemm-w1a4`,`./gemm-w2a2`.
+ Run `./conv-w1a2`, `./conv-w1a3`, `./conv-w1a4_small`, `./conv-w1a4_large`,`./conv-w2a2_small`, `./conv-w2a2_large`
> Note that 
> + for GEMM kernel, we profile the GEMM shape as `[M, N, K]` as `[64, N, K]`, where `N=K=[128,256,384,...,1024]`.
> + for CONV kernel, we profile the CONV shape with on feature map with `[H, W] = [16,16]` and the kernel size is `[O, C, K, K] = [O, C, 3, 3]`, where `O=C=[128,256,384,...,1024]`. 
> + `conv-w1a4_small` is for `w1a4` in `IN=COUT=[128,..., 640]`, 
> + `conv-w1a4_large` is for `w1a4` in `IN=COUT>=640`
> + `conv-w2a2_small` is for `w2a2` in `IN=COUT=[128,..., 640]`
> + `conv-w2a2_large` is for `w2a2` in `IN=COUT>=640`
<!-- ···
For APMM, please use the following versions. Note: M64, N=K=128, 256, ..., 1024.
apmm-w1a2: V71. [Same]
apmm-w1a3: V87. [Speed is slightly (6.5%) slower than reported.]
apmm-w1a4: V78. [Same]
apmm-w2a2: V72. [Current speed is slightly faseter than reported speed.]
apmm-w1a5: V88. [Speed is 15% lower than reported.]
apmm-w1a8: V77. [Same]
apmm-w2a6: V89. [Speed is 13% lower than reported.]
apmm-w2a8: V76. [Same] -->

<!-- For AP-Conv, please use the following versions. Note: H=W=16, CIN=COUT=128, 256, ..., 1024 -->

<!-- ap-conv-w1a2: V51. [Speed is 16% faster than reported.] -->
<!-- ap-conv-w1a3: V59. [For CIN=COUT=128, ..., 512, same as reported. For larger size, need to use w1a4]

ap-conv-w1a4_small: V53 & V54. 
[For CIN=COUT=128,..., 640, use V53. For larger size, use V54. Overall, current speed is 16% faster than reported.]

ap-conv-w2a2: V48 & V49. [For CIN=COUT=128,..., 640, use V48. For larger size, use V49. Overall, current speed is 17% faster than reported.] -->
<!-- ap-conv-w1a5: V91. [15% faster than reported.]
ap-conv-w1a8: V55 & V56. [For CIN=COUT=128,256, use V55. For larger size, use V56. Overall, current speed is 38% faster than reported.]
ap-conv-w2a6: V92. [10% slower than reported.]
ap-conv-w2a8: V93. [19% faster than reported.]
··· -->

## CUTLASS -- GEMM kernel
+ `cd bench_cutlass/`
+ `make all`
+ `./run-gemm.py`

## CUTLASS -- CONV kernel
+ `cd bench_cutlass/`
+ `make all`
+ `./run-conv.py`

## APNN-TC -- NN model  
+ Build and run the network with `w1a2` APNN (Table-2). 
+ `cd APNN-TC && make`
+ `./alexnet` to run alexnet with `w1a2`.
+ `./vgg` to run VGG-variant in `w1a2`. 
+ `./resnet18` to run ResNet18 in `w1a2`.

## CUTLASS -- NN model  
+ Build and run CUTLASS baseline.
+ Run `cd cutlass_baselines && make`
+  Select the precision (FP32, FP16, INT8) of CUTLASS, `cd cutlass_baselines/src/config.h` and comment out other two unused bitwidth.
```
#define BIT_WIDTH 32
// #define BIT_WIDTH 16
// #define BIT_WIDTH 8
```

+ Run `./alexnet` to run AlexNet in (FP32, FP16, INT8).
+ Run `./vgg_variant` to run VGG-variant in (FP32, FP16, INT8).
+ Run `./resnet18` to run ResNet18 in (FP32, FP16, INT8).