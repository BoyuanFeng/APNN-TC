# APNN-TC for SC21


## Clone this project.
```
git clone git@github.com:YukeWang96/APNN_TC_sc21.git
```

## OS & Compiler:
+ `Ubuntu 16.04+`
+ `gcc >= 7.5`
+ `make >= 4.2.1`
+ `CUDA >= 11.0`
+ `libjpeg`

## Files & Directory
+ `cutlass_baselines/`: CUTLASS baselines NN models, including FP21, FP16, and INT8.
+ `cutlass/`: CUTLASS header and source files.
+ `APNN-TC/`: our APNN-TC NN low-bit design with `w1a2` for demonstration.

## Compile
+ Build and run CUTLASS baseline.
```
cd cutlass_baselines
make
```

+ Build and run the network with `w1a2` APNN (Table-2). 
> +  `cd APNN-TC && make`
> + `./alexnet` to run alexnet with `w1a2`.
> + `./vgg` to run VGG-variant in `w1a2`.
> + `./resnet18` to run ResNet18 in `w1a2`.
