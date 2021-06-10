# APNN-TC for SC21


# Clone this project.
```
git clone git@github.com:YukeWang96/APNN_TC_sc21.git
```

# OS & Compiler:
+ `Ubuntu 16.04+`
+ `gcc >= 7.5`
+ `make >= 4.2.1`
+ `CUDA >= 11.0`
+ `libjpeg`

# Files & Directory
+ `cutlass_baselines/`: CUTLASS baselines NN models, including FP21, FP16, and INT8.
+ `cutlass/`: CUTLASS core kernel files.
+ `APNN-TC/`: our key low-bit design.


# Compile
+ `cutlass_baselines/`
```
cd cutlass_baselines
make
```
+ `APNN-TC/`
```
cd APNN-TC
make
```