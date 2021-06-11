INC = 	-I../cutlass/include \
		-I../cutlass/tools/util/include \
		-I../cutlass/examples/common

FLAG = -std=c++11 -O3 -w -arch=sm_86 

DEP = src/gemm.cuh src/config.h

all: alexnet  \
	 vgg_variant \
	 resnet18

alexnet: src/alexnet.cu $(DEP)
	nvcc $(INC) $(FLAG) $< -o $@ 

vgg_variant: src/vgg_variant.cu $(DEP)
	nvcc $(INC) $(FLAG) $< -o $@ 

resnet18: src/resnet18.cu $(DEP)
	nvcc $(INC) $(FLAG) $< -o $@ 

clean:
	rm -r alexnet \
		  vgg_variant \
		  resnet18
