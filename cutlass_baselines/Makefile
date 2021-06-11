INC = 	-I../cutlass/include \
		-I../cutlass/tools/util/include \
		-I../cutlass/examples/common
FLAG = -std=c++11 -O3 -w -arch=sm_86 

all: vgg_variant 
# alexnet

alexnet: alexnet.cu gemm.cuh config.h
	nvcc $(INC) $(FLAG) $< -o $@ 

vgg_variant: vgg_variant.cu gemm.cuh config.h
	nvcc $(INC) $(FLAG) $< -o $@ 


clean:
	rm -r alexnet 