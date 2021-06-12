INC = 	-I../cutlass/include \
		-I../cutlass/tools/util/include \
		-I../cutlass/examples/common \
		-Icudnn/include

LIB = -L cudnn/lib64\
 		-L /usr/local/lib

LK = -lcudnn

FLAG = -std=c++11 -O3 -w -arch=sm_86 

DEP = src/gemm.cuh src/config.h

all: alexnet  \
	 vgg_variant \
	 resnet18

alexnet: src/alexnet.cu $(DEP)
	nvcc $(INC) $(LIB) $(FLAG) $< -o $@ $(LK)

vgg_variant: src/vgg_variant.cu $(DEP)
	nvcc $(INC) $(LIB) $(FLAG) $< -o $@ $(LK)

resnet18: src/resnet18.cu $(DEP)
	nvcc $(INC) $(LIB) $(FLAG) $< -o $@ $(LK)

clean:
	rm -r alexnet \
		  vgg_variant \
		  resnet18
