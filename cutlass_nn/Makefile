INC = 	-I../cutlass/include \
		-I../cutlass/tools/util/include \
		-I../cutlass/examples/common \
		-Icudnn/include

LIB = -L cudnn/lib64\
 		-L /usr/local/lib

LK = -lcudnn

FLAG = -std=c++11 -O3 -w -arch=sm_86 

DEP = src/gemm.cuh src/config.h src/layer.cuh

all: alexnet.bin  \
	 vgg_variant.bin \
	 resnet18.bin

alexnet.bin: src/alexnet.cu $(DEP)
	nvcc $(INC) $(LIB) $(FLAG) $< -o $@ $(LK)

vgg_variant.bin: src/vgg_variant.cu $(DEP)
	nvcc $(INC) $(LIB) $(FLAG) $< -o $@ $(LK)

resnet18.bin: src/resnet18.cu $(DEP)
	nvcc $(INC) $(LIB) $(FLAG) $< -o $@ $(LK)

clean:
	rm -rf *.bin