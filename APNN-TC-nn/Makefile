NVCC = nvcc
NVCC_FLAG = -std=c++11 -O3 -w  -arch=sm_86 -rdc=true 
LIBS = -ljpeg
INC = -I../APNN-TC-lib/bmmaTensorCoreGemm/src \
	-I../APNN-TC-lib/Common
DEP = src/param.h src/kernel.cuh src/data.h src/data.cpp src/utility.h

all: alexnet.bin vgg_variant.bin resnet18.bin

alexnet.bin: src/alexnet.cu $(DEP)
	$(NVCC) $(NVCC_FLAG) $(INC) -o $@ src/alexnet.cu src/data.cpp $(LIBS)

vgg_variant.bin: src/vgg.cu $(DEP)
	$(NVCC) $(NVCC_FLAG) $(INC) -o $@ src/vgg.cu src/data.cpp $(LIBS)

resnet18.bin: src/resnet18.cu $(DEP)
	$(NVCC) $(NVCC_FLAG) $(INC) -o $@ src/resnet18.cu src/data.cpp $(LIBS)

.PHONY: clean

clean:
	rm *.bin
