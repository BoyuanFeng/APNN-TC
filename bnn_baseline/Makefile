NVCC = nvcc
NVCC_FLAG = -std=c++11 -O3 -w  -arch=sm_86 -rdc=true 
LIBS = -ljpeg

all: alexnet.bin \
	vgg.bin \
	resnet.bin

alexnet.bin: alexnet.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ alexnet.cu data.cpp $(LIBS)

vgg.bin: vgg.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ vgg.cu data.cpp $(LIBS)

resnet.bin: resnet.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ resnet.cu data.cpp $(LIBS)

clean:
	rm *.bin

