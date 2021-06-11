NVCC = nvcc
NVCC_FLAG = -std=c++11 -O3 -w  -arch=sm_86 -maxrregcount=64 -rdc=true 
LIBS = -ljpeg
INC = -I../APNN_sc21_backup/src/bmmaTensorCoreGemm/src \
	-I../APNN_sc21_backup/src/Common
DEP = src/param.h src/kernel.cuh src/data.h src/data.cpp src/utility.h

all: alexnet vgg resnet18

alexnet: src/alexnet.cu $(DEP)
	$(NVCC) $(NVCC_FLAG) $(INC) -o $@ src/alexnet.cu src/data.cpp $(LIBS)

vgg: src/vgg.cu $(DEP)
	$(NVCC) $(NVCC_FLAG) $(INC) -o $@ src/vgg.cu src/data.cpp $(LIBS)

resnet18: src/resnet18.cu $(DEP)
	$(NVCC) $(NVCC_FLAG) $(INC) -o $@ src/resnet18.cu src/data.cpp $(LIBS)

clean:
	rm alexnet \
		vgg \
		resnet18 

