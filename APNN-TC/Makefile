NVCC = nvcc
NVCC_FLAG = -std=c++11 -O3 -w  -arch=sm_86 -maxrregcount=64 -rdc=true 
LIBS = -ljpeg
INC = -I../APNN_sc21_backup/src/bmmaTensorCoreGemm/src -I../APNN_sc21_backup/src/Common

all: imagenet_alexnet
# imagenet_vgg imagenet_resnet

# all: 
# 	imagenet_alexnet \
# 	imagenet_vgg \
# 	imagenet_resnet
	 
imagenet_alexnet: alexnet.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) $(INC) -o $@ alexnet.cu data.cpp $(LIBS)

imagenet_vgg: imagenet_vgg.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) $(INC) -o $@ imagenet_vgg.cu data.cpp $(LIBS)

imagenet_resnet: imagenet_resnet.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) $(INC) -o $@ imagenet_resnet.cu data.cpp $(LIBS)

clean:
	rm imagenet_resnet \
		imagenet_alexnet \
		imagenet_vgg 

