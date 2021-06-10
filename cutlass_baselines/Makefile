INC = -I../cutlass/include -I../cutlass/tools/util/include -Icutlass/examples/common
FLAG = -std=c++11 -O3 -w -arch=sm_86 

all: alexnet

alexnet: alexnet.cu 
	nvcc $(INC) $(FLAG) $< -o $@ 

clean:
	rm -r alexnet 