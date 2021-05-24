all:
	nvcc -arch=sm_86 -std=c++11 -O2 -Icudnn/include -L cudnn/lib64 -L /usr/local/lib test.cu -o test -lcudnn
