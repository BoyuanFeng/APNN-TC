INC = 	-I../cutlass/include\
		-I../cutlass/tools/util/include \
		-I../cutlass/examples/common

FLAG= -std=c++11 -O3 -w -arch=sm_86 

all: bench_gemm.bin \
	 bench_conv.bin \

bench_gemm.bin: bench_gemm.cu 
	nvcc $(INC) $(FLAG) $< -o $@

bench_conv.bin: bench_conv.cu 
	nvcc $(INC) $(FLAG) $< -o $@ 
	
clean:
	rm -rf *.bin 