INC=-Icutlass/include -Icutlass/tools/util/include -Icutlass/examples/common
FLAG= -std=c++11 -O3 -w -arch=sm_86 

all: bench_gemm \
	bench_conv \
	alexnet \

bench_gemm: bench_gemm.cu 
	nvcc $(INC) $(FLAG) $< -o $@

bench_conv: bench_conv.cu 
	nvcc $(INC) $(FLAG) $< -o $@ 

alexnet: alexnet.cu 
	nvcc $(INC) $(FLAG) $< -o $@ 

clean:
	rm -r bench_gemm \
		  bench_conv \