INC=-Icutlass/include -Icutlass/tools/util/include -Icutlass/examples/common
FLAG= -std=c++11 -O3 -w -arch=sm_86 

all: bench_gemm \
	bench_conv \
	# bench_batched_gemm \
	# cublas_gemm

bench_gemm: bench_gemm.cu 
	nvcc $(INC) $(FLAG) $< -o $@

bench_conv: bench_conv.cu 
	nvcc $(INC) $(FLAG) $< -o $@ 

# bench_batched_gemm: bench_batched_gemm.cu 
# 	nvcc $(INC) $(FLAG) $< -o $@

# cublas_gemm: cublas_gemm.cu
# 	nvcc $(FLAG) $< -o $@ -lcublas

clean:
	rm -r bench_gemm \
		bench_conv \
		# bench_batched_gemm \
		# cublas_gemm