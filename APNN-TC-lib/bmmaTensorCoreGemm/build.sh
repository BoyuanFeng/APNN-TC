rm -f *.out
nvcc -ccbin g++ \
    -m64 -gencode \
    arch=compute_86,code=compute_86 \
    -I../Common \
    -o 67_bmmaTensorCoreGemm.out \
    src/67_bmmaTensorCoreGemm.cu
 
# nvcc -ccbin g++ -m64 \
#      --compiler-options '-fPIC' \
#      -rdc=true \
#      -gencode arch=compute_86,code=compute_86 \
#      -I../Common \
#      --shared \
#      -o lib/lib0_bmmaTensorCoreGemm.so \
#      src/0_bmmaTensorCoreGemm.cu \

# nvcc -ccbin g++ -m64 \
#     --compiler-options '-fPIC' \
#     -rdc=true\
#     -I../Common \
#     -Iinc/ \
#     -gencode arch=compute_86,code=compute_86 \
#     src/main.cu \
#     -o main.out  \
#     -Llib/ -l0_bmmaTensorCoreGemm

# export LD_LIBRARY_PATH=/home/yuke/APNN_TC_sc21/APNN_sc21_backup/src/bmmaTensorCoreGemm/lib:$LD_LIBRARY_PATH 
# ./main.out