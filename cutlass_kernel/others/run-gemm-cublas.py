#!/usr/bin/env python3
import os

M_N_K_list = [
    # 128,
    # 256,
    # 384,
    512,
    # 640,
    # 768,
    # 896,
    1024,
    # 2048,
    # 4096,
    # 8192,
    # 16384,
    # 32768,
    # 65536,
]

for m in M_N_K_list:
    os.system("./cublas_gemm {} {} {}".format(m, m, m))