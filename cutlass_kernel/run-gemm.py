#!/usr/bin/env python3
import os

B = 64

N_K_list = [
    128,
    256,
    384,
    512,
    640,
    768,
    896,
    1024
]

for N_K in N_K_list:
    os.system("./bench_gemm.bin {} {} {}".format(B, N_K, N_K))