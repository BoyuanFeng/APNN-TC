#!/usr/bin/env python3
import os

M_N_K_list = [
    128,
    256,
    384,
    512,
    640,
    768,
    896,
    1024
]

for m in M_N_K_list:
    os.system("./bench_gemm {} {} {}".format(64, m, m))