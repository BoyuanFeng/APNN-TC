#!/usr/bin/env python3
import os
import sys

batch_count = 1

M_N_K_list = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    # 16384,
    # 32768,
    # 65536,
]

for m in M_N_K_list:
    os.system("./bench_batched_gemm {} {} {} {}".format(m, m, m, batch_count))