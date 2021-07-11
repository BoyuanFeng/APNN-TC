0:
Basic imma code from CUDA SDK without modification.

1:
Clean up imma code. Will implement bmma from here.

2:
An initial attempt. Still some bugs.
CUDA error at bmmaTensorCoreGemm.cu:418 code=700(cudaErrorIllegalAddress) "cudaEventSynchronize(stop)"

3:
SHMEM and BMMA seems to be correct. Writing back to D is still being developed.

4:
Add writing back to D. Have all workloads (e.g., read, compute, write). Can compile and run. Speed is 400TOPS.
2x faster than Li Ang's BMMASN-Bin (~200TOPS, on a matrix of MxNxK: 8192x8192x65536). 4x faster than Li Ang's BMMA_bin implementation (~120TOPS).

Overall, the kernel takes 21 ms (404TOPS).
Loading from GL takes 9 ms. 
BMMA takes 1.2 ms. [去除这块，有918TOPS]
Writing back to SHMEM takes 6 ms. [去除这块，有787TOPS]
Writing back to GL takes 5 ms. [去除这块，有528TOPS]

Note:
    a. The address of writing back to D has not been carefully checked.
    b. The correctness of the result is not validated.
    c. The performance may still be improved. E.g., i) Increase the block size; (ii) reduce the GL write according to the required number of bits in the workload.

5:
This version should be logically correct. It can also compile and run without any errors or warnings.
This version is 520 TOPS, since some memory access errors are corrected.

The correctness of the result is not validated yet.


6:
Add validation code.

The computation results is not correct.

A初始化成0x0000000F, B初始化成0xFFFFFFFF时，偶数行对，奇数行出错。

7:
Backup. 2021-1-17. 9:59 PM.

8:
Fix a bug in block_i and block_j.

9:
Fix a bug in load_matrix_sync.

Now, it computes correctly for matrix size of 64x64xK, K has been tested from 1024 ~ 8192.
Note that this corresponds to computation on a single block.

However, it still does not work for other matrix sizes (e.g., 128x128x1024).

10:
Fix a bug in write to GL.

Now, it computes correctly for various matrix sizes, since I have manually tested for the following size with rand() initialized A and B:
    64x128x1024, 64x128x4096, 128x128x4096, 256x128x4096, 256x256x8192, 4096x4096x8192.

For matrix of 4096x4096x8192, the speed is 496.79TOPS.

11:
Cleanup.

12:
Repeat 200 times to measure the performance.
This is the complete version for bmma with 64x64 tiling block.
We name this version as BMMA_64

13:
BMMA_64 with reduce.
We name this version as BMMA_64_Reduce.
Computation result is correct.

14:
Initial version of BMMA_32.
Computation result is wrong now.

15:
A working version of BMMA_32.
The computation result is correct.
8 Warp. 速度整体都比BMMA_64慢。

16:
Initial version of BMMA_32.
4 warp. 比version 15快。但整体比version 12慢。

17:
Initial version of BMMA_32.
2 warp. 比version 16快。但整体比version 12慢。

18:
BMMA_64 which writing directly from FRAG to GL without SHMEM.
Modified from version 12.

19:
BMMA_64 which uses stride 8 in wmma::store_matrix_sync.
Modified from version 18.
Slightly faster than version 18, especially on 512 and 1024.

20:
BMMA_64 which uses new format (i.e., continuous GL memory layout).
Modified from version 19.
This is 10% faster than version 19.

21:
BMMA_64. 加了async_memcpy. 
Modified from Version 20.
反而变慢了很多。

22:
BMMA_128. tiling size: 128x128.
Modified from Version 12
计算结果是对的。

23:
BMMA_128 which writing directly from FRAG to GL without SHMEM.
Modified from Version 22.
在1024时比22快。其他没有测。

24:
BMMA_128 which uses new format (i.e., continuous GL memory layout).
Modified from Version 23.
整体比v23快。注意到，在小matrix (e.g., 512x512)上，BMMA_128显著慢于BMMA_64.

25:
This is a micro-benchmark on the performance of an Ampere new feature on "Asynchronous data movement".
https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/

Using this new feature, the time for reading the same data from GL to SHMEM increases from 0.10735 ms to 0.112201 ms.

I asked a question here. But no one answers.
https://forums.developer.nvidia.com/t/fine-grained-address-control-in-cooperative-groups-memcpy-async/168768

Considering that this cannot be a significant contribution, I will not investigate more on this feature.

26:
A uncomplete CONV.
Can compile and run with no error.
Given a 256x256x128 image, COUT as 128, 2-bit for both weights and features, the TOPS is 372.
Given a 32x32x128 image, COUT as 128, 2bit for both weights and features, the TOPS is 215.

27:
A complete CONV with reduce.
Results it not validated.
Given a 32x32x128 image, COUT as 128, 2bit for both weights and features, the TOPS is 121. Much slower than 4-bit CUTLASS CONV with 146 TOPS.

28:
BMMA 128. Copied from V23. 
Now, it computes correctly for various matrix sizes, since I have manually tested for the following size with rand() initialized A and B:
    1024, 2048, 4096, 8192
后面的APMM都是基于这个改的。

29:
APNN-2bit x 2bit. Modified from V28. Tiling Size: 128x128
Results are validated.

30:
APNN-W1-A2. Modified from V13. Tiling Size: 64x64
Results are validated.

31:
APNN-W1-A2. Modified from V29. Tiling Size: 128x128
Results are validated.

32:
APNN-W1-A4. Modified from V31. Tiling Size: 128x128
Results are validated.

33:
APNN-W1-A4. Modified from V30. Tiling Size: 64x64
Results are validated.

34:
APNN-W1-A8. Tiling Size: 64x64.
Results are not validated.

35:
APNN-W1-A8. Tiling Size: 128x128.
Results are not validated.

36:
APMM-W2-A8. Tiling Size: 128x128.
Results are not validated.

37:
APMM-W2-A8. Tiling Size: 64x64.
Results are not validated.

38:
APMM-W3-A1. Tiling Size: 64x64.
Results are not validated.

39:
APMM-W3-A1. Tiling Size: 128x128.
Results are not validated.

40:
APMM-W5-A1. Tiling Size: 128x128
Results are not validated.

41:
APMM-W5-A1. Tiling Size: 64x64
Results are not validated.

42:
APMM-W6-A2. Tiling Size: 64x64
Results are not validated.

43:
APMM-W6-A2. Tiling Size: 128x128
Results are not validated.

*************************APMM code before this line does node support CHUNK_K = 1!!!!!  **********************************

44:
Conv-w2-a2. When reading image patch, use 32bit instead of 128bit. 还加了skew, 去除bank conflict.
32bit和128bit没有performance区别。skew可以从0.019ms快到0.0181ms.

45:
Conv-w2-a2. 从GL直接读shared memory, 而不是先读一个image patch到GL.
0.0168 ms

46:
Conv-w2-a2. 不做pack. 直接把32-bit integer 写出去。
0.0152ms。已经比cutlass-int4快了

47:
Conv-w2-a2. Combine 45 & 46.
0.0141 ms. 比cutlass-int4快1.17x.

48:
Conv-w2-a2. Complete computation. Passed cuda-memcheck with 0 error. Tiling Size: 64x64

Result is validated.

49:
Conv-w2-a2.  Tiling Size: 128x128.

Result is validated.


50:
Conv-w2-a2.  Tiling Size: 128x128.

Some trick on FRAG. The performance improvement is not significant.

51:
Conv-w1-a2. Tiling Size: 64x64.

Result is validated.

52:
Conv-w1-a2. Tiling Size: 128x128.

Result is validated.


53:
Conv-w1-a4. Tiling Size: 64x64.
Result is validated.


54:
Conv-w1-a4. Tiling Size: 128x128.
Result is validated.

55:
Conv-w1a8. Tiling Size: 64x64.
Result is validated.

56:
Conv-w1a8. Tiling Size: 128x128.
Result is validated.

57:
Conv-w2a8. Tiling Size: 64x64.


58:
Conv-w2a8. Tiling Size: 128x128.


59:
Conv-w1a3. Tiling Size: 64x64.

60:
Conv-w1a3. Tiling Size: 128x128.

61:
Conv-w1a5. Tiling Size: 64x64.

62:
Conv-w1a5. Tiling Size: 128x128.

63:
Conv-w2a6. Tiling Size: 64x64.

64:
Conv-w2a6. Tiling Size: 128x128.

65:
APMM-w1a2. With the same data layout as Conv. 32-bit output. 64x64.
Results validated.

66:
APMM-w1a2. With the same data layout as Conv. Added quantization and packing. 64x64.
Results validated.

67:
APConv-w1a2. Added quantization and packing. 64x64.
Results validated.

68:
APConv-w1a2. Added quantization and packing. Added 2x2 pooling. 64x64.
Results validated.

----------------------------

69:
APConv-w2a2. Added quantization and packing. Added 2x2 pooling. 64x64.
Results validated.

70:
APConv-w2a2. Added quantization and packing. 64x64.
Results validated.

71:
APMM-w1a2. With the same data layout as Conv. 32-bit output. 64x64.
Results validated.  Support CHUNK_K = 1.
This version is different from v65 in terms of GL loading. Speed is slightly faster than v65.

72:
APMM-w2a2. With the same data layout as Conv. 32-bit output. 64x64.
Results validated. Support CHUNK_K = 1.

73:
APMM-w2a2. With the same data layout as Conv. Added quantization and packing. 64x64.
Results validated. Support CHUNK_K = 1.

74:
APConv-w2a8. Added quantization and packing. 64x64.
Results validated

75:
APConv-w2a8. Added quantization and packing. Added 2x2 pooling. 64x64.
Results NOT validated

76:
APMM-w2a8. With the same data layout as Conv. 32-bit output. 64x64.
Results validated. Support CHUNK_K = 1.

77:
APMM-w1a8. With the same data layout as Conv. 32-bit output. 64x64.
Results validated. Support CHUNK_K = 1.

78:
APMM-w1a4. With the same data layout as Conv. 32-bit output. 64x64.
Results validated. Support CHUNK_K = 1.

79:
APMM-W3-A1. Tiling Size: 64x64.
Results are not validated. Support CHUNK_K = 1.

80:
APMM-W5-A1. Tiling Size: 64x64.
Results are not validated. Support CHUNK_K = 1.

81:
APMM-W6-A2. Tiling Size: 64x64.
Results are not validated. Support CHUNK_K = 1.

82:
APConv-w1a2. Only TC Computation without bit combination and bit decomposition. Tiling Size: 64x64.
Memory access is the same as V51. Only commented out the bit combination code.
The result is not meanful since it does not have bit combination.
I wrote this code only to measure the performance overhead of bit combination.

83:
APMM-w4a4. With the same data layout as Conv. 32-bit output. 64x64.
Results validated. Support CHUNK_K = 1.

84:
APMM-w1a1. With the same data layout as Conv. 32-bit output. 64x64.
No need to reduce since it is w1a1.
Writing directly from FRAG to GL without SHMEM.
Results validated. Support CHUNK_K = 1.