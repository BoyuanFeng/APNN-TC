Batch = 8
H = 32
W = 32
CIN = 128
COUT = 128
bmma_ms_avg = 0.0165

TOPS = ((Batch * 9.0 * CIN * H * W * COUT * 2)/(bmma_ms_avg/1000.0)) / 1e12
print("Conv2: ", TOPS)



Batch = 8
H = 16
W = 16
CIN = 128
COUT = 256
bmma_ms_avg = 0.0248


TOPS = ((Batch * 9.0 * CIN * H * W * COUT * 2)/(bmma_ms_avg/1000.0)) / 1e12
print("Conv3: ", TOPS)

