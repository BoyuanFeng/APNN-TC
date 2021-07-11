__host__ void init_host_matrices(uint8_t *a, uint8_t *b, int *c);

__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B,
                                  const int *C, int *D, int alpha, int beta);

__global__ void simple_wmma_gemm_imma(const uint8_t *a, const uint8_t *b,
                                      const int *c, int *d, int m_ld, int n_ld,
                                      int k_ld, int alpha, int beta);