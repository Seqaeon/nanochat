// The b1 WMMA GEMM, shared by two compilation paths:
//   kernels/b1_probe.cu     -> nvcc, links cuBLAS, standalone A/B binary
//   scripts/o3_kernel_gate.py --backend nvrtc -> runtime-compiled, NO nvcc needed
// The nvrtc path exists because `nvidia-cuda-nvcc-cu12` ships only ptxas, not the
// nvcc frontend, so a CUDA *runtime* container cannot get nvcc from pip at all.
// libnvrtc.so and mma.h both ship in the wheels torch already depends on.
#pragma once
#include <mma.h>
// NVRTC has no libstdc++, so <cstdint> is unavailable there. The kernel only needs
// two fixed-width types; declare them directly under the RTC compiler.
#ifdef __CUDACC_RTC__
typedef unsigned int uint32_t;
typedef int int32_t;
#else
#include <cstdint>
#endif

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

#ifdef USE_XOR
#define BITOP bmmaBitOpXOR
#else
#define BITOP bmmaBitOpAND
#endif

#define WM 8
#define WN 8
#define WK 128
#define BM 64
#define BN 64
#define BK 512
#define BKW (BK/32)

extern "C" __global__ void b1_gemm(const uint32_t* __restrict__ A,
                        const uint32_t* __restrict__ B,
                        int32_t* __restrict__ C,
                        int M, int N, int K) {
  const int kw = K / 32;
  __shared__ uint32_t As[BM][BKW];
  __shared__ uint32_t Bs[BN][BKW];

  int tid = threadIdx.x;
  int warp = tid / 32;
  int wm = warp / 2;
  int wn = warp % 2;
  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc[2][4];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++) wmma::fill_fragment(acc[i][j], 0);

  for (int k0 = 0; k0 < kw; k0 += BKW) {
    for (int idx = tid; idx < BM * BKW; idx += blockDim.x) {
      int r = idx / BKW, c = idx % BKW;
      As[r][c] = A[(blockRow + r) * kw + k0 + c];
    }
    for (int idx = tid; idx < BN * BKW; idx += blockDim.x) {
      int r = idx / BKW, c = idx % BKW;
      Bs[r][c] = B[(blockCol + r) * kw + k0 + c];
    }
    __syncthreads();

    for (int kk = 0; kk < BK; kk += WK) {
      wmma::fragment<wmma::matrix_a, WM, WN, WK, precision::b1, wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b, WM, WN, WK, precision::b1, wmma::col_major> bf[4];
      for (int i = 0; i < 2; i++) {
        const uint32_t* pu = &As[wm * 16 + i * 8][0] + (kk / 32);
        wmma::load_matrix_sync(af[i], reinterpret_cast<const precision::b1*>(pu), BK);
      }
      for (int j = 0; j < 4; j++) {
        const uint32_t* pu = &Bs[wn * 32 + j * 8][0] + (kk / 32);
        wmma::load_matrix_sync(bf[j], reinterpret_cast<const precision::b1*>(pu), BK);
      }
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
          wmma::bmma_sync(acc[i][j], af[i], bf[j], acc[i][j],
                          BITOP, bmmaAccumulateOpPOPC);
    }
    __syncthreads();
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++) {
      int r = blockRow + wm * 16 + i * 8;
      int c = blockCol + wn * 32 + j * 8;
      wmma::store_matrix_sync(&C[r * N + c], acc[i][j], N, wmma::mem_row_major);
    }
}


