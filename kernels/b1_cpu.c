// Starting CPU inference kernel for the native binary transformer.
//
// WHY THIS EXISTS
//   Everything in nanochat/binary.py SIMULATES binary arithmetic in bf16: sign_ste
//   emits +-1 floats and F.linear is an ordinary GEMM. That is correct for training
//   (you cannot backprop through a popcount) but it means "we built a fully binary
//   transformer" was, until this file, a claim about a simulation. This computes the
//   same function with actual bit operations, and b1_cpu.py asserts the two agree
//   EXACTLY, which is what makes the claim real.
//
//   CPU rather than GPU on purpose. Measured on an i7-11800H with AVX512_VPOPCNTDQ,
//   single core: 637 / 866 / 1733 GOPS at K = 512 / 1792 / 3584, against tuned
//   OpenBLAS fp32 at 126 / 116 / 105 GFLOPS. 5.1x / 7.5x / 16.5x, improving with
//   width, which is the direction the design goes. On GPU the same idea buys
//   1.13-1.58x because it fights cuBLAS and FlashAttention on their home ground.
//
// Build: gcc -O3 -march=native -shared -fPIC kernels/b1_cpu.c -o kernels/libb1cpu.so
#include <stdint.h>
#include <string.h>
#ifdef __AVX512VPOPCNTDQ__
#include <immintrin.h>
#endif

// Pack a +-1 float array (rows x cols) into bits, 1 = positive. Row-major, each row
// padded to a whole number of uint64 words.
void b1_pack(const float *src, uint64_t *dst, int rows, int cols) {
    int words = (cols + 63) / 64;
    memset(dst, 0, (size_t)rows * words * 8);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (src[(size_t)r * cols + c] > 0.0f)
                dst[(size_t)r * words + c / 64] |= 1ULL << (c % 64);
}

// dot of two packed +-1 rows over `bits` bits, returned as the INTEGER
//     <a, b> = bits - 2 * hamming(a, b)
// which is exactly what sign(a) @ sign(b) computes in the fp simulation.
static inline int32_t b1_dot(const uint64_t *a, const uint64_t *b, int words, int bits) {
    int64_t ham = 0;
    int w = 0;
#ifdef __AVX512VPOPCNTDQ__
    __m512i acc = _mm512_setzero_si512();
    for (; w + 8 <= words; w += 8) {
        __m512i x = _mm512_xor_si512(_mm512_loadu_si512((const void *)(a + w)),
                                     _mm512_loadu_si512((const void *)(b + w)));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(x));
    }
    ham += _mm512_reduce_add_epi64(acc);
#endif
    for (; w < words; w++) ham += __builtin_popcountll(a[w] ^ b[w]);
    return (int32_t)(bits - 2 * ham);
}

// out[m][n] = <A[m], B[n]>, the popcount form of a binary linear layer.
// A is (M, K) activations, B is (N, K) weights, both packed. out is int32 (M, N).
void b1_linear(const uint64_t *A, const uint64_t *B, int32_t *out,
               int M, int N, int K) {
    int words = (K + 63) / 64;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            out[(size_t)m * N + n] = b1_dot(A + (size_t)m * words,
                                            B + (size_t)n * words, words, K);
}

// Majority bundling of `n` packed +-1 vectors of `bits` bits, with a per-element
// integer threshold. This is BundledResidual and the value-side of HammingAttention.
void b1_bundle(const uint64_t *src, uint64_t *dst, const int32_t *threshold,
               int n, int bits) {
    int words = (bits + 63) / 64;
    memset(dst, 0, (size_t)words * 8);
    for (int c = 0; c < bits; c++) {
        int32_t s = 0;
        for (int i = 0; i < n; i++)
            s += (src[(size_t)i * words + c / 64] >> (c % 64)) & 1ULL ? 1 : -1;
        if (s - (threshold ? threshold[c] : 0) >= 0)
            dst[c / 64] |= 1ULL << (c % 64);
    }
}
