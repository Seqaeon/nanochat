"""Modal runners for the fully-binary-transformer Phase 0 oracles.

WRITTEN FOR A MODAL NOTEBOOK, not the `modal run` CLI.  There is no
`@app.local_entrypoint`; every function is called with `.remote()` from a cell.

    import modal
    from modal_binary_oracles import app, kernel_gate, sensitivity_scan, upload_help

    print(upload_help())                      # one-time volume setup, run from your laptop

    with modal.enable_output(), app.run():
        print(kernel_gate.with_options(gpu="A100-40GB").remote())

    with modal.enable_output(), app.run():
        print(sensitivity_scan.with_options(gpu="L4").remote(eval_steps=40, batch=8))

Paste this whole file into a cell, or upload it and import it.

--------------------------------------------------------------------------------
WHICH GPU
--------------------------------------------------------------------------------
kernel_gate (O3) is architecture-sensitive.  `b1` MMA with BOTH the AND and XOR
operands exists on Turing (sm_75) and Ampere (sm_80/sm_86).  Ada dropped INT1, and
Hopper removed the XOR operand from the hardware and emulates it up to 5x slower.

    T4    sm_75   valid, cheapest
    A10   sm_86   valid, same architecture as the local dev machine
    A100  sm_80   USE THIS: valid b1 and enough memory to carry Phase 1 after
    L4 / L40S     Ada, INT1 dropped                -> do not use for O3
    H100 / H200 / B200 / B300                      -> AND only, XOR emulated;
                                                      never take a headline number here

sensitivity_scan (O5) is forward-only on a 126M model, so any of them works.  L4 is
plenty and cheap.  Locally this needs batch 1 to fit in 3.68 GB and one optimizer
step never completed in 34 minutes; on an L4 the whole sweep is minutes.

--------------------------------------------------------------------------------
WHAT THE KERNEL GATE IS MEASURING AGAINST
--------------------------------------------------------------------------------
From scripts/o4_cost_model.py, at matched inference bytes the binary model issues
24.5x the MACs of the dense one.  So a b1 kernel must reach >= 24.5x over bf16 --
38% of the Ampere spec ceiling -- for the binary model to TIE on wall clock.
Locally the same binary measured 1.93x and 4.19x in two power states, which is why
this has to be re-run somewhere with a stable power budget.
"""
import subprocess
import modal

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_bf16.h>

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;
#ifdef USE_XOR
#define REFOP(a,b) ((a)^(b))
#define BITOP bmmaBitOpXOR
#else
#define REFOP(a,b) ((a)&(b))
#define BITOP bmmaBitOpAND
#endif

#define WM 8
#define WN 8
#define WK 128
#define BM 64
#define BN 64
#define BK 512
#define BKW (BK/32)

__global__ void b1_gemm(const uint32_t* __restrict__ A,
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


static double bench_b1(uint32_t* dA, uint32_t* dB, int32_t* dC, int M,int N,int K,int iters){
  dim3 grid(N/BN, M/BM), block(256);
  cudaEvent_t s,f; cudaEventCreate(&s); cudaEventCreate(&f);
  cudaEventRecord(s);
  for(int i=0;i<iters;i++) b1_gemm<<<grid,block>>>(dA,dB,dC,M,N,K);
  cudaEventRecord(f); cudaEventSynchronize(f);
  float ms; cudaEventElapsedTime(&ms,s,f);
  cudaEventDestroy(s); cudaEventDestroy(f);
  return 2.0*M*N*K*iters/(ms*1e-3)/1e12;
}
static double bench_bf16(cublasHandle_t h,__nv_bfloat16* a,__nv_bfloat16* b,float* c,int M,int N,int K,int iters){
  float alpha=1.f,beta=0.f;
  cudaEvent_t s,f; cudaEventCreate(&s); cudaEventCreate(&f);
  cudaEventRecord(s);
  for(int i=0;i<iters;i++)
    cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,b,CUDA_R_16BF,N,a,CUDA_R_16BF,K,&beta,c,CUDA_R_32F,N,
                 CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaEventRecord(f); cudaEventSynchronize(f);
  float ms; cudaEventElapsedTime(&ms,s,f);
  cudaEventDestroy(s); cudaEventDestroy(f);
  return 2.0*M*N*K*iters/(ms*1e-3)/1e12;
}

int main(int argc, char** argv) {
  int M=4096,N=4096,K=4096,rounds=6,iters=20;
  if(argc>=4){M=atoi(argv[1]);N=atoi(argv[2]);K=atoi(argv[3]);}
  if(argc>=5) rounds=atoi(argv[4]);
  int kw=K/32;
  std::vector<uint32_t> hA((size_t)M*kw),hB((size_t)N*kw);
  std::mt19937 rng(0);
  for(auto&x:hA)x=rng(); for(auto&x:hB)x=rng();

  uint32_t *dA,*dB; int32_t* dC; __nv_bfloat16 *fA,*fB; float* fC;
  #define CK(x) do{cudaError_t _e=(x); if(_e!=cudaSuccess){printf("CUDA_FAIL %s\n",cudaGetErrorString(_e));return 1;}}while(0)
  CK(cudaMalloc(&dA,hA.size()*4)); CK(cudaMalloc(&dB,hB.size()*4)); CK(cudaMalloc(&dC,(size_t)M*N*4));
  CK(cudaMalloc(&fA,(size_t)M*K*2)); CK(cudaMalloc(&fB,(size_t)K*N*2)); CK(cudaMalloc(&fC,(size_t)M*N*4));
  CK(cudaMemcpy(dA,hA.data(),hA.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB.data(),hB.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMemset(fA,0,(size_t)M*K*2)); CK(cudaMemset(fB,0,(size_t)K*N*2));

  dim3 grid(N/BN,M/BM), block(256);
  b1_gemm<<<grid,block>>>(dA,dB,dC,M,N,K);
  CK(cudaDeviceSynchronize());
  std::vector<int32_t> hC((size_t)M*N);
  CK(cudaMemcpy(hC.data(),dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
  int bad=0;
  for(int t=0;t<64;t++){int m=(t*37)%M,n=(t*53)%N;int ref=0;
    for(int w=0;w<kw;w++) ref+=__builtin_popcount(REFOP(hA[(size_t)m*kw+w],hB[(size_t)n*kw+w]));
    if(ref!=hC[(size_t)m*N+n]) bad++;}
  printf("bitop=%s shape=%dx%dx%d correctness=%s\n",
#ifdef USE_XOR
    "XOR",
#else
    "AND",
#endif
    M,N,K, bad?"FAIL":"PASS");

  cublasHandle_t h; cublasCreate(&h);
  bench_b1(dA,dB,dC,M,N,K,3); bench_bf16(h,fA,fB,fC,M,N,K,3);
  printf("round   b1_TOPS  bf16_TFLOPS  ratio\n");
  std::vector<double> rs;
  for(int r=0;r<rounds;r++){
    double t1=bench_b1(dA,dB,dC,M,N,K,iters);
    double t2=bench_bf16(h,fA,fB,fC,M,N,K,iters);
    double t1b=bench_b1(dA,dB,dC,M,N,K,iters);
    double b1=(t1+t1b)/2;
    printf("%5d %9.2f %12.2f %6.2fx\n",r,b1,t2,b1/t2);
    rs.push_back(b1/t2);
  }
  std::sort(rs.begin(),rs.end());
  printf("median ratio (unoptimised b1 kernel vs cuBLAS bf16): %.2fx\n", rs[rs.size()/2]);
  return 0;
}
"""

app = modal.App("binary-transformer-oracles")
vol = modal.Volume.from_name("nanochat-binary", create_if_missing=True)

kernel_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch")
)

oracle_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pyarrow", "tiktoken", "tokenizers",
                 "rustbpe", "filelock", "requests")
)


def upload_help():
    """One-time volume population.  Run these from the machine holding the repo."""
    return """
modal volume create nanochat-binary   # (create_if_missing already does this)

# the repo code (nanochat/ + scripts/ + tokenizer/), ~a few MB
modal volume put nanochat-binary ./nanochat            /repo/nanochat
modal volume put nanochat-binary ./scripts             /repo/scripts
modal volume put nanochat-binary ./tokenizer           /repo/tokenizer

# the dense depth-8 V=32,768 checkpoint, 335 MB
modal volume put nanochat-binary ./out/dense_d8_V32k_model_001014.pt /ckpt.pt

# TWO parquet shards minimum: the dataloader takes paths[:-1] for train and
# paths[-1:] for val, so a single shard leaves the val split empty.
modal volume put nanochat-binary ./data/shard_00000.parquet /data/shard_00000.parquet
modal volume put nanochat-binary ./data/shard_00001.parquet /data/shard_00001.parquet
"""


def _sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout, flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[:4000], flush=True)
    return r.stdout


@app.function(image=kernel_image, gpu="A100-40GB", timeout=1800)
def kernel_gate(shapes=None, rounds=6):
    """O3: is a b1 GEMM actually faster than bf16, and by how much?"""
    import torch
    out = []
    p = torch.cuda.get_device_properties(0)
    arch = f"{p.major}{p.minor}"
    banner = (f"device: {p.name}  sm_{arch}  {p.total_memory/2**30:.1f} GiB  "
              f"{p.multi_processor_count} SMs")
    print(banner, flush=True)
    out.append(banner)

    xor_native = int(arch) < 90
    note = f"b1 XOR: {'native' if xor_native else 'EMULATED (removed in sm_90) - do not headline'}"
    print(note, flush=True); out.append(note)
    if int(arch) == 89:
        w = "WARNING: sm_89 (Ada) dropped INT1 tensor ops; this run may fail or fall back"
        print(w, flush=True); out.append(w)

    with open("/root/b1.cu", "w") as f:
        f.write(CUDA_SRC)
    _sh(f"nvcc -O3 -arch=sm_{arch} /root/b1.cu -lcublas -o /root/b1_and")
    if xor_native:
        _sh(f"nvcc -O3 -arch=sm_{arch} -DUSE_XOR /root/b1.cu -lcublas -o /root/b1_xor")

    out.append(_sh("nvidia-smi -q -d POWER | grep -iE 'current power limit|default power limit'"))
    out.append(_sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader"))

    if shapes is None:
        shapes = [
            ("square 4096", "4096 4096 4096"),
            ("ffn d=512",   "8192 2048 512"),
            ("ffn d=1024",  "8192 4096 1024"),
            ("ffn d=2048",  "8192 8192 2048"),
            ("head V=32k",  "8192 32768 512"),
            ("head V=131k", "4096 131072 512"),
        ]
    for label, dims in shapes:
        out.append(f"--- {label} ({dims}) AND ---")
        out.append(_sh(f"/root/b1_and {dims} {rounds}"))
        if xor_native and label.startswith("square"):
            out.append(f"--- {label} ({dims}) XOR ---")
            out.append(_sh(f"/root/b1_xor {dims} {rounds}"))

    out.append("BREAK-EVEN: >= 24.5x needed to tie on wall clock at matched bytes.")
    return "\n".join(x for x in out if x)


@app.function(image=oracle_image, gpu="L4", volumes={"/vol": vol}, timeout=7200)
def sensitivity_scan(eval_steps=40, batch=8, seq=2048, scale=("row", "none"),
                     binarise=("weights", "acts", "both"), only=None):
    """O5: where in a whole transformer is floating point load-bearing?"""
    import sys, os
    sys.path.insert(0, "/vol/repo")
    os.chdir("/vol/repo")
    argv = ["o5", "--ckpt", "/ckpt.pt", "--data-dir", "/vol/data",
            "--tokenizer-dir", "/vol/repo/tokenizer",
            "--eval-steps", str(eval_steps), "--batch", str(batch), "--seq", str(seq),
            "--scale", *scale, "--binarise", *binarise]
    if only:
        argv += ["--only", *only]
    sys.argv = argv
    import io as _io, contextlib
    buf = _io.StringIO()
    from scripts.o5_sensitivity import main
    with contextlib.redirect_stdout(buf):
        main()
    text = buf.getvalue()
    print(text, flush=True)
    return text
