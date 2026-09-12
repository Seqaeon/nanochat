#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_bf16.h>


#ifdef USE_XOR
#define REFOP(a,b) ((a)^(b))
#else
#define REFOP(a,b) ((a)&(b))
#endif
#include "b1_gemm.cuh"

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
