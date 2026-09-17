// MUSA tensor-core SSD chunk scan specialization.
//
// The implementation follows dashboard submission 77994: stage the causal
// S=C*B^T tiles and Z/state operands once, then use WMMA 16x16x32 for the
// triangular S*Z and C*state products. It is deliberately an exact-shape
// kernel (T=128,H=64,D=64,N=128,G=8); all other shapes remain on Triton.
#include <cstdint>
#include <musa_bf16.h>
#include <musa_runtime.h>
#include <mma.h>
#include <torch/all.h>
#include <torch/extension.h>
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {
constexpr int T=128,H=64,DH=64,N=128,G=8,KT=32;
using namespace mtmusa::wmma;

__device__ __forceinline__ void smat_tile(const __mt_bfloat16* b,
                                          const __mt_bfloat16* c,
                                          __half* sm, int tile, int lane) {
  int g=tile>>6, t0=((tile>>3)&7)*16, u0=(tile&7)*16;
  fragment<accumulator,16,16,KT,float> a; fill_fragment(a,0.f);
  int kend=(u0<=t0+15)?N:0;
  for(int k=0;k<kend;k+=KT) {
    fragment<matrix_a,16,16,KT,__mt_bfloat16,row_major> fa;
    fragment<matrix_b,16,16,KT,__mt_bfloat16,col_major> fb;
    load_matrix_sync(fa,c+(int64_t)(t0*G+g)*N+k,G*N);
    load_matrix_sync(fb,b+(int64_t)(u0*G+g)*N+k,G*N);
    mma_sync(a,fa,fb,a);
  }
#pragma unroll
  for(int q=0;q<8;q++) {
    int t=t0+(lane>>3)+((q>>1)<<2), u=u0+(lane&7)+((q&1)<<3);
    sm[(int64_t)(g*T+t)*T+u]=u<=t?__float2half(a.x[q]):__float2half(0.f);
  }
}

__global__ __launch_bounds__(256) void stage_kernel(
    const float* state,const __mt_bfloat16* x,const float* dt,
    const float* A,const __mt_bfloat16* b,const __mt_bfloat16* c,
    __half* z,float* e,__half* q,__half* sm) {
  int blk=blockIdx.x, tid=threadIdx.x;
  if(blk<H) {
    int h=blk,u=tid>>1,d0=(tid&1)*32;
    __shared__ float cs[T];
    if(tid==0){float s=0;for(int v=0;v<T;v++){s+=dt[v*H+h];cs[v]=s;}}
    __syncthreads();
    float arg=A[h]*cs[u], ev=__expf(arg), w=dt[u*H+h]/ev;
    if((tid&1)==0)e[u*H+h]=ev;
    const __mt_bfloat16* xr=x+(int64_t)(u*H+h)*DH+d0;
#pragma unroll
    for(int j=0;j<8;j++){
      float4 v=__bfloat1642float4(*reinterpret_cast<const __bfloat164*>(xr+4*j));
      int64_t o=(int64_t)(h*DH+d0+4*j)*T+u;
      z[o]=__float2half(w*v.x);z[o+T]=__float2half(w*v.y);z[o+2*T]=__float2half(w*v.z);z[o+3*T]=__float2half(w*v.w);
    }
  } else if(blk<2*H) {
    int h=blk-H,d=tid>>2; int64_t o=(int64_t)(h*DH+d)*N+(tid&3)*32;
#pragma unroll
    for(int j=0;j<32;j++)q[o+j]=__float2half(state[o+j]);
  } else {
    smat_tile(b,c,sm,(blk-2*H)*8+(tid>>5),tid&31);
  }
}

template<int WPB> __global__ __launch_bounds__(WPB*32) void main_kernel(
    const __half* sm,const __half* z,const __half* q,const __half* c,
    const float* e,const __mt_bfloat16* x,const float* D,__mt_bfloat16* y) {
  int tid=threadIdx.x,lane=tid&31,tile=blockIdx.x*WPB+(tid>>5);
  int t0=(tile>>8)*16,j0=(tile&255)*16,g=(tile&255)>>5;
  fragment<accumulator,16,16,KT,float> a;fill_fragment(a,0.f);
  for(int k=0;k<t0+16;k+=KT){
    fragment<matrix_a,16,16,KT,__half,row_major> fa;
    fragment<matrix_b,16,16,KT,__half,col_major> fb;
    load_matrix_sync(fa,sm+(int64_t)(g*T+t0)*T+k,T);
    load_matrix_sync(fb,z+(int64_t)j0*T+k,T);mma_sync(a,fa,fb,a);
  }
  for(int k=0;k<N;k+=KT){
    fragment<matrix_a,16,16,KT,__half,row_major> fa;
    fragment<matrix_b,16,16,KT,__half,col_major> fb;
    load_matrix_sync(fa,c+(int64_t)(t0*G+g)*N+k,G*N);
    load_matrix_sync(fb,q+(int64_t)j0*N+k,N);mma_sync(a,fa,fb,a);
  }
#pragma unroll
  for(int qv=0;qv<8;qv++){
    int t=t0+(lane>>3)+((qv>>1)<<2),j=j0+(lane&7)+((qv&1)<<3),h=j>>6;
    int64_t o=(int64_t)(t*H+h)*DH+(j&63);float xv=__bfloat162float(x[o]);
    y[o]=__float2bfloat16_rn(__fmaf_rn(a.x[qv],e[t*H+h],D[h]*xv));
  }
}
}

torch::Tensor musa_ssd_chunk_scan_tce(const torch::Tensor& state,
    const torch::Tensor& x,const torch::Tensor& dt,const torch::Tensor& A,
    const torch::Tensor& B,const torch::Tensor& C,const torch::Tensor& D) {
  TORCH_CHECK(state.device().is_privateuseone() && x.sizes()==torch::IntArrayRef({T,H,DH}) && state.sizes()==torch::IntArrayRef({H,DH,N}));
  TORCH_CHECK(dt.sizes()==torch::IntArrayRef({T,H}) && A.sizes()==torch::IntArrayRef({H}) && B.sizes()==torch::IntArrayRef({T,G,N}) && C.sizes()==torch::IntArrayRef({T,G,N}) && D.sizes()==torch::IntArrayRef({H}));
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && B.scalar_type()==torch::kBFloat16 && C.scalar_type()==torch::kBFloat16 && state.scalar_type()==torch::kFloat32 && dt.scalar_type()==torch::kFloat32 && A.scalar_type()==torch::kFloat32 && D.scalar_type()==torch::kFloat32);
  auto opts=x.options(); auto y=torch::empty({T,H,DH},opts); auto z=torch::empty({H*DH,T},opts.dtype(torch::kFloat16)); auto e=torch::empty({T,H},dt.options()); auto q=torch::empty({H,DH,N},opts.dtype(torch::kFloat16)); auto sm=torch::empty({G,T,T},opts.dtype(torch::kFloat16));
  auto st=c10::musa::getCurrentMUSAStream(); stage_kernel<<<2*H+64,256,0,st>>>(state.data_ptr<float>(),reinterpret_cast<const __mt_bfloat16*>(x.data_ptr()),dt.data_ptr<float>(),A.data_ptr<float>(),reinterpret_cast<const __mt_bfloat16*>(B.data_ptr()),reinterpret_cast<const __mt_bfloat16*>(C.data_ptr()),reinterpret_cast<__half*>(z.data_ptr()),e.data_ptr<float>(),reinterpret_cast<__half*>(q.data_ptr()),reinterpret_cast<__half*>(sm.data_ptr()));
  main_kernel<4><<<(T/16)*(H*DH/16)/4,128,0,st>>>(reinterpret_cast<const __half*>(sm.data_ptr()),reinterpret_cast<const __half*>(z.data_ptr()),reinterpret_cast<const __half*>(q.data_ptr()),reinterpret_cast<const __half*>(C.data_ptr()),e.data_ptr<float>(),reinterpret_cast<const __mt_bfloat16*>(x.data_ptr()),D.data_ptr<float>(),reinterpret_cast<__mt_bfloat16*>(y.data_ptr()));
  TORCH_CHECK(musaGetLastError()==musaSuccess,"SSD TCE launch failed"); return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("musa_ssd_chunk_scan_tce",&musa_ssd_chunk_scan_tce);}
