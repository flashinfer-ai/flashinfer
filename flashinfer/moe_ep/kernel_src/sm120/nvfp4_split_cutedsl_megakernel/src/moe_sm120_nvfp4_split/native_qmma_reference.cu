// Independent SM120 NVFP4 oracle. Direct global loads, one warp per 16x8 tile.
// PTX ISA: mma.m16n8k64 E2M1 fragments and scale_vec::4X selectors.
#include <cuda_runtime.h>
#include <stdint.h>
__device__ uint32_t word(const uint8_t* p,int row,int rows,int stride,int off) {
 return row<rows ? *reinterpret_cast<const uint32_t*>(p+(int64_t)row*stride+off) : 0;
}
__global__ void reference(const uint8_t* x,const uint8_t* sx,const uint8_t* w,
                          const uint8_t* sw,float* y,int m,int n,int k) {
 int lane=threadIdx.x&31, g=lane>>2, t=lane&3;
 int ni=blockIdx.x*16, mi=blockIdx.y*8;
 float d0=0,d1=0,d2=0,d3=0;
 for(int kk=0;kk<k;kk+=64) {
  uint32_t a0=word(w,ni+g,n,k/2,kk/2+t*4);
  uint32_t a1=word(w,ni+g+8,n,k/2,kk/2+t*4);
  uint32_t a2=word(w,ni+g,n,k/2,kk/2+t*4+16);
  uint32_t a3=word(w,ni+g+8,n,k/2,kk/2+t*4+16);
  uint32_t b0=word(x,mi+g,m,k/2,kk/2+t*4);
  uint32_t b1=word(x,mi+g,m,k/2,kk/2+t*4+16);
  uint32_t sa=word(sw,ni+g+(t&1)*8,n,k/16,kk/16);
  uint32_t sb=word(sx,mi+g,m,k/16,kk/16);
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
 }
 int row=mi+t*2,col=ni+g;
 if(row<m && col<n)y[(int64_t)row*n+col]=d0;
 if(row+1<m && col<n)y[(int64_t)(row+1)*n+col]=d1;
 if(row<m && col+8<n)y[(int64_t)row*n+col+8]=d2;
 if(row+1<m && col+8<n)y[(int64_t)(row+1)*n+col+8]=d3;
}
extern "C" int native_qmma(const void* x,const void* sx,const void* w,const void* sw,
                           void* y,int m,int n,int k,void* stream) {
 reference<<<dim3((n+15)/16,(m+7)/8),32,0,(cudaStream_t)stream>>>
 ((const uint8_t*)x,(const uint8_t*)sx,(const uint8_t*)w,(const uint8_t*)sw,(float*)y,m,n,k);
 return (int)cudaGetLastError();
}
