#include <cuda_runtime.h>
#include "gin_full_state_v3.h"
using U=unsigned long long;
using Net=ncclGin;
__device__ __forceinline__ int find_window(EqState* s,U ptr,U bytes){
 for(int i=0;i<s->count;++i)if(ptr>=s->windows[i].base && ptr-s->windows[i].base+bytes<=s->windows[i].bytes)return i;
 asm volatile("trap;");return -1;
}
extern "C" __device__ __noinline__ void megamoe_gin_full_put(U state_ptr,int context,int slot,int peer,U dst,U src,U bytes,int blocking){
 auto* s=(EqState*)state_ptr;Net gin(s->comm,context);auto team=ncclTeamWorld(s->comm);
 int d=find_window(s,dst,bytes),a=find_window(s,src,bytes);
 if(bytes==8){
  // All eight-byte messages in the audited ring/owner-plan paths are commit
  // records. Each source slot has exactly one writer. Strict VA atomic add
  // publishes the exact original 64-bit record, after prior payload puts.
  U key=src;unsigned index=(src>>3 ^ src>>17)&4095;unsigned tries=0;
  while(true){U prev=atomicCAS(s->shadow_keys+index,0ull,key);if(prev==0 || prev==key)break;
   index=(index+1)&4095;if(++tries==4096){asm volatile("trap;");return;}}
  U value=*(const U*)src,previous=s->shadow_values[index];
  gin.signal(team,peer,ncclGin_StrongVASignalAdd{s->windows[d].window,dst-s->windows[d].base,value-previous});
  s->shadow_values[index]=value;
 }else{
  gin.put(team,peer,s->windows[d].window,dst-s->windows[d].base,s->windows[a].window,src-s->windows[a].base,bytes);
 }
 s->pending[slot].peer=peer;s->pending[slot].valid=1;
 if(blocking && !(context>=24 && context<40))gin.flush(ncclCoopThread());
}
// Progress/source retirement only. Remote consumption and safe reuse are
// proved by original receive commits, cyclic ACKs and terminal credit events.
// This helper deliberately does not implement a stand-alone remote quiet.
extern "C" __device__ __noinline__ void megamoe_gin_full_flush(U state_ptr,int context,int slot){
 auto*s=(EqState*)state_ptr;Net gin(s->comm,context);
 gin.flush(ncclCoopThread());s->pending[slot].valid=0;
}
extern "C" __device__ __noinline__ void megamoe_gin_put_commit(U state_ptr,int context,int slot,int peer,U dst,U src,U bytes,U signal_dst,U signal_src){
 auto*s=(EqState*)state_ptr;Net gin(s->comm,context);auto team=ncclTeamWorld(s->comm);
 int d=find_window(s,dst,bytes),a=find_window(s,src,bytes),sig=find_window(s,signal_dst,8);
 U key=signal_src;unsigned index=(key>>3 ^ key>>17)&4095,tries=0;
 while(true){U prev=atomicCAS(s->shadow_keys+index,0ull,key);if(prev==0 || prev==key)break;index=(index+1)&4095;if(++tries==4096){asm volatile("trap;");return;}}
 U value=*(const U*)signal_src,previous=s->shadow_values[index];
 gin.put(team,peer,s->windows[d].window,dst-s->windows[d].base,s->windows[a].window,src-s->windows[a].base,bytes,ncclGin_StrongVASignalAdd{s->windows[sig].window,signal_dst-s->windows[sig].base,value-previous});
 s->shadow_values[index]=value;s->pending[slot].peer=peer;s->pending[slot].valid=1;
}

// VA signal consumers use NCCL's public acquire/rolling-comparison API.
extern "C" __device__ __noinline__ void megamoe_gin_wait_signal(U state,U addr,U expected,int bits){
 auto*s=(EqState*)state;int d=find_window(s,addr,8);Net gin(s->comm,0);
 gin.waitSignal(ncclCoopThread(),s->windows[d].window,addr-s->windows[d].base,
                expected,bits,cuda::memory_order_acquire);
}
extern "C" __device__ __noinline__ U megamoe_gin_read_signal(U state,U addr){
 auto*s=(EqState*)state;int d=find_window(s,addr,8);Net gin(s->comm,0);
 return gin.readSignal(s->windows[d].window,addr-s->windows[d].base,64,cuda::memory_order_acquire);
}

// All 128 CTA threads participate. Publish only after every payload store is
// system-visible; the following commit put remains the protocol's release.
extern "C" __device__ __noinline__ void megamoe_lsa_put(U state,int peer,U dst,U src,U bytes){
 auto*s=(EqState*)state;int d=find_window(s,dst,bytes);
 char* target=(char*)ncclGetPeerPointer(s->windows[d].window,dst-s->windows[d].base,peer);
 if(!target){asm volatile("trap;");return;}
 __syncthreads();
 if(((dst|src|bytes)&15)==0){
  for(U i=threadIdx.x;i<bytes/16;i+=blockDim.x)((uint4*)target)[i]=((const uint4*)src)[i];
 }else if(((dst|src|bytes)&7)==0){
  for(U i=threadIdx.x;i<bytes/8;i+=blockDim.x)((volatile U*)target)[i]=((const U*)src)[i];
 }else{
  for(U i=threadIdx.x;i<bytes;i+=blockDim.x)target[i]=((const char*)src)[i];
 }
 __threadfence_system();__syncthreads();
}
extern "C" __device__ __noinline__ void megamoe_system_fence(){__threadfence_system();}
