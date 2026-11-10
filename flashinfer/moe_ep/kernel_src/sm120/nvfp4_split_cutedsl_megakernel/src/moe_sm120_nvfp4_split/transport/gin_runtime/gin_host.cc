// Host-only bridge loads the exact private NCCL .so, not PyTorch's global NCCL.
#define NCCL_HOSTLIB_ONLY
#include <cuda_runtime_api.h>
#include "gin_full_state_v3.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#define EQ_STR1(x) #x
#define EQ_STR(x) EQ_STR1(x)
static thread_local std::string error;
static void* library=nullptr;
static std::string path;
extern "C" int eq_set_library(const char* value) {
  if(library || !value || !*value) return -1;
  path=value; return 0;
}
template<class T> static T symbol(const char* name) {
  if(!library) {library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);if(!library)throw std::runtime_error(dlerror());}
  void* p=dlsym(library,name);if(!p)throw std::runtime_error(std::string("Missing NCCL symbol ")+name);
  return reinterpret_cast<T>(p);
}
#define EQ_CALL(name,...) check(symbol<decltype(&name)>(EQ_STR(name))(__VA_ARGS__),EQ_STR(name))
static void check(ncclResult_t r,const char* what) {
  if(r!=ncclSuccess)throw std::runtime_error(std::string(what)+" NCCL status "+std::to_string((int)r));
}
static void cuda_check(cudaError_t r,const char* what) {
  if(r!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(r));
}
struct Host {ncclComm_t comm=nullptr; EqState state{}; EqState* device=nullptr; bool dev_created=false;};
#define EQ_BEGIN try { error.clear();
#define EQ_END return 0;}catch(const std::exception& e){error=e.what();return -1;}
extern "C" const char* eq_error(){return error.c_str();}
extern "C" int eq_version(int* out) {
  EQ_BEGIN
  EQ_CALL(ncclGetVersion,out);
  if(*out!=23203)throw std::runtime_error("Expected private NCCL2.32.3");
  // Verify versioned entry points without initializing CUDA or a communicator.
  (void)symbol<decltype(&ncclDevCommCreate)>(EQ_STR(ncclDevCommCreate));
  (void)symbol<decltype(&ncclCommWindowRegister)>(EQ_STR(ncclCommWindowRegister));
  EQ_END
}
extern "C" int eq_unique_id(void* uid) {EQ_BEGIN EQ_CALL(ncclGetUniqueId,(ncclUniqueId*)uid);EQ_END}
// Private torch MemPool hooks: NCCL windows require cuMem-backed allocations.
// These hooks affect only the explicit experimental pool, never the default.
extern "C" void* eq_pool_allocate(size_t bytes,int device,cudaStream_t) {
  try {
    cuda_check(cudaSetDevice(device),"pool device");
    void* ptr=nullptr;EQ_CALL(ncclMemAlloc,&ptr,bytes);return ptr;
  }catch(const std::exception& e){error=e.what();return nullptr;}
}
extern "C" void eq_pool_free(void* ptr,size_t,int device,cudaStream_t) {
  try {cuda_check(cudaSetDevice(device),"pool free device");EQ_CALL(ncclMemFree,ptr);}
  catch(const std::exception& e){error=e.what();fprintf(stderr,"GIN private pool free: %s\n",e.what());}
}
extern "C" int eq_create(void* uid,int rank,int world,int network,void** handle,unsigned long long* device,int* info) {
  EQ_BEGIN
  Host* h=new Host();*handle=h;
  EQ_CALL(ncclCommInitRank,&h->comm,world,*reinterpret_cast<ncclUniqueId*>(uid),rank);
  ncclDevCommRequirements req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if(network) { req.ginForceEnable=true;req.ginSignalCount=4*world;req.ginStrongSignalsRequired=true;req.ginVaSignalsRequired=true;req.ginContextCount=40;req.ginQueueDepth=1024;
  req.ginConnectionType=NCCL_GIN_CONNECTION_FULL;req.ginExclusiveContexts=false; }
  EQ_CALL(ncclDevCommCreate,h->comm,&req,&h->state.comm);h->dev_created=true;
  ncclCommProperties props=NCCL_COMM_PROPERTIES_INITIALIZER;
  EQ_CALL(ncclCommQueryProperties,h->comm,&props);
  info[0]=props.ginType;info[1]=h->state.comm.ginContextCount;
  info[2]=h->state.comm.ginConnectionCount;info[3]=h->state.comm.ginNetDeviceTypes[0];
  info[4]=h->state.comm.rank;info[5]=h->state.comm.nRanks;
  if(network && (props.ginType!=NCCL_GIN_TYPE_GDAKI || info[1]!=40 || info[2]!=1 || info[3]!=NCCL_GIN_TYPE_GDAKI))
    throw std::runtime_error("Require exactly one GDAKI connection with forty contexts; actual type="+std::to_string(info[0])+" contexts="+std::to_string(info[1])+" connections="+std::to_string(info[2])+" deviceType="+std::to_string(info[3]));
  cuda_check(cudaMalloc((void**)&h->device,sizeof(EqState)),"state allocation");
  cuda_check(cudaMemcpy(h->device,&h->state,sizeof(EqState),cudaMemcpyHostToDevice),"state publication");
  *device=reinterpret_cast<unsigned long long>(h->device);
  EQ_END
}
extern "C" int eq_register(void* handle,unsigned long long address,unsigned long long bytes) {
  EQ_BEGIN
  Host* h=(Host*)handle;int i=h->state.count;if(i>=64)throw std::runtime_error("window registry full");
  auto& w=h->state.windows[i];w.base=address;w.bytes=bytes;
  // Every rank registers the identical logical allocation in the same order.
  // VA signals must be STRICT_ORDERING; no ordinary stores while active.
  EQ_CALL(ncclCommWindowRegister,h->comm,(void*)address,bytes,&w.window,NCCL_WIN_STRICT_ORDERING);
  ++h->state.count;
  cuda_check(cudaMemcpy(h->device,&h->state,sizeof(EqState),cudaMemcpyHostToDevice),"window publication");
  EQ_END
}
extern "C" int eq_destroy(void* handle) {
  EQ_BEGIN
  Host* h=(Host*)handle;
  cuda_check(cudaDeviceSynchronize(),"final device drain");
  for(int i=h->state.count-1;i>=0;--i)EQ_CALL(ncclCommWindowDeregister,h->comm,h->state.windows[i].window);
  if(h->dev_created)EQ_CALL(ncclDevCommDestroy,h->comm,&h->state.comm);
  if(h->device)cuda_check(cudaFree(h->device),"state free");
  if(h->comm)EQ_CALL(ncclCommDestroy,h->comm);
  delete h;
  EQ_END
}

extern "C" int eq_peer(void* handle, int index, int peer, unsigned long long* address) {
 EQ_BEGIN
 Host* h=(Host*)handle;
 if(index<0 || index>=h->state.count)throw std::runtime_error("Invalid window");
 void* ptr=nullptr;
 EQ_CALL(ncclGetPeerDevicePointer,h->state.windows[index].window,0,peer,&ptr);
 *address=reinterpret_cast<unsigned long long>(ptr);
 EQ_END
}
