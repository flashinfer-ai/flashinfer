typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "A 64-bit CUDA host ABI is required");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) Dsv4TensorMap { uint64_t opaque[16]; };
struct __align__(64) Dsv4TensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Dsv4TensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(Dsv4TensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) Dsv4TensorMapPack { Dsv4TensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(Dsv4TensorMap) >= alignof(CUtensorMap), "Dsv4TensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define DSV4_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_MASKS_OFF 0
#define SMEM_MASKS_STAGE_BYTES 80
#define SMEM_MASKS_STRIDE 80
#define SMEM_TOTAL 128
#define THREADS 64

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(64) void
kernel_dsv4_t7_direct_route_order(int* __restrict__ route_experts, int* __restrict__ route_order, int route_count)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    unsigned int* masks = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int masks_addr = smem + 0;

    // === Task calls (dependency order) ===
    int expert = 0;
    if (tid < route_count) {
        expert = route_experts[tid];
    }
    unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, (expert & 1) != 0);
    unsigned int votes = _vote_0;
    if (lane == 0) {
        masks[warp * 10] = votes;
    }
    unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, (expert >> 1 & 1) != 0);
    unsigned int votes_0 = _vote_1;
    if (lane == 0) {
        masks[warp * 10 + 1] = votes_0;
    }
    unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, (expert >> 2 & 1) != 0);
    unsigned int votes_1 = _vote_2;
    if (lane == 0) {
        masks[warp * 10 + 2] = votes_1;
    }
    unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, (expert >> 3 & 1) != 0);
    unsigned int votes_2 = _vote_3;
    if (lane == 0) {
        masks[warp * 10 + 3] = votes_2;
    }
    unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, (expert >> 4 & 1) != 0);
    unsigned int votes_3 = _vote_4;
    if (lane == 0) {
        masks[warp * 10 + 4] = votes_3;
    }
    unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, (expert >> 5 & 1) != 0);
    unsigned int votes_4 = _vote_5;
    if (lane == 0) {
        masks[warp * 10 + 5] = votes_4;
    }
    unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, (expert >> 6 & 1) != 0);
    unsigned int votes_5 = _vote_6;
    if (lane == 0) {
        masks[warp * 10 + 6] = votes_5;
    }
    unsigned int _vote_7 = __ballot_sync(0xFFFFFFFF, (expert >> 7 & 1) != 0);
    unsigned int votes_6 = _vote_7;
    if (lane == 0) {
        masks[warp * 10 + 7] = votes_6;
    }
    unsigned int _vote_8 = __ballot_sync(0xFFFFFFFF, (expert >> 8 & 1) != 0);
    unsigned int votes_7 = _vote_8;
    if (lane == 0) {
        masks[warp * 10 + 8] = votes_7;
    }
    unsigned int _vote_9 = __ballot_sync(0xFFFFFFFF, tid < route_count);
    unsigned int valid = _vote_9;
    if (lane == 0) {
        masks[warp * 10 + 9] = valid;
    }
    __syncthreads();
    unsigned int low = masks[9];
    unsigned int high = masks[19];
    unsigned int less_low = 0;
    unsigned int less_high = 0;
    unsigned int low_bits = masks[8];
    unsigned int high_bits = masks[18];
    if ((expert >> 8 & 1) != 0) {
        less_low = less_low | low & (low_bits ^ (unsigned int)-1);
        less_high = less_high | high & (high_bits ^ (unsigned int)-1);
    } else {
        low_bits = low_bits ^ (unsigned int)-1;
        high_bits = high_bits ^ (unsigned int)-1;
    }
    low = low & low_bits;
    high = high & high_bits;
    unsigned int low_bits_8 = masks[7];
    unsigned int high_bits_9 = masks[17];
    if ((expert >> 7 & 1) != 0) {
        less_low = less_low | low & (low_bits_8 ^ (unsigned int)-1);
        less_high = less_high | high & (high_bits_9 ^ (unsigned int)-1);
    } else {
        low_bits_8 = low_bits_8 ^ (unsigned int)-1;
        high_bits_9 = high_bits_9 ^ (unsigned int)-1;
    }
    low = low & low_bits_8;
    high = high & high_bits_9;
    unsigned int low_bits_10 = masks[6];
    unsigned int high_bits_11 = masks[16];
    if ((expert >> 6 & 1) != 0) {
        less_low = less_low | low & (low_bits_10 ^ (unsigned int)-1);
        less_high = less_high | high & (high_bits_11 ^ (unsigned int)-1);
    } else {
        low_bits_10 = low_bits_10 ^ (unsigned int)-1;
        high_bits_11 = high_bits_11 ^ (unsigned int)-1;
    }
    low = low & low_bits_10;
    high = high & high_bits_11;
    unsigned int low_bits_12 = masks[5];
    unsigned int high_bits_13 = masks[15];
    if ((expert >> 5 & 1) != 0) {
        less_low = less_low | low & (low_bits_12 ^ (unsigned int)-1);
        less_high = less_high | high & (high_bits_13 ^ (unsigned int)-1);
    } else {
        low_bits_12 = low_bits_12 ^ (unsigned int)-1;
        high_bits_13 = high_bits_13 ^ (unsigned int)-1;
    }
    low = low & low_bits_12;
    high = high & high_bits_13;
    unsigned int low_bits_14 = masks[4];
    unsigned int high_bits_15 = masks[14];
    if ((expert >> 4 & 1) != 0) {
        less_low = less_low | low & (low_bits_14 ^ (unsigned int)-1);
        less_high = less_high | high & (high_bits_15 ^ (unsigned int)-1);
    } else {
        low_bits_14 = low_bits_14 ^ (unsigned int)-1;
        high_bits_15 = high_bits_15 ^ (unsigned int)-1;
    }
    low = low & low_bits_14;
    high = high & high_bits_15;
    unsigned int low_bits_16 = masks[3];
    unsigned int high_bits_17 = masks[13];
    if ((expert >> 3 & 1) != 0) {
        less_low = less_low | low & (low_bits_16 ^ (unsigned int)-1);
        less_high = less_high | high & (high_bits_17 ^ (unsigned int)-1);
    } else {
        low_bits_16 = low_bits_16 ^ (unsigned int)-1;
        high_bits_17 = high_bits_17 ^ (unsigned int)-1;
    }
    low = low & low_bits_16;
    high = high & high_bits_17;
    unsigned int low_bits_18 = masks[2];
    unsigned int high_bits_19 = masks[12];
    if ((expert >> 2 & 1) != 0) {
        less_low = less_low | low & (low_bits_18 ^ (unsigned int)-1);
        less_high = less_high | high & (high_bits_19 ^ (unsigned int)-1);
    } else {
        low_bits_18 = low_bits_18 ^ (unsigned int)-1;
        high_bits_19 = high_bits_19 ^ (unsigned int)-1;
    }
    low = low & low_bits_18;
    high = high & high_bits_19;
    unsigned int low_bits_20 = masks[1];
    unsigned int high_bits_21 = masks[11];
    if ((expert >> 1 & 1) != 0) {
        less_low = less_low | low & (low_bits_20 ^ (unsigned int)-1);
        less_high = less_high | high & (high_bits_21 ^ (unsigned int)-1);
    } else {
        low_bits_20 = low_bits_20 ^ (unsigned int)-1;
        high_bits_21 = high_bits_21 ^ (unsigned int)-1;
    }
    low = low & low_bits_20;
    high = high & high_bits_21;
    unsigned int low_bits_22 = masks[0];
    unsigned int high_bits_23 = masks[10];
    if ((expert & 1) != 0) {
        less_low = less_low | low & (low_bits_22 ^ (unsigned int)-1);
        less_high = less_high | high & (high_bits_23 ^ (unsigned int)-1);
    } else {
        low_bits_22 = low_bits_22 ^ (unsigned int)-1;
        high_bits_23 = high_bits_23 ^ (unsigned int)-1;
    }
    low = low & low_bits_22;
    high = high & high_bits_23;
    unsigned int earlier = ((unsigned int)1 << (unsigned int)lane) - (unsigned int)1;
    int _popc_0 = __popc(low & earlier);
    int equal_before = _popc_0;
    if (warp == 1) {
        int _popc_1 = __popc(low);
        int _popc_2 = __popc(high & earlier);
        equal_before = _popc_1 + _popc_2;
    }
    int _popc_3 = __popc(less_low);
    int _popc_4 = __popc(less_high);
    int rank = _popc_3 + _popc_4 + equal_before;
    if (tid < route_count) {
        route_order[rank] = tid;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"

