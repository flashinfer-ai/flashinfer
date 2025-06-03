#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <tuple>
#include <type_traits>

#include "../exception.h"
#include "../logging.h"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace trtllm_moe_allreduce_fusion {

namespace details {

static constexpr int kBytesPerAccess = 16;
static constexpr int kOneShotMaxToken = 128;
static constexpr int kBarrierFlagCount = 256;

}  // namespace details

enum class FP4QuantizationSFLayout {
  // Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
  // blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8
  // values. The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and
  // columns to the nearest multiple of 4.
  //
  // The scale factor block rows map to data block rows in an interleaved pattern:
  // For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
  // Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
  //
  // Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
  SWIZZLED,
  // Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen
  // kernels standard.
  LINEAR
};

namespace utils {
#define FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(std::optional<int> batchIdx, int rowIdx,
                                                       int colIdx, std::optional<int> numRows,
                                                       int numCols, SFType* SFout,
                                                       FP4QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    if (layout == FP4QuantizationSFLayout::SWIZZLED) {
      // SF vector index (16 elements share one SF in the K dimension).
      // numRows and numCols are unpadded.
      int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
      int32_t mIdx = rowIdx;

      auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numCols);
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else if (layout == FP4QuantizationSFLayout::LINEAR) {
      // Linear row-major layout, no padding required.
      int32_t KTileIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;

      int32_t numKTiles = numCols / CVT_FP4_SF_VEC_SIZE;
      int64_t mTileStride = numKTiles;

      int64_t BTileStride = numRows.value_or(0) * mTileStride;

      int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else {
      return nullptr;
    }
  }
#endif
  return nullptr;
}
}  // namespace utils

// struct __device_builtin__ __builtin_align__(16) float4
// {
//     float x, y, z, w;
// };

template <typename T>
struct AllReduceFusionParams {
  int nranks;
  int rank;
  // size = token_num * hidden_dim
  int size;
  int hidden_dim;
  void** workspace;
  void* allreduce_in;
  void* residual_in;
  void* residual_out;
  void* norm_out;
  void* quant_out;
  void* scale_out;
  void* rms_gamma;
  float rms_eps;
  // todo(review): why float* scale_factor in trt-llm?
  float scale_factor;
  FP4QuantizationSFLayout layout = FP4QuantizationSFLayout::SWIZZLED;
  cudaStream_t stream;
};

template <typename T>
struct MoeReductionAllReduceFusionParams : public AllReduceFusionParams<T> {
  // * moe reduction specific params
  // Refer to kernel implementation on layout of those params
  // number of active experts on current device

  // todo(review): why int* moe_reduction_device_num_experts = nullptr; in trt-llm?
  int moe_reduction_device_num_experts = 0;
  // per token per expert fp32 scale
  float* moe_reduction_scale_input = nullptr;
  // per token per expert input
  void* moe_reduction_active_experts_token_input = nullptr;
  // per token input
  void* moe_reduction_token_input = nullptr;
};

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
    flag_value = *flag_ptr;
    int comm_size = reinterpret_cast<int*>(workspace[NRanks * 3])[3];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] =
          reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int clear_size;
  int flag_value;
};

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> vec_add(const vec_t<T, VEC_SIZE>& a,
                                                      const vec_t<T, VEC_SIZE>& b) {
  vec_t<T, VEC_SIZE> ret;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    ret[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
  }
  return ret;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> rms_norm(vec_t<T, VEC_SIZE> const& residual,
                                                       vec_t<T, VEC_SIZE> const& gamma,
                                                       float const eps, int hidden_dim) {
  namespace cg = cooperative_groups;
  __shared__ float s_val;
  vec_t<T, VEC_SIZE> norm_out;
  cg::cluster_group cluster = cg::this_cluster();
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float v = static_cast<float>(residual[i]);
    acc += v * v;
  }
  utils::blockReduceSumV2<float, 1>(&acc);
  if (cluster.num_blocks() > 1) {
    if (threadIdx.x == 0) {
      s_val = acc;
      acc = 0.f;
    }
    cluster.sync();
    if (threadIdx.x == 0) {
      for (int i = 0; i < cluster.num_blocks(); ++i) {
        acc += *cluster.map_shared_rank(&s_val, i);
      }
    }
    cluster.sync();
  }
  if (threadIdx.x == 0) {
    s_val = rsqrtf(acc / hidden_dim + eps);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    norm_out[i] =
        static_cast<T>(static_cast<float>(residual[i]) * s_val * static_cast<float>(gamma[i]));
  }
  return norm_out;
}

template <bool ResidualOut, bool NormOut, bool QuantOut, typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void fused_op(vec_t<T, VEC_SIZE> const& val, int access_id, int token_id,
                                         int access_id_in_token, AllReduceFusionParams<T>& params) {
  vec_t<T, VEC_SIZE> residual_val;
  vec_t<T, VEC_SIZE> gamma_val;
  residual_val.load(reinterpret_cast<vec_t<T, VEC_SIZE>*>(params.residual_in) + access_id);
  gamma_val.load(reinterpret_cast<vec_t<T, VEC_SIZE>*>(params.rms_gamma) + access_id_in_token);
  residual_val = vec_add<T, VEC_SIZE>(val, residual_val);
  if constexpr (ResidualOut) {
    residual_val.store(reinterpret_cast<vec_t<T, VEC_SIZE>*>(params.residual_out) + access_id);
  }
  vec_t<T, VEC_SIZE> norm_val =
      rms_norm<T, VEC_SIZE>(residual_val, gamma_val, params.rms_eps, params.hidden_dim);
  if constexpr (NormOut) {
    norm_val.store(reinterpret_cast<vec_t<T, VEC_SIZE>*>(params.norm_out) + access_id);
  }
  if constexpr (QuantOut) {
    vec_t<T, VEC_SIZE> pack_val;
    pack_val.cast_load(norm_val);
    auto sf_out = utils::cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 2>(
        std::nullopt /* batchIdx */, token_id, access_id_in_token, std::nullopt /* numRows */,
        params.hidden_dim, reinterpret_cast<uint32_t*>(params.scale_out), params.layout);
    reinterpret_cast<uint32_t*>(params.quant_out)[access_id] =
        cvt_warp_fp16_to_fp4(pack_val, *params.scale_factor, sf_out);
  }
}

template <typename T>
struct neg_zero {
  static constexpr T value = -T(0);
};

template <>
struct neg_zero<half> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __half value = __half_raw{neg_zero_bits};
};

template <>
struct neg_zero<nv_bfloat16> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __nv_bfloat16 value = __nv_bfloat16_raw{neg_zero_bits};
};

template <typename T>
__device__ static constexpr T neg_zero_v = neg_zero<T>::value;

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ bool has_neg_zero(const vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    if (vec[i] == neg_zero_v<T>) {
      return true;
    }
  }
  return false;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void remove_neg_zero(vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    vec[i] = (vec[i] == neg_zero_v<T>) ? static_cast<T>(0.f) : vec[i];
  }
}

template <typename T>
__device__ __forceinline__ void set_neg_zero(T* addr) {
  vec_t<T, details::kBytesPerAccess / sizeof(T)> val;
  val.fill(neg_zero_v<T>);
  val.store_global_volatile(addr);
}

__device__ __forceinline__ float4 ld_global_volatile(float4* addr) {
  float4 val;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
               : "l"(addr));
  return val;
}

int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    auto status = cudaGetDevice(&device_id);
    FLASHINFER_CHECK(status == cudaSuccess, "cudaGetDevice failed with error code " +
                                                std::string(cudaGetErrorString(status)));
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    sm_count = device_prop.multiProcessorCount;
  }
  return sm_count;
}

bool use_oneshot(int token_num) { return token_num <= details::kOneShotMaxToken; }

/////////////////////////////////////////////////////////////////
//                  * MoE Reduction Fusion *                   //
/////////////////////////////////////////////////////////////////

template <typename T, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
__global__ void moereduce_allreduce_fusion_kernel_oneshot_lamport(
    MoeReductionAllReduceFusionParams<T> params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  cg::grid_group grid = cg::this_grid();

  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

  // Each token is handled by one cluster
  // which token is handled by current cluster
  int token_id = grid.cluster_rank();
  // total number of token
  int num_token = params.size / params.hidden_dim;
  // Each thread handle kElemsPerAccess num elem in token. Total cluster.num_threads() to handle one
  // token For current token, which kElemsPerAccess is handled by current thread (in unit of
  // kElemsPerAccess)
  int access_id_in_token = cluster.thread_rank();
  // Across all token, which kElemsPerAccess is handled by current thread (in unit of
  // kElemsPerAccess)
  int access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
  // Persistent kernel
  // stride to next token handled by current cta
  int token_stride = grid.num_clusters();
  // stride in unit of kElemsPerAccess
  int access_stride = token_stride * params.hidden_dim / VEC_SIZE;
  // Total number of access in unit of kElemsPerAccess to handle (token_num * hidden_dim)
  // This is within one rank

  int tot_access = params.size / VEC_SIZE;
  vec_t<T, VEC_SIZE> clear_vec;
  clear_vec.fill(neg_zero_v<T>);

  cudaGridDependencySynchronize();
  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / VEC_SIZE;

  // * MoE related
  int threadid_in_cluster = cluster.thread_rank();
  // Start Offset within one token's hidden_size of element
  // Current thread handle token[thread_offset_within_token : thread_offset_within_token +
  // kElemsPerAccess]

  // todo(review): review this
  int thread_offset_within_token = threadid_in_cluster * VEC_SIZE;

  // todo(review): review this
  union ACC_TYPE {
    vec_t<T, VEC_SIZE> packed;
    T unpacked[VEC_SIZE];
  };

  // Persistent Kernel
  // Each cluster iterate through all token it need to handle
  for (int token_id = grid.cluster_rank(); token_id < num_token; token_id += grid.num_clusters()) {
    if (thread_offset_within_token >= params.hidden_dim) {
      break;
    }

    // * MoE Reduce
    // Offset within (num_token, hidden_size) in unit of element
    int thread_offset_across_token = token_id * params.hidden_dim + thread_offset_within_token;

    ACC_TYPE accumulator;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      accumulator.unpacked[i] = static_cast<T>(0);
    }

    // * Iterate through all active expert
    int num_actexp = *(params.moe_reduction_device_num_experts);
    for (int actexp_i = 0; actexp_i < num_actexp; ++actexp_i) {
      // * Load active expert i's token j's partial data
      // Offset within (num_act_exp, num_token, hidden_size) in unit of element
      int thread_offset_across_actexp_token =
          actexp_i * (params.hidden_dim * num_token) + thread_offset_across_token;
      ACC_TYPE actexp_i_data;

      actexp_i_data.packed = reinterpret_cast<vec_t<T, VEC_SIZE> const*>(
          params.moe_reduction_active_experts_token_input)[thread_offset_across_actexp_token /
                                                           VEC_SIZE];

      // * Load active expert i's token j's scale
      int thread_offset_scale = actexp_i * num_token + token_id;
      float actexp_i_token_j_scale =
          reinterpret_cast<float const*>(params.moe_reduction_scale_input)[thread_offset_scale];

      // * acc += scale(data)
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        // assume computation is done in ScaleType
        accumulator.unpacked[i] += static_cast<T>(
            (static_cast<float>(actexp_i_data.unpacked[i]) * actexp_i_token_j_scale));
      }
    }

    // * FC2 + reduced(gGEMM2)
    ACC_TYPE fc2_data;
    fc2_data.packed = reinterpret_cast<vec_t<T, VEC_SIZE> const*>(
        params.moe_reduction_token_input)[thread_offset_across_token / VEC_SIZE];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      accumulator.unpacked[i] += fc2_data.unpacked[i];
    }

    // * AR Store
    int access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
    int idx = access_id;
    vec_t<T, VEC_SIZE> val;
    val.load(accumulator.packed);

#pragma unroll
    if (has_neg_zero(val)) {
      val.fill(0.f);
    }
    // for (int i = 0; i < 4; ++i) {
    //   // Handle two bf16/fp16 at one time
    //   if (is_neg_zero(val[i])) {
    //     val[i] = 0.f;
    //   }
    // }
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      // STG.128 to remote rank
      reinterpret_cast<vec_t<T, VEC_SIZE>*>(comm.data_bufs[r])[params.rank * tot_access + idx] =
          val;
    }
  }

  // * Clear previous buffer
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    reinterpret_cast<vec_t<T, VEC_SIZE>*>(comm.clear_buf)[idx] = clear_vec;
  }

  // * AR Load + Fusion
  for (int idx = access_id, tidx = token_id; idx < tot_access;
       idx += access_stride, tidx += token_stride) {
    // * AR Load
    vec_t<T, VEC_SIZE> vals[NRanks];
    bool done = false;
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        // LDG.128 from local rank
        vals[r].load_global_volatile(&reinterpret_cast<vec_t<T, VEC_SIZE>*>(
            comm.data_bufs[params.rank])[r * tot_access + idx]);
        done &= !has_neg_zero(vals[r]);
      }
    }
    vec_t<T, VEC_SIZE> sum_val = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
      sum_val = vec_add<T, VEC_SIZE>(sum_val, vals[r]);
    }

    // * Fuse
    fused_op<ResidualOut, NormOut, QuantOut, T, VEC_SIZE>(sum_val, idx, tidx, access_id_in_token,
                                                          params);
  }
  comm.update(params.size * NRanks);
  cudaTriggerProgrammaticLaunchCompletion();
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
cudaError_t launch_oneshot_moereduce_lamport(MoeReductionAllReduceFusionParams<T> const& params,
                                             cudaLaunchConfig_t& cfg) {
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &cfg,
      moereduce_allreduce_fusion_kernel_oneshot_lamport<T, NRanks, ResidualOut, NormOut, QuantOut>,
      params));
  return cudaSuccess;
}

template <typename T, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
cudaError_t moereduction_allreduce_fusion_kernel_launcher(
    MoeReductionAllReduceFusionParams<T> const& params, bool launch_with_pdl) {
  int token_num = params.size / params.hidden_dim;
  bool oneshot = use_oneshot(token_num);
  // Only support one shot
  FLASHINFER_CHECK(oneshot, "only support one shot");
  // Each token is handled by one cluster
  int cluster_num = token_num;
  // Total number of threads (within one cluster) that's need to handle one token
  // given that each thread handle kElemsPerAccess
  int threads_per_token = params.hidden_dim * sizeof(T) / details::kBytesPerAccess;
  // Total number of warp (within one cluster) that's need to handle one token
  // given that each thread handle kElemsPerAccess
  int warps_per_token = (threads_per_token + 31) / 32;
  int cluster_size = 8;
  while (warps_per_token % cluster_size != 0) {
    cluster_size /= 2;
  }
  int block_size = warps_per_token / cluster_size * 32;
  FLASHINFER_CHECK(block_size <= 1024 && cluster_size > 0,
                   "block_size <= 1024 && cluster_size > 0");
  int sm_count = get_sm_count();
  int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;
  cudaLaunchConfig_t cfg;
  cudaLaunchAttribute attribute[2];
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = 2;
  if (oneshot) {
    FLASHINFER_CUDA_CALL(
        (launch_oneshot_moereduce_lamport<T, NRanks, ResidualOut, NormOut, QuantOut>(params, cfg)));
  }
  return cudaSuccess;
}

template <typename T>
cudaError_t moereduction_allreduce_fusion_op(MoeReductionAllReduceFusionParams<T> const& params,
                                             bool launch_with_pdl) {
  FLASHINFER_CHECK(params.residual_in && params.rms_gamma, "residual_in and rms_gamma must be set");
  FLASHINFER_CHECK(params.moe_reduction_scale_input &&
                       params.moe_reduction_active_experts_token_input &&
                       params.moe_reduction_token_input,
                   "moe_reduction_scale_input, moe_reduction_active_experts_token_input and "
                   "moe_reduction_token_input must be set");
  FLASHINFER_CHECK(params.size % params.hidden_dim == 0, "size must be a multiple of hidden_dim");
  FLASHINFER_CHECK(params.hidden_dim * sizeof(T) % details::kBytesPerAccess == 0,
                   "hidden_dim * sizeof(T) must be a multiple of kBytesPerAccess");
  if (params.residual_out && not params.norm_out && params.quant_out) {
    // pattern1: AR+Add_RMS+Quant
    // [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
    // [m, 7168] bf16 residual_out, [m, 7168] fp4 quant_out
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 2, true, false, true>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 4, true, false, true>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 8, true, false, true>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 16, true, false, true>(
        params, launch_with_pdl)));
    return cudaSuccess;
  } else if (not params.residual_out && params.norm_out && not params.quant_out) {
    // pattern2: AR+AddRMS
    // [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
    // [m, 7168] bf16 norm_out
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 2, false, true, false>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 4, false, true, false>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 8, false, true, false>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 16, false, true, false>(
        params, launch_with_pdl)));
    return cudaSuccess;
  } else if (params.residual_out && params.norm_out && not params.quant_out) {
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 2, true, true, false>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 4, true, true, false>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 8, true, true, false>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 16, true, true, false>(
        params, launch_with_pdl)));
    return cudaSuccess;
  } else if (params.residual_out && params.norm_out && params.quant_out) {
    // for test
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 2, true, true, true>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 4, true, true, true>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 8, true, true, true>(
        params, launch_with_pdl)));
    FLASHINFER_CUDA_CALL((moereduction_allreduce_fusion_kernel_launcher<T, 16, true, true, true>(
        params, launch_with_pdl)));
    return cudaSuccess;
  }
  FLASHINFER_CHECK(false, "allreduce_fusion_kernel: unsupported pattern!");
  return cudaErrorNotSupported;
}
}  // namespace trtllm_moe_allreduce_fusion

}  // namespace flashinfer
