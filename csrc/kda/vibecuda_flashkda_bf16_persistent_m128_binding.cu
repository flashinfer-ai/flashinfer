// VibeCUDA binding TU for the persistent FlashKDA M128 prefill kernel
// (vibecuda_flashkda_bf16_persistent_m128.cu). The kernel matches the frozen
// persistent-M128 schedule with plain const-void* TMA descriptor params.
//
// Persistent dispatch: packed workloads whose (sequence, head) task count
// exceeds the physical SM count leave the direct M128 route underfilled /
// wave-chained (e.g. 384 tasks on 152 SMs = 2.53 waves with a serial
// last-task tail on every SM). This TU plans balanced task bins on device
// (one 32-thread block) so the persistent kernel launches exactly sm_count
// workers, each chewing a balanced chunk-count bin with no inter-CTA waves.
// Buffers are capacity-grown workspace scratch; their contents are
// recomputed on every call.
#include "vibecuda_flashkda_tma.cuh"

// The generated standalone source declares its own fixed-width typedefs;
// isolate them so they do not collide with this TU's CUDA headers.
#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
#define FlashKDATensorMap flashkda_generated_FlashKDATensorMap
#define FlashKDATensorMapPack flashkda_generated_FlashKDATensorMapPack
#define CUtensorMap flashkda_generated_CUtensorMap
#include "vibecuda_flashkda_bf16_persistent_m128.cu"
#undef CUtensorMap
#undef FlashKDATensorMapPack
#undef FlashKDATensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace kda_flash {

static_assert(THREADS == 1024);
static_assert(SMEM_TOTAL == 221696);

// Device-side persistent task planner. Fast path (uniform sequence lengths):
// contiguous head-grouped bins written fully in parallel (~2us), which is
// exactly the load-balanced split for equal chunk counts. Slow path (ragged):
// greedy LPT over the descending-length order with ALL bin state kept in
// registers distributed across the warp (bin w lives in slot w/32 of lane
// w%32) plus a register-staged per-block chunk LUT, so the serial per-task
// step is shuffles + register compares only - no shared memory or
// __syncthreads inside the loop.-workers.
__global__ void __launch_bounds__(32)
    kernel_flashkda_persistent_plan(const long long* __restrict__ cu_seqlens,
                                    const int* __restrict__ seq_order, int num_seqs, int num_heads,
                                    int sm_count, int* __restrict__ task_ids,
                                    int* __restrict__ task_offsets,
                                    int* __restrict__ choice_scratch) {
  const int lane = threadIdx.x;
  const int total_tasks = num_seqs * num_heads;
  const int workers = (sm_count < total_tasks) ? sm_count : total_tasks;

  // Uniform detection: check the length of the first sequence against every
  // other sequence, in parallel.
  const long long len0 = cu_seqlens[seq_order[0] + 1] - cu_seqlens[seq_order[0]];
  bool uniform = true;
  for (int s = lane; s < num_seqs; s += 32) {
    const int seq_idx = seq_order[s];
    if (cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx] != len0) {
      uniform = false;
    }
  }
  uniform = __all_sync(0xffffffffu, uniform);

  if (uniform) {
    // Parallel contiguous split: worker w owns [w*T/W, (w+1)*T/W); the task
    // ordering inside bins is identity (tasks already head-grouped).
    for (int w = lane; w <= workers; w += 32) {
      task_offsets[w] = (int)(((long long)w * total_tasks) / workers);
    }
    for (int pos = lane; pos < total_tasks; pos += 32) {
      task_ids[pos] = pos;
    }
    return;
  }

  // Ragged: serpentine (zigzag) assignment. The task order is the caller's
  // stable descending-length order, so ping-ponging the bin index across
  // successive rows of `workers` tasks approximates LPT balance (classic
  // sorted-list break) with a pure index function - fully parallel, no
  // serial assignment loop at all.
  {
    __shared__ int s_cnt2[160];
    __shared__ int s_off2[161];
    if (lane < 5) {
      for (int w = lane; w < workers; w += 5) {
        s_cnt2[w] = 0;
      }
    }
    __syncwarp();
    for (int pos = lane; pos < total_tasks; pos += 32) {
      const int row = pos / workers;
      const int col = pos - row * workers;
      const int w = (row & 1) ? (workers - 1 - col) : col;
      choice_scratch[pos] = w;
      atomicAdd(&s_cnt2[w], 1);
    }
    __syncwarp();
    if (lane == 0) {
      int running = 0;
      for (int w = 0; w < workers; ++w) {
        s_off2[w] = running;
        task_offsets[w] = running;
        running += s_cnt2[w];
        s_cnt2[w] = 0;  // cursor reuse
      }
      s_off2[workers] = running;
      task_offsets[workers] = running;
    }
    __syncwarp();
    for (int pos = lane; pos < total_tasks; pos += 32) {
      const int w = choice_scratch[pos];
      task_ids[s_off2[w] + atomicAdd(&s_cnt2[w], 1)] = pos;
    }
    return;
  }
}

void RunPersistentM128(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                       const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                       const void* dt_bias_ptr, const void* cu_seqlens_ptr,
                       const void* seq_order_ptr, void* task_ids_ptr, void* task_offsets_ptr,
                       void* choice_scratch_ptr, const void* initial_state_ptr, void* out_ptr,
                       void* final_state_ptr, void* descriptor_storage_ptr, int64_t token_count,
                       int64_t num_seqs, int64_t prepare_descriptors, int64_t num_heads,
                       int64_t use_initial_state, int64_t store_final_state, double scale,
                       double lower_bound, int64_t beta_tma_rows, int64_t beta_tma_dim1,
                       int64_t sm_count, int64_t cuda_stream) {
  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  kda_flash::CheckCuda(
      cudaFuncSetAttribute(kernel_flashkda_bf16_persistent_m128,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
      "cudaFuncSetAttribute(kernel_flashkda_bf16_persistent_m128)");

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointersAll<128>(
      q_ptr, k_ptr, v_ptr, g_ptr, beta_tma_ptr, out_ptr, /*scratch_out_ptr=*/nullptr, token_count,
      num_heads, beta_tma_rows * beta_tma_dim1, beta_tma_dim1, descriptor_storage_ptr,
      prepare_descriptors, stream);

  const int64_t total_tasks = num_seqs * num_heads;
  KDA_FFI_CHECK(total_tasks > 0, "persistent m128 requires at least one task");
  const int64_t workers = (sm_count < total_tasks) ? sm_count : total_tasks;
  KDA_FFI_CHECK(workers <= 160, "persistent m128 plan supports at most 160 workers");

  kernel_flashkda_persistent_plan<<<1, 32, 0, stream>>>(  // NOLINT: single-block LPT planner
      reinterpret_cast<const long long*>(cu_seqlens_ptr),
      reinterpret_cast<const int*>(seq_order_ptr), static_cast<int>(num_seqs),
      static_cast<int>(num_heads), static_cast<int>(workers), reinterpret_cast<int*>(task_ids_ptr),
      reinterpret_cast<int*>(task_offsets_ptr), reinterpret_cast<int*>(choice_scratch_ptr));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_persistent_plan launch");

  const dim3 grid(static_cast<uint32_t>(workers), 1, 1);
  const dim3 block(THREADS, 1, 1);
  kernel_flashkda_bf16_persistent_m128<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(q_ptr)), tma.q,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(k_ptr)), tma.k,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(v_ptr)), tma.v,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(g_ptr)), tma.g,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(beta_ptr)), tma.beta,
      reinterpret_cast<float*>(const_cast<void*>(A_log_ptr)),
      reinterpret_cast<float*>(const_cast<void*>(dt_bias_ptr)),
      reinterpret_cast<long long*>(const_cast<void*>(cu_seqlens_ptr)),
      reinterpret_cast<int*>(const_cast<void*>(seq_order_ptr)),
      reinterpret_cast<int*>(task_ids_ptr), reinterpret_cast<int*>(task_offsets_ptr),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(initial_state_ptr)),
      reinterpret_cast<__nv_bfloat16*>(out_ptr), tma.out,
      reinterpret_cast<__nv_bfloat16*>(final_state_ptr), static_cast<int>(num_heads),
      static_cast<int>(use_initial_state), static_cast<int>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_persistent_m128 launch");
}

}  // namespace kda_flash
