// Query the common buffer layout without compiling a GEMM specialization.
#include <flashinfer/megamoe/eplb_layout.cuh>

#include "megamoe_sm90_common.cuh"

namespace {

// One CTA prepares one N128/K128 record after Python validates the scale domain.
template <uint32_t RecordBytes>
__global__ void predecode_mxfp4_tiles(const uint8_t* packed, const uint8_t* scales, uint8_t* output,
                                      uint32_t n, uint32_t k, bool interleave) {
  if constexpr (RecordBytes == 8448) {
    const uint32_t k_blocks = k / 128u;
    const uint32_t k_block = blockIdx.x % k_blocks;
    const uint32_t expert_n_tile = blockIdx.x / k_blocks;
    const uint32_t expert = expert_n_tile / (n / 128u);
    const uint32_t n_tile = expert_n_tile % (n / 128u);
    uint8_t* record = output + static_cast<uint64_t>(blockIdx.x) * 8448u;
    const auto matrix_row_of = [&](uint32_t row) {
      const uint32_t logical_row = n_tile * 128u + row;
      const uint32_t raw_row = interleave ? logical_row / 16u * 8u + (logical_row & 7u) +
                                                ((logical_row & 8u) ? n / 2u : 0u)
                                          : logical_row;
      return static_cast<uint64_t>(expert) * n + raw_row;
    };
    for (uint32_t slot = threadIdx.x; slot < 2048u; slot += 256u) {
      const uint32_t half = slot / 1024u;
      const uint32_t warp = (slot / 256u) % 4u;
      const uint32_t lane = (slot / 8u) % 32u;
      const uint32_t group = (slot / 2u) % 4u;
      const uint32_t row_pair = slot % 2u;
      const uint32_t row = half * 64u + warp * 16u + lane / 4u + row_pair * 8u;
      const uint64_t matrix_row = matrix_row_of(row);
      uint32_t word = 0;
#pragma unroll
      for (uint32_t side = 0; side < 2; ++side) {
#pragma unroll
        for (uint32_t byte_idx = 0; byte_idx < 4; ++byte_idx) {
          const uint32_t col =
              k_block * 128u + group * 32u + (lane % 4u) * 4u + side * 16u + byte_idx;
          const uint8_t byte = packed[matrix_row * (k / 2u) + col / 8u * 4u + (col & 3u)];
          const uint32_t code = (col & 4u) ? byte & 15u : byte >> 4u;
          word |= (code & 7u) << ((side * 4u + byte_idx) * 4u);
          word |= ((code >> 3u) & 1u) << (3u + byte_idx * 8u + side * 4u);
        }
      }
      const uint32_t physical = half * 4096u + warp * 1024u + lane * 16u + (group / 2u) * 512u +
                                ((group % 2u) * 2u + row_pair) * 4u;
      *reinterpret_cast<uint32_t*>(record + physical) = word;
    }
    if (threadIdx.x < 64u) {
      const uint32_t half = threadIdx.x / 32u;
      const uint32_t warp = (threadIdx.x / 8u) % 4u;
      const uint32_t row_group = threadIdx.x % 8u;
      uint32_t sf_word = 0;
#pragma unroll
      for (uint32_t row_pair = 0; row_pair < 2; ++row_pair) {
        const uint32_t row = half * 64u + warp * 16u + row_group + row_pair * 8u;
        const uint64_t matrix_row = matrix_row_of(row);
#pragma unroll
        for (uint32_t group = 0; group < 4; ++group) {
          const uint32_t sf = scales[matrix_row * (k / 32u) + k_block * 4u + group];
          sf_word |= (sf - 121u) << (group * 4u + row_pair * 16u);
        }
      }
      const uint32_t physical = 8192u + half * 128u + warp * 32u + row_group * 4u;
      *reinterpret_cast<uint32_t*>(record + physical) = sf_word;
    }

  } else {
    const uint32_t k_blocks = k / 128;
    const uint32_t k_block = blockIdx.x % k_blocks;
    const uint32_t expert_n_tile = blockIdx.x / k_blocks;
    const uint32_t expert = expert_n_tile / (n / 128);
    const uint32_t n_tile = expert_n_tile % (n / 128);
    const uint32_t stored_column = threadIdx.x & 127;
#pragma unroll 4
    for (uint32_t row = threadIdx.x >> 7; row < 128; row += 2) {
      const uint32_t logical_row = n_tile * 128 + row;
      const uint32_t raw_row =
          interleave ? logical_row / 16 * 8 + (logical_row & 7) + ((logical_row & 8) ? n / 2 : 0)
                     : logical_row;
      const uint32_t column = k_block * 128 + (stored_column ^ ((row & 7) << 4));
      const uint64_t matrix_row = static_cast<uint64_t>(expert) * n + raw_row;
      const uint8_t byte = packed[matrix_row * (k / 2) + column / 8 * 4 + (column & 3)];
      const uint32_t code = (column & 4) ? byte & 15u : byte >> 4;
      const uint32_t magnitude = code & 7u;
      const int base = __byte_perm(0x3c383000u, 0x4c484440u, magnitude) & 255;
      const int exponent = static_cast<int>(scales[matrix_row * (k / 32) + column / 32]) - 127;
      const uint32_t value = magnitude ? base + exponent * 8 : 0;
      output[static_cast<uint64_t>(blockIdx.x) * 16384 + row * 128 + stored_column] =
          static_cast<uint8_t>(value | ((code & 8u) << 4));
    }
  }
}

}  // namespace

void megamoe_sm90_predecode_mxfp4(tvm::ffi::TensorView packed, tvm::ffi::TensorView scales,
                                  tvm::ffi::TensorView output, bool interleave) {
  CHECK_CUDA(packed);
  CHECK_CUDA(scales);
  CHECK_CUDA(output);
  CHECK_CONTIGUOUS(packed);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(output);
  const DLDataType u8{kDLUInt, 8, 1};
  TVM_FFI_ICHECK_EQ(packed.dtype(), u8);
  TVM_FFI_ICHECK_EQ(scales.dtype(), u8);
  TVM_FFI_ICHECK_EQ(output.dtype(), u8);
  TVM_FFI_ICHECK_EQ(packed.ndim(), 3);
  TVM_FFI_ICHECK_EQ(scales.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.ndim(), 3);
  const int64_t e = packed.size(0), n = packed.size(1), k = packed.size(2) * 2;
  TVM_FFI_ICHECK_GT(e, 0);
  TVM_FFI_ICHECK_GT(n, 0);
  TVM_FFI_ICHECK_GT(k, 0);
  TVM_FFI_ICHECK_EQ(n % 128, 0);
  TVM_FFI_ICHECK_EQ(k % 128, 0);
  TVM_FFI_ICHECK_EQ(scales.size(0), e);
  TVM_FFI_ICHECK_EQ(scales.size(1), n);
  TVM_FFI_ICHECK_EQ(scales.size(2), k / 32);
  TVM_FFI_ICHECK_EQ(output.size(0), e);
  TVM_FFI_ICHECK_EQ(output.size(1), n / 128 * (k / 128));
  const int64_t record_bytes = output.size(2);
  TVM_FFI_ICHECK(record_bytes == 8448 || record_bytes == 16384)
      << "prepared records must contain 8448 or 16384 bytes";
  TVM_FFI_ICHECK_EQ(packed.device().device_id, scales.device().device_id);
  TVM_FFI_ICHECK_EQ(packed.device().device_id, output.device().device_id);
  const int64_t tiles = e * (n / 128) * (k / 128);
  TVM_FFI_ICHECK_LE(tiles, 0x7fffffffLL);
  if (record_bytes == 8448) {
    predecode_mxfp4_tiles<8448>
        <<<static_cast<unsigned>(tiles), 256, 0, get_stream(output.device())>>>(
            static_cast<const uint8_t*>(packed.data_ptr()),
            static_cast<const uint8_t*>(scales.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()), static_cast<uint32_t>(n),
            static_cast<uint32_t>(k), interleave);
  } else {
    predecode_mxfp4_tiles<16384>
        <<<static_cast<unsigned>(tiles), 256, 0, get_stream(output.device())>>>(
            static_cast<const uint8_t*>(packed.data_ptr()),
            static_cast<const uint8_t*>(scales.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()), static_cast<uint32_t>(n),
            static_cast<uint32_t>(k), interleave);
  }
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

tvm::ffi::Array<int64_t> megamoe_sm90_symm_buffer_layout(int64_t ranks, int64_t experts,
                                                         int64_t capacity, int64_t topk,
                                                         int64_t hidden, int64_t intermediate,
                                                         int64_t pool, int64_t sf_pool,
                                                         int64_t remote_nodes) {
  TVM_FFI_ICHECK(remote_nodes >= 0 and ranks % (remote_nodes + 1) == 0)
      << "ranks must split evenly across the nodes";
  return flashinfer::megamoe::detail::symm_buffer_layout(ranks, experts, capacity, topk, hidden,
                                                         intermediate, pool, sf_pool, remote_nodes);
}

// Word offsets of the replica-plan workspace (`eplb::Workspace`), then the
// planner's shared-memory size in bytes:
// [num_words, slot_expert, copies, expert_base, local_counter, copy_list,
//  copy_count, copy_done, remap_done, flags, hist, local_hist, plan_ticket,
//  plan_ready, plan_generation, plan_smem_bytes]
tvm::ffi::Array<int64_t> megamoe_sm90_eplb_workspace_layout(int64_t num_ranks, int64_t num_experts,
                                                            int64_t num_redundant_slots) {
  TVM_FFI_ICHECK(num_ranks > 0 and num_experts > 0 and num_experts % num_ranks == 0)
      << "num_experts must divide across ranks";
  TVM_FFI_ICHECK(num_redundant_slots > 0 and
                 num_redundant_slots <= flashinfer::megamoe::eplb::Workspace::kMaxRedundantSlots)
      << "num_redundant_slots must be in [1, 32]";
  TVM_FFI_ICHECK_LE(num_ranks, flashinfer::megamoe::eplb::Workspace::kMaxRanks)
      << "the in-kernel planner supports at most 32 ranks";
  using flashinfer::megamoe::eplb::Workspace;
  const Workspace ws(num_ranks, num_experts, num_redundant_slots);
  const int64_t smem = flashinfer::megamoe::eplb::plan_smem_bytes(
      num_ranks, num_experts, num_redundant_slots, Workspace::kNumPlanThreads / 32);
  return tvm::ffi::Array<int64_t>(
      {ws.num_words(), ws.slot_expert_offset(), ws.copies_offset(), ws.expert_base_offset(),
       ws.local_counter_offset(), ws.copy_list_offset(), ws.copy_count_offset(),
       ws.copy_done_offset(), ws.remap_done_offset(), ws.flags_offset(), ws.hist_offset(),
       ws.local_hist_offset(), ws.plan_ticket_offset(), ws.plan_ready_offset(),
       ws.plan_generation_offset(), smem});
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_symm_buffer_layout, megamoe_sm90_symm_buffer_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_eplb_workspace_layout,
                              megamoe_sm90_eplb_workspace_layout);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_predecode_mxfp4, megamoe_sm90_predecode_mxfp4);
