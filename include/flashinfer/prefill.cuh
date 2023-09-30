#ifndef FLASHINFER_PREFILL_CUH_
#define FLASHINFER_PREFILL_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "cp_async.cuh"
#include "layout.cuh"
#include "mma.cuh"
#include "permuted_smem.cuh"
#include "rope.cuh"
#include "state.cuh"
#include "utils.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;

constexpr size_t warp_size = 32;

namespace {

template <bool row_major, typename T>
__device__ __forceinline__ void apply_llama_rope(T *x_first_half, T *x_second_half,
                                                 const float *freq_first_half,
                                                 const float *freq_second_half, size_t offset) {
#pragma unroll
  for (size_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos_first_half, sin_first_half, cos_second_half, sin_second_half, tmp;
    size_t i, j;
    if constexpr (row_major) {
      // 0 1 | 4 5
      // ---------
      // 2 3 | 6 7
      i = ((reg_id % 4) / 2);
      j = (reg_id / 4);
    } else {
      // 0 1 | 2 3
      // ---------
      // 4 5 | 6 7
      i = reg_id / 4;
      j = (reg_id % 4) / 2;
    }
    __sincosf(float(offset + 8 * i) * freq_first_half[2 * j + reg_id % 2], &sin_first_half,
              &cos_first_half);
    __sincosf(float(offset + 8 * i) * freq_second_half[2 * j + reg_id % 2], &sin_second_half,
              &cos_second_half);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = tmp * cos_first_half - (float)x_second_half[reg_id] * sin_first_half;
    x_second_half[reg_id] = (float)x_second_half[reg_id] * cos_second_half + tmp * sin_second_half;
  }
}

}  // namespace

template <size_t num_frags_y, size_t num_frags_z, size_t num_warps, size_t stride, QKVLayout layout,
          typename T>
__device__ __forceinline__ void produce_kv(permuted_smem_t<stride, T> *smem, T *gmem,
                                           const tensor_info_t<layout> &qkv_info,
                                           size_t kv_idx_base, size_t kv_len, size_t head_idx) {
  size_t tx = threadIdx.x, ty = threadIdx.y;
  auto block = cg::this_thread_block();
#pragma unroll
  for (size_t step = 0; step < num_frags_z * num_frags_y / num_warps; ++step) {
    size_t i = 8 * ((step * num_warps + ty) / (num_frags_y / 2)) + tx / 4,
           j = ((step * num_warps + ty) % (num_frags_y / 2)) * 4 + tx % 4;
    size_t kv_idx = kv_idx_base + i, feat_idx = j * bank_capacity<T>();
    smem->load_bank_async(i, j, gmem + qkv_info.get_kv_elem_offset(kv_idx, head_idx, feat_idx),
                          kv_idx < kv_len);
  }
  block.sync();
}

template <QKVLayout layout, RotaryMode rotary_mode, size_t num_frags_y, size_t num_frags_z,
          size_t num_stages_smem, size_t num_warps, typename DTypeIn, typename DTypeOut>
__global__ void SinglePrefillWithKVCacheKernel(DTypeIn *__restrict__ q, DTypeIn *__restrict__ k,
                                               DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
                                               float *__restrict__ tmp,
                                               tensor_info_t<layout> qkv_info, float sm_scale,
                                               float rope_inv_scale, float rope_inv_theta) {
  size_t qo_len = qkv_info.qo_len;
  size_t kv_len = qkv_info.kv_len;
  size_t tx = threadIdx.x, ty = threadIdx.y;
  size_t bx = blockIdx.x, head_idx = blockIdx.y;
  size_t num_heads = gridDim.y;
  auto block = cg::this_thread_block();
  cg::thread_block_tile g = cg::tiled_partition<4>(block);

  constexpr size_t head_dim = num_frags_y * mma::frag_size;
  static_assert(num_frags_z * num_frags_y % num_warps == 0);

  extern __shared__ uint8_t smem[];

  uint32_t q_frag[num_frags_y][4];
  uint32_t kv_frag[num_frags_y][num_frags_z][4];
  float x_frag[num_frags_z][8];
  uint32_t att_frag[num_frags_z][4];
  float o_frag[num_frags_y][8]{0.f};
  float m_prev[2]{-5e4};
  float m[2]{-5e4};
  float d[2]{0.f};
  float rope_freq[num_frags_y][4];
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (size_t j = 0; j < 4; ++j) {
        rope_freq[fy][j] =
            rope_inv_scale *
            __powf(rope_inv_theta,
                   float(2 * ((fy * mma::frag_size + (j / 2) * 8 + (tx % 4) * 2 + (j % 2)) %
                              (head_dim / 2))) /
                       float(head_dim));
      }
    }
  }
  block.sync();

  // cooperative fetch q fragment from gmem to reg
  permuted_smem_t<4, DTypeIn> q_smem((DTypeIn *)smem);
  size_t q_idx = (bx * num_warps + ty) * mma::frag_size + tx / 4;
#pragma unroll
  for (size_t fyo = 0; fyo < num_frags_y / 2; ++fyo) {
    // load q fragment from gmem to smem
    size_t feat_idx = (fyo * 4 + tx % 4) * bank_capacity<DTypeIn>();
    size_t i = ty * mma::frag_size + tx / 4, j = tx % 4;
    if (q_idx < qo_len) {
      q_smem.load_bank(i, j, q + qkv_info.get_qo_elem_offset(q_idx, head_idx, feat_idx));
    }
    if (q_idx + 8 < qo_len) {
      q_smem.load_bank(i + 8, j, q + qkv_info.get_qo_elem_offset(q_idx + 8, head_idx, feat_idx));
    }
    // load q fragment from smem to reg
    i = ty * mma::frag_size + tx % mma::frag_size;
    j = tx / mma::frag_size;
    q_smem.ldmatrix_m8n8x4(q_frag[fyo * 2], i, j);
    q_smem.ldmatrix_m8n8x4(q_frag[fyo * 2 + 1], i, j + 2);
  }
  block.sync();
  if constexpr (rotary_mode == RotaryMode::kLlama) {
    // apply rotary embedding
#pragma unroll
    for (size_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
      apply_llama_rope<true, DTypeIn>((DTypeIn *)q_frag[fyi],
                                      (DTypeIn *)q_frag[fyi + num_frags_y / 2], rope_freq[fyi],
                                      rope_freq[fyi + num_frags_y / 2], q_idx + (kv_len - qo_len));
    }
  }
  block.sync();

  // multiply q by sm_scale
  vec_t<DTypeIn, 2> sm_scale2;
  sm_scale2.fill(sm_scale);
#pragma unroll
  for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
    for (size_t reg_id = 0; reg_id < 4; ++reg_id) {
      vec_t<DTypeIn, 2> tmp;
      tmp.load((DTypeIn *)&q_frag[fy][reg_id]);
      tmp.data = tmp.data * sm_scale2.data;
      tmp.store((DTypeIn *)&q_frag[fy][reg_id]);
    }
  }
  block.sync();

  permuted_smem_t<head_dim / bank_capacity<DTypeIn>(), DTypeIn> k_smem[num_stages_smem];
  permuted_smem_t<head_dim / bank_capacity<DTypeIn>(), DTypeIn> v_smem[num_stages_smem];
#pragma unroll
  for (size_t i = 0; i < num_stages_smem; ++i) {
    k_smem[i].base =
        (DTypeIn *)(smem + i * num_frags_z * mma::frag_size * head_dim * sizeof(DTypeIn));
    v_smem[i].base = (DTypeIn *)(smem + (num_stages_smem + i) * num_frags_z * mma::frag_size *
                                            head_dim * sizeof(DTypeIn));
  }

  size_t producer_kv_idx_base = 0, copy_stage_idx = 0;
#pragma unroll
  for (size_t iter = 0; iter < num_stages_smem - 1; ++iter) {
    produce_kv<num_frags_y, num_frags_z, num_warps>(k_smem + iter, k, qkv_info,
                                                    producer_kv_idx_base, kv_len, head_idx);
    produce_kv<num_frags_y, num_frags_z, num_warps>(v_smem + iter, v, qkv_info,
                                                    producer_kv_idx_base, kv_len, head_idx);

    cp_async::commit_group();
    producer_kv_idx_base += mma::frag_size * num_frags_z;
    copy_stage_idx += 1;
  }

  size_t consumer_kv_idx_base = 0, compute_stage_idx = 0;
  size_t effective_kv_len = min(kv_len, kv_len - qo_len + ((bx + 1) * num_warps) * mma::frag_size);
#pragma unroll 8
  for (size_t iter = 0; iter < (effective_kv_len + (mma::frag_size * num_frags_z - 1)) /
                                   (mma::frag_size * num_frags_z);
       ++iter) {
    produce_kv<num_frags_y, num_frags_z, num_warps>(k_smem + copy_stage_idx, k, qkv_info,
                                                    producer_kv_idx_base, kv_len, head_idx);
    produce_kv<num_frags_y, num_frags_z, num_warps>(v_smem + copy_stage_idx, v, qkv_info,
                                                    producer_kv_idx_base, kv_len, head_idx);
    cp_async::commit_group();
    cp_async::wait_group<num_stages_smem - 1>();
    block.sync();
    // init x_frag with -inf or 0 by applying mask
#pragma unroll
    for (size_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (size_t reg_id = 0; reg_id < 8; ++reg_id) {
        size_t q_idx = (bx * num_warps + ty) * mma::frag_size + 8 * ((reg_id % 4) / 2) + tx / 4,
               kv_idx = consumer_kv_idx_base + fz * mma::frag_size + 8 * (reg_id / 4) +
                        2 * (tx % 4) + reg_id % 2;
        bool predicate = (q_idx - qo_len < kv_idx - kv_len || q_idx >= qo_len || kv_idx >= kv_len);
        x_frag[fz][reg_id] = predicate ? -5e4 : 0.f;
      }
    }

    // load k tile from smem to reg
#pragma unroll
    for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (size_t fz = 0; fz < num_frags_z; ++fz) {
        size_t i = mma::frag_size * fz + 8 * (tx / 16) + tx % 8, j = fy * 2 + (tx % 16) / 8;
        k_smem[compute_stage_idx].ldmatrix_m8n8x4(kv_frag[fy][fz], i, j);
      }
    }
    if constexpr (rotary_mode == RotaryMode::kLlama) {
      // apply rotary embedding
#pragma unroll
      for (size_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
#pragma unroll
        for (size_t fz = 0; fz < num_frags_z; ++fz) {
          size_t kv_idx = consumer_kv_idx_base + fz * mma::frag_size + tx / 4;
          apply_llama_rope<false, DTypeIn>(
              (DTypeIn *)kv_frag[fyi][fz], (DTypeIn *)kv_frag[fyi + num_frags_y / 2][fz],
              rope_freq[fyi], rope_freq[fyi + num_frags_y / 2], kv_idx);
        }
      }
    }
    block.sync();

    // compute q*k^T
#pragma unroll
    for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (size_t fz = 0; fz < num_frags_z; ++fz) {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(x_frag[fz], q_frag[fy], kv_frag[fy][fz]);
      }
    }
    block.sync();

    // compute m,d states in online softmax
#pragma unroll
    for (size_t j = 0; j < 2; ++j) {
      m_prev[j] = m[j];
#pragma unroll
      for (size_t fz = 0; fz < num_frags_z; ++fz) {
        float m_local = max(max(x_frag[fz][j * 2 + 0], x_frag[fz][j * 2 + 1]),
                            max(x_frag[fz][j * 2 + 4], x_frag[fz][j * 2 + 5]));
        float d_local =
            __expf(x_frag[fz][j * 2 + 0] - m_local) + __expf(x_frag[fz][j * 2 + 1] - m_local) +
            __expf(x_frag[fz][j * 2 + 4] - m_local) + __expf(x_frag[fz][j * 2 + 5] - m_local);
#pragma unroll
        for (size_t offset = 2; offset > 0; offset /= 2) {
          merge_md(m_local, d_local, g.shfl_down(m_local, offset), g.shfl_down(d_local, offset));
        }
        m_local = g.shfl(m_local, 0);
        d_local = g.shfl(d_local, 0);
        merge_md(m[j], d[j], m_local, d_local);
      }
    }
    block.sync();

    // compute att_frag
#pragma unroll
    for (size_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (size_t reg_id = 0; reg_id < 4; ++reg_id) {
        vec_t<DTypeIn, 2> tmp;
#pragma unroll
        for (size_t j = 0; j < 2; ++j) {
          tmp[j] = __expf(x_frag[fz][reg_id * 2 + j] - m[reg_id % 2]);
        }
        tmp.store((DTypeIn *)&att_frag[fz][reg_id]);
      }
    }

    // load v tile from smem to reg
#pragma unroll
    for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (size_t fz = 0; fz < num_frags_z; ++fz) {
        size_t i = mma::frag_size * fz + tx % 16, j = fy * 2 + tx / 16;
        v_smem[compute_stage_idx].ldmatrix_m8n8x4_trans(kv_frag[fy][fz], i, j);
      }
    }

    // scale o_frag
#pragma unroll
    for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (size_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fy][reg_id] *= __expf(m_prev[(reg_id % 4) / 2] - m[(reg_id % 4) / 2]);
      }
    }
    block.sync();

    // compute sfm*v
#pragma unroll
    for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (size_t fz = 0; fz < num_frags_z; ++fz) {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(o_frag[fy], att_frag[fz],
                                                           kv_frag[fy][fz]);
      }
    }
    block.sync();
    copy_stage_idx = (copy_stage_idx + 1) % num_stages_smem;
    compute_stage_idx = (compute_stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += mma::frag_size * num_frags_z;
    consumer_kv_idx_base += mma::frag_size * num_frags_z;
  }

  // divide d
#pragma unroll
  for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
    for (size_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[fy][reg_id] *= __fdividef(1.f, d[(reg_id % 4) / 2]);
    }
  }

  // write back
  permuted_smem_t<head_dim / bank_capacity<DTypeOut>(), DTypeOut> o_smem((DTypeIn *)smem);
  if constexpr (std::is_same<DTypeOut, float>::value) {
    // TODO(Zihao)
  } else if constexpr (sizeof(DTypeOut) == 2) {
    uint32_t o_frag_f16[num_frags_y][4];
#pragma unroll
    for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (size_t reg_id = 0; reg_id < 4; ++reg_id) {
        vec_t<DTypeOut, 2> tmp;
        tmp.cast_load(&o_frag[fy][reg_id * 2]);
        tmp.store((DTypeOut *)&o_frag_f16[fy][reg_id]);
      }
    }
#pragma unroll
    for (size_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (size_t reg_id = 0; reg_id < 4; ++reg_id) {
        size_t i = ty * mma::frag_size + 8 * (reg_id % 2) + tx / 4, j = fy * 2 + (reg_id / 2);
        ((uint32_t *)o_smem.get_ptr(i, j))[tx % 4] = o_frag_f16[fy][reg_id];
      }
    }
#pragma unroll
    for (size_t fy = 0; fy < num_frags_y; ++fy) {
      size_t i = ty * mma::frag_size + tx % 16, j = fy * 2 + tx / 16;
      size_t o_idx = (bx * num_warps + ty) * mma::frag_size + tx % 16,
             feat_idx = (fy * 2 + tx / 16) * bank_capacity<DTypeOut>();
      if (o_idx < qo_len) {
        o_smem.store_bank(i, j, o + qkv_info.get_qo_elem_offset(o_idx, head_idx, feat_idx));
      }
    }
  } else {
    // NOTE(Zihao): Not implemented yet.
  }
}

template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCache(DTypeIn *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *tmp,
                                     size_t num_heads, size_t qo_len, size_t kv_len,
                                     size_t head_dim, QKVLayout layout = QKVLayout::kNHD,
                                     RotaryMode rotary_mode = RotaryMode::kNone,
                                     float rope_scale = 1.f, float rope_theta = 1e4,
                                     cudaStream_t stream = nullptr) {
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;

  SWITCH_HEAD_DIM(
      head_dim, HEAD_DIM,
      {SWITCH_ROTARY_MODE(
          rotary_mode, ROTARY_MODE, {SWITCH_LAYOUT(layout, LAYOUT, {
            constexpr size_t num_frags_y = HEAD_DIM / mma::frag_size;
            constexpr size_t num_frags_z = 2;
            constexpr size_t num_warps = 8UL;
            constexpr size_t num_stages_smem = 4;
            constexpr size_t num_rows_per_cta = num_warps * mma::frag_size;
            auto kernel =
                SinglePrefillWithKVCacheKernel<LAYOUT, ROTARY_MODE, num_frags_y, num_frags_z,
                                               num_stages_smem, num_warps, DTypeIn, DTypeOut>;
            dim3 nblks((qo_len + (num_rows_per_cta - 1)) / (num_rows_per_cta), num_heads);
            dim3 nthrs(32, num_warps);
            size_t smem_size =
                2 * num_stages_smem * num_frags_z * mma::frag_size * head_dim * sizeof(DTypeIn);
            tensor_info_t<LAYOUT> qkv_info(qo_len, kv_len, num_heads, HEAD_DIM);
            void *args[] = {(void *)&q,
                            (void *)&k,
                            (void *)&v,
                            (void *)&o,
                            (void *)&tmp,
                            (void *)&qkv_info,
                            (void *)&sm_scale,
                            (void *)&rope_inv_scale,
                            (void *)&rope_inv_theta};
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            FLASHINFER_CUDA_CALL(
                cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
          })})});
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_