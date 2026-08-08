

#include <cooperative_groups.h>

#include <cassert>
#include <flashinfer/dsv3_ops/common.cuh>

namespace cg = cooperative_groups;
constexpr int TopK = 2048;

template <int NClusters, bool PDL_ENABLED, bool PRE_HISTOGRAM>
__device__ __forceinline__ void fast_topk_cuda_v4(
    const float* __restrict__ logits,  // Input logits [max_num_pages * 64]
    int* __restrict__ output_indices,  // Output top-k indices [TopK]
    const int* __restrict__ pre_hist,  // Histogram of the first bin, if
                                       // PRE_HISTOGRAM is enabled, [256]
    const int seq_len, const int num_cached) {
  constexpr int RADIX = 256;

  __shared__ int shared_hist[3][RADIX];

  // we may assume logits is aligned to 64 * 4 bytes, each warp handles a page
  //
  // First scan entire logits, processing bits [0-8), filling in histogram
  // from histogram, fill in cached indices, and compute the next histogram
  //
  // Then for bits [8-16), [16-24), [24-32), we have the histogram and cached
  // indices, compute next histogram and cached indices.

  // cache 2 * num_cached uint16_t (indices), 2 * num_cached float (logits),
  // TopK uint16_t (final indices)
  extern __shared__ uint8_t shared_cache[];
  alignas(128) __shared__ int shared_final_idx_count;      // number of topk indices in s_topk_inds
  alignas(128) __shared__ int shared_num_cached_count[2];  // number of cached indices
  alignas(128) __shared__ int shared_threshold_bin;
  uint32_t* s_cached_logit_bits = (uint32_t*)shared_cache;
  int* s_cached_indices = (int*)(2 * num_cached + s_cached_logit_bits);
  int* s_topk_inds = (int*)(2 * num_cached + s_cached_indices);

  auto cluster = cg::this_cluster();
  const int block_id = cluster.block_rank();
  const bool radix_thread = threadIdx.x < RADIX;
  constexpr bool DEBUG = false;
  constexpr bool RUN_PHASE1 = true;
  constexpr bool ENABLE_CLUSTER_SAFETY = true;

  auto get_threshold_bin = [&](int hist_idx, int& k_remaining, bool sum_hist = true) {
    __syncthreads();
    // first reduce cum sum locally
    int cum_val = cum_sum(shared_hist[hist_idx]);
    if (NClusters > 1 && sum_hist) {
      if (radix_thread) {
        shared_hist[hist_idx][threadIdx.x] = cum_val;
      }
      cluster.sync();  // now first block in cluster has its local cum sum

      if (radix_thread) {
#pragma unroll
        for (int cl = 0; cl < NClusters - 1; cl++) {
          cum_val += cluster.map_shared_rank(&shared_hist[hist_idx][threadIdx.x],
                                             (cl + block_id + 1) % NClusters)[0];
        }
        shared_hist[2][threadIdx.x] = cum_val;
      }
    } else {
      if (radix_thread) {
        shared_hist[2][threadIdx.x] = cum_val;
      }
    }

    __syncthreads();

    int cum_val1 = threadIdx.x < RADIX - 1 ? shared_hist[2][threadIdx.x + 1] : 0;

    if (radix_thread && cum_val > k_remaining && cum_val1 <= k_remaining) {
      shared_threshold_bin = threadIdx.x;
      if (DEBUG) {
        printf(
            "block_id %d: threshold_bin %d. cum_sum_thres %d, cum_sum_thres+1 "
            "%d, topk_val %d\n",
            block_id, threadIdx.x, cum_val, cum_val1, shared_final_idx_count);
      }
    }

    __syncthreads();
    const int threshold_bin = shared_threshold_bin;
    if (threshold_bin < RADIX - 1) {
      k_remaining -= shared_hist[2][threshold_bin + 1];
    }
    return threshold_bin;
  };

  if (PDL_ENABLED) cudaGridDependencySynchronize();

  if (radix_thread) {
    if (PRE_HISTOGRAM) {
      shared_hist[0][threadIdx.x] = pre_hist[threadIdx.x];
    } else {
      shared_hist[0][threadIdx.x] = 0;
    }
    shared_hist[1][threadIdx.x] = 0;
  }
  if (threadIdx.x == 0) {
    shared_final_idx_count = 0;
    shared_num_cached_count[0] = 0;
  }

  if (!PRE_HISTOGRAM) {
    __syncthreads();
    for (int i = threadIdx.x + block_id * 1024; i < seq_len; i += 1024 * NClusters) {
      float res = logits[i];
      auto bin = convert_to_uint32_v2(res) >> 24 & 0xff;
      atomicAdd(shared_hist[0] + bin, 1);
    }
  }

  int top_k_remaining = TopK;
  if (RUN_PHASE1) {
    // get_threshold_bin synchronizes
    // if we use the PRE_HISTOGRAM then we don't need to sum the first histogram
    // across clusters
    const int threshold_bin = get_threshold_bin(0, top_k_remaining, !PRE_HISTOGRAM);

    auto compute_phase1 = [&](float logit, int i) {
      uint32_t bits = convert_to_uint32_v2(logit);
      int bin = (bits >> 24);

      if (bin > threshold_bin) {
        int topk_offset = atomicAdd(&shared_final_idx_count, 1);
        if (topk_offset < TopK) {
          s_topk_inds[topk_offset] = i;
        }
      }

      if (bin == threshold_bin) {
        int cached_offset = atomicAdd(&shared_num_cached_count[0], 1);
        if (cached_offset < num_cached) {
          s_cached_indices[cached_offset] = i;
          s_cached_logit_bits[cached_offset] = bits;
          atomicAdd(shared_hist[1] + (bits >> 16 & 0xff), 1);
        }
      }
    };
    if (ENABLE_CLUSTER_SAFETY && NClusters > 1) {
      cluster.barrier_arrive();  // arrive at this point to signal that we are
                                 // done with shared_hist[0], which we need in
                                 // distributed shared memory to communicate
                                 // histograms within the cluster, otherwise there
                                 // is a race condition that produces slightly wrong
                                 // results, for slightly better performance
    }

    run_vectorized<NClusters, 1024, 4, 2>(logits, seq_len, block_id, compute_phase1);
  }
#pragma unroll
  for (int t = 1; t <= 3; t++) {
    const int phase = t % 2;
    // we are now using histogram buffers at phase, and cached_offsets at phase
    // ^ 1 clear the next stage to prepare

    if (ENABLE_CLUSTER_SAFETY && NClusters > 1) {
      cluster.barrier_wait();  // wait because we need to clear shared_hist for this
                               // next histogram; there is a dependency between
                               // get_threshold_bin and phase ^ 1 of the histogram
    }
    if (radix_thread) {
      shared_hist[phase ^ 1][threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
      shared_num_cached_count[phase] = 0;
    }
    // get_threshold_bin synchronizes the block
    const int threshold_bin = get_threshold_bin(phase, top_k_remaining);
    if (ENABLE_CLUSTER_SAFETY && NClusters > 1 && t < 3) {
      cluster.barrier_arrive();  // same reasoning as above, for the last
                                 // iteration there is no dependency because no
                                 // more calls to get_threshold_bin
    }
    int buf_len = min(num_cached, shared_num_cached_count[phase ^ 1]);
    if (DEBUG && threadIdx.x == 0) {
      printf("block_id %d, num_cached %d, \n", block_id, buf_len);
    }
    // using cached indices, it's a local slice so don't partition between
    // blocks
    for (int i = threadIdx.x; i < buf_len; i += 1024) {
      uint32_t bits = s_cached_logit_bits[i + (phase ^ 1) * num_cached];
      int cached_idx = s_cached_indices[i + (phase ^ 1) * num_cached];
      int bin = (bits >> (24 - t * 8)) & 0xff;

      if (bin > threshold_bin) {
        int topk_offset = atomicAdd(&shared_final_idx_count, 1);
        if (topk_offset < TopK) {
          s_topk_inds[topk_offset] = cached_idx;
        } else {
          break;
        }
      }
      if (bin == threshold_bin && t < 3) {
        int cached_offset = atomicAdd(&shared_num_cached_count[phase], 1);
        if (cached_offset < num_cached) {
          s_cached_indices[cached_offset + phase * num_cached] = cached_idx;
          s_cached_logit_bits[cached_offset + phase * num_cached] = bits;
          atomicAdd(shared_hist[phase ^ 1] + (bits >> (24 - (t + 1) * 8) & 0xff), 1);
        }
      }
    }

    // it could be that at the last stage, we have say S topk indices, T indices
    // above the threshold_bin and S + T < TopK. Then the rest of the topk items
    // must come from threshold_bin
    if (t == 3) {
      if (top_k_remaining > 0) {
        for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
          int bin = s_cached_logit_bits[i] & 0xff;
          int cached_idx = s_cached_indices[i];
          if (bin == threshold_bin) {
            int topk_offset = atomicAdd(&shared_final_idx_count, 1);
            if (topk_offset < TopK) {
              s_topk_inds[topk_offset] = cached_idx;
            } else {
              break;
            }
          }
        }
      }
    }
  }
  __syncthreads();
  // now shared_final_idx_count contains the local topK in each
  // block in a cluster, and
  // s_topk_inds[0:shared_final_idx_count] contains the local
  // slice.
  if (NClusters > 1) {
    int topk_start = 0;
    int topk_num = shared_final_idx_count;

    cluster.sync();  // sync to get the shared_final_idx_count across CTAs in
                     // current cluster

    if (block_id > 0) {
      if (threadIdx.x == 0) {
        topk_start += atomicAdd(cluster.map_shared_rank(&shared_final_idx_count, 0), topk_num);
        shared_final_idx_count = topk_start;
      }
      __syncthreads();
      topk_start = shared_final_idx_count;
    }

    if (DEBUG && threadIdx.x == 0) {
      printf("block rank %d, topk start %d, topk num %d\n", block_id, topk_start, topk_num);
    }

    cluster.sync();  // sync so CTAs don't exit when other CTAs depend on
                     // shared_final_idx_count
    for (int i = threadIdx.x; i < min(TopK, topk_num); i += 1024) {
      if (i + topk_start < TopK) {
        output_indices[i + topk_start] = s_topk_inds[i];
      }
    }

  } else {
    for (int i = threadIdx.x; i < TopK; i += 1024) {
      output_indices[i] = s_topk_inds[i];
    }
  }
}

template <int NClusters, bool PDL_ENABLED, bool PRE_HISTOGRAM>
__global__ __launch_bounds__(1024) void __cluster_dims__(NClusters, 1, 1)
    fast_topk_clusters_kernel(const float* __restrict__ logits,  // [batchsize, max_num_pages * 64]
                              int* __restrict__ output_indices, int* __restrict__ seq_lens,
                              int* __restrict__ pre_hist, int logit_stride, int indices_stride,
                              int num_cached) {
  int cluster_id = blockIdx.x / NClusters;
  int logit_offset = cluster_id * logit_stride;
  int ind_offset = cluster_id * indices_stride;
  int seq_len = seq_lens[cluster_id];
  if (seq_len <= TopK) {
    for (int i = threadIdx.x + (blockIdx.x % NClusters) * 1024; i < TopK; i += 1024 * NClusters) {
      if (i < seq_len) {
        output_indices[ind_offset + i] = i;
      } else {
        output_indices[ind_offset + i] = -1;
      }
    }

  } else {
    fast_topk_cuda_v4<NClusters, PDL_ENABLED, PRE_HISTOGRAM>(
        logits + logit_offset, output_indices + ind_offset, pre_hist + cluster_id * 256, seq_len,
        num_cached);
  }
}

constexpr int MAX_SMEM_CARVEOUT = 227 * 1000;

#define DISPATCH_TOPK(CLUSTERS, PDL)                                                               \
  if (num_clusters == CLUSTERS && pdl_enabled == PDL) {                                            \
    if (pre_hist == nullptr) {                                                                     \
      setup_kernel_smem_once<fast_topk_clusters_kernel<CLUSTERS, PDL, false>,                      \
                             MAX_SMEM_CARVEOUT>();                                                 \
      kernel = (void*)&fast_topk_clusters_kernel<CLUSTERS, PDL, false>;                            \
    } else {                                                                                       \
      setup_kernel_smem_once<fast_topk_clusters_kernel<CLUSTERS, PDL, true>, MAX_SMEM_CARVEOUT>(); \
      kernel = (void*)&fast_topk_clusters_kernel<CLUSTERS, PDL, true>;                             \
    }                                                                                              \
  }

void launch_fast_topk_clusters(const float* logits, int* indices, int* seq_lens, int* pre_hist,
                               int batch_size, int logit_stride, int indices_stride, int num_cached,
                               int num_clusters, bool pdl_enabled, cudaStream_t stream) {
  int extern_shared_mem =
      (num_cached * 2 * sizeof(float) + num_cached * 2 * sizeof(int) +
       TopK * sizeof(int));  // 2 * num_cached float, 2 * num_cached int, topk int

  void* args[7] = {&logits,       &indices,        &seq_lens,  &pre_hist,
                   &logit_stride, &indices_stride, &num_cached};

  void* kernel = nullptr;
  // DISPATCH_TOPK(1, false);
  DISPATCH_TOPK(2, false);
  DISPATCH_TOPK(4, false);
  DISPATCH_TOPK(8, false);
  DISPATCH_TOPK(1, true);
  DISPATCH_TOPK(2, true);
  DISPATCH_TOPK(4, true);
  DISPATCH_TOPK(8, true);

  if (kernel == nullptr) {
    num_clusters = 1;
    pdl_enabled = false;
    DISPATCH_TOPK(1, false);
  }

  cudaLaunchConfig_t config;
  config.numAttrs = 0;
  cudaLaunchAttribute attribute[1];
  if (pdl_enabled) {
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs += 1;
  }
  config.blockDim = 1024;
  config.dynamicSmemBytes = extern_shared_mem;
  config.gridDim = batch_size * num_clusters;
  config.stream = stream;
  config.attrs = attribute;

  cudaLaunchKernelExC(&config, kernel, args);
}
