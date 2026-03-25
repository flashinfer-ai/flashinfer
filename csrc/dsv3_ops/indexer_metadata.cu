#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#include <cassert>
#include <flashinfer/dsv3_ops/common.cuh>

namespace cg = cooperative_groups;

template <int tile_size>
__global__ void indexer_kernel_metadata(
    const int* __restrict__ seq_lens,  // [batch_size]
    const int batch_size,
    int4* __restrict__ sm_mapping,  // [gridDim.x * num_physical_sms]
    int num_physical_sms) {
  // each physical_sms group processes local_batch_size number of
  // batches, we assume that there is at least one sm per batch, so
  // local_batch_size <= num_physical_sms, also, the number of threads per block
  // is greater than or equal to num_physical_sms
  const int block_batch_sz = (batch_size + gridDim.x - 1) / gridDim.x;
  const int local_batch_start = block_batch_sz * blockIdx.x;
  const int local_batch_end = min(batch_size, local_batch_start + block_batch_sz);
  const int local_batch_size = max(0, local_batch_end - local_batch_start);

  int seq_len = 0;
  if (threadIdx.x + local_batch_start < local_batch_end) {
    seq_len = seq_lens[threadIdx.x + local_batch_start];
  }

  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<tile_size>(block);

  int pages = (seq_len + 63) / 64;
  int num_pages = cg::reduce(tile, pages, cg::plus<int>());
  int pages_per_sm = (num_pages + num_physical_sms - 1) / num_physical_sms;

  // allocate 1 sm per batch, real_pages_now is the number of pages left over.
  int real_pages_now = max(0, pages - pages_per_sm);
  int num_real_pages_now = cg::reduce(tile, real_pages_now, cg::plus<int>());

  int extra_sms = num_physical_sms - local_batch_size;
  int extra_pages_per_sm = extra_sms == 0 ? 0 : (num_real_pages_now + extra_sms - 1) / extra_sms;

  int sms_per_batch = threadIdx.x < local_batch_size
                          ? 1 + (extra_pages_per_sm == 0 ? 0 : real_pages_now / extra_pages_per_sm)
                          : 0;

  int sm_block_scan = cg::exclusive_scan(tile, sms_per_batch);
  extern __shared__ int sm_block_sum[];

  int* sm_batch_idx = sm_block_sum + num_physical_sms;
  int* shared_sms_per_batch = sm_block_sum + 2 * num_physical_sms;
  int* batch_pages = sm_block_sum + 3 * num_physical_sms;
  if (threadIdx.x < num_physical_sms) {
    sm_block_sum[threadIdx.x] = sm_block_scan;
    sm_batch_idx[threadIdx.x] = 0;
    shared_sms_per_batch[threadIdx.x] = sms_per_batch;
    batch_pages[threadIdx.x] = pages;
  }
  __syncthreads();
  for (int sm_id = threadIdx.x; sm_id < num_physical_sms; sm_id += tile_size) {
    int batch_id = -1;
    int local_offset = -1;
    for (int i = 0; i < local_batch_size; ++i) {
      bool within_bucket =
          i < local_batch_size - 1 ? sm_block_sum[i] <= sm_id && sm_id < sm_block_sum[i + 1] : true;
      if (within_bucket) {
        batch_id = i;
        local_offset = atomicAdd(sm_batch_idx + batch_id, 1);
        break;
      }
    }
    int sms_per_batch = shared_sms_per_batch[batch_id];
    int pages = batch_pages[batch_id];
    int pages_per_sm = (pages + sms_per_batch - 1) / sms_per_batch;
    int4 data;
    data.x = batch_id + local_batch_start;
    data.y = local_offset;
    data.z = local_offset * pages_per_sm;
    data.w = max(0, min(pages_per_sm, batch_pages[batch_id] - data.z));

    sm_mapping[blockIdx.x * num_physical_sms + sm_id] = data;
  }
}

void launch_indexer_kernel_metadata(int* seq_lens, int batch_size, int num_physical_sms,
                                    int* sm_mapping, cudaStream_t stream) {
  int num_blocks = (batch_size + num_physical_sms - 1) / num_physical_sms;

  int warps = (num_physical_sms + 31) / 32;
  int shared_size = sizeof(int) * 4 * num_physical_sms;
  if (warps <= 4) {
    indexer_kernel_metadata<128><<<num_blocks, 128, shared_size, stream>>>(
        seq_lens, batch_size, (int4*)sm_mapping, num_physical_sms);
  } else if (warps <= 8) {
    indexer_kernel_metadata<256><<<num_blocks, 256, shared_size, stream>>>(
        seq_lens, batch_size, (int4*)sm_mapping, num_physical_sms);
  } else if (warps <= 16) {
    indexer_kernel_metadata<512><<<num_blocks, 512, shared_size, stream>>>(
        seq_lens, batch_size, (int4*)sm_mapping, num_physical_sms);
  } else {
    assert(false && "too many physical sms");
  }
}
