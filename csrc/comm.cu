// adapted from https://github.com/efeslab/Nanoflow/blob/d6b381e58110a8b5d08cfabd4a55c0d5d0ebef57/pipeline/src/comm.cu
#include <comm.h>
#include "rms_norm.cuh"

extern "C" __global__ void __launch_bounds__(1024)
    allgather(mscclpp::SmChannelDeviceHandle* sm_channels, mscclpp::DeviceSyncer* syncers,
              const uint64_t n_parallel_sm_blocks, const uint64_t local_offset,
              const uint64_t* offsets, const uint64_t nelem_per_channel) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int peer = bid / n_parallel_sm_blocks; // assert len(sm_channels) % n_parallel_sm_blocks == 0
    const int peer_block_idx = bid % n_parallel_sm_blocks;

    if (peer_block_idx == 0 && tid == 0) {
        sm_channels[peer].signal();
        sm_channels[peer].wait();
    }

    syncers[peer].sync(n_parallel_sm_blocks);

    sm_channels[peer].put(local_offset * sizeof(int), nelem_per_channel * sizeof(int),
                          tid + peer_block_idx * blockDim.x, n_parallel_sm_blocks * blockDim.x);
    // sm_channels[peer].get(offsets[peer] * sizeof(int), nelem_per_channel * sizeof(int),
    //                       tid + peer_block_idx * blockDim.x, n_parallel_sm_blocks * blockDim.x);
    
    syncers[peer].sync(n_parallel_sm_blocks);

    if (peer_block_idx == 0 && tid == 0) {
        sm_channels[peer].signal();
        sm_channels[peer].wait();
    }
}

__forceinline__ __device__ void copy(cutlass::half_t* input, cutlass::half_t* output, const uint64_t nelem,
                                     const uint32_t threadId, const uint32_t numThreads) {
    const uintptr_t input_val = reinterpret_cast<uintptr_t>(input);
    const uintptr_t output_val = reinterpret_cast<uintptr_t>(output);
    int4* input_ptr = reinterpret_cast<int4*>((input_val + sizeof(int4) - 1) / sizeof(int4) * sizeof(int4));
    int4* output_ptr = reinterpret_cast<int4*>((output_val + sizeof(int4) - 1) / sizeof(int4) * sizeof(int4));

    // assert input % sizeof(int4) == output % sizeof(int4)
    const uint64_t nFirstElem = (reinterpret_cast<uintptr_t>(input_ptr) - input_val) / sizeof(cutlass::half_t);
    if (threadId < nFirstElem && threadId < nelem) {
        output[threadId] = input[threadId];
    }

    const uint64_t nelem4 = (nelem - nFirstElem) / (sizeof(int4) / sizeof(cutlass::half_t));
    for (uint64_t i = threadId; i < nelem4; i += numThreads) {
        output_ptr[i] = input_ptr[i];
    }

    const uint64_t nLastElem = (nelem - nFirstElem) % (sizeof(int4) / sizeof(cutlass::half_t));
    if (threadId < nLastElem) {
        const uint64_t offset = nelem - nLastElem + threadId;
        output[offset] = input[offset];
    }
}

extern "C" __forceinline__ __device__ void
    allgatherKernel(mscclpp::SmChannelDeviceHandle* sm_channels, 
                    mscclpp::DeviceSyncer* syncers,
                    const int nchannels, const uint64_t local_offset, const uint64_t nelem_per_shard,
                    cutlass::half_t* input, cutlass::half_t* output) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int n_parallel_sm_blocks = gridDim.x / nchannels;
    const int peer = bid / n_parallel_sm_blocks;
    const int peer_block_idx = bid % n_parallel_sm_blocks;
    
    if (peer < nchannels) {
        sm_channels[peer].put(local_offset * sizeof(cutlass::half_t), nelem_per_shard * sizeof(cutlass::half_t),
                              tid + peer_block_idx * blockDim.x, n_parallel_sm_blocks * blockDim.x);

        syncers[peer].sync(n_parallel_sm_blocks);
    }

    if (input != output) {
        copy(&input[local_offset], &output[local_offset], nelem_per_shard, tid + bid * blockDim.x, blockDim.x * gridDim.x);
    }
}

extern "C" __global__ void __launch_bounds__(1024)
    allgatherKernelWithoutSync(mscclpp::SmChannelDeviceHandle* sm_channels, 
							   mscclpp::DeviceSyncer* syncers,
							   const int nchannels, const uint64_t local_offset, const uint64_t nelem_per_shard,
							   cutlass::half_t* input, cutlass::half_t* output) {
    allgatherKernel(sm_channels, syncers, nchannels, local_offset, nelem_per_shard, input, output);
}

extern "C" __global__ void __launch_bounds__(1024)
    allgatherKernelEntryPoint(mscclpp::SmChannelDeviceHandle* sm_channels, 
                              mscclpp::DeviceSyncer* syncers,
                              const int nchannels, const uint64_t local_offset, const uint64_t nelem_per_shard,
                              cutlass::half_t* input, cutlass::half_t* output) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n_parallel_sm_blocks = gridDim.x / nchannels;
    const int peer = bid / n_parallel_sm_blocks;
    const int peer_block_idx = bid % n_parallel_sm_blocks;
    if (peer < nchannels) {
        if (peer_block_idx == 0 && tid == 0) {
            sm_channels[peer].signal();
            sm_channels[peer].wait();
        }
        syncers[peer].sync(n_parallel_sm_blocks);
    }

    allgatherKernel(sm_channels, syncers, nchannels, local_offset, nelem_per_shard, input, output);

    if (peer < nchannels && peer_block_idx == 0 && tid == 0) {
        sm_channels[peer].signal();
        sm_channels[peer].wait();
    }
}

extern "C" __global__ void __launch_bounds__(1024)
    columnwiseAllgatherKernelEntryPoint(mscclpp::SmChannelDeviceHandle* sm_channels,
                                        mscclpp::DeviceSyncer* syncers,
                                        const bool sync, const int nchannels,
                                        const uint64_t input_ncols, const uint64_t output_ncols,
                                        const uint64_t output_row_offset, const uint64_t nrows,
                                        cutlass::half_t* input, cutlass::half_t* output) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n_parallel_sm_blocks = gridDim.x / nchannels;
    const int peer = bid / n_parallel_sm_blocks;
    const int peer_block_idx = bid % n_parallel_sm_blocks;
    if (sync && peer < nchannels) {
        if (peer_block_idx == 0 && tid == 0) {
            sm_channels[peer].signal();
            sm_channels[peer].wait();
        }
        syncers[peer].sync(n_parallel_sm_blocks);
    }

    // assert input % sizeof(int4) == sm_channels[peer].data % sizeof(int4) == 0
    int4* input_ptr = reinterpret_cast<int4*>(input);
    const uint32_t nelem = input_ncols * nrows;
    constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
    const uint64_t nelem4 = nelem / n_half_per_int4;

    if (peer < nchannels) {
        for (uint64_t i = tid + blockDim.x * peer_block_idx; i < nelem4;
             i += blockDim.x * n_parallel_sm_blocks) {
            const uint64_t dest_col = (i * n_half_per_int4) % input_ncols + output_row_offset;
            const uint64_t dest_row = (i * n_half_per_int4) / input_ncols;
            const uint64_t dest_off = dest_row * output_ncols + dest_col;
            sm_channels[peer].write<int4>(dest_off / n_half_per_int4, input_ptr[i]);
        }
        syncers[peer].sync(n_parallel_sm_blocks);
    }

    // assert input != output: columwise allgather cannot be inplace
    // assert output % sizeof(int4) == 0
    int4* output_ptr = reinterpret_cast<int4*>(output);
    for (uint64_t i = tid + bid * blockDim.x; i < nelem4; i += blockDim.x * gridDim.x) {
        const uint64_t dest_col = (i * n_half_per_int4) % input_ncols + output_row_offset;
        const uint64_t dest_row = (i * n_half_per_int4) / input_ncols;
        const uint64_t dest_off = dest_row * output_ncols + dest_col;
        output_ptr[dest_off / n_half_per_int4] = input_ptr[i];
    }

    if (sync && peer < nchannels && peer_block_idx == 0 && tid == 0) {
        sm_channels[peer].signal();
        sm_channels[peer].wait();
    }
}

extern "C" __forceinline__ __device__ void
    reduceScatterKernel(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
                        mscclpp::DeviceSyncer* syncer,
                        const int rank, const int nranks, const uint64_t nelem_per_shard,
                        cutlass::half_t* input, cutlass::half_t* output) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;
    const uint64_t local_offset = rank * nelem_per_shard;

    constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
    const uint64_t local_offset4 = (local_offset + n_half_per_int4 - 1) / n_half_per_int4;
    const uint64_t nFirstElem = local_offset4 * n_half_per_int4 - local_offset;
    if (threadId < nFirstElem && threadId < nelem_per_shard) {
        const uint64_t offset = local_offset + threadId;
        cutlass::half_t tmp = input[offset];
        for (int i = 0; i < nranks - 1; ++i) {
            tmp += sm_input_buff_channels[(rank + i) % (nranks - 1)].read<cutlass::half_t>(offset);
        }
        output[offset] = tmp;
    }

    int4* input4 = &reinterpret_cast<int4*>(input)[local_offset4];
    int4* output4 = &reinterpret_cast<int4*>(output)[local_offset4];
    const uint64_t nelem4 = (nelem_per_shard - nFirstElem) / n_half_per_int4;
    for (uint64_t offset = threadId; offset < nelem4; offset += gridDim.x * blockDim.x) {
        int4 tmp = input4[offset];
        for (int i = 0; i < nranks - 1; ++i) {
            int4 val = sm_input_buff_channels[(rank + i) % (nranks - 1)].read<int4>(local_offset4 + offset);
            *reinterpret_cast<__half2*>(&tmp.x) += *reinterpret_cast<__half2*>(&val.x);
            *reinterpret_cast<__half2*>(&tmp.y) += *reinterpret_cast<__half2*>(&val.y);
            *reinterpret_cast<__half2*>(&tmp.z) += *reinterpret_cast<__half2*>(&val.z);
            *reinterpret_cast<__half2*>(&tmp.w) += *reinterpret_cast<__half2*>(&val.w);
        }
        output4[offset] = tmp;
    }

    const uint64_t nLastElem = (nelem_per_shard - nFirstElem) % n_half_per_int4;
    if (threadId < nLastElem) {
        const uint64_t offset = local_offset + nelem_per_shard - nLastElem + threadId;
        cutlass::half_t tmp = input[offset];
        for (int i = 0; i < nranks - 1; ++i) {
            tmp += sm_input_buff_channels[(rank + i) % (nranks - 1)].read<cutlass::half_t>(offset);
        }
        output[offset] = tmp;
    }

// for (uint64_t offset = threadId; offset < nelem4; offset += gridDim.x * blockDim.x) {
//         half* tmp = reinterpret_cast<half*>(input4+offset);
//         float accum[] = {tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]};
//         for (int i = 0; i < nranks - 1; ++i) {
//             int4 val = sm_input_buff_channels[(rank + i) % (nranks - 1)].read<int4>(local_offset4 + offset);
//             half* half_ptr = reinterpret_cast<half*>(&val);
//             accum[0] +=  __half2float(half_ptr[0]);
//             accum[1] +=  __half2float(half_ptr[1]);
//             accum[2] +=  __half2float(half_ptr[2]);
//             accum[3] +=  __half2float(half_ptr[3]);
//             accum[4] +=  __half2float(half_ptr[4]);
//             accum[5] +=  __half2float(half_ptr[5]);
//             accum[6] +=  __half2float(half_ptr[6]);
//             accum[7] +=  __half2float(half_ptr[7]);
//         }
//         half* half_output_ptr = reinterpret_cast<half*> (output4 + offset);
//         half_output_ptr[0] = __float2half(accum[0]);
//         half_output_ptr[1] = __float2half(accum[1]);
//         half_output_ptr[2] = __float2half(accum[2]);
//         half_output_ptr[3] = __float2half(accum[3]);
//         half_output_ptr[4] = __float2half(accum[4]);
//         half_output_ptr[5] = __float2half(accum[5]);
//         half_output_ptr[6] = __float2half(accum[6]);
//         half_output_ptr[7] = __float2half(accum[7]);
//     }

//     const uint64_t nLastElem = (nelem_per_shard - nFirstElem) % n_half_per_int4;
//     if (threadId < nLastElem) {
//         const uint64_t offset = local_offset + nelem_per_shard - nLastElem + threadId;
//         float tmp = __half2float(*reinterpret_cast<half*> (&input[offset]));
//         for (int i = 0; i < nranks - 1; ++i) {
//             tmp += sm_input_buff_channels[(rank + i) % (nranks - 1)].read<cutlass::half_t>(offset);
//         }
//         output[offset] = tmp;
//     }
    syncer->sync(gridDim.x);
}

extern "C" __global__ void __launch_bounds__(1024)
    reduceScatterKernelWithoutSync(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
                        		   mscclpp::DeviceSyncer* syncer,
                        		   const int rank, const int nranks, const uint64_t nelem_per_shard,
                        		   cutlass::half_t* input, cutlass::half_t* output) {
    reduceScatterKernel(sm_input_buff_channels, syncer, rank, nranks, nelem_per_shard, input, output);
}

extern "C" __global__ void __launch_bounds__(1024)
    reduceScatterKernelEntryPoint(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
                                  mscclpp::DeviceSyncer* syncer,
                                  const int rank, const int nranks, const uint64_t nelem_per_shard,
                                  cutlass::half_t* input, cutlass::half_t* output) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;
    if (threadId < nranks - 1) {
        sm_input_buff_channels[threadId].signal();
        sm_input_buff_channels[threadId].wait();
    }
    syncer->sync(gridDim.x);

    reduceScatterKernel(sm_input_buff_channels, syncer, rank, nranks, nelem_per_shard, input, output);
    
    if (threadId < nranks - 1) {
        sm_input_buff_channels[tid].signal();
        sm_input_buff_channels[tid].wait();
    }
}

extern "C" __global__ void __launch_bounds__(1024)
    allreduceKernelWithoutSync(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
                    		   mscclpp::SmChannelDeviceHandle* sm_output_buff_channels,
                    		   mscclpp::DeviceSyncer* syncers,
                    		   mscclpp::DeviceSyncer* global_syncer,
                    		   const int nchannels, const uint64_t local_offset,
                    		   const int rank, const int nranks, const uint64_t nelem_per_shard,
                    		   cutlass::half_t* input, cutlass::half_t* output) {
    reduceScatterKernel(sm_input_buff_channels, global_syncer, rank, nranks, nelem_per_shard, input, output);
    allgatherKernel(sm_output_buff_channels, syncers, nchannels, local_offset, nelem_per_shard, output, output);
}

extern "C" __global__ void __launch_bounds__(1024)
    allreduceKernelEntryPoint(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
                              mscclpp::SmChannelDeviceHandle* sm_output_buff_channels,
                              mscclpp::DeviceSyncer* syncers,
                              mscclpp::DeviceSyncer* global_syncer,
                              const int nchannels, const uint64_t local_offset,
                              const int rank, const int nranks, const uint64_t nelem_per_shard,
                              cutlass::half_t* input, cutlass::half_t* output) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;
    if (threadId < nranks - 1) {
        sm_input_buff_channels[threadId].signal();
        sm_input_buff_channels[threadId].wait();
    }
    global_syncer->sync(gridDim.x);

    reduceScatterKernel(sm_input_buff_channels, global_syncer, rank, nranks, nelem_per_shard, input, output);
    allgatherKernel(sm_output_buff_channels, syncers, nchannels, local_offset, nelem_per_shard, output, output);

    const int n_parallel_sm_blocks = gridDim.x / nchannels;
    const int peer = bid / n_parallel_sm_blocks;
    const int peer_block_idx = bid % n_parallel_sm_blocks;
    if (peer < nchannels && peer_block_idx == 0 && tid == 0) {
        sm_output_buff_channels[peer].signal();
        sm_output_buff_channels[peer].wait();
    }
}

extern "C" __global__ void __launch_bounds__(1024) 
    allreduceKernelWithLNEntryPoint(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
                                    mscclpp::SmChannelDeviceHandle* sm_output_buff_channels,
                                    mscclpp::DeviceSyncer* syncers,
                                    mscclpp::DeviceSyncer* global_syncer,
                                    const int nchannels, const uint64_t local_offset,
                                    const int rank, const int nranks, const uint64_t nelem_per_shard,
                                    cutlass::half_t* input, cutlass::half_t* output, cutlass::half_t* output_before_ln,
                                    half* ln_weight, int rows, int columns, float epsilon) 
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;
    if (threadId < nranks - 1) {
        sm_input_buff_channels[threadId].signal();
        sm_input_buff_channels[threadId].wait();
    }
    global_syncer->sync(gridDim.x);

    reduceScatterKernel(sm_input_buff_channels, global_syncer, rank, nranks, nelem_per_shard, input, output_before_ln);
    __syncthreads();
    auto local_output_shards_before_ln = output_before_ln + local_offset;
    auto local_output_shards_of_ln = output + local_offset;
    rmsnorm_device<half, half2>((float4 *)local_output_shards_of_ln, (float4 *)local_output_shards_before_ln,
                    (float4 *)ln_weight, rows, columns, epsilon);
    __syncthreads();
    allgatherKernel(sm_output_buff_channels, syncers, nchannels, local_offset, nelem_per_shard, output, output);

    const int n_parallel_sm_blocks = gridDim.x / nchannels;
    const int peer = bid / n_parallel_sm_blocks;
    const int peer_block_idx = bid % n_parallel_sm_blocks;
    if (peer < nchannels && peer_block_idx == 0 && tid == 0) {
        sm_output_buff_channels[peer].signal();
        sm_output_buff_channels[peer].wait();
    }
}

extern "C" __global__ void __launch_bounds__(1024)
    asyncAllgatherKernelStartEntryPoint(mscclpp::SimpleProxyChannelDeviceHandle* proxy_channels,
                                        mscclpp::SmChannelDeviceHandle* sm_sync_channels,
                                        const int nchannels,
                                        const uint64_t local_offset,
                                        const uint64_t nelem_per_shard) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;

    if (threadId < nchannels) {
        sm_sync_channels[threadId].signal();
        sm_sync_channels[threadId].wait();
        proxy_channels[threadId].putWithSignal(local_offset * sizeof(cutlass::half_t), nelem_per_shard * sizeof(cutlass::half_t));
    }
}

extern "C" __global__ void __launch_bounds__(1024)
	asyncAllgatherKernelFinishEntryPoint(mscclpp::SimpleProxyChannelDeviceHandle* proxy_channels,
										 const int nchannels) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;

    if (threadId < nchannels) proxy_channels[threadId].wait();
}

__forceinline__ __device__ uint64_t get_scatch_start(const int src_rank, const uint64_t nelem_per_shard, const int nFirstElem) {
    constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
    const uint64_t scratch_reserved_start = src_rank * (nelem_per_shard + n_half_per_int4 * 2);
    const uint64_t scratch_true_start = (scratch_reserved_start + 2 * n_half_per_int4 - 1) / n_half_per_int4 * n_half_per_int4 - nFirstElem;
    return scratch_true_start;
}

extern "C" __global__ void __launch_bounds__(1024)
	asyncReduceScatterKernelStartEntryPoint(mscclpp::SimpleProxyChannelDeviceHandle* proxy_channels,
                                            mscclpp::SmChannelDeviceHandle* sm_sync_channels,
                                            const int rank, const int nranks, const uint64_t nelem_per_shard) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;

    if (threadId < nranks - 1) {
        sm_sync_channels[threadId].signal();
        sm_sync_channels[threadId].wait();

        constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
        const int remoteRank = threadId >= rank ? threadId + 1 : threadId;
        const uint64_t remote_offset = remoteRank * nelem_per_shard;
        const uint64_t remote_offset4 = (remote_offset + n_half_per_int4 - 1) / n_half_per_int4;
        const int nFirstElem = remote_offset4 * n_half_per_int4 - remote_offset;
        proxy_channels[threadId].putWithSignal(
            get_scatch_start(rank, nelem_per_shard, nFirstElem) * sizeof(cutlass::half_t),
            remoteRank * nelem_per_shard * sizeof(cutlass::half_t),
            nelem_per_shard * sizeof(cutlass::half_t));
    }
}

extern "C" __global__ void __launch_bounds__(1024)
	asyncReduceScatterKernelFinishEntryPoint(mscclpp::SimpleProxyChannelDeviceHandle* proxy_channels,
                                             mscclpp::DeviceSyncer* syncer,
											 const int rank, const int nranks, const uint64_t nelem_per_shard,
											 cutlass::half_t* scratch, cutlass::half_t* input, cutlass::half_t* output) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;
    const uint64_t local_offset = rank * nelem_per_shard;

    if (threadId < nranks - 1) proxy_channels[threadId].wait();
    syncer->sync(gridDim.x);

    constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
    const uint64_t local_offset4 = (local_offset + n_half_per_int4 - 1) / n_half_per_int4;
    const uint64_t nFirstElem = local_offset4 * n_half_per_int4 - local_offset;
    if (threadId < nFirstElem && threadId < nelem_per_shard) {
        cutlass::half_t tmp = input[local_offset + threadId];
        for (int r = 0; r < nranks; ++r) {
            if (r == rank) continue;
            tmp += scratch[get_scatch_start(r, nelem_per_shard, nFirstElem) + threadId];
        }
        output[local_offset + threadId] = tmp;
    }

    int4* input4 = &reinterpret_cast<int4*>(input)[local_offset4];
    int4* output4 = &reinterpret_cast<int4*>(output)[local_offset4];
    const uint64_t nelem4 = (nelem_per_shard - nFirstElem) / n_half_per_int4;
    for (uint64_t offset = threadId; offset < nelem4; offset += gridDim.x * blockDim.x) {
        int4 tmp = input4[offset];
        for (int r = 0; r < nranks; ++r) {
            if (r == rank) continue;
            int4 val = reinterpret_cast<int4*>(&scratch[get_scatch_start(r, nelem_per_shard, nFirstElem) + nFirstElem])[offset];
            *reinterpret_cast<__half2*>(&tmp.x) += *reinterpret_cast<__half2*>(&val.x);
            *reinterpret_cast<__half2*>(&tmp.y) += *reinterpret_cast<__half2*>(&val.y);
            *reinterpret_cast<__half2*>(&tmp.z) += *reinterpret_cast<__half2*>(&val.z);
            *reinterpret_cast<__half2*>(&tmp.w) += *reinterpret_cast<__half2*>(&val.w);
        }
        output4[offset] = tmp;
    }

    const uint64_t nLastElem = (nelem_per_shard - nFirstElem) % n_half_per_int4;
    if (threadId < nLastElem) {
        const uint64_t offset = nelem_per_shard - nLastElem + threadId;
        cutlass::half_t tmp = input[local_offset + offset];
        for (int r = 0; r < nranks; ++r) {
            if (r == rank) continue;
            tmp += scratch[get_scatch_start(r, nelem_per_shard, nFirstElem) + offset];
        }
        output[local_offset + offset] = tmp;
    }

    syncer->sync(gridDim.x);
}

extern "C" __global__ void __launch_bounds__(1024)
	syncDevices(mscclpp::SmChannelDeviceHandle* sm_sync_channels, const int nchannels) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;

    if (threadId < nchannels) {
        sm_sync_channels[threadId].signal();
        sm_sync_channels[threadId].wait();
    }
}

extern "C" __global__ void __launch_bounds__(1024)
	asyncReduceKernel(const int rank, const int nranks, const uint64_t nelem_per_shard,
					  cutlass::half_t** scratches, cutlass::half_t* input, cutlass::half_t* output) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threadId = tid + bid * blockDim.x;
    const uint64_t local_offset = rank * nelem_per_shard;

    constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
    const uint64_t local_offset4 = (local_offset + n_half_per_int4 - 1) / n_half_per_int4;
    const uint64_t nFirstElem = local_offset4 * n_half_per_int4 - local_offset;
    if (threadId < nFirstElem && threadId < nelem_per_shard) {
        cutlass::half_t tmp = input[local_offset + threadId];
        for (int i = 0; i < nranks - 1; ++i) {
            tmp += scratches[i][n_half_per_int4 - nFirstElem + threadId];
        }
        output[local_offset + threadId] = tmp;
    }

    int4* input4 = &reinterpret_cast<int4*>(input)[local_offset4];
    int4* output4 = &reinterpret_cast<int4*>(output)[local_offset4];
    const uint64_t nelem4 = (nelem_per_shard - nFirstElem) / n_half_per_int4;
    for (uint64_t offset = threadId; offset < nelem4; offset += gridDim.x * blockDim.x) {
        int4 tmp = input4[offset];
        for (int i = 0; i < nranks - 1; ++i) {
            int4 val = reinterpret_cast<int4*>(&scratches[i][n_half_per_int4])[offset];
            *reinterpret_cast<__half2*>(&tmp.x) += *reinterpret_cast<__half2*>(&val.x);
            *reinterpret_cast<__half2*>(&tmp.y) += *reinterpret_cast<__half2*>(&val.y);
            *reinterpret_cast<__half2*>(&tmp.z) += *reinterpret_cast<__half2*>(&val.z);
            *reinterpret_cast<__half2*>(&tmp.w) += *reinterpret_cast<__half2*>(&val.w);
        }
        output4[offset] = tmp;
    }

    const uint64_t nLastElem = (nelem_per_shard - nFirstElem) % n_half_per_int4;
    if (threadId < nLastElem) {
        const uint64_t offset = nelem_per_shard - nLastElem + threadId;
        cutlass::half_t tmp = input[local_offset + offset];
        for (int i = 0; i < nranks - 1; ++i) {
            tmp += scratches[i][n_half_per_int4 - nFirstElem + offset];
        }
        output[local_offset + offset] = tmp;
    }
}

void setupChannels(mscclpp::Communicator* comm, std::vector<mscclpp::SmChannel>& smChannels, 
                   int rank, int nranks, void* buff, size_t buffBytes) {
    const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc;
    mscclpp::RegisteredMemory buffRegMem = comm->registerMemory(buff, buffBytes, allTransports);

    std::vector<std::shared_ptr<mscclpp::Connection>> connections;
    std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
    std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;

    for (int r = 0; r < nranks; ++r) {
        if (r == rank) continue;

        mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
        connectionFutures.push_back(comm->connectOnSetup(r, 0, transport));

        comm->sendMemoryOnSetup(buffRegMem, r, 0);
        auto remoteMemory = comm->recvMemoryOnSetup(r, 0);
        remoteRegMemories.push_back(remoteMemory);
    }
    comm->setup();
    std::transform(
        connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
        [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) { return future.get(); });
    
    std::unordered_map<size_t, std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
    for (size_t cid = 0; cid < connections.size(); ++cid) {
        smSemaphores.emplace(cid, std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm, connections[cid]));
    }
    comm->setup();

    for (size_t cid = 0; cid < connections.size(); ++cid) {
        if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
            smChannels.emplace_back(smSemaphores[cid], remoteRegMemories[cid].get(), buffRegMem.data());
        }
    }
}
