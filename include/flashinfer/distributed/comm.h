// adapted from https://github.com/efeslab/Nanoflow/blob/d6b381e58110a8b5d08cfabd4a55c0d5d0ebef57/pipeline/include/comm.h
#pragma once
#include <cutlass/half.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/proxy_channel_device.hpp>

#include "helper.h"

extern "C" __global__ void __launch_bounds__(1024)
	allgather(mscclpp::SmChannelDeviceHandle* sm_channels,
			  mscclpp::DeviceSyncer* syncers,
			  const uint64_t n_parallel_sm_blocks,
			  const uint64_t local_offset,
			  const uint64_t* offsets,
			  const uint64_t nelem_per_channel);

extern "C" __global__ void __launch_bounds__(1024)
    allgatherKernelWithoutSync(mscclpp::SmChannelDeviceHandle* sm_channels, 
							   mscclpp::DeviceSyncer* syncers,
							   const int nchannels, const uint64_t local_offset, const uint64_t nelem_per_shard,
							   cutlass::half_t* input, cutlass::half_t* output);

extern "C" __global__ void __launch_bounds__(1024)
    reduceScatterKernelWithoutSync(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
                        		   mscclpp::DeviceSyncer* syncer,
                        		   const int rank, const int nranks, const uint64_t nelem_per_shard,
                        		   cutlass::half_t* input, cutlass::half_t* output);

extern "C" __global__ void __launch_bounds__(1024)
    allreduceKernelWithoutSync(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
                    		   mscclpp::SmChannelDeviceHandle* sm_output_buff_channels,
                    		   mscclpp::DeviceSyncer* syncers,
                    		   mscclpp::DeviceSyncer* global_syncer,
                    		   const int nchannels, const uint64_t local_offset,
                    		   const int rank, const int nranks, const uint64_t nelem_per_shard,
                    		   cutlass::half_t* input, cutlass::half_t* output);

extern "C" __global__ void __launch_bounds__(1024)
	allgatherKernelEntryPoint(mscclpp::SmChannelDeviceHandle* sm_channels,
							  mscclpp::DeviceSyncer* syncers,
							  const int nchannels,
							  const uint64_t local_offset,
							  const uint64_t nelem_per_shard,
							  cutlass::half_t* input,
							  cutlass::half_t* output);

extern "C" __global__ void __launch_bounds__(1024)
    columnwiseAllgatherKernelEntryPoint(mscclpp::SmChannelDeviceHandle* sm_channels,
                                        mscclpp::DeviceSyncer* syncers,
                                        const bool sync, const int nchannels,
                                        const uint64_t input_ncols, const uint64_t output_ncols,
                                        const uint64_t output_row_offset, const uint64_t nrows,
                                        cutlass::half_t* input, cutlass::half_t* output);

extern "C" __global__ void __launch_bounds__(1024)
	reduceScatterKernelEntryPoint(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
								  mscclpp::DeviceSyncer* syncer,
								  const int rank,
								  const int nranks,
								  const uint64_t nelem_per_shard,
								  cutlass::half_t* input,
								  cutlass::half_t* output);

extern "C" __global__ void __launch_bounds__(1024)
	allreduceKernelEntryPoint(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
							  mscclpp::SmChannelDeviceHandle* sm_output_buff_channels,
							  mscclpp::DeviceSyncer* syncers,
							  mscclpp::DeviceSyncer* global_syncer,
							  const int nchannels,
							  const uint64_t local_offset,
							  const int rank,
							  const int nranks,
							  const uint64_t nelem_per_shard,
							  cutlass::half_t* input,
							  cutlass::half_t* output);

extern "C" __global__ void __launch_bounds__(1024) 
    allreduceKernelWithLNEntryPoint(mscclpp::SmChannelDeviceHandle* sm_input_buff_channels,
                                    mscclpp::SmChannelDeviceHandle* sm_output_buff_channels,
                                    mscclpp::DeviceSyncer* syncers,
                                    mscclpp::DeviceSyncer* global_syncer,
                                    const int nchannels, const uint64_t local_offset,
                                    const int rank, const int nranks, const uint64_t nelem_per_shard,
                                    cutlass::half_t* input, cutlass::half_t* output, cutlass::half_t* output_of_ln, 
									half* ln_weight, int rows, int columns, float epsilon);

extern "C" __global__ void __launch_bounds__(1024)
	asyncAllgatherKernelStartEntryPoint(mscclpp::SimpleProxyChannelDeviceHandle* proxy_channels,
										mscclpp::SmChannelDeviceHandle* sm_sync_channels,
										const int nchannels,
										const uint64_t local_offset,
										const uint64_t nelem_per_shard);

extern "C" __global__ void __launch_bounds__(1024)
	asyncAllgatherKernelFinishEntryPoint(mscclpp::SimpleProxyChannelDeviceHandle* proxy_channels,
										 const int nchannels);

extern "C" __global__ void __launch_bounds__(1024)
	asyncReduceScatterKernelStartEntryPoint(mscclpp::SimpleProxyChannelDeviceHandle* proxy_channels,
											mscclpp::SmChannelDeviceHandle* sm_sync_channels,
											const int rank, const int nranks, const uint64_t nelem_per_shard);

extern "C" __global__ void __launch_bounds__(1024)
	asyncReduceScatterKernelFinishEntryPoint(mscclpp::SimpleProxyChannelDeviceHandle* proxy_channels,
											 mscclpp::DeviceSyncer* syncer,
											 const int rank, const int nranks, const uint64_t nelem_per_shard,
											 cutlass::half_t* scratch, cutlass::half_t* input, cutlass::half_t* output);

extern "C" __global__ void __launch_bounds__(1024)
	syncDevices(mscclpp::SmChannelDeviceHandle* sm_sync_channels, const int nchannels);

extern "C" __global__ void __launch_bounds__(1024)
	asyncReduceKernel(const int rank, const int nranks, const uint64_t nelem_per_shard,
					  cutlass::half_t** scratches, cutlass::half_t* input, cutlass::half_t* output);

void setupChannels(mscclpp::Communicator* comm,
				   std::vector<mscclpp::SmChannel>& smChannels,
				   int rank,
				   int nranks,
				   void* buff,
				   size_t buffBytes);
