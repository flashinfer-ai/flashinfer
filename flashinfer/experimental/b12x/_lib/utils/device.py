# Copyright (c) 2025 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA device capabilities and launch geometry."""

import cuda.bindings.driver as cuda
import cutlass
import functools
import torch
from b12x._lib.utils.dtypes import _hardware_info_cache, _num_sm_cache


def get_num_sm(device: torch.device) -> int:
    """Return the device SM count without exposing lru_cache to Dynamo."""
    if device.type != "cuda":
        return 1
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    sm_count = _num_sm_cache.get(index)
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(index).multi_processor_count
        _num_sm_cache[index] = sm_count
    return sm_count


@torch._dynamo.disable
def current_cuda_stream() -> cuda.CUstream:
    """Return the current Torch CUDA stream as a CUDA driver stream handle."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def cuda_stream_to_int(stream: object | None) -> int | None:
    """Return a raw CUDA stream handle, preserving None as a caller fallback."""
    if stream is None:
        return None
    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is not None:
        return int(cuda_stream)
    return int(stream)


def cuda_stream_from_int_or_current(stream_int: int | None) -> cuda.CUstream:
    """Use a supplied raw CUDA stream handle, or fall back to Torch's current stream."""
    if stream_int is None:
        return current_cuda_stream()
    return cuda.CUstream(int(stream_int))


def get_hardware_info() -> "cutlass.utils.HardwareInfo":
    """Get cached HardwareInfo singleton.

    HardwareInfo queries CUDA device capabilities, which can be expensive.
    This function caches the singleton to avoid repeated queries.
    """
    global _hardware_info_cache
    if _hardware_info_cache is None:
        _hardware_info_cache = cutlass.utils.HardwareInfo()
    return _hardware_info_cache


@functools.cache
def get_max_active_clusters(cluster_size: int) -> int:
    """Get max active clusters for a given cluster size (cached).

    Args:
        cluster_size: Product of cluster_shape_mn dimensions.

    Returns:
        Maximum number of active clusters supported by hardware.
    """
    if cluster_size == 1:
        # A one-CTA cluster is bounded exactly by the SM count. Avoid the
        # CUTLASS occupancy probe, which requires a live CUDA context only to
        # recover that same value.
        return get_num_sm(torch.device("cuda"))
    return get_hardware_info().get_max_active_clusters(cluster_size)
