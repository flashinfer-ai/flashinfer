# Copyright (c) 2026 by FlashInfer team.
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

"""Framework-side facilities shared by production Kernel wrappers."""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float8E4M3FN
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor


class _GraphSafeDLPack:
    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        # stream=-1 skips producer sync; CuTe launches on the current captured stream.
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def to_cute(tensor: torch.Tensor, alignment: int) -> cute.Tensor:
    if tensor.dtype == torch.float8_e4m3fn:
        # CuTe DSL's from_dlpack has no Float8E4M3FN entry in its dtype map
        # (NVIDIA/cutlass#3716). Export the same storage as bytes and restore
        # the element type; keep the graph-safe adapter so export never syncs.
        result = from_dlpack(
            _GraphSafeDLPack(tensor.view(torch.uint8).detach()),
            assumed_align=alignment,
        )
        result.element_type = Float8E4M3FN
        return result
    return from_dlpack(
        _GraphSafeDLPack(tensor.detach()),
        assumed_align=alignment,
    )


def to_cute_dynamic(
    tensor: torch.Tensor,
    alignment: int,
    *,
    divisibility: int,
) -> cute.Tensor:
    return to_cute(tensor, alignment).mark_compact_shape_dynamic(
        mode=0,
        divisibility=divisibility,
    )


def make_fake_dynamic_compact_tensor(
    dtype,
    *,
    alignment: int,
    divisibility: int,
) -> cute.Tensor:
    return make_fake_compact_tensor(
        dtype,
        (cute.sym_int32(divisibility=divisibility),),
        assumed_align=alignment,
    )


def is_dense_on(t: torch.Tensor, device: torch.device, alignment: int) -> bool:
    """Contiguous, on ``device``, with an ``alignment``-byte base address."""
    return t.device == device and t.is_contiguous() and t.data_ptr() % alignment == 0


def require_mnnvl_multicast_device() -> torch.device:
    """The current device, if it supports the MNNVL multicast kernels."""
    import torch.distributed._symmetric_memory as symm_mem

    from ..mnnvl import is_multicast_supported

    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] < 10:
        raise RuntimeError("requires a Blackwell GPU")
    if symm_mem.get_backend(device) is None:
        raise RuntimeError("PyTorch Symmetric Memory is unavailable")
    if not is_multicast_supported(device.index):
        raise RuntimeError("NVLink multicast is unavailable")
    return device


def current_cu_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def collective_support(
    local_supported: bool,
    group,
    device: torch.device,
    flag: torch.Tensor | None = None,
) -> bool:
    """Agree on a dispatch choice during warmup, before CUDA graph capture.

    Every rank must call this, including ranks whose local check failed.
    Keep the agreed dispatch for graph replay; this helper synchronizes the CPU.
    ``flag`` is an optional preallocated one-element int32 vote buffer, for
    votes on whether an allocation succeeded.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Agree on collective support before CUDA graph capture")
    import torch.distributed as dist

    if flag is None:
        vote = torch.tensor(int(local_supported), dtype=torch.int32, device=device)
    else:
        vote = flag.fill_(int(local_supported))
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    return bool(vote.item())
