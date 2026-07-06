"""
Copyright (c) 2025 by FlashInfer team.

The underlying CUDA kernel is adapted from ThunderKittens' NVLink all-to-all:
https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/parallel/all_to_all/all_to_all.cu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from types import SimpleNamespace
from typing import List

import torch

from ..jit.comm import gen_ulysses_a2a_module
from ..utils import register_custom_op

# world sizes for which the fused-transpose kernel is instantiated
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)


@functools.cache
def get_ulysses_a2a_module():
    module = gen_ulysses_a2a_module().build_and_load()

    @register_custom_op(
        "flashinfer::init_ulysses_a2a",
        mutates_args=[],
    )
    def init_ulysses_a2a(
        out_ipc_ptrs: List[int],
        signal_ipc_ptrs: List[int],
        rank: int,
        world_size: int,
        full_nvlink: bool,
    ) -> int:
        return module.init_ulysses_a2a(
            out_ipc_ptrs, signal_ipc_ptrs, rank, world_size, full_nvlink
        )

    @register_custom_op("flashinfer::dispose_ulysses_a2a", mutates_args=[])
    def dispose_ulysses_a2a(fa: int) -> None:
        module.dispose_ulysses_a2a(fa)

    @register_custom_op("flashinfer::ulysses_a2a", mutates_args=["out"])
    def ulysses_a2a(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        B: int,
        S_local: int,
        H: int,
        D: int,
        mode: int,
    ) -> None:
        module.ulysses_a2a(fa, inp, out, B, S_local, H, D, mode)

    return SimpleNamespace(
        init_ulysses_a2a=init_ulysses_a2a,
        dispose_ulysses_a2a=dispose_ulysses_a2a,
        ulysses_a2a=ulysses_a2a,
    )


def init_ulysses_a2a(
    out_ipc_ptrs: List[int],
    signal_ipc_ptrs: List[int],
    rank: int,
    world_size: int,
    full_nvlink: bool,
) -> int:
    r"""Initialize the fused-transpose Ulysses NVLink-P2P all-to-all backend.

    .. note::
        Advanced / internal API. Prefer
        :class:`~flashinfer.comm.UlyssesCommunicator`, which selects the
        backend from the actual GPU topology before any IPC allocation or JIT
        compilation, owns the IPC workspace lifecycle, and validates operands.
        This raw entry point assumes the caller has already verified all-pairs
        NVLink P2P.

    The kernel is a *push* model: each rank writes the head/sequence blocks
    destined for its peers directly into the peers' IPC-shared output staging
    buffers over NVLink, with the Ulysses layout permutation folded into the
    write addresses. Only the output staging buffers and the signal buffers must
    be IPC-shared (allocate them with
    :func:`flashinfer.comm.create_shared_buffer`); the input tensor is read
    locally and needs no registration.

    Parameters
    ----------
    out_ipc_ptrs : list[int]
        Per-rank device pointers (opened via CUDA IPC) to the output staging
        buffers, ordered by rank. Each must be at least as large as the
        all-to-all output for this group.
    signal_ipc_ptrs : list[int]
        Per-rank device pointers to the signal buffers used for the inter-GPU
        barrier. Each buffer must be ``meta_size()`` bytes (same ``Signal``
        layout as the vLLM custom all-reduce).
    rank : int
        Current rank within the Ulysses group.
    world_size : int
        Ulysses group size; must be one of ``(2, 4, 6, 8)``.
    full_nvlink : bool
        ``True`` when every pair of ranks is connected via NVLink. The push
        kernel requires all-pairs P2P access; callers must gate on this.

    Returns
    -------
    int
        Opaque handle (``fa``) to pass to subsequent ``ulysses_a2a`` calls.
        Free it with :func:`dispose_ulysses_a2a`.

    Note
    ----
    ``init`` zeroes this rank's own signal buffer. Callers must issue a
    process-group barrier (e.g. ``torch.distributed.barrier``) after all ranks
    return from init and before the first all-to-all call.
    """
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"ulysses a2a only supports world size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if not full_nvlink:
        raise ValueError(
            "full_nvlink=False is not supported: the fused kernel pushes over "
            "all-pairs NVLink P2P and has no non-P2P path. Use "
            "UlyssesCommunicator(backend='auto') for topology-aware NCCL "
            "fallback instead."
        )
    return get_ulysses_a2a_module().init_ulysses_a2a(
        out_ipc_ptrs, signal_ipc_ptrs, rank, world_size, full_nvlink
    )


def dispose_ulysses_a2a(fa: int) -> None:
    r"""Release a handle returned by :func:`init_ulysses_a2a`."""
    get_ulysses_a2a_module().dispose_ulysses_a2a(fa)


def ulysses_a2a(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    B: int,
    S_local: int,
    H: int,
    D: int,
    mode: int,
) -> None:
    r"""Fused-transpose Ulysses all-to-all.

    .. note::
        Advanced / internal API. Prefer
        :meth:`UlyssesCommunicator.scatter_heads` (``mode == 0``) and
        :meth:`UlyssesCommunicator.gather_heads` (``mode == 1``), which derive
        the geometry from the tensor shapes and validate operands.

    The result for this rank is written into ``out`` (bit-identical to the
    equivalent NCCL all-to-all followed by the layout permutation).

    ``mode == 0`` (input a2a): ``inp [B, S_local, H, D] -> out [B, S_global, H_local, D]``

    ``mode == 1`` (output a2a): ``inp [B, S_global, H_local, D] -> out [B, S_local, H, D]``

    where ``H`` is the *global* head count, ``H_local = H // world_size`` and
    ``S_global = S_local * world_size``. Both tensors must be contiguous CUDA
    tensors of the same dtype (float32/float16/bfloat16). All ranks must call
    with consistent geometry in the same order; a mismatch is a collective
    failure (hang or corruption), as with any collective.
    """
    for name, t in (("inp", inp), ("out", out)):
        if not (isinstance(t, torch.Tensor) and t.is_cuda):
            raise ValueError(f"{name} must be a CUDA tensor")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if inp.device != out.device:
        raise ValueError(f"inp is on {inp.device} but out is on {out.device}")
    if inp.dtype != out.dtype:
        raise ValueError(f"inp dtype {inp.dtype} != out dtype {out.dtype}")
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 or 1, got {mode}")
    if min(B, S_local, H, D) <= 0:
        raise ValueError(f"B/S_local/H/D must be positive, got {(B, S_local, H, D)}")
    expected = B * S_local * H * D
    if inp.numel() != expected or out.numel() != expected:
        raise ValueError(
            f"inp/out numel ({inp.numel()}/{out.numel()}) do not match "
            f"B*S_local*H*D = {expected}"
        )
    get_ulysses_a2a_module().ulysses_a2a(fa, inp, out, B, S_local, H, D, mode)
