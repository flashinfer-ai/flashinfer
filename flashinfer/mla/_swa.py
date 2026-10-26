# SPDX-License-Identifier: Apache-2.0
"""Page64 BF16 sliding-window MLA. SM120's packed FP8 sparse API is separate."""

import functools
import math
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..autotuner import AutoTuner, TunableRunner, TuningConfig
from ..jit.mla import gen_mla_swa_sm90_module
from ..trace.templates.attention import mla_swa_decode_trace
from ..utils import register_custom_op, register_fake_op

_DEFAULT_TACTICS = {
    (1, 1): 0,
    (2, 1): 1,
    (4, 1): 2,
    (8, 1): 3,
    (16, 1): 3,
    (1, 4): 2,
    (2, 4): 3,
    (4, 4): 3,
    (8, 4): 4,
    (16, 4): 5,
}


@flashinfer_api
def get_mla_swa_workspace_size_bytes(
    batch_size: int, q_len_per_req: int, num_q_heads: int = 64
) -> int:
    """Return byte workspace capacity for every SM90 MLA+SWA tuning candidate.

    Parameters
    ----------
    batch_size : int
        Local batch size, one of 1, 2, 4, 8, 16.
    q_len_per_req : int
        Query length per request, 1 or 4.
    num_q_heads : int
        Local query head count; currently 64.

    Returns
    -------
    int
        Capacity for FP32 partial output and LSE with up to nine splits,
        including 256-byte padding. The final output is not part of workspace.
        Even DSM/direct calls require this capacity so autotune can change tactics.
    """
    if (batch_size, q_len_per_req) not in _DEFAULT_TACTICS or num_q_heads != 64:
        raise ValueError("Qualified shapes: B1/2/4/8/16, q_len1/4, Hq64")
    rows = batch_size * q_len_per_req * num_q_heads
    return (rows * 9 * 1024 * 4 + 255) // 256 * 256 + (rows * 9 * 4 + 255) // 256 * 256


@functools.cache
def _get_module():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up MLA+SWA eagerly before CUDA Graph capture")
    return gen_mla_swa_sm90_module().build_and_load()


@functools.cache
def _device_supported(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (9, 0) and bool(
        _get_module().device_supported(device.index)
    )


def _validate(
    query,
    kv_cache,
    workspace,
    block_tables,
    seq_lens,
    out,
    window_left,
    sm_scale,
    backend,
):
    if backend not in ("auto", "sm90"):
        raise ValueError("MLA+SWA currently supports backend='auto' or 'sm90'")
    tensors = (query, kv_cache, workspace, block_tables, seq_lens, out)
    if any(not t.is_cuda or t.device != query.device for t in tensors):
        raise ValueError("All tensors must be on the same CUDA device")
    if query.ndim != 4 or query.shape[-2:] != (64, 1088):
        raise ValueError("query must be [B, q_len_per_req, 64, 1088]")
    b, sq, h, _ = query.shape
    required = get_mla_swa_workspace_size_bytes(b, sq, h)
    if (
        query.dtype != torch.bfloat16
        or kv_cache.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
    ):
        raise ValueError("Q, KV and output must be BF16")
    if query.stride()[-3:] != (h * 1088, 1088, 1) or query.stride(0) < sq * h * 1088:
        raise ValueError("Q rows must be contiguous and batches must not overlap")
    if (
        kv_cache.ndim != 3
        or kv_cache.shape[1:] != (64, 1088)
        or not kv_cache.is_contiguous()
        or kv_cache.shape[0] < 1
    ):
        raise ValueError("kv_cache must be contiguous [num_pages, 64, 1088]")
    if (
        block_tables.ndim != 2
        or block_tables.shape[0] != b
        or block_tables.shape[1] < 1
        or block_tables.stride(1) != 1
        or block_tables.stride(0) < block_tables.shape[1]
    ):
        raise ValueError("block_tables must be non-overlapping [B, max_pages] rows")
    if (
        seq_lens.shape != (b,)
        or not seq_lens.is_contiguous()
        or block_tables.dtype != torch.int32
        or seq_lens.dtype != torch.int32
    ):
        raise ValueError("block_tables/seq_lens must be CUDA int32; seq_lens is [B]")
    if (
        workspace.dtype != torch.uint8
        or workspace.ndim != 1
        or not workspace.is_contiguous()
        or workspace.numel() < required
    ):
        raise ValueError(
            f"workspace must be a contiguous uint8 vector with at least {required} bytes"
        )
    if out.shape != (b, sq, h, 1024) or not out.is_contiguous():
        raise ValueError("out must be contiguous [B, q_len_per_req, 64, 1024]")
    if window_left != 512 or not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError(
            "Only window_left=512 and positive finite sm_scale are qualified"
        )
    if any(t.data_ptr() % 16 for t in (query, kv_cache, workspace, out)):
        raise ValueError("Q/KV/workspace/out require 16-byte alignment")

    # Writable buffers cannot alias inputs or each other, including strided views.
    def extent(t):
        n = 1 + sum(
            (s - 1) * stride for s, stride in zip(t.shape, t.stride(), strict=False)
        )
        return t.data_ptr(), t.data_ptr() + n * t.element_size()

    for target in (workspace, out):
        lo, hi = extent(target)
        for source in tensors:
            if source is target:
                continue
            other_lo, other_hi = extent(source)
            if max(lo, other_lo) < min(hi, other_hi):
                raise ValueError(
                    "Writable MLA+SWA buffers must not overlap any other tensor"
                )
    if not _device_supported(query.device):
        raise ValueError(
            "Requires SM90 cluster launch and sufficient opt-in shared memory"
        )


@register_custom_op("flashinfer::mla_swa_sm90", mutates_args=("workspace", "out"))
def _run(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    window_left: int,
    tactic: int,
) -> None:
    _get_module().run(
        query,
        kv_cache,
        workspace,
        block_tables,
        seq_lens,
        out,
        sm_scale,
        window_left,
        tactic,
    )


@register_fake_op("flashinfer::mla_swa_sm90")
def _run_fake(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    window_left: int,
    tactic: int,
) -> None:
    pass


def _profiling_inputs(inputs):
    # Never profile random int32 page IDs or mutate caller metadata. No host
    # metadata inspection is required; all dimensions stay exact (no bucketing).
    q, kv, workspace, table, lengths, out = inputs
    b, sq = q.shape[:2]
    table = (
        torch.arange(b * table.shape[1], device=q.device, dtype=torch.int32).reshape(
            table.shape
        )
        % kv.shape[0]
    )
    context = min(table.shape[1] * 64, 4127)
    lengths = torch.full((b,), context, device=q.device, dtype=torch.int32)
    return [q, kv, workspace, table, lengths, out]


class _Runner(TunableRunner):
    def __init__(self, scale, window_left):
        self.scale = scale
        self.window_left = window_left

    def get_cache_key_extras(self, inputs):
        return ("sm90-bf16-page64-fp32-v1", self.scale, self.window_left)

    def get_valid_tactics(self, inputs, profile):
        return list(range(6))

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        if do_preparation:
            _get_module()
            return inputs[-1]
        if tactic == -1:
            tactic = _DEFAULT_TACTICS[tuple(inputs[0].shape[:2])]
        _run(*inputs, self.scale, self.window_left, tactic)
        return inputs[-1]


@functools.cache
def _runner(scale, window_left):
    return _Runner(scale, window_left)


_TUNING_CONFIG = TuningConfig(use_cuda_graph=True, inputs_pre_hook=_profiling_inputs)


@flashinfer_api(trace=mla_swa_decode_trace)
def batch_decode_with_kv_cache_mla_swa(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    window_left: int,
    sm_scale: float,
    out: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Causal sliding-window MLA decode over a shared latent Page64 KV cache.

    Parameters
    ----------
    query : torch.Tensor
        BF16 ``[B, q_len_per_req, 64, 1088]``; B1/2/4/8/16 and q_len1/4.
    kv_cache : torch.Tensor
        BF16 contiguous ``[num_pages, 64, 1088]``. The first 1024 features
        are both latent K and V; the last 64 features are positional K.
        Attended K/V values must be finite. Masked page slots need not be
        initialized; they do not contribute even if they contain NaN or Inf.
    workspace_buffer : torch.Tensor
        Caller-owned CUDA uint8 buffer sized by :func:`get_mla_swa_workspace_size_bytes`.
        Do not share writable workspace across concurrent calls/streams.
    block_tables : torch.Tensor
        CUDA int32 ``[B, max_pages]``, mapping logical pages to physical pages.
    seq_lens : torch.Tensor
        CUDA int32 ``[B]``, total KV lengths **including** the query tokens.
        Query i has position ``seq_lens[b] - q_len_per_req + i``. Valid
        page IDs must be in range and each nonempty length must fit its table;
        invalid metadata values are not checked synchronously. Zero-length
        dummy rows produce zero output.
    window_left : int
        Number of past tokens; currently 512 (513 including the current token).
    sm_scale : float
        Positive finite attention scale, supplied by the model.
    out : torch.Tensor, optional
        Contiguous BF16 ``[B, q_len_per_req, 64, 1024]``. If omitted, allocated.
        Pass a persistent output and warm up eagerly before Graph capture.
    backend : str
        ``"auto"`` or ``"sm90"``. Both use the SM90 implementation; there is
        no hidden fallback to a different operator/precision on other devices.

    Returns
    -------
    torch.Tensor
        The supplied or newly allocated output. All split reductions retain
        FP32 until the final BF16 write. Use ``flashinfer.autotune(True)``
        before capture to search the six tactics; otherwise a frozen LUT is used.
    """
    if out is None:
        out = torch.empty(
            (*query.shape[:-1], 1024), device=query.device, dtype=query.dtype
        )
    scale = float(sm_scale)
    _validate(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        out,
        window_left,
        scale,
        backend,
    )
    inputs = [query, kv_cache, workspace_buffer, block_tables, seq_lens, out]
    runner, tactic = AutoTuner.get().choose_one(
        "mla_swa_sm90", [_runner(scale, window_left)], _TUNING_CONFIG, inputs
    )
    return runner(inputs, tactic=tactic)
