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

"""Fused fp8 paged-MQA top-k sparse-prefill indexer (SM103).

One call scores and selects: for every prefill query row it scores all
causally visible compressed KV blocks with fp32
``sum_h relu(q_h . k_block)`` over 4 MQA heads of head_dim 128, selects the
top 512 blocks (ties broken by LOWER block index -- a deterministic part of
the contract, not a checker detail), expands each selected block to its 4
token indices, appends the causal tail of the open block, and emits one
packed int32 row per query: ascending valid token indices in columns
0..2050, -1 padding, valid count in column 2051. The output is bitwise
deterministic across runs of any given build.

The kernel is validated per CUDA toolkit, and dispatch refuses to serve
when the current toolkit (``torch.version.cuda``) is not in
``VALIDATED_CUDA_VERSIONS`` (fallback reason ``unvalidated_toolkit``):
behavior under other compilers is measured to differ and is not
guaranteed. On the validated toolkit (nvcc 13.3) the selection matches an
fp32 reference bit-for-bit on every verified row. Measured under nvcc 12.9
and 13.0, the top-512 set can differ from that reference by a score-TIED
boundary block at the selection threshold (set difference <= 1 block per
row, packed format/causal tail/ordering/count exact), and nvcc 13.0
additionally showed intermittent instability under investigation — hence
the refusal rather than a weaker contract.

There is no other FlashInfer path serving this fused contract today:
``fp8_paged_mqa_logits`` + ``FP8MQATopKWrapper`` cover the decode-side
logits and a single-request prefill selection over a different (fused)
cache layout, and the wrapper refuses multi-request batches.

Generated-kernel provenance: agent-generated kernel (search-campaign
winner), source-vetted (producer/consumer ordering via CUDA Programmatic
Dependent Launch; exact iterative selection fallback behind every
specialized route) and verified bit-exact against an fp32 reference
per workload row on GB300.
"""

import functools
import os as _os

import torch

from ..jit.fp8_paged_mqa_topk_indexer import (
    gen_fp8_paged_mqa_topk_indexer_module,
)

KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"


@functools.cache
def _jit_nvcc_release() -> str:
    """The CUDA release of the nvcc the JIT will compile with ("" when it
    cannot be determined). torch.version.cuda describes torch's build, not
    the compiler that builds this kernel, and the validated-toolkit
    guarantee is about the compiler."""
    import re
    import subprocess

    try:
        txt = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return ""
    m = re.search(r"release (\d+\.\d+)", txt)
    return m.group(1) if m else ""


DISPATCH_STATS = {"specialized": 0, "rejected": 0}
LAST_FALLBACK_REASONS: list = []
_BUILT = {"ready": False}

PACKED_WIDTH = 2052  # 2048 expanded slots + 3 causal-tail slots + count col
BACKENDS = ("specialized",)
# CUDA toolkits (torch.version.cuda prefixes) the kernel is validated on.
# Dispatch refuses off this list; extend it only after re-running the full
# per-toolkit validation (reference exactness + sanitizers on the scored
# regime).
VALIDATED_CUDA_VERSIONS = ("13.3", "13.1",)


@functools.cache
def get_fp8_paged_mqa_topk_indexer_module():
    mod = gen_fp8_paged_mqa_topk_indexer_module().build_and_load()
    _BUILT["ready"] = True
    return mod


def _supported_with_reasons(qi, ck_cache, cpage_table, cu_seqlens, pos, visible, out):
    """Guard order: GPU-free (kill switch, dtype, geometry) -> layout ->
    device. Returns [] when servable, else reasons. No device syncs."""
    if _os.environ.get(KILL_SWITCH_ENV, "0") == "1":
        return ["kill_switch:" + KILL_SWITCH_ENV + "=1"]
    if not (
        torch.version.cuda and torch.version.cuda.startswith(VALIDATED_CUDA_VERSIONS)
    ):
        return [
            "unvalidated_toolkit:cuda_"
            + str(torch.version.cuda)
            + ";validated="
            + ",".join(VALIDATED_CUDA_VERSIONS)
        ]
    _nvcc = _jit_nvcc_release()
    if _nvcc and not _nvcc.startswith(VALIDATED_CUDA_VERSIONS):
        # the environment's nvcc (which compiles the JIT module) differs
        # from torch's CUDA build and is not a validated toolkit
        return [
            "unvalidated_toolkit:nvcc_"
            + _nvcc
            + ";validated="
            + ",".join(VALIDATED_CUDA_VERSIONS)
        ]
    r = []
    if (
        qi.dim() != 3
        or qi.shape[1] != 4
        or qi.shape[2] != 128
        or qi.dtype != torch.float8_e4m3fn
    ):
        r.append("qi_not_Tx4x128_fp8e4m3")
    if (
        ck_cache.dim() != 4
        or ck_cache.shape[1] != 64
        or ck_cache.shape[2] != 1
        or ck_cache.shape[3] != 128
        or ck_cache.dtype != torch.float8_e4m3fn
    ):
        r.append("ck_cache_not_Px64x1x128_fp8e4m3")
    if cpage_table.dim() != 2 or cpage_table.dtype != torch.int32:
        r.append("cpage_table_not_2d_int32")
    if (
        cu_seqlens.dim() != 1
        or cu_seqlens.dtype != torch.int32
        or (cpage_table.dim() == 2 and cu_seqlens.shape[0] != cpage_table.shape[0] + 1)
    ):
        r.append("cu_seqlens_not_Bplus1_int32")
    # qi.shape[0] is only meaningful once qi's rank is right: return the
    # geometry errors first instead of raising on the shape read.
    if r:
        return r
    if pos.dim() != 1 or pos.dtype != torch.int32 or pos.shape[0] != qi.shape[0]:
        r.append("pos_not_T_int32")
    if (
        visible.dim() != 1
        or visible.dtype != torch.int32
        or visible.shape[0] != qi.shape[0]
    ):
        r.append("visible_not_T_int32")
    if out is not None and (
        out.dtype != torch.int32
        or out.dim() != 2
        or out.shape[0] != qi.shape[0]
        or out.shape[1] != PACKED_WIDTH
        or not out.is_contiguous()
        or out.device != qi.device
    ):
        r.append("out_not_TxPW_int32_contiguous")
    if cpage_table.shape[0] == 0:
        r.append("batch_size_zero")
    for _name, _t in (
        ("qi", qi),
        ("ck_cache", ck_cache),
        ("cpage_table", cpage_table),
        ("cu_seqlens", cu_seqlens),
        ("pos", pos),
        ("visible", visible),
    ):
        if not _t.is_contiguous():
            r.append(_name + "_not_contiguous")
        if _t.device != qi.device:
            r.append(_name + "_device_mismatch")
    for _name, _t in (("qi", qi), ("ck_cache", ck_cache)):
        if _t.data_ptr() % 16:
            r.append(_name + "_not_16B_aligned")
    if out is not None and out.data_ptr() % 16:
        r.append("out_not_16B_aligned")
    if out is not None:
        _o0 = out.data_ptr()
        _o1 = _o0 + out.numel() * out.element_size()
        for _nm, _t in (
            ("qi", qi),
            ("ck_cache", ck_cache),
            ("cpage_table", cpage_table),
            ("cu_seqlens", cu_seqlens),
            ("pos", pos),
            ("visible", visible),
        ):
            _t0 = _t.data_ptr()
            _t1 = _t0 + _t.numel() * _t.element_size()
            if out.device == _t.device and _o0 < _t1 and _t0 < _o1:
                r.append("out_overlaps_" + _nm)
    if r:
        return r
    if qi.device.type != "cuda":
        return ["qi_not_cuda"]
    if torch.cuda.get_device_capability(qi.device) != (10, 3):
        return ["not_sm103"]
    if torch.cuda.is_current_stream_capturing() and not _BUILT["ready"]:
        return ["module_not_built_before_graph_capture"]
    return []


def fp8_paged_mqa_topk_indexer(
    qi: torch.Tensor,
    ck_cache: torch.Tensor,
    cpage_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pos: torch.Tensor,
    visible: torch.Tensor,
    *,
    out: torch.Tensor = None,
    backend: str = "specialized",
):
    """Score + select + emit in one fused call (see module docstring).

    Parameters
    ----------
    qi : torch.Tensor
        ``(total_q, 4, 128)`` fp8 E4M3 indexer queries, varlen-packed over
        the batch.
    ck_cache : torch.Tensor
        ``(num_cpages, 64, 1, 128)`` fp8 E4M3 compressed keys, paged 64
        blocks per page.
    cpage_table : torch.Tensor
        ``(batch_size, max_cpages)`` int32: compressed block ``b`` of
        request ``r`` lives at paged row
        ``cpage_table[r, b // 64] * 64 + b % 64``.
    cu_seqlens : torch.Tensor
        ``(batch_size + 1,)`` int32 cumulative query lengths.
    pos : torch.Tensor
        ``(total_q,)`` int32 request-local token positions.
    visible : torch.Tensor
        ``(total_q,)`` int32 causally visible compressed-block counts
        (``(pos + 1) // 4`` for standard causal prefill).
    out : torch.Tensor, optional
        ``(total_q, 2052)`` int32 contiguous destination; allocated when
        omitted.
    backend : str
        Only ``"specialized"`` exists for this fused contract. A call the
        allowlist cannot serve raises ``ValueError`` with the fallback
        reasons (there is no stock path to swap to); the kill switch
        ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` makes every call raise.

    Returns
    -------
    torch.Tensor
        ``(total_q, 2052)`` int32 packed rows: ascending selected token
        indices, ``-1`` padding, valid count in column 2051.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    reasons = _supported_with_reasons(
        qi, ck_cache, cpage_table, cu_seqlens, pos, visible, out
    )
    LAST_FALLBACK_REASONS[:] = reasons
    if reasons:
        DISPATCH_STATS["rejected"] += 1
        raise ValueError(
            "fp8_paged_mqa_topk_indexer cannot serve this call; "
            f"fallback_reasons={reasons}"
        )
    try:
        get_fp8_paged_mqa_topk_indexer_module()
    except Exception as e:  # pragma: no cover - environment-dependent
        DISPATCH_STATS["rejected"] += 1
        raise ValueError(
            "fp8_paged_mqa_topk_indexer: specialized module unavailable "
            f"({type(e).__name__}) and no stock path exists"
        ) from e
    DISPATCH_STATS["specialized"] += 1
    if out is None:
        out = torch.empty(
            (qi.shape[0], PACKED_WIDTH), dtype=torch.int32, device=qi.device
        )
    mod = get_fp8_paged_mqa_topk_indexer_module()
    with torch.cuda.device(qi.device):
        mod.fp8_paged_mqa_topk_indexer_run(
            qi, ck_cache, cpage_table, cu_seqlens, pos, visible, out
        )
    return out
