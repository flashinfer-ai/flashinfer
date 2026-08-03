"""
Copyright (c) 2026 by FlashInfer team.

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
from typing import List, Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit.mxfp8_attention_sm120 import gen_mxfp8_attention_sm120_module
from .trace.templates.mxfp8_attention_sm120 import (
    mxfp8_attention_sm120_fwd_trace,
    mxfp8_attention_sm120_run_trace,
)
from .utils import supported_compute_capability


_BLOCK_M = 128
_BLOCK_N = 64
_HEAD_DIM = 128
_SUPPORTED_OUT_DTYPES = (torch.float16, torch.bfloat16)


@functools.cache
def get_mxfp8_attention_sm120_module():
    return gen_mxfp8_attention_sm120_module().build_and_load()


def _check_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got device={tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _padded_starts(lens: List[int]) -> List[int]:
    """Per-request starts when every request is padded to a 128-row multiple."""
    starts = [0]
    for length in lens:
        starts.append(starts[-1] + _cdiv(length, _BLOCK_M) * _BLOCK_M)
    return starts


def _scatter_index(indptr: List[int], padded_starts: List[int], device) -> torch.Tensor:
    """Destination rows of the real tokens inside the padded ragged buffer."""
    pad_offsets = [p - r for p, r in zip(padded_starts[:-1], indptr[:-1], strict=True)]
    lens = [indptr[i + 1] - indptr[i] for i in range(len(indptr) - 1)]
    total = indptr[-1]
    if total == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    offsets = torch.tensor(pad_offsets, dtype=torch.int64, device=device)
    lens_t = torch.tensor(lens, dtype=torch.int64, device=device)
    return torch.arange(
        total, dtype=torch.int64, device=device
    ) + torch.repeat_interleave(offsets, lens_t)


def _build_lpt_work_lists(
    qo_lens: List[int],
    kv_lens: List[int],
    qo_starts: List[int],
    kv_starts: List[int],
    num_qo_heads: int,
    causal: bool,
    num_sm: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Host LPT (longest-processing-time) assignment of (request, q_head, q_tile)
    work items to persistent CTAs; cost = number of 64-key blocks the tile scans."""
    import heapq

    works = []  # (req, qhead, qtile, cost)
    for r, (ql, kl) in enumerate(zip(qo_lens, kv_lens, strict=True)):
        nqt = _cdiv(ql, _BLOCK_M)
        nkt = _cdiv(kl, _BLOCK_N)
        offset = kl - ql  # slice-3 append: queries sit at the END of the kv range
        for hq in range(num_qo_heads):
            for qt in range(nqt):
                eff = (
                    min(nkt, _cdiv((qt + 1) * _BLOCK_M + offset, _BLOCK_N))
                    if causal
                    else nkt
                )
                works.append((r, hq, qt, eff))
    works.sort(key=lambda w: -w[3])  # stable sort, longest first

    heap = [(0, c) for c in range(num_sm)]
    heapq.heapify(heap)
    cta: List[List[tuple]] = [[] for _ in range(num_sm)]
    for w in works:
        load, c = heapq.heappop(heap)
        cta[c].append(w)
        heapq.heappush(heap, (load + w[3], c))

    work_indptr = [0] * (num_sm + 1)
    head_i, qtile_i, qip, kip, ql_v, kl_v, bi = [], [], [], [], [], [], []
    for c in range(num_sm):
        work_indptr[c + 1] = work_indptr[c] + len(cta[c])
        for r, hq, qt, _ in cta[c]:
            head_i.append(hq)
            qtile_i.append(qt)
            qip.append(qo_starts[r])
            kip.append(kv_starts[r])
            ql_v.append(qo_lens[r])
            kl_v.append(kv_lens[r])
            bi.append(r)

    def _to_i32(values: List[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device)

    return [
        _to_i32(work_indptr),
        _to_i32(head_i),
        _to_i32(qtile_i),
        _to_i32(qip),
        _to_i32(kip),
        _to_i32(ql_v),
        _to_i32(kl_v),
        _to_i32(bi),
    ]


def _check_qkv(name: str, tensor: torch.Tensor) -> None:
    _check_cuda_contiguous(name, tensor)
    if tensor.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name} must be torch.float8_e4m3fn, got {tensor.dtype}")
    if tensor.ndim != 3 or tensor.shape[2] != _HEAD_DIM:
        raise ValueError(
            f"{name} must have shape [total_tokens, num_heads, {_HEAD_DIM}], "
            f"got {tuple(tensor.shape)}"
        )


class MXFP8AttentionSM120Wrapper:
    r"""Plan/run wrapper for ragged varlen FP8 prefill attention on SM120/SM121.

    ``plan`` does the batch-composition work once per step (pads each request to
    128-row multiples, builds the persistent scheduler's LPT work lists); ``run``
    is tensors-only and is meant to be called once per layer. Causal masking follows
    the FlashInfer slice-3 (append) convention: request-local query ``m`` attends
    keys ``[0, m + kv_len - qo_len]``, covering prefix-cache hits and chunked-prefill
    continuation natively.

    .. note::
        ``plan`` moves ``qo_indptr``/``kv_indptr`` to the host (one D2H sync) to
        build the padding layout and LPT work lists; this mirrors the host-side
        planning of the other FlashInfer prefill wrappers.

    Example
    -------
    >>> wrapper = MXFP8AttentionSM120Wrapper()
    >>> wrapper.plan(qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, causal=True)
    >>> for q, k, v in layers:  # fp8 ragged tensors, per-layer scales
    ...     out, lse = wrapper.run(q, k, v, q_scale=qs, k_scale=ks, v_scale=vs)
    """

    def __init__(self) -> None:
        self._plan_info = None

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        causal: bool = False,
    ) -> None:
        r"""Plan for a ragged batch composition.

        Parameters
        ----------
        qo_indptr, kv_indptr : torch.Tensor
            int32 ``[batch + 1]`` ragged offsets (real, unpadded).
        num_qo_heads, num_kv_heads : int
            Head counts; GQA requires ``num_qo_heads % num_kv_heads == 0``.
        causal : bool, optional
            Whether ``run`` applies the (offset) causal mask. Affects the scheduler's
            cost model, so it is fixed at plan time.
        """
        _check_cuda_contiguous("qo_indptr", qo_indptr)
        _check_cuda_contiguous("kv_indptr", kv_indptr)
        if qo_indptr.dtype != torch.int32 or kv_indptr.dtype != torch.int32:
            raise ValueError("qo_indptr/kv_indptr must be torch.int32")
        if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
            raise ValueError("qo_indptr/kv_indptr must be 1-D")
        if num_kv_heads < 1 or num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_qo_heads ({num_qo_heads}) must be a multiple of "
                f"num_kv_heads ({num_kv_heads})"
            )
        if qo_indptr.device != kv_indptr.device:
            raise ValueError("qo_indptr/kv_indptr must be on the same device")

        qo_indptr_cpu = qo_indptr.cpu().tolist()
        kv_indptr_cpu = kv_indptr.cpu().tolist()
        batch = len(qo_indptr_cpu) - 1
        if len(kv_indptr_cpu) != batch + 1:
            raise ValueError("qo_indptr/kv_indptr batch sizes disagree")
        qo_lens = [qo_indptr_cpu[i + 1] - qo_indptr_cpu[i] for i in range(batch)]
        kv_lens = [kv_indptr_cpu[i + 1] - kv_indptr_cpu[i] for i in range(batch)]
        for r in range(batch):
            if qo_lens[r] < 0 or kv_lens[r] < 0:
                raise ValueError("indptr must be non-decreasing")
            if causal and qo_lens[r] > kv_lens[r]:
                raise ValueError(
                    f"causal append requires qo_len <= kv_len, got request {r}: "
                    f"{qo_lens[r]} > {kv_lens[r]}"
                )

        device = qo_indptr.device
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        qo_starts = _padded_starts(qo_lens)
        kv_starts = _padded_starts(kv_lens)
        self._plan_info = {
            "device": device,
            "batch": batch,
            "total_q": qo_indptr_cpu[-1],
            "total_kv": kv_indptr_cpu[-1],
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "causal": bool(causal),
            "sq_pad": qo_starts[-1],
            "sk_pad": kv_starts[-1],
            "dst_q": _scatter_index(qo_indptr_cpu, qo_starts, device),
            "dst_kv": _scatter_index(kv_indptr_cpu, kv_starts, device),
            "work_lists": _build_lpt_work_lists(
                qo_lens,
                kv_lens,
                qo_starts,
                kv_starts,
                num_qo_heads,
                causal,
                num_sm,
                device,
            ),
        }

    @supported_compute_capability([120, 121])
    @flashinfer_api(trace=mxfp8_attention_sm120_run_trace)
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale: Optional[float] = None,
        q_scale: float = 1.0,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Run the attention for one layer (tensors only; see :meth:`plan`).

        Q/K/V are ragged ``torch.float8_e4m3fn`` tensors whose per-tensor scales fold
        as ``softmax(Q K^T * (sm_scale * q_scale * k_scale)) * V * v_scale``.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Ragged FP8 tensors ``[total_q, num_qo_heads, 128]`` /
            ``[total_kv, num_kv_heads, 128]``; token totals must match the indptr
            passed to :meth:`plan`.
        sm_scale : Optional[float], optional
            Softmax score scale, defaults to ``head_dim ** -0.5``.
        q_scale, k_scale, v_scale : float, optional
            Per-tensor dequantization scales of the FP8 inputs (default 1.0).
        out, lse : Optional[torch.Tensor], optional
            Optional output ``[total_q, num_qo_heads, 128]`` and log-sum-exp
            ``[total_q, num_qo_heads]`` buffers.
        out_dtype : torch.dtype, optional
            Output dtype (``torch.float16`` or ``torch.bfloat16``) when ``out`` is
            omitted.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Attention output and LSE ``[total_q, num_qo_heads]`` (float32, natural-log
            domain of the folded scores).
        """
        if self._plan_info is None:
            raise RuntimeError("plan() must be called before run()")
        p = self._plan_info
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            _check_qkv(name, tensor)
        if (
            q.device != p["device"]
            or k.device != p["device"]
            or v.device != p["device"]
        ):
            raise ValueError("q/k/v must be on the plan's device")
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        if v.shape != k.shape:
            raise ValueError(
                f"k/v shapes must match, got {tuple(k.shape)} vs {tuple(v.shape)}"
            )
        if total_q != p["total_q"] or total_kv != p["total_kv"]:
            raise ValueError(
                f"token totals {(total_q, total_kv)} do not match the plan's "
                f"{(p['total_q'], p['total_kv'])}"
            )
        if num_qo_heads != p["num_qo_heads"] or num_kv_heads != p["num_kv_heads"]:
            raise ValueError("head counts do not match the plan")
        if out_dtype not in _SUPPORTED_OUT_DTYPES:
            raise ValueError(f"out_dtype must be float16/bfloat16, got {out_dtype}")
        if sm_scale is None:
            sm_scale = head_dim**-0.5

        device = p["device"]
        sq_pad, sk_pad = p["sq_pad"], p["sk_pad"]
        dst_q, dst_kv = p["dst_q"], p["dst_kv"]

        # Zero the padded tails: pad bytes sit inside the last loaded K/V block and
        # must not form NaN fp8 patterns (0 * NaN = NaN in the PV MMA).
        q_pad = torch.zeros(
            (sq_pad, num_qo_heads, head_dim), dtype=q.dtype, device=device
        )
        k_pad = torch.zeros(
            (sk_pad, num_kv_heads, head_dim), dtype=k.dtype, device=device
        )
        v_pad = torch.zeros_like(k_pad)
        if total_q > 0:
            q_pad[dst_q] = q
        if total_kv > 0:
            k_pad[dst_kv] = k
            v_pad[dst_kv] = v

        o_pad = torch.empty(
            (sq_pad, num_qo_heads, head_dim), dtype=torch.float32, device=device
        )
        lse_pad = torch.empty(
            (num_qo_heads, sq_pad), dtype=torch.float32, device=device
        )
        l_pad = torch.empty_like(lse_pad)

        module = get_mxfp8_attention_sm120_module()
        # The kernel's V smem atom is Sk-major, so V must arrive Sk-contiguous
        # ([num_kv_heads, head_dim, Sk_pad]); see the binding for why.
        v_pad_t = v_pad.permute(1, 2, 0).contiguous()
        module.fwd(
            q_pad,
            k_pad,
            v_pad_t,
            *p["work_lists"],
            o_pad,
            lse_pad,
            l_pad,
            float(sm_scale) * float(q_scale) * float(k_scale),
            float(v_scale),
            p["causal"],
        )

        if out is None:
            out = torch.empty(
                (total_q, num_qo_heads, head_dim), dtype=out_dtype, device=device
            )
        else:
            _check_cuda_contiguous("out", out)
            if tuple(out.shape) != (total_q, num_qo_heads, head_dim):
                raise ValueError(
                    f"out shape {tuple(out.shape)} must be "
                    f"{(total_q, num_qo_heads, head_dim)}"
                )
        if lse is None:
            lse = torch.empty(
                (total_q, num_qo_heads), dtype=torch.float32, device=device
            )
        else:
            _check_cuda_contiguous("lse", lse)
            if tuple(lse.shape) != (total_q, num_qo_heads):
                raise ValueError(
                    f"lse shape {tuple(lse.shape)} must be {(total_q, num_qo_heads)}"
                )
        if total_q > 0:
            out.copy_(o_pad[dst_q].to(out.dtype))
            lse.copy_(lse_pad[:, dst_q].transpose(0, 1))
        return out, lse


@supported_compute_capability([120, 121])
@flashinfer_api(trace=mxfp8_attention_sm120_fwd_trace)
def mxfp8_attention_sm120_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: Optional[float] = None,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Ragged varlen FP8 prefill attention on SM120/SM121 (one-shot plan+run).

    Convenience wrapper around :class:`MXFP8AttentionSM120Wrapper` for single-shot
    use; serving integrations should prefer the wrapper so the per-batch host work
    (padding layout + LPT work lists) is amortized across layers. See
    :meth:`MXFP8AttentionSM120Wrapper.run` for the full contract.

    .. note::
        This function moves ``qo_indptr``/``kv_indptr`` to the host (one sync)
        on every call.
    """
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    wrapper = MXFP8AttentionSM120Wrapper()
    wrapper.plan(qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, causal=causal)
    return wrapper.run(
        q,
        k,
        v,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        out=out,
        lse=lse,
        out_dtype=out_dtype,
    )
