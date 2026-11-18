# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``backend="specialized"`` plan for :class:`FP8MQATopKWrapper`.

Serves the wrapper's selective continuous-K TopK contract with the
agent-generated SM103 kernel behind
:func:`flashinfer.attn_scores.fp8_paged_mqa_topk_indexer`, and extends the
wrapper to MULTI-REQUEST batches (``qo_indptr``/``kv_indptr`` of shape
``[B + 1]``), which the ``cutedsl`` backend refuses.

Contract and restrictions (validated eagerly, never silently relaxed):

- FP8 only (``qkv_dtype="fp8"``), ``num_heads=4``, ``head_dim=128``,
  ``top_k=512``, ``compression_ratio=4``, fp32 logits/weights,
  ``math_mode`` None or ``"relu_strict"`` (scoring is fp32 accumulation
  with per-head relu).
- FULL-PREFILL chunks only: every request must have ``q_len == kv_len``
  (all of a request's tokens are query rows, the kernel's campaign
  contract — its per-row block space is sized from the query extent).
  Chunked prefill (a short query window over a longer KV context) is
  rejected at ``plan``; measured on GB300, the kernel's visible-block
  coverage degrades to the query extent on such calls, so serving them
  would be silently wrong.
- ``k_scale`` must be all-ones and per-row head ``weights`` must be a
  positive per-row constant: the kernel scores ``sum_h relu(q_h . k)``,
  and a per-row positive constant scales every candidate's score equally,
  leaving the top-k selection identical. Checked eagerly (outside
  capture); captured replays assume the precondition holds.
- Key storage must hold a multiple of 64 rows (the kernel's page size in
  compressed keys), and every request's KV segment must start at a
  multiple of 256 original tokens — prefixes are PACKED, so every
  ``kv_indptr`` value except the final one must be a multiple of 256
  (64 compressed keys): the continuous
  key storage is then addressable zero-copy through the kernel's 64-block
  page table, and K mutations after ``prepare_pool`` stay visible.
- ``plan`` performs a host sync (prefixes are read on host) and is NOT
  capturable on this backend; ``run`` is capturable after an eager
  warmup. ``host_length_hints`` is rejected (the ``cutedsl`` backend
  honors it; ``backend="auto"`` demotes on the rejection rather than
  silently ignoring the bound).
- ``plan`` releases the previous plan's device tensors (page table,
  prefix metadata, scratch). A CUDA graph captured against an earlier
  plan holds raw addresses into that storage, so discard such graphs
  before replanning; replaying one across a replan is undefined. This is
  the wrapper rule — keep capture inputs alive until the caller discards
  its graphs — applied to plan-owned tensors: states are not retained
  across replans because that retention would grow without bound.
- Environment validation mirrors the functional entry's dispatch
  guards AT CONSTRUCTION: unvalidated CUDA toolkits (torch build or JIT
  nvcc, outside ``VALIDATED_CUDA_VERSIONS``), the kill switch, and
  non-SM103 devices all raise here, so ``backend="auto"`` demotes instead
  of resolving to a plan whose every ``run`` would refuse.

Output follows the wrapper contract: shared int32 ``[Q, top_k]`` storage
of request-local COMPRESSED-KEY indices, ascending, ``-1`` padded. The
functional entry's appended open-block causal tail (which is not a
complete compressed key) is excluded.
"""

from __future__ import annotations

import torch

import os as _os

from .prefill_indexer import (
    KILL_SWITCH_ENV,
    PACKED_WIDTH,
    VALIDATED_CUDA_VERSIONS,
    _toolkit_validation_reasons,
    fp8_paged_mqa_topk_indexer,
    get_fp8_paged_mqa_topk_indexer_module,
)

_RATIO = 4
_PAGE_BLOCKS = 64
_TOPK = 512
_SENTINEL = torch.iinfo(torch.int32).max


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


class SpecializedTopKPlan:
    """Plan/run lifecycle for the specialized wrapper backend."""

    def __init__(
        self,
        *,
        max_context_length,
        max_chunk_size,
        max_batch_size,
        device,
        num_heads,
        head_dim,
        top_k,
        compression_ratio,
        math_mode,
        num_keys,
        qkv_dtype,
        logits_dtype,
        weights_dtype,
        model_config,
    ):
        # Environment gates mirror the functional entry's dispatch guards,
        # applied at construction so an unservable environment fails here --
        # which lets backend="auto" demote to cutedsl instead of building a
        # sticky plan whose every run() raises.
        _require(
            not _toolkit_validation_reasons(),
            "backend='specialized' requires a validated CUDA toolkit "
            f"(cuda={torch.version.cuda}, validated={VALIDATED_CUDA_VERSIONS}; "
            "the JIT nvcc is gated identically)",
        )
        _require(
            _os.environ.get(KILL_SWITCH_ENV, "0") != "1",
            f"backend='specialized' is disabled ({KILL_SWITCH_ENV}=1)",
        )
        _dev = torch.device(device)
        _require(
            _dev.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(_dev) == (10, 3),
            f"backend='specialized' requires an SM103 CUDA device (device={device})",
        )
        _require(qkv_dtype == "fp8", "backend='specialized' supports qkv_dtype='fp8'")
        _require(num_heads == 4, "backend='specialized' supports num_heads=4")
        _require(head_dim == 128, "backend='specialized' supports head_dim=128")
        _require(top_k == _TOPK, "backend='specialized' supports top_k=512")
        _require(
            compression_ratio == _RATIO,
            "backend='specialized' supports compression_ratio=4",
        )
        _require(
            math_mode in (None, "relu_strict"),
            "backend='specialized' supports math_mode None or 'relu_strict'",
        )
        _require(
            logits_dtype == torch.float32 and weights_dtype == torch.float32,
            "backend='specialized' supports fp32 logits and weights",
        )
        _require(model_config is None, "backend='specialized' takes model_config=None")
        self._device = torch.device(device)
        self._max_context_length = int(max_context_length)
        self._max_chunk_size = int(max_chunk_size)
        self._max_batch_size = int(max_batch_size)
        self._num_keys = (
            int(num_keys)
            if num_keys is not None
            else self._max_context_length // _RATIO
        )
        self.math_mode = "relu_strict"
        self.error = 0
        self.repair_error = 0
        self.active_shape = None
        # prepared K bindings keyed by data_ptr: padded paged views
        # (NCP, 64, 1, 128). data_ptr is host metadata, so run's binding
        # lookup stays safe during graph capture.
        self._pools = {}
        self._state = None  # per-plan device tensors
        self._out = None  # shared output storage

    # -- lifecycle ---------------------------------------------------------

    def prepare_pool(self, k, *, k_scale):
        _require(
            k.is_cuda and k.dtype == torch.float8_e4m3fn and k.dim() == 2,
            "specialized prepare_pool expects CUDA fp8 K of shape [N, 128]",
        )
        _require(
            k.shape[1] == 128 and k.is_contiguous(), "K must be [N, 128] contiguous"
        )
        _require(
            k_scale.is_cuda and k_scale.dtype == torch.float32,
            "specialized prepare_pool expects fp32 k_scale",
        )
        _require(
            not torch.cuda.is_current_stream_capturing(),
            "prepare_pool is not capturable",
        )
        _require(
            bool(torch.all(k_scale == 1.0).item()),
            "backend='specialized' requires unit k_scale (raw fp8 scoring)",
        )
        _require(
            k.shape[0] % _PAGE_BLOCKS == 0,
            "backend='specialized' requires key storage in multiples of 64 "
            "rows (zero-copy paged view; pad the storage allocation)",
        )
        self._pools[k.data_ptr()] = k.view(-1, _PAGE_BLOCKS, 1, 128)
        return None

    def clear_pools(self):
        self._pools = {}
        return None

    def input_buffers(self):
        raise NotImplementedError(
            "backend='specialized' takes q/weights directly at run()"
        )

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        *,
        num_query_rows=None,
        next_n=None,
        host_length_hints=None,
    ):
        _require(
            not torch.cuda.is_current_stream_capturing(),
            "backend='specialized' plan is not capturable (host sync)",
        )
        _require(self._pools, "prepare_pool must run before plan")
        _require(
            host_length_hints is None,
            "backend='specialized' does not serve host_length_hints "
            "(the cutedsl backend honors them; this rejection lets "
            "backend='auto' demote instead of silently ignoring the bound)",
        )
        # Resolve the JIT module now (one-time build, cached): a validated
        # nvcc can report its version and still fail to build the module,
        # and discovering that at run() is too late for backend='auto' to
        # demote to a working backend.
        try:
            get_fp8_paged_mqa_topk_indexer_module()
        except Exception as e:
            raise ValueError(
                f"backend='specialized' kernel module unavailable ({type(e).__name__})"
            ) from e
        qo = qo_indptr.to("cpu", torch.int64).tolist()
        kv = kv_indptr.to("cpu", torch.int64).tolist()
        _require(
            len(qo) == len(kv) and len(qo) >= 2, "qo/kv_indptr must share shape [B+1]"
        )
        # Per-request checks keep both arrays strictly increasing, so
        # non-negative first values make every value non-negative. A
        # negative KV start passes the 256-alignment check (-256 % 256 ==
        # 0) and would plan a negative page number addressing outside the
        # prepared pool.
        _require(qo[0] >= 0 and kv[0] >= 0, "qo/kv_indptr must be non-negative")
        b = len(qo) - 1
        _require(b <= self._max_batch_size, "batch exceeds max_batch_size")
        dev = self._device
        pos_rows, vis_rows, cu, table_rows = [], [], [0], []
        max_pages = 0
        for r in range(b):
            q_len = qo[r + 1] - qo[r]
            kv_len = kv[r + 1] - kv[r]
            _require(q_len >= 1 and kv_len >= q_len, "invalid prefixes")
            _require(
                q_len == kv_len,
                "backend='specialized' serves full-prefill chunks only "
                "(q_len == kv_len per request); chunked prefill is outside "
                "the kernel's validated contract",
            )
            _require(q_len <= self._max_chunk_size, "q_len exceeds max_chunk_size")
            _require(
                kv_len <= self._max_context_length,
                "kv_len exceeds max_context_length",
            )
            _require(
                kv[r] % (_RATIO * _PAGE_BLOCKS) == 0,
                "specialized plans require each request's KV segment to "
                "start at a multiple of 256 original tokens (64 keys)",
            )
            if next_n is not None:
                _require(q_len == next_n, "next_n mismatch")
            key_start = kv[r] // _RATIO
            n_keys = kv_len // _RATIO  # complete compressed keys only
            # every prepared binding must cover the plan: run() may be
            # called with any of them, so check the smallest
            _require(
                key_start + n_keys
                <= min(p.shape[0] for p in self._pools.values()) * _PAGE_BLOCKS,
                "prepared pool too small for planned prefixes",
            )
            # request-local original-token positions of the query rows
            first = kv_len - q_len
            pos_rows.append(torch.arange(first, kv_len, dtype=torch.int32))
            vis_rows.append(((pos_rows[-1] + 1) // _RATIO).to(torch.int32))
            cu.append(cu[-1] + q_len)
            n_pages = max(1, -(-n_keys // _PAGE_BLOCKS))
            table_rows.append(
                torch.arange(n_pages, dtype=torch.int32) + key_start // _PAGE_BLOCKS
            )
            max_pages = max(max_pages, n_pages)
        table = torch.zeros(b, max_pages, dtype=torch.int32)
        for r, row in enumerate(table_rows):
            table[r, : row.shape[0]] = row
        t = cu[-1]
        if num_query_rows is not None:
            _require(t <= int(num_query_rows), "live rows exceed num_query_rows")
        self._state = dict(
            T=t,
            B=b,
            pos=torch.cat(pos_rows).to(dev),
            visible=torch.cat(vis_rows).to(dev),
            cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=dev),
            page_table=table.to(dev),
            scratch=torch.empty(t, PACKED_WIDTH, dtype=torch.int32, device=dev),
        )
        self.active_shape = (t, b)
        return None

    def run(self, q, weights, k, *, k_scale, out=None, q_scale=None):
        _require(self._state is not None, "plan must run before run")
        _require(q_scale is None, "q_scale must be None for FP8")
        st = self._state
        t = st["T"]
        _require(
            q.is_cuda and q.dtype == torch.float8_e4m3fn and q.dim() == 3,
            "q must be CUDA fp8 [Q, 4, 128]",
        )
        _require(q.shape[0] >= t and q.shape[1] == 4 and q.shape[2] == 128, "bad q")
        # binding lookup by data_ptr (host metadata, capture-safe): each
        # prepared K stays addressable, and an unknown binding is rejected
        # in captured runs too, instead of recording another binding's keys
        pool = self._pools.get(k.data_ptr())
        _require(pool is not None, "run must use a prepared K binding")
        if not torch.cuda.is_current_stream_capturing():
            _require(
                bool(torch.all(k_scale == 1.0).item()),
                "backend='specialized' requires unit k_scale at run as well",
            )
            _require(
                weights.is_cuda
                and weights.dtype == torch.float32
                and weights.shape[0] >= t
                and bool(
                    torch.all(weights[:t] == weights[:t, :1]).item()
                    and torch.all(weights[:t, 0] > 0).item()
                ),
                "backend='specialized' requires a positive per-row-constant "
                "weights matrix (selection is then weight-invariant)",
            )
        packed = fp8_paged_mqa_topk_indexer(
            q[:t],
            pool,
            st["page_table"],
            st["cu_seqlens"],
            st["pos"],
            st["visible"],
            out=st["scratch"],
        )
        # derive ascending compressed-key indices: one per selected complete
        # block (its first expanded token), excluding the open-block tail
        toks = packed[:, : PACKED_WIDTH - 1]
        ids = toks >> 2
        mask = (toks >= 0) & ((toks & 3) == 0) & (ids < st["visible"].unsqueeze(1))
        cand = torch.where(mask, ids, torch.full_like(ids, _SENTINEL))
        sel = cand.sort(dim=1).values[:, :_TOPK]
        if out is None:
            # the wrapper contract returns the full supplied Q extent
            # ([Q, top_k] with -1 in the inactive rows), not just the
            # plan's t live rows
            nq = q.shape[0]
            if self._out is None or self._out.shape[0] < nq:
                self._out = torch.empty(
                    nq, _TOPK, dtype=torch.int32, device=self._device
                )
            out = self._out[:nq]
        # out may carry more rows than the plan's t live rows (the wrapper
        # contract allows a [Q, top_k] destination with Q >= t): write the
        # live rows and pad the rest, instead of a shape-mismatched copy_.
        live = torch.where(sel != _SENTINEL, sel, torch.full_like(sel, -1))
        out[:t].copy_(live)
        out[t:].fill_(-1)
        return out
