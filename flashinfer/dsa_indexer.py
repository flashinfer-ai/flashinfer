"""
Copyright (c) 2024 by FlashInfer team.

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
from typing import Optional

import torch

from .jit.dsa_indexer import gen_dsa_indexer_module

__all__ = [
    "dsa_indexer_topk",
    "seed_prep",
    "scan",
    "select",
    "get_dsa_indexer_module",
]


@functools.cache
def get_dsa_indexer_module():
    return gen_dsa_indexer_module().build_and_load()


def dsa_indexer_topk(
    sample_logits: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_end: torch.Tensor,
    top_k: int,
    *,
    num_buckets: int = 256,
    cand_cap: int = 32768,
    refresh_every: int = 64,
) -> torch.Tensor:
    dev = q.device
    num_q = q.shape[0]
    origin = torch.empty(num_q, dtype=torch.float32, device=dev)
    inv_delta = torch.empty(num_q, dtype=torch.float32, device=dev)
    th_bucket = torch.empty(num_q, dtype=torch.int32, device=dev)
    bcount = torch.zeros(num_q, num_buckets, dtype=torch.int32, device=dev)
    cand_val = torch.empty(num_q, cand_cap, dtype=torch.float32, device=dev)
    cand_idx = torch.empty(num_q, cand_cap, dtype=torch.int32, device=dev)
    cand_cnt = torch.empty(num_q, dtype=torch.int32, device=dev)
    seed_prep(
        sample_logits, num_buckets, top_k, cand_cap, origin, inv_delta,
        th_bucket, bcount, cand_val, cand_idx, cand_cnt,
    )
    cu_start = torch.zeros(num_q, dtype=torch.int32, device=dev)
    scan(
        q, kv, kv_scales, weights, cu_start, cu_end, origin, inv_delta,
        th_bucket, cand_val, cand_idx, cand_cnt, bcount, num_buckets, top_k,
        refresh_every=refresh_every,
    )
    return select(cand_val, cand_idx, cand_cnt, th_bucket, num_buckets, top_k)


def seed_prep(
    slog: torch.Tensor,
    num_buckets: int,
    top_k: int,
    cand_cap: int,
    origin: torch.Tensor,
    inv_delta: torch.Tensor,
    th_bucket: torch.Tensor,
    bcount: torch.Tensor,
    cand_val: torch.Tensor,
    cand_idx: torch.Tensor,
    cand_cnt: torch.Tensor,
    headroom: float = 0.0,
    probe_stride_tok: int = 0,
    hist_stride: int = 1,
) -> None:
    get_dsa_indexer_module().seed_prep(
        slog, num_buckets, top_k, cand_cap, 0, float(headroom),
        probe_stride_tok, hist_stride, origin, inv_delta, th_bucket, bcount,
        cand_val, cand_idx, cand_cnt,
    )


def scan(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_start: torch.Tensor,
    cu_end: torch.Tensor,
    origin: torch.Tensor,
    inv_delta: torch.Tensor,
    th_bucket: torch.Tensor,
    cand_val: torch.Tensor,
    cand_idx: torch.Tensor,
    cand_cnt: torch.Tensor,
    bcount: torch.Tensor,
    num_buckets: int,
    top_k: int,
    refresh_every: int = -1,
    num_kv_splits: int = -1,
    probe_group: int = 0,
    probe_add_max: int = 0,
) -> None:
    get_dsa_indexer_module().scan(
        q, kv, kv_scales, weights, cu_start, cu_end, origin, inv_delta, th_bucket,
        cand_val, cand_idx, cand_cnt, bcount, num_buckets, top_k, refresh_every,
        num_kv_splits, probe_group, probe_add_max,
    )


def select(
    cand_val: torch.Tensor,
    cand_idx: torch.Tensor,
    cand_cnt: torch.Tensor,
    th_bucket: torch.Tensor,
    num_buckets: int,
    top_k: int,
    out_idx: Optional[torch.Tensor] = None,
    out_val: Optional[torch.Tensor] = None,
    probe_group: int = 0,
    probe_add_max: int = 0,
) -> torch.Tensor:
    R = cand_val.shape[0]
    dev = cand_val.device
    if out_idx is None:
        out_idx = torch.empty(R, top_k, dtype=torch.int32, device=dev)
    if out_val is None:
        out_val = torch.empty(R, top_k, dtype=torch.float32, device=dev)
    zero = torch.zeros(R, dtype=torch.float32, device=dev)
    one = torch.ones(R, dtype=torch.float32, device=dev)
    get_dsa_indexer_module().select(
        cand_val, cand_idx, cand_cnt, zero, one, th_bucket, num_buckets,
        top_k, probe_group, probe_add_max, None, out_val, out_idx,
    )
    return out_idx
