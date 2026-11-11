# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Thin adapter for the ResInfer production BF16-output decode path."""

import torch

from .host import run_paged_fp4_decode_qmul4_tmem
from .kv_layout import NvFp4PagedKVPool


def decode(q, k, v, ks, vs, tables, lengths, sm_scale, key_scale, value_scale, splits):
    ks, vs = ks.view(torch.float8_e4m3fn), vs.view(torch.float8_e4m3fn)
    # The donor takes a four-dimensional view of interleaved V scale bytes.
    vs = vs.view(*k.shape[:3], q.shape[-1] // 16)
    pool = NvFp4PagedKVPool(k.shape[1], q.shape[-1], k, v, ks, vs)
    out = torch.empty_like(q)
    return run_paged_fp4_decode_qmul4_tmem(
        q,
        pool,
        v,
        vs,
        tables,
        lengths,
        sm_scale * key_scale,
        out,
        value_scale=value_scale,
        plan=(splits, "none" if splits == 1 else "kernel"),
    )
