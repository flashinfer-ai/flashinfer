# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import functools
import math

from .launch import current_arch
from .utils import (
    CP_ONLY_KERNELS,
    INFER_KERNELS,
    MIN_LCP,
    TRAIN_KERNELS,
    StructFeatures,
    load_coefs,
    predict_kernel,
    seq_chunks,
    struct_features,
    warmup_from_gate_avg,
)

MAX_EVALS = 3


@functools.lru_cache(maxsize=4)
def _default_coefs(path: str | None, arch: str) -> dict:
    return load_coefs(path, arch=arch)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(
    coefs: dict, feat: StructFeatures, is_train: bool = True, baseline: bool = False
) -> float:
    priced = TRAIN_KERNELS if is_train else INFER_KERNELS
    return sum(
        predict_kernel(coefs, feat, k)
        for k in priced
        if not (baseline and k in CP_ONLY_KERNELS)
    )


def wave_candidates(chunks, H: int, P: int, min_S: int = MIN_LCP):
    cmax, total = max(chunks), sum(chunks)
    if min_S > cmax:
        return
    last = None  # consecutive waves can share an S once the steps get small
    for w in range(1, max(1, math.ceil(total * H / (min_S * P))) + 1):
        s = min(cmax, max(min_S, math.ceil(total * H / (w * P))))
        if last is None or s < last:
            last = s
            yield s
        if s <= min_S:
            return


def decide(
    cu_seqlens=None,
    seq_lens=None,
    num_chunks=None,
    num_v_heads=None,
    coefs=None,
    coefs_path=None,
    P=None,
    chunk=None,
    margin=0.05,
    is_train=True,
    min_S=MIN_LCP,
    max_evals=MAX_EVALS,
    candidates=None,
    prune=True,
    debug=False,
    warmup_per_head=None,
    g=None,
    arch=None,
):
    if coefs is None:
        coefs = _default_coefs(coefs_path, arch or current_arch())
    if P is None:
        P = coefs["P"]
    if chunk is None:
        chunk = coefs["chunk"]
    if num_v_heads is None:
        raise ValueError("num_v_heads is required")

    if num_chunks is not None:
        chunks = [int(c) for c in num_chunks]
    elif cu_seqlens is not None:
        chunks = seq_chunks(cu_seqlens, chunk)
    elif seq_lens is not None:
        chunks = [math.ceil(int(x) / chunk) for x in seq_lens]
    else:
        raise ValueError("provide one of cu_seqlens / seq_lens / num_chunks")
    if not chunks or min(chunks) <= 0:
        raise ValueError(f"invalid sequence lengths: {chunks}")

    H, cmax = int(num_v_heads), max(chunks)
    if warmup_per_head is None:
        if g is not None:
            avg_per_head = g.float().mean(dim=1).reshape(-1).tolist()
            warmup_per_head = warmup_from_gate_avg(avg_per_head, chunk)
        else:
            warmup_per_head = [math.inf] * H
    feat_kw = dict(
        warmup_per_head=warmup_per_head, is_train=is_train, backend=coefs["backend"]
    )
    base_feat = struct_features(chunks, cmax, H, P, **feat_kw)
    base_t = predict(coefs, base_feat, is_train=is_train, baseline=True)

    if candidates is not None:
        cands, budget = sorted(set(candidates)), len(candidates)
    else:
        cands, budget = wave_candidates(chunks, H, P, min_S), max(1, max_evals)

    sweep, best_feat, best_t = [], None, float("inf")
    for S in cands:
        feat = struct_features(chunks, S, H, P, **feat_kw)
        t = predict(coefs, feat, is_train=is_train, baseline=not feat.enable_cp)
        sweep.append((S, t))
        if feat.enable_cp and t < best_t:
            best_feat, best_t = feat, t
        if best_feat is None:
            continue
        if len(sweep) >= budget:
            break
    sweep.sort()

    if best_feat is None:
        use_cp, best_lcp = False, None
    else:
        use_cp, best_lcp = best_t < base_t * (1.0 - margin), int(best_feat.S)
    lcp = best_lcp if use_cp else None

    if not debug:
        return use_cp, lcp
    return {
        "use_cp": use_cp,
        "lcp": lcp,
        "best_cp_lcp": best_lcp,
        "best_cp_ms": best_t,  # inf when no CP point exists
        "nocp_pred_ms": base_t,
        "sweep": sweep,
    }
