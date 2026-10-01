#!/usr/bin/env python3
"""Empirically pin down the approximation in the vendored SGLang DSv4 top-k.

The kernel (csrc/sglang_dsv4/sgl_kernel/deepseek_v4/topk_impl.cuh) bins every
score by its fp16 round-to-nearest image, top 12 ordered bits (4096 bins), finds
the bin B holding the k-th largest, emits everything above B exactly, and stores
at most kMaxNumTie = 2048 members of B (first arrival) before an exact radix
select among the stored ones.  Prediction: exact iff |B| <= 2048; otherwise the
output holds at most min(k - A, |B| - 2048) indices that are not in the true
top-k (A = count above B), each of them still inside B.  Every run below prints
the observed worst per-row count of selected values STRICTLY below the true
k-th value (``below_kth``) next to that bound, plus the relaxed-contract check
(nothing below B, everything above B present) and a determinism check.  The
count is by value, not by index set against torch.topk: under ties an exact
backend may legitimately pick other members of the k-th value.
"""

from __future__ import annotations

import sys

import torch

import flashinfer

DEV = torch.device("cuda")


def coarse_bin_f32(vals):
    b16 = vals.half().view(torch.int16).to(torch.int32) & 0xFFFF
    mask = torch.where(
        (b16 & 0x8000) != 0, torch.full_like(b16, 0xFFFF), torch.full_like(b16, 0x8000)
    )
    return ((b16 ^ mask) & 0xFFFF) >> 4


def run(backend, logits, lengths, k, approx=False):
    out = torch.empty(logits.shape[0], k, dtype=torch.int32, device=DEV)
    kw = {"backend": backend, "out_indices": out}
    if approx:
        kw["approx_ties"] = True
    flashinfer.top_k_varlen(logits, lengths, k, **kw)
    torch.cuda.synchronize()
    return out


def analyse(tag, logits, lengths, k, backend, approx=False):
    try:
        out1 = run(backend, logits, lengths, k, approx)
        out2 = run(backend, logits, lengths, k, approx)
    except Exception as e:  # noqa: BLE001
        print(
            f"  {tag:34s} {backend + ('~' if approx else ''):20s} skip({type(e).__name__}: {str(e)[:60]})"
        )
        return
    lens = lengths.tolist()
    worst_below_kth = 0
    worst_bound = 0
    below = 0
    missed_above = 0
    dups = 0
    pads = 0
    nondet_rows = 0
    for r in range(logits.shape[0]):
        ne = min(logits.shape[1], lens[r])
        kk = min(k, ne)
        row = logits[r, :ne]
        sel = out1[r]
        valid = sel[sel >= 0].long()
        pads += int((sel < 0).sum())
        if valid.numel() and valid.unique().numel() != valid.numel():
            dups += valid.numel() - valid.unique().numel()
        if not torch.equal(out1[r], out2[r]):
            nondet_rows += 1
        if kk == 0:
            continue
        finite_row = torch.nan_to_num(row, nan=float("inf"))  # torch ranks NaN on top
        ref_vals = torch.topk(finite_row, kk).values
        kth = ref_vals[-1]
        bins = coarse_bin_f32(finite_row)
        bk = coarse_bin_f32(kth.reshape(1))[0]
        above = int((bins > bk).sum())
        in_bin = int((bins == bk).sum())
        bound = min(max(kk - above, 0), max(in_bin - 2048, 0))
        # by value, not index set: tied k-th members are interchangeable
        below_kth = int((finite_row[valid] < kth).sum())
        worst_below_kth = max(worst_below_kth, below_kth)
        worst_bound = max(worst_bound, bound)
        if valid.numel():
            below += int((bins[valid] < bk).sum())
            above_idx = (bins > bk).nonzero(as_tuple=True)[0]
            if above_idx.numel():
                missed_above += int((~torch.isin(above_idx, valid)).sum())
    name = backend + ("~" if approx else "")
    print(
        f"  {tag:34s} {name:20s} worst_below_kth={worst_below_kth:4d} (bound {worst_bound:4d})"
        f"  below_bin={below} missed_above={missed_above} dups={dups} pads={pads} nondeterministic_rows={nondet_rows}"
    )
    sys.stdout.flush()


def main():
    print(f"# device {torch.cuda.get_device_name()}  flashinfer {flashinfer.__file__}")
    torch.manual_seed(3)
    N = 16384
    backends = [
        ("sglang", False),
        ("radix_primitives", False),
        ("radix_primitives", True),
        ("walkfirst_primitives", False),
    ]

    # A: one fp16 coarse bin holding 2048 distinct fp32 values (the span 2**-12
    # at 1.0 is 2**11 fp32 ulps), each repeated ~L/2048 times; |B| = L, A = 0
    for k in (512, 2048):
        for L in (2048, 2049, 2304, 4096, 8192):
            rows = 8
            logits = torch.full((rows, N), -3.0, device=DEV)
            logits[:, :L] = 1.0 + torch.rand(rows, L, device=DEV) * 2.0**-12
            for r in range(rows):
                logits[r] = logits[r][torch.randperm(N, device=DEV)]
            # lengths cover the whole row so every in-bin value is a candidate
            lengths = torch.full((rows,), N, dtype=torch.int32, device=DEV)
            tag = f"A onebin |B|={L} k={k}"
            for be, ap in backends:
                analyse(tag, logits.contiguous(), lengths, k, be, ap)

    # B: mixed row: |B| = 2100 in-bin values, plus 300 values ABOVE the bin (A = 300)
    k = 512
    rows = 8
    logits = torch.randn(rows, N, device=DEV) * 0.01  # tiny background, far below
    logits[:, :2100] = 1.0 + torch.rand(rows, 2100, device=DEV) * 2.0**-12
    logits[:, 2100:2400] = 2.0 + torch.rand(rows, 300, device=DEV)
    for r in range(rows):
        logits[r] = logits[r][torch.randperm(N, device=DEV)]
    lengths = torch.full((rows,), N, dtype=torch.int32, device=DEV)
    for be, ap in backends:
        analyse("B |B|=2100 A=300 k=512", logits.contiguous(), lengths, k, be, ap)

    # C: values above the fp16 range (> 65504 all alias to the +inf bin)
    logits = torch.randn(rows, N, device=DEV)
    logits[:, :3000] = 70000.0 + torch.rand(rows, 3000, device=DEV) * 10000.0
    for r in range(rows):
        logits[r] = logits[r][torch.randperm(N, device=DEV)]
    for be, ap in backends:
        analyse("C 3000 values > 65504, k=512", logits.contiguous(), lengths, k, be, ap)

    # D: NaNs present (100 per row); torch ranks NaN first
    logits = torch.randn(rows, N, device=DEV)
    logits[:, :100] = float("nan")
    for r in range(rows):
        logits[r] = logits[r][torch.randperm(N, device=DEV)]
    for be, ap in backends:
        analyse("D 100 NaN per row, k=512", logits.contiguous(), lengths, k, be, ap)

    # E: realistic: randn rows at trace-like lengths (no ties expected)
    logits = torch.randn(rows, N, device=DEV)
    lengths = torch.tensor(
        [505, 829, 1109, 1482, 1825, 2085, 4096, 16384], dtype=torch.int32, device=DEV
    )
    for be, ap in backends:
        analyse(
            "E randn, trace lengths, k=512", logits.contiguous(), lengths, k, be, ap
        )

    # F: ReLU-sparse rows (exact zeros at the boundary) at trace-like lengths
    z = 0.9
    c = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(z)).item()
    logits = torch.relu(torch.randn(rows, N, device=DEV) - c)
    for be, ap in backends:
        analyse(
            "F relu z=0.9, trace lengths, k=512",
            logits.contiguous(),
            lengths,
            k,
            be,
            ap,
        )


if __name__ == "__main__":
    main()
