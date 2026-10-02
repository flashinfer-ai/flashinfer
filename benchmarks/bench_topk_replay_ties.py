#!/usr/bin/env python3
"""Replay recorded SGLang decode-length traces through ``top_k_varlen`` backends
under CUDA graphs, with a tie-density knob.

Purpose: measure what SGLang's tie truncation (its fixed 2048-entry tie buffer)
buys it, and whether exactness costs our backends anything on the same rows.
Every backend sees the same logits and lengths; backends are timed
interleaved per step so drift hits all of them alike.

Two modes:

* trace replay (default): steps sampled evenly from a JSON
  ``{"steps": [[compressed_len, ...], ...]}`` file; logits pattern selects the
  tie density (``randn`` = no ties, ``relu<z>`` = ReLU'd Gaussian with exact-zero
  fraction ``z``, so the boundary sits in the zero bin when fewer than k scores
  are positive).
* ``--adv``: adversarial rows whose boundary coarse bin holds far more than 2048
  candidates (``const``, ``onebin``, ``dups50``), where SGLang truncates and an
  exact backend must resolve or fall back.

Backend spec strings: ``sglang``, ``sglang_raw`` (the vendored kernel alone),
``walkfirst`` (the walk-first kernel; its short-row arms serve rows <= 16K),
``walkfirst~`` (``approx_ties=True``, only meaningful once the kernel honours it),
``radix_primitives``, ``radix_primitives~`` (approx_ties=True), ``radix``,
``radix_filter``.  A ``_paged`` suffix (``walkfirst_paged``, ``sglang_paged``,
``sglang_raw_paged``) passes a random injective page table with ``page_size`` =
``TIES_PAGE_SIZE`` (default 64), so the kernel emits physical KV slots (the fused
DSA decode contract) -- the apples-to-apples column against SGLang's own fused
kernel; the slots are mapped back through the table and validated like raw
indices (never as index sets against another run: under ties any subset of the
tied columns is exact).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time

DEV = torch.device("cuda")
REPS = int(os.environ.get("TIES_REPS", "2"))


# ---------------------------------------------------------------------------
# logits patterns
# ---------------------------------------------------------------------------
def make_logits(pattern: str, rows: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(rows, n, device=DEV, generator=g)
    if pattern == "randn":
        return x
    if pattern.startswith("relu"):
        z = float(pattern[4:])
        c = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(z)).item()
        return torch.relu(x - c)  # P(exact zero) == z
    if pattern == "const":
        return torch.ones(rows, n, device=DEV)
    if pattern == "onebin":
        # fp32 values inside ONE fp16 12-bit coarse bin (bin width at 1.0 is
        # 2**-6; we use a 2**-12 span): 2048 distinct values, each repeated
        return 1.0 + torch.rand(rows, n, device=DEV, generator=g) * 2.0**-12
    if pattern == "dups50":
        m = torch.rand(rows, n, device=DEV, generator=g) < 0.5
        return torch.where(m, torch.full_like(x, 5.0), x)
    raise ValueError(pattern)


# ---------------------------------------------------------------------------
# validation: exact multiset, or SGLang's relaxed contract
# ---------------------------------------------------------------------------
def coarse_bin_f32(vals: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp16 (RN) -> ordered 16-bit key -> top 12 bits (the sglang bin)."""
    b16 = vals.half().view(torch.int16).to(torch.int32) & 0xFFFF
    mask = torch.where(
        (b16 & 0x8000) != 0,
        torch.full_like(b16, 0xFFFF),
        torch.full_like(b16, 0x8000),
    )
    return ((b16 ^ mask) & 0xFFFF) >> 4


def validate(logits, lengths, k, idx, relaxed: bool, rows_to_check: int = 16) -> str:
    """'' when the output satisfies the contract, else a short reason."""
    lens = lengths.tolist()
    for r in range(min(rows_to_check, logits.shape[0])):
        ne = min(logits.shape[1], int(lens[r]))
        kk = min(k, ne)
        sel = idx[r]
        if kk == 0:
            if int((sel >= 0).sum()) != 0:
                return f"row{r}: writes in an empty row"
            continue
        valid = sel[sel >= 0]
        if valid.numel() != kk or int(valid.max()) >= ne:
            return f"row{r}: {valid.numel()} valid of {kk}, max {int(valid.max()) if valid.numel() else -1}"
        if valid.unique().numel() != kk:
            return f"row{r}: duplicate indices"
        row = logits[r, :ne]
        ref = torch.topk(row, kk).values
        got = row[valid.long()]
        if relaxed:
            bins = coarse_bin_f32(row)
            bk = coarse_bin_f32(ref[-1].reshape(1))[0]
            if bool((bins[valid.long()] < bk).any()):
                return f"row{r}: picked below the boundary bin"
            above = (bins > bk).nonzero(as_tuple=True)[0]
            if above.numel() and not bool(torch.isin(above, valid.long()).all()):
                return f"row{r}: missed an element above the boundary bin"
        else:
            if not torch.equal(torch.sort(got, descending=True).values, ref):
                return f"row{r}: value multiset differs from torch.topk"
    return ""


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
PAGE_SIZE = int(os.environ.get("TIES_PAGE_SIZE", "64"))


def parse_spec(spec: str):
    approx = spec.endswith("~")
    base = spec[:-1] if approx else spec
    paged = base.endswith("_paged")
    if paged:
        base = base[: -len("_paged")]
    backend = "walkfirst_primitives" if base == "walkfirst" else base
    return backend, approx, paged


def make_page_table(rows: int, n: int, page_size: int, seed: int) -> torch.Tensor:
    """One distinct random physical page per (row, page): injective per row, so
    a physical slot maps back to exactly one column."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    pages = (n + page_size - 1) // page_size
    pool = 4 * pages + 7
    return torch.stack(
        [torch.randperm(pool, generator=g, device=DEV)[:pages] for _ in range(rows)]
    ).to(torch.int32)


def unmap_columns(phys: torch.Tensor, pt: torch.Tensor, page_size: int):
    """Physical slots -> columns through each row's injective table (-1 kept);
    None when a slot's page is not in its row's table."""
    rows, pages = pt.shape
    shift = page_size.bit_length() - 1
    pool = int(pt.max()) + 1
    inv = torch.full((rows, pool), -1, dtype=torch.int64, device=pt.device)
    inv.scatter_(
        1, pt.long(), torch.arange(pages, device=pt.device).expand(rows, pages)
    )
    p64 = phys.to(torch.int64)
    page = inv.gather(1, (p64 >> shift).clamp(0, pool - 1))
    if bool(((page < 0) & (phys >= 0)).any()):
        return None
    col = (page << shift) | (p64 & (page_size - 1))
    return torch.where(phys >= 0, col, p64)


def _sglang_raw_fn(logits, lengths, k, out_i, pt=None, page_size=0):
    """The vendored SGLang kernel alone: lengths pre-clamped, zero offsets
    pre-built, no per-call host tensor ops (the FlashInfer wrapper used to add
    two small device ops per call that cost ~1.5 us each under graph replay).
    With a page table: the varlen entry (per-row lengths derived in-kernel)
    with upstream's fused page-table transform pass."""
    from flashinfer.topk_varlen.topk_varlen import _get_sglang_dsv4_topk_module
    from flashinfer.utils import device_support_pdl

    mod = _get_sglang_dsv4_topk_module(logits.device)
    lens = lengths.clamp(min=0, max=logits.shape[1]).to(torch.int32).contiguous()
    zero = torch.zeros(logits.shape[0], dtype=torch.int32, device=logits.device)
    pdl = device_support_pdl(logits.device)

    if pt is None:

        def fn():
            mod.sglang_dsv4_topk_ragged(logits, lens, zero, out_i, pdl)

    else:

        def fn():
            mod.sglang_dsv4_topk_varlen(logits, lens, out_i, 1, 1, pt, page_size, pdl)

    return fn


def _radix_cutlass_raw_fn(logits, lengths, k, out_i):
    """The shared radix ragged-transform kernel alone, with the per-row lengths,
    zero offsets and row-state buffer prepared once (what `_run_radix_cutlass`
    recomputes per call with three device ops)."""
    from flashinfer.topk import get_topk_module
    from flashinfer.topk_varlen.topk_varlen import _get_cache_buf, _scratch_tag

    n = logits.shape[1]
    lens = lengths.clamp(min=0, max=n).to(torch.int32).contiguous()
    offsets = torch.zeros(logits.shape[0], dtype=torch.int32, device=logits.device)
    row_states = _get_cache_buf(
        f"radix_topk_row_states_{_scratch_tag(logits.device)}",
        1024 * 1024,
        logits.device,
        zero_init=True,
    )
    mod = get_topk_module()

    def fn():
        mod.radix_topk_ragged_transform(
            logits, out_i, offsets, lens, row_states, k, False, 0, False
        )

    return fn


def time_spec(spec, logits, lengths, k, out_i):
    backend, approx, paged = parse_spec(spec)
    try:
        kw = {"backend": backend, "out_indices": out_i}
        if approx:
            kw["approx_ties"] = True
        pt = None
        if paged:
            pt = make_page_table(logits.shape[0], logits.shape[1], PAGE_SIZE, seed=97)
            kw["page_table"] = pt
            kw["page_size"] = PAGE_SIZE

        if backend == "sglang_raw":
            fn = _sglang_raw_fn(
                logits, lengths, k, out_i, pt, PAGE_SIZE if paged else 0
            )
            backend = "sglang"
        elif backend == "radix_cutlass_raw":
            fn = _radix_cutlass_raw_fn(logits, lengths, k, out_i)
            backend = "radix_cutlass"
        else:

            def fn():
                flashinfer.top_k_varlen(logits, lengths, k, **kw)

        fn()
        torch.cuda.synchronize()
        relaxed = approx or backend == "sglang"
        if paged:
            # physical slots: map back through the table, then the usual check
            cols = unmap_columns(out_i, pt, PAGE_SIZE)
            if cols is None:
                return None, "WRONG[paged slot outside the row's table]"
            bad = validate(logits, lengths, k, cols, relaxed)
        else:
            bad = validate(logits, lengths, k, out_i, relaxed)
        if bad:
            return None, f"WRONG[{bad}]"
        best = 1e18
        for _ in range(REPS):
            ts = bench_gpu_time(fn, use_cuda_graph=True, num_iters_within_graph=10)
            best = min(best, sorted(ts)[len(ts) // 2] * 1000.0)
        return best, "ok"
    except Exception as e:  # noqa: BLE001
        return None, f"skip({type(e).__name__}: {str(e)[:80]})"


# ---------------------------------------------------------------------------
def header(args):
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"# device {torch.cuda.get_device_name()} torch {torch.__version__}")
    try:
        import importlib.metadata as md

        print(f"# nvidia-cutlass-dsl {md.version('nvidia-cutlass-dsl')}")
    except Exception:  # noqa: BLE001
        pass
    print(f"# flashinfer {flashinfer.__file__}")
    print(f"# args {vars(args)}")
    print(f"# REPS={REPS}; time = min over reps of the median graph-replay us per call")
    sys.stdout.flush()


def run_trace(args):
    with open(args.steps) as fh:
        steps = json.load(fh)["steps"]
    n_steps = len(steps)
    pick = sorted(
        {
            int(round(i * (n_steps - 1) / max(args.num_steps - 1, 1)))
            for i in range(args.num_steps)
        }
    )
    print(f"# trace {args.steps}: {n_steps} steps, sampled {len(pick)}")
    for pattern in args.patterns:
        print(f"\n=== pattern={pattern} K={args.k} N={args.n} ===")
        print(
            f"{'step':>5s} {'bs':>3s} {'minL':>5s} {'maxL':>5s} | "
            + " ".join(f"{s:>18s}" for s in args.backends)
        )
        acc = {s: [] for s in args.backends}
        status = {s: {} for s in args.backends}
        for si in pick:
            lens = [min(max(int(v), 0), args.n) for v in steps[si]]
            rows = len(lens)
            if rows == 0:
                continue
            logits = make_logits(pattern, rows, args.n, seed=1000 + si)
            lengths = torch.tensor(lens, dtype=torch.int32, device=DEV)
            out_i = torch.empty(rows, args.k, dtype=torch.int32, device=DEV)
            cells = []
            for spec in args.backends:
                t, st = time_spec(spec, logits, lengths, args.k, out_i)
                status[spec][st] = status[spec].get(st, 0) + 1
                if t is not None:
                    acc[spec].append(t)
                    cells.append(f"{t:18.2f}")
                else:
                    cells.append(f"{st[:18]:>18s}")
            print(
                f"{si:5d} {rows:3d} {min(lens):5d} {max(lens):5d} | " + " ".join(cells)
            )
            sys.stdout.flush()
        print("--- mean us/call over sampled steps (equal step weight):")
        ref = statistics.mean(acc["sglang"]) if acc.get("sglang") else None
        for spec in args.backends:
            if acc[spec]:
                m = statistics.mean(acc[spec])
                rel = f"  sglang/this = {ref / m:5.2f}x" if ref else ""
                print(
                    f"  {spec:20s} {m:8.2f} us  (n={len(acc[spec])}){rel}  status={status[spec]}"
                )
            else:
                print(f"  {spec:20s}   n/a   status={status[spec]}")
        sys.stdout.flush()


def run_adv(args):
    rows = args.adv_rows
    for pattern in args.patterns:
        for L in args.adv_lengths:
            L = min(L, args.n)
            print(
                f"\n=== ADV pattern={pattern} K={args.k} N={args.n} rows={rows} L={L} ==="
            )
            logits = make_logits(pattern, rows, args.n, seed=7)
            lengths = torch.full((rows,), L, dtype=torch.int32, device=DEV)
            out_i = torch.empty(rows, args.k, dtype=torch.int32, device=DEV)
            for spec in args.backends:
                t, st = time_spec(spec, logits, lengths, args.k, out_i)
                cell = f"{t:8.2f} us" if t is not None else "   n/a   "
                print(f"  {spec:20s} {cell}  {st}")
                sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--steps",
        default=os.environ.get("TOPK_TRACE_STEPS"),
        help='JSON {"steps": [[compressed_len, ...], ...]} recorded from the serving engine '
        "(one list per decode step); default from $TOPK_TRACE_STEPS",
    )
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument(
        "--patterns", nargs="+", default=["randn", "relu0.5", "relu0.75", "relu0.9"]
    )
    ap.add_argument(
        "--backends",
        nargs="+",
        default=[
            "sglang",
            "walkfirst",
            "radix_primitives",
            "radix_primitives~",
        ],
    )
    ap.add_argument("--adv", action="store_true")
    ap.add_argument("--adv-rows", type=int, default=16)
    ap.add_argument(
        "--adv-lengths", nargs="+", type=int, default=[2048, 4096, 8192, 16384]
    )
    args = ap.parse_args()
    if not args.adv and not args.steps:
        ap.error("--steps (or $TOPK_TRACE_STEPS) is required for trace replay")
    header(args)
    if args.adv:
        run_adv(args)
    else:
        run_trace(args)


if __name__ == "__main__":
    main()
