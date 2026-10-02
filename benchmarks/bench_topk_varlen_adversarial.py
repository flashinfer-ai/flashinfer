"""Adversarial top_k_varlen perf suite with output-validation fairness gate.

Hunts configs/distributions where radix_primitives loses >3% to ANY other
backend (sglang, radix DSL, GVR-with-oracle, radix_cutlass).  Every
backend's output is validated against torch.topk per cell on the first
``rows_to_check`` (4) rows only: exact backends must reproduce the exact
top-k value multiset, the vendored sglang kernel (approximate on tie
overflows by design) is held to its relaxed tie contract (see ``validate``).
A violation is reported as DNF(wrong) and excluded from the loss comparison.

Usage:
    python benchmarks/bench_topk_varlen_adversarial.py all   # P1..P6
    python benchmarks/bench_topk_varlen_adversarial.py p1    # one family
    FI_ONLY_PRIM=1 ... p1     # time only radix_primitives

Attack families target the kernel's structural cliffs:
  P1 tie-overflow distributions (refine re-reads: constant/one-bin/2-valued/
     quantized/inf-flood at large N)
  P2 dispatch-gate bands (just below the MC gate at 65536; just above the
     scan-collect gate at 32768)
  P3 wave-quantization batches (just past 148 SMs and past 296)
  P4 extreme ragged skew (one full row + many tiny rows, large batch)
  P5 tiny-N mega-batch (CTA over-provisioning)
  P6 k:N pressure (k = N/2)

All backends measured through the public API (uniform overhead), CUDA-graph
timing, min-of-R protocol.  Reports every cell where prim is >3% slower than
the best competitor.
"""

import os
import sys

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time

DEV = "cuda"
LOSS_THRESH = 1.03
REPS = 5
FAILS = []
FAMILIES = ("all", "p1", "p2", "p3", "p4", "p5", "p6")
ONLY_PRIM = os.environ.get("FI_ONLY_PRIM") == "1"


def gen(pattern, rows, N, dtype, k, seed):
    torch.manual_seed(seed)
    if pattern == "randn":
        x = torch.randn(rows, N, device=DEV) * 2.0
    elif pattern == "constant":
        x = torch.full((rows, N), 1.2345, device=DEV)
    elif pattern == "two_values":
        x = torch.where(torch.rand(rows, N, device=DEV) < 0.5, 1.0, -1.0)
    elif pattern == "one_bin":
        x = 1.0 + torch.rand(rows, N, device=DEV) * 2**-12
    elif pattern == "quantized4":
        x = torch.randint(0, 4, (rows, N), device=DEV).float()
    elif pattern == "inf_flood":
        x = torch.randn(rows, N, device=DEV)
        x = torch.where(
            torch.rand(rows, N, device=DEV) < 0.2,
            torch.full_like(x, float("inf")),
            x,
        )
    elif pattern == "tiecap_edge":
        # exactly ~TIE_CAP+few elements in the threshold bin
        x = torch.randn(rows, N, device=DEV) * 2.0
        ranks = torch.rand(rows, N, device=DEV).argsort(-1).argsort(-1)
        x = torch.where(ranks < k - 1, torch.full_like(x, 100.0), x)
        x = torch.where(
            (ranks >= k - 1) & (ranks < k - 1 + 2100), torch.full_like(x, 50.0), x
        )
    else:
        raise ValueError(pattern)
    return x.to(dtype).contiguous()


def oracle_pre_idx(logits, seq_lens, k):
    lf = logits.float()
    g = seq_lens.shape[0]
    pre = torch.zeros(g, k, dtype=torch.int32, device=DEV)
    sl = seq_lens.cpu().tolist()
    N = logits.shape[1]
    for i in range(g):
        ne = min(N, int(sl[i]))
        kk = min(k, ne)
        pre[i, :kk] = torch.topk(lf[i, :ne], kk).indices.int() - 1
    return pre


def _coarse_bin(vals, dtype):
    """The kernels' coarse histogram bin per value (fp32 via fp16 cvt.rn,
    12-bit; 16-bit dtypes on their own pattern, 13-bit)."""
    if dtype == torch.float32:
        b16 = vals.half().view(torch.int16).to(torch.int32) & 0xFFFF
        shift = 4
    else:
        b16 = vals.to(dtype).view(torch.int16).to(torch.int32) & 0xFFFF
        shift = 3
    mask = torch.where(
        (b16 & 0x8000) != 0,
        torch.full_like(b16, 0xFFFF),
        torch.full_like(b16, 0x8000),
    )
    return ((b16 ^ mask) & 0xFFFF) >> shift


def validate(backend, logits, seq_lens, k, indices, rows_to_check=4, relaxed=False):
    """Fairness gate over the first ``rows_to_check`` rows of the cell only
    (the rest are timed unvalidated).  Exact mode: the output must be the
    exact top-k value multiset.  Relaxed mode (``approx_ties`` contract, also
    sglang's inherent behavior): full row of unique in-range indices, every
    value STRICTLY above the exact k-th value's coarse bin must be selected,
    and nothing below that bin may be selected -- i.e. the only freedom is
    which members of the boundary bin fill the remaining slots."""
    lf = logits.float()
    sl = seq_lens.cpu().tolist()
    N = logits.shape[1]
    for row in range(min(rows_to_check, logits.shape[0])):
        ne = min(N, int(sl[row]))
        kk = min(k, ne)
        sel = indices[row]
        sel = sel[(sel >= 0) & (sel < ne)].long()
        if backend == "radix_cutlass":
            # cutlass leaves masked-region indices instead of -1 pads
            sel = sel[:kk]
        if sel.numel() < kk:
            return False
        if relaxed:
            if sel[:kk].unique().numel() != kk:
                return False
            row_vals = logits[row, :ne]
            bins = _coarse_bin(row_vals, logits.dtype)
            kth = torch.topk(lf[row, :ne], kk).values[-1]
            bkth = _coarse_bin(kth.to(logits.dtype).reshape(1), logits.dtype)[0]
            sel_bins = bins[sel[:kk]]
            if bool((sel_bins < bkth).any()):
                return False
            above = (bins > bkth).nonzero(as_tuple=True)[0]
            if above.numel() and not bool(torch.isin(above, sel[:kk]).all()):
                return False
        else:
            got = torch.sort(lf[row, :ne][sel[:kk]], descending=True).values
            ref = torch.topk(lf[row, :ne], kk).values
            if not torch.equal(got, ref):
                return False
    return True


def time_backend(backend, logits, seq_lens, k, pre_idx=None):
    out_i = torch.empty(logits.shape[0], k, dtype=torch.int32, device=DEV)
    kw = {}
    if backend in ("gvr", "gvr_2"):
        if pre_idx is None:
            return None
        kw["pre_idx"] = pre_idx
    try:

        def fn():
            flashinfer.top_k_varlen(
                logits, seq_lens, k, backend=backend, out_indices=out_i, **kw
            )

        print(f"      [{backend}] first call", flush=True)
        fn()
        torch.cuda.synchronize()
        print(f"      [{backend}] validate", flush=True)
        # sglang truncates tie overflows by design: hold it to the relaxed
        # contract, every exact backend to the exact multiset
        if not validate(
            backend, logits, seq_lens, k, out_i, relaxed=(backend == "sglang")
        ):
            return "DNF(wrong)"
        best = 1e18
        for r in range(REPS):
            print(f"      [{backend}] graph rep {r}", flush=True)
            ts = bench_gpu_time(fn, use_cuda_graph=True, num_iters_within_graph=10)
            best = min(best, sorted(ts)[len(ts) // 2] * 1000)
        return best
    except Exception as e:
        return f"skip({type(e).__name__})"


def attack(tag, pattern, dtype, k, N, seq_list, seed, use_oracle=False):
    rows = len(seq_list)
    logits = gen(pattern, rows, N, dtype, k, seed)
    seq_t = torch.tensor(seq_list, dtype=torch.int32, device=DEV)
    pre = oracle_pre_idx(logits, seq_t, k) if use_oracle else None
    res = {}
    backends = ["radix_primitives", "radix", "radix_cutlass", "radix_filter"]
    if N % 4 == 0 and k <= 2048:
        backends.append("walkfirst_primitives")
    if dtype == torch.float32 and N % 4 == 0 and k <= 2048:
        backends.append("sglang")
    if use_oracle and k in (512, 1024, 2048):
        backends.append("gvr")
    if dtype == torch.float32 and k in (512, 1024, 2048) and N % 4 == 0:
        backends.append(
            "gvr_2"
        )  # identity anchor unless an oracle is in play (hints are inert)
    if ONLY_PRIM:
        backends = ["radix_primitives"]
    if os.environ.get("FI_BACKENDS"):
        backends = os.environ["FI_BACKENDS"].split(",")
    for be in backends:
        pre_be = pre
        if be == "gvr_2" and pre_be is None:
            pre_be = (
                torch.arange(k, dtype=torch.int32, device=DEV)
                .expand(rows, k)
                .contiguous()
            )
        res[be] = time_backend(be, logits, seq_t, k, pre_be)
    p = res.get("radix_primitives")
    line = f"{tag:44s} " + " ".join(
        f"{be[:9]}={v:7.2f}" if isinstance(v, float) else f"{be[:9]}={v}"
        for be, v in res.items()
    )
    verdict = ""
    if isinstance(p, float):
        rivals = {
            be: v
            for be, v in res.items()
            if be != "radix_primitives" and isinstance(v, float)
        }
        if rivals:
            best_be, best_v = min(rivals.items(), key=lambda kv: kv[1])
            if p > best_v * LOSS_THRESH:
                verdict = f"  << LOSS {-(1 - p / best_v) * 100:.1f}% to {best_be}"
                FAILS.append((tag, p, best_be, best_v, p / best_v))
    print(line + verdict, flush=True)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only not in FAMILIES:
        sys.exit(
            f"unknown family {only!r}\n"
            f"usage: {os.path.basename(sys.argv[0])} [{'|'.join(FAMILIES)}]"
        )
    print(
        f"flashinfer: {flashinfer.__file__} device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    if only in ("all", "p1"):
        print("== P1 tie-overflow distributions ==", flush=True)
        for pat in (
            "constant",
            "one_bin",
            "two_values",
            "quantized4",
            "inf_flood",
            "tiecap_edge",
        ):
            for N, b in (
                (65536, 1),
                (65536, 64),
                (131072, 1),
                (131072, 256),
                (32768, 16),
            ):
                attack(
                    f"P1 {pat} fp32 N={N} b={b} k=2048",
                    pat,
                    torch.float32,
                    2048,
                    N,
                    [N] * b,
                    seed=101,
                )
                if pat in ("constant", "one_bin", "two_values"):
                    attack(
                        f"P1 {pat} bf16 N={N} b={b} k=1024",
                        pat,
                        torch.bfloat16,
                        1024,
                        N,
                        [N] * b,
                        seed=102,
                    )

    if only in ("all", "p2"):
        print("== P2 dispatch-gate bands ==", flush=True)
        for N in (40960, 49152, 57344, 63488, 33792, 36864):
            for b in (1, 4, 16):
                attack(
                    f"P2 randn fp32 N={N} b={b} k=2048",
                    "randn",
                    torch.float32,
                    2048,
                    N,
                    [N] * b,
                    seed=103,
                )
        for N in (40960, 57344):
            attack(
                f"P2 randn bf16 N={N} b=1 k=1024",
                "randn",
                torch.bfloat16,
                1024,
                N,
                [N],
                seed=104,
            )

    if only in ("all", "p3"):
        print("== P3 wave quantization ==", flush=True)
        for b in (148, 152, 160, 296, 304, 448):
            for N in (32768, 65536):
                attack(
                    f"P3 randn fp32 N={N} b={b} k=2048",
                    "randn",
                    torch.float32,
                    2048,
                    N,
                    [N] * b,
                    seed=105,
                )

    if only in ("all", "p4"):
        print("== P4 extreme ragged skew (oracle GVR in play) ==", flush=True)
        for b, N in ((256, 131072), (256, 65536), (64, 131072)):
            skew = [N] + [2112] * (b - 1)
            attack(
                f"P4 skew1long fp32 N={N} b={b} k=2048",
                "randn",
                torch.float32,
                2048,
                N,
                skew,
                seed=106,
                use_oracle=True,
            )
            skew4 = [N] * 4 + [2112] * (b - 4)
            attack(
                f"P4 skew4long fp32 N={N} b={b} k=2048",
                "randn",
                torch.float32,
                2048,
                N,
                skew4,
                seed=107,
                use_oracle=True,
            )

    if only in ("all", "p5"):
        print("== P5 tiny-N mega-batch ==", flush=True)
        for N, k, b in (
            (256, 64, 4096),
            (1024, 256, 2048),
            (512, 128, 8192),
            (2048, 512, 1024),
        ):
            attack(
                f"P5 randn fp32 N={N} k={k} b={b}",
                "randn",
                torch.float32,
                k,
                N,
                [N] * b,
                seed=108,
            )
            attack(
                f"P5 randn bf16 N={N} k={k} b={b}",
                "randn",
                torch.bfloat16,
                k,
                N,
                [N] * b,
                seed=109,
            )

    if only in ("all", "p6"):
        print("== P6 k:N pressure ==", flush=True)
        for N, k, b in (
            (4096, 2048, 1),
            (4096, 2048, 256),
            (8192, 2048, 64),
            (2560, 1024, 512),
        ):
            attack(
                f"P6 randn fp32 N={N} k={k} b={b}",
                "randn",
                torch.float32,
                k,
                N,
                [N] * b,
                seed=110,
            )
            attack(
                f"P6 twoval fp32 N={N} k={k} b={b}",
                "two_values",
                torch.float32,
                k,
                N,
                [N] * b,
                seed=111,
            )

    print("\n===== ADVERSARIAL PERF VERDICT =====", flush=True)
    if not FAILS:
        print("No >3% losses found to any backend.", flush=True)
    else:
        print(f"{len(FAILS)} losing cells found:", flush=True)
        for tag, p, be, v, r in sorted(FAILS, key=lambda x: -x[4]):
            print(
                f"  -{(r - 1) * 100:5.1f}%  {tag}: prim {p:.2f} vs {be} {v:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
