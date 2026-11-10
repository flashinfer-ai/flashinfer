"""Cost of ``recurrent_kda(batch_invariant=True)`` next to today's ``recurrent_kda``, one measurement per process.

``--mode decode``: one decode step as the serving frameworks call it (packed ``[1, B]`` tokens, ``cu_seqlens``, an fp32
pool addressed by ``ssm_state_indices``) replayed as a CUDA graph, at chunk phase ``--phase`` (``0`` .. ``15``, ``mixed``
= a fixed random phase per request, ``chunk`` = the sixteen phases in sequence, reported as their time average). The
baseline is today's decode kernel alone: the sum of its own CUPTI records inside the operator's call on a bf16 and on an
fp32 pool, in the packed ``[1, B]`` and the ``[B, 1]`` layout, the fastest of the four; the operator's whole call is
printed as context.

``--mode prefill``: ``--batch`` sequences of ``--tokens`` tokens (``--ragged`` adds ``7 * i % 16`` tokens to sequence
``i``), the class against ``backend="auto"`` on the same fp32 pool, eager calls.

``--mode extend``: ``--tokens`` tokens appended to every request's open chunk at ``--phase``
(``recurrent_kda_batch_invariant_extend``) against the operator's ``[B, T]`` call and ``T`` single-token decode kernels.

Every time is CUPTI GPU time with a cold L2 (a 512 MiB flush before each iteration), the median of ``--iters``. ``--order``
lists the variants in measurement order so two processes can run them in opposite orders (A B B A). ``--grid`` runs
the decode grid of the pull request that added the class, one fresh process per (point, order), and prints the rows.
"""

import argparse
import json
import statistics
import subprocess
import sys

import torch

D = 128
LB = -5.0
FLUSH_BYTES = 512 << 20
BASELINES = {
    "bf16": "bf16 pool, packed [1, B]",
    "fp32": "fp32 pool, packed [1, B]",
    "bf16_fixed": "bf16 pool, [B, 1]",
    "fp32_fixed": "fp32 pool, [B, 1]",
}


class Recorder:
    """One CUPTI activity session; ``take()`` returns the (name, start, end) records since the last call."""

    def __init__(self):
        from cupti import cupti

        self.cupti = cupti
        self.records = []
        self.kinds = (
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.MEMCPY,
            cupti.ActivityKind.MEMSET,
        )

    def _completed(self, activities):
        kernel = self.cupti.ActivityKind.CONCURRENT_KERNEL
        for a in activities:
            if a.kind in self.kinds:
                name = a.name if a.kind == kernel else str(a.kind).split(".")[-1]
                self.records.append((name, a.start, a.end))

    def __enter__(self):
        self.cupti.activity_register_callbacks(
            lambda: (8 * 1024 * 1024, 0), self._completed
        )
        for kind in self.kinds:
            self.cupti.activity_enable(kind)
        return self

    def __exit__(self, *exc):
        self.cupti.activity_flush_all(1)
        for kind in self.kinds:
            self.cupti.activity_disable(kind)

    def take(self):
        self.cupti.activity_flush_all(1)
        r, self.records = self.records, []
        return r


def _is_decode_kernel(name):
    n = name.lower()
    return "kda" in n and "index" not in n and "copy" not in n


def measure(rec, fn, iters, pre=None, steps=1):
    """Median over ``iters`` of (span, kernel sum, decode-kernel sum) in microseconds, summed over ``steps`` calls of
    ``fn`` that each start from a cold L2; ``pre`` runs untimed before each iteration."""
    buf = torch.empty(FLUSH_BYTES, dtype=torch.int8, device="cuda")
    out = []
    for it in range(iters + 3):
        if pre:
            pre()
        tot = [0.0, 0.0, 0.0]
        for _ in range(steps):
            buf.zero_()
            torch.cuda.synchronize()
            rec.take()
            fn()
            torch.cuda.synchronize()
            r = rec.take()
            tot[0] += (max(e for _, _, e in r) - min(s for _, s, _ in r)) / 1e3
            tot[1] += sum(e - s for _, s, e in r) / 1e3
            tot[2] += sum(e - s for n, s, e in r if _is_decode_kernel(n)) / 1e3
        if it >= 3:
            out.append(tot)
    return [statistics.median(x[i] for x in out) / steps for i in range(3)]


def capture(fn, pre=None):
    for _ in range(3):
        if pre:
            pre()
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    if pre:
        pre()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _rows(gen, n, h, offset=0.0):
    x = torch.randn(1, n, h, D, device="cuda", generator=gen) + offset
    return x.to(torch.bfloat16)


def _inputs(gen, n, h):
    return (
        _rows(gen, n, h),
        _rows(gen, n, h),
        _rows(gen, n, h),
        _rows(gen, n, h, -6.0),
        torch.randn(1, n, h, device="cuda", generator=gen).to(torch.bfloat16),
    )


def _cu(lens):
    return torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64, device="cuda"
    )


def run_point(a):
    import flashinfer
    from flashinfer.batch_invariant.kda.api import recurrent_kda_batch_invariant_extend
    from flashinfer.batch_invariant.kda.pool import get_pool
    from flashinfer.kda import recurrent_kda

    h, B = a.heads, a.batch
    gen = torch.Generator(device="cuda").manual_seed(1)
    fused = dict(
        A_log=(0.1 * torch.randn(h, device="cuda", generator=gen)).float(),
        dt_bias=(0.1 * torch.randn(h, D, device="cuda", generator=gen)).float(),
        lower_bound=LB,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
    )
    ids = torch.arange(B, dtype=torch.int32, device="cuda")
    row = dict(
        mode=a.mode,
        heads=h,
        B=B,
        phase=a.phase,
        tokens=a.tokens,
        order=a.order,
        device=torch.cuda.get_device_name(0),
        flashinfer=flashinfer.__version__,
    )
    pool0 = 0.05 * torch.randn(B, h, D, D, device="cuda", generator=gen)

    with Recorder() as rec:
        if a.mode == "prefill":
            lens = [a.tokens + ((7 * i) % 16 if a.ragged else 0) for i in range(B)]
            x = _inputs(gen, sum(lens), h)
            cu = _cu(lens)
            pool = pool0.clone()

            def reset():
                pool.copy_(pool0)

            def call(**kw):
                recurrent_kda(
                    *x,
                    initial_state=pool,
                    cu_seqlens=cu,
                    ssm_state_indices=ids,
                    **kw,
                    **fused,
                )

            for var in a.order.split(","):
                kw = dict(batch_invariant=True) if var == "bi" else dict(backend="auto")
                row[f"{var}_us"], row[f"{var}_kernels_us"], _ = measure(
                    rec, lambda: call(**kw), a.iters, pre=reset
                )
            row["ratio"] = row["auto_kernels_us"] / row["bi_kernels_us"]
            print("ROW," + json.dumps(row), flush=True)
            return

        if a.phase == "mixed":
            phases = torch.randint(
                0, 16, (B,), generator=torch.Generator().manual_seed(7)
            ).tolist()
        else:
            phases = [0 if a.phase == "chunk" else int(a.phase)] * B
        lens = [16 + p for p in phases]
        pool = pool0.clone()
        recurrent_kda(
            *_inputs(gen, sum(lens), h),
            initial_state=pool,
            cu_seqlens=_cu(lens),
            ssm_state_indices=ids,
            batch_invariant=True,
            **fused,
        )
        ep = get_pool(pool)
        live = [pool, *ep.mutable()]
        snap = [t.clone() for t in live]

        def restore():
            for t, c in zip(live, snap, strict=True):
                t.copy_(c)

        T = 1 if a.mode == "decode" else a.tokens
        tok = _inputs(gen, B * T, h)
        cu1 = torch.arange(B + 1, dtype=torch.int64, device="cuda")

        def operator_step(state, toks=None):
            recurrent_kda(
                *(toks or tok),
                initial_state=state,
                cu_seqlens=cu1,
                ssm_state_indices=ids,
                **fused,
            )

        def fixed_step(state, toks):
            recurrent_kda(*toks, initial_state=state, ssm_state_indices=ids, **fused)

        for var in a.order.split(","):
            if var == "bi" and a.mode == "decode":
                out = torch.empty(1, B, h, D, dtype=torch.bfloat16, device="cuda")
                g = capture(
                    lambda: recurrent_kda(
                        *tok,
                        initial_state=pool,
                        cu_seqlens=cu1,
                        ssm_state_indices=ids,
                        output=out,
                        batch_invariant=True,
                        **fused,
                    ),
                    pre=restore,
                )
                steps = 16 if a.phase == "chunk" else 1
                row["bi_us"], _, _ = measure(
                    rec, g.replay, a.iters, pre=restore, steps=steps
                )
            elif var == "bi":
                btk = tuple(t.view(B, T, *t.shape[2:]) for t in tok)
                row["bi_us"], row["bi_kernels_us"], _ = measure(
                    rec,
                    lambda: recurrent_kda_batch_invariant_extend(
                        *btk,
                        A_log=fused["A_log"],
                        dt_bias=fused["dt_bias"],
                        initial_state=pool,
                        lower_bound=LB,
                        ssm_state_indices=ids,
                    ),
                    a.iters,
                    pre=restore,
                )
            elif var in BASELINES:
                state = pool0.to(torch.bfloat16) if "bf16" in var else pool0.clone()
                one = tuple(t[:, :B] for t in tok)
                if var.endswith("fixed"):
                    fix = tuple(t[0, :, None] for t in one)
                    g = capture(lambda: fixed_step(state, fix))
                else:
                    g = capture(lambda: operator_step(state, one))
                row[f"{var}_served_us"], _, row[f"{var}_kernel_us"] = measure(
                    rec, g.replay, a.iters
                )
            elif var == "multi":  # the operator's [B, T] call on an fp32 pool
                state = pool0.clone()
                btk = tuple(t.view(B, T, *t.shape[2:]) for t in tok)

                def multi():
                    recurrent_kda(
                        *btk, initial_state=state, ssm_state_indices=ids, **fused
                    )

                row["multi_us"], row["multi_kernels_us"], _ = measure(
                    rec, multi, a.iters
                )
    base = min((row[f"{v}_kernel_us"], v) for v in BASELINES if f"{v}_kernel_us" in row)
    assert base[0] > 0.0, "no decode kernel record matched the baseline filter"
    row["baseline"] = f"decode kernel alone, {BASELINES[base[1]]}"
    if a.mode == "decode":
        row["base_us"] = base[0]
        row["ratio"] = base[0] / row["bi_us"]
    else:
        cands = [(T * base[0], f"{T} x {row['baseline']}")]
        if "multi_kernels_us" in row:
            cands.append((row["multi_kernels_us"], "[B, T] call, kernels"))
        row["base_us"], row["baseline"] = min(cands)
        row["ratio"] = row["base_us"] / row["bi_kernels_us"]
    print("ROW," + json.dumps(row), flush=True)


def run_grid(a):
    orders = (
        "bi,bf16,fp32,bf16_fixed,fp32_fixed",
        "fp32_fixed,bf16_fixed,fp32,bf16,bi",
    )
    print(
        "| heads | B | phase | baseline [us] | batch_invariant=True [us] | ratio | legs |"
    )
    print("|---|---|---|---|---|---|---|")
    for h in a.grid_heads:
        for B in a.grid_batch:
            for phase in a.grid_phases:
                if phase == "mixed" and B < 8:
                    continue
                legs = []
                for order in orders:
                    cmd = [
                        sys.executable,
                        __file__,
                        "--mode",
                        "decode",
                        "--heads",
                        str(h),
                    ]
                    cmd += ["--batch", str(B), "--phase", phase, "--order", order]
                    cmd += ["--iters", str(a.iters)]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    rows = [x for x in p.stdout.splitlines() if x.startswith("ROW,")]
                    if not rows:
                        print(f"FAILED {h} {B} {phase}: {p.stderr[-400:]}", flush=True)
                        continue
                    legs.append(json.loads(rows[0][4:]))
                    if a.jsonl:
                        with open(a.jsonl, "a") as f:
                            f.write(rows[0][4:] + "\n")
                if len(legs) == 2:
                    base = (legs[0]["base_us"] * legs[1]["base_us"]) ** 0.5
                    ours = (legs[0]["bi_us"] * legs[1]["bi_us"]) ** 0.5
                    print(
                        f"| {h} | {B} | {phase} | {base:.1f} | {ours:.1f} | {base / ours:.3f} | "
                        f"{legs[0]['ratio']:.3f} / {legs[1]['ratio']:.3f} |",
                        flush=True,
                    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["decode", "prefill", "extend"], default="decode")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--phase", default="8")
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--ragged", action="store_true")
    ap.add_argument("--order", default="")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--grid-heads", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--grid-batch", type=int, nargs="+", default=[8, 16, 64, 128, 256])
    ap.add_argument(
        "--grid-phases", nargs="+", default=["0", "8", "15", "mixed", "chunk"]
    )
    ap.add_argument("--jsonl", default="")
    a = ap.parse_args()
    if not a.order:
        a.order = {
            "decode": "bi,bf16,fp32,bf16_fixed,fp32_fixed",
            "prefill": "bi,auto",
            "extend": "bi,bf16,fp32,bf16_fixed,fp32_fixed,multi",
        }[a.mode]
    if a.grid:
        run_grid(a)
    else:
        run_point(a)


if __name__ == "__main__":
    main()
