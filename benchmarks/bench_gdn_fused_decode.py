# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the specialized fused GDN decode kernels vs the composable path.

**What the "stock" column here is, and what it is not.**  The baseline in
this script is ``gdn_fused_decode_step``'s own composable torch
implementation -- the executable specification of the op.  It is *not* the
multi-kernel chain a serving framework runs today (upstream
``causal_conv1d_update`` + rearrange + ``gated_delta_rule_decode_*`` and
their neighbours), which lives in the framework and cannot be imported
here.  The composable path is a correctness path: nobody tunes it, and it
is slower than the serving chain.  So the ratios this script prints are an
in-repo development signal -- "did this kernel get faster than the
specification" -- and must **not** be quoted as a speedup over serving.
The number that may be quoted against serving comes from an external
harness that times the framework's real chain, and ultimately from an
end-to-end A/B.

Compares the specialized kernels
(``flashinfer/gdn_kernels/experimental/gdn_fused_decode_registry.json``)
against that composable
path on the registered workload signatures, on a single GPU, in one
process.  The op takes no backend argument and has no environment gate, so
both arms are produced through the registry -- the same mechanism
production dispatch uses:

* ``default`` phase -- the shipped registry.  Each registered signature is
  timed once per registered impl and once through the public op.  The
  per-impl columns come from calling the impl modules directly: this
  benchmark lives inside the package and may reach into it, which is
  exactly why per-impl numbers belong here rather than in an external
  harness.  The internal preference order is attested (a signature served
  by both impls must run the CuTe-DSL one).
* ``composable`` phase -- the registry is emptied in process, so the public
  op serves every signature from the composable torch path.  Composable-only
  operation is proven, not assumed: the routing probe must return False for
  every signature and no impl's ``launch_count()`` may move across the whole
  phase.

CUDA-graph replay time is the headline number for these decode shapes (vLLM
decode runs inside captured graphs); eager time is the sanity column.  The
op mutates its state pools in place, so timed values drift between
iterations -- the memory traffic is value-independent, which is what the
timing measures.

**Cache regime: every number here is cold-L2**, measured with one call per
CUDA graph.  That is the serving regime -- in a decode step every other
layer runs between two visits to the same weights, so nothing this op
touches stays resident -- and it is why this script times with its own event
loop instead of ``flashinfer.testing.utils.bench_gpu_time`` (see
``_median_ms``).

Usage::

    python benchmarks/bench_gdn_fused_decode.py \
        [--output results.json] [--iters 200] [--warmup 20]
"""

import argparse
import json
import math

SPECIALIZED_GDN_MODULE = (
    "flashinfer.gdn_kernels.experimental.gdn_fused_decode_specialized"
)
# Shipped impls, in the dispatcher's internal preference order.  These are
# module names, not public API: the op exposes no way to ask for one.
SHIPPED_IMPLS = ("cutedsl_sm120_pdl", "cuda_sm120_persistent")
POOL = 48
_L2_BUFFER = None


def _shipped_registry_rows():
    """Read the packaged registry JSON straight off disk.

    Read as a file rather than through the dispatch module so the workload
    list is the SHIPPED one even while the composable phase has the
    in-process registry emptied.
    """
    from importlib import resources

    payload = json.loads(
        resources.files("flashinfer.gdn_kernels.experimental")
        .joinpath("gdn_fused_decode_registry.json")
        .read_text()
    )
    assert payload["op"] == "gdn_fused_decode_step"
    assert payload["schema_version"] == 1
    return payload["workloads"]


def _signatures(rows):
    """Distinct signatures (registry rows minus impl), with their impls."""
    signatures = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in (
                "b",
                "hidden",
                "n_ba",
                "qkv_dim",
                "h_q",
                "hv",
                "d",
                "conv_width",
                "conv_state_len",
                "conv_layout",
            )
        )
        signatures.setdefault(key, {"row": dict(row), "impls": []})
        signatures[key]["impls"].append(row["impl"])
    return signatures


def _make_inputs(row, seed=42):
    """Build one input set for a registry row, on the row's own geometry.

    Distributions match ``tests/gdn/test_fused_decode.py::_make_inputs`` so a
    timing run and the correctness suite exercise the same numeric regime:
    small-scale ``randn`` activations, a row-strided ``mixed_qkv`` view like
    the one serving passes, an SD or DS conv pool per ``row["conv_layout"]``,
    and one distinct state-pool slot per batch row.
    """
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    B = int(row["b"])
    # state_indices walks the pool downwards from POOL-1, so a batch larger
    # than the pool runs off the bottom, where torch's negative-index wrap
    # silently aliases two batch rows onto one state slot instead of failing.
    # A registry row wide enough to do that must fail loudly here.
    assert B <= POOL, f"registry row b={B} exceeds the benchmark state pool ({POOL})"
    hidden, n_ba = int(row["hidden"]), int(row["n_ba"])
    qkv_dim, hv, d = int(row["qkv_dim"]), int(row["hv"]), int(row["d"])
    conv_width = int(row["conv_width"])
    state_len = int(row["conv_state_len"])
    if row["conv_layout"] == "SD":
        # vLLM's default pool allocation: (state_len, dim) physical rows,
        # consumed as the transposed [P, qkv_dim, state_len] view.
        conv_state = (
            torch.randn(POOL, state_len, qkv_dim, device="cuda").bfloat16() * 0.5
        ).transpose(-1, -2)
    else:
        conv_state = (
            torch.randn(POOL, qkv_dim, state_len, device="cuda").bfloat16() * 0.5
        )
    return {
        "hidden_states": torch.randn(B, hidden, device="cuda").bfloat16() * 0.5,
        "w_ba": torch.randn(hidden, n_ba, device="cuda").bfloat16() * 0.02,
        "mixed_qkv": torch.randn(B, qkv_dim, device="cuda").bfloat16() * 0.5,
        "conv_weight": torch.randn(qkv_dim, conv_width, device="cuda").bfloat16() * 0.3,
        "conv_bias": torch.randn(qkv_dim, device="cuda").bfloat16() * 0.1,
        "conv_state": conv_state,
        "A_log": torch.randn(hv, device="cuda").float() * 0.5,
        "dt_bias": torch.randn(hv, device="cuda").bfloat16() * 0.1,
        "scale": 1.0 / math.sqrt(d),
        "ssm_state": torch.randn(POOL, hv, d, d, device="cuda").float() * 0.05,
        "state_indices": torch.arange(POOL - 1, POOL - 1 - B, -1, device="cuda").int(),
        "use_qk_l2norm": True,
    }


def _impl_call(impl, inputs):
    """Closure running ONE impl module directly, bypassing dispatch.

    The public op chooses its implementation and offers no override, so
    this is the only way to attribute a number to a specific kernel -- and
    it is legitimate precisely because this benchmark ships inside the
    package whose internals it is timing.  ``impl.execute`` takes the op's
    tensors positionally and does not take ``use_qk_l2norm`` (the guard
    already established it is True for every registered signature).
    """

    def call():
        """One direct launch of this impl on the prepared inputs."""
        impl.execute(
            inputs["hidden_states"],
            inputs["w_ba"],
            inputs["mixed_qkv"],
            inputs["conv_weight"],
            inputs["conv_bias"],
            inputs["conv_state"],
            inputs["A_log"],
            inputs["dt_bias"],
            float(inputs["scale"]),
            inputs["ssm_state"],
            inputs["state_indices"],
        )

    return call


def _l2_buffer():
    """Scratch buffer large enough to evict L2 between timed calls."""
    import torch

    global _L2_BUFFER
    if _L2_BUFFER is None:
        _L2_BUFFER = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
    return _L2_BUFFER


def _median_ms(fn, iters, warmup, replay=None):
    """Median wall time of one call, measured COLD-L2.

    Deliberately not ``flashinfer.testing.utils.bench_gpu_time``: its graph
    path packs ``num_iters_within_graph`` (default 10) calls into a single
    graph, and it silently disables its own ``cold_l2_cache`` when the timed
    callable takes no tensor arguments -- which is exactly how a closure like
    this one is written.  Both inflate a decode kernel: in a real decode step
    every other layer runs between two visits to the same weights, so nothing
    this op touches is L2-resident, and serving replays ONE call per graph.
    """
    import torch

    buf = _l2_buffer()
    body = replay or fn
    for _ in range(warmup):
        body()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        buf.zero_()  # evict L2 -- the serving cache regime
        starts[i].record()
        body()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    n = len(times)
    return times[n // 2] if n % 2 else 0.5 * (times[n // 2 - 1] + times[n // 2])


def _time_call(fn, args):
    """(eager_ms, graph_ms) medians for fn(), both cold-L2.

    The graph column captures exactly ONE call, so it is the number a
    serving engine replays; ``graph_ms`` and ``eager_ms`` therefore differ
    only by launch overhead, not by cache regime.
    """
    import torch

    eager_ms = _median_ms(fn, args.iters, args.warmup)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            fn()
    except Exception:
        torch.cuda.synchronize()
        return eager_ms, None
    graph_ms = _median_ms(fn, args.iters, args.warmup, replay=graph.replay)
    return eager_ms, graph_ms


def _sig_fields(key):
    """Expand a signature key tuple back into its named registry fields."""
    b, hidden, n_ba, qkv_dim, h_q, hv, d, conv_width, state_len, layout = key
    return {
        "b": b,
        "hidden": hidden,
        "n_ba": n_ba,
        "qkv_dim": qkv_dim,
        "h_q": h_q,
        "hv": hv,
        "d": d,
        "conv_width": conv_width,
        "conv_state_len": state_len,
        "conv_layout": layout,
    }


def _empty_registry(specialized_gdn):
    """Context manager emptying the in-process registry.

    This is how the composable arm is produced: not an environment
    variable (there is none) and not a private flag, but the very mechanism
    dispatch consults.  With no rows, every signature misses the registry
    and ``gdn_fused_decode_step`` is exactly its composable torch path.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        """Swap the registry loader for one returning no rows, then restore."""
        original = specialized_gdn.load_gdn_fused_decode_registry
        specialized_gdn.load_gdn_fused_decode_registry = tuple
        try:
            yield
        finally:
            specialized_gdn.load_gdn_fused_decode_registry = original

    return _ctx()


def run_default_phase(args, signatures, specialized_gdn):
    """Time the shipped registry: per impl, and through the public op."""
    import torch

    from flashinfer import gdn_fused_decode_step

    results = {"phase": "default", "registry": "shipped", "rows": []}

    for key, entry in signatures.items():
        record = _sig_fields(key)
        # Per-impl columns.  The op deliberately exposes no backend
        # selector, so an in-repo benchmark that wants to compare its own
        # implementations calls the impl modules directly -- a development
        # tool reaching into the package it lives in, not a public API.
        for impl_name in entry["impls"]:
            impl = specialized_gdn._load_impl(impl_name)
            if impl is None:
                record[f"{impl_name}_unavailable"] = True
                continue
            inputs = _make_inputs(entry["row"])
            call = _impl_call(impl, inputs)
            # Eager warm-up: the first non-capturing dispatch compiles this
            # (batch, scale, conv layout) variant (prerequisite for graph
            # timing; vLLM's profile run plays this role in serving).
            call()
            torch.cuda.synchronize()
            eager_ms, graph_ms = _time_call(call, args)
            record[f"{impl_name}_eager_ms"] = eager_ms
            record[f"{impl_name}_graph_ms"] = graph_ms

        # The column that answers "what does a caller actually get?": the
        # public op, choosing for itself.
        inputs = _make_inputs(entry["row"])

        def call():
            """One call through the public op, which picks its own impl."""
            gdn_fused_decode_step(**inputs)

        call()
        torch.cuda.synchronize()
        eager_ms, graph_ms = _time_call(call, args)
        record["dispatched_eager_ms"] = eager_ms
        record["dispatched_graph_ms"] = graph_ms
        results["rows"].append(record)

    # Attest the internal preference order: a signature served by BOTH impls
    # must run the CuTe-DSL one.  The signature has to be one both impls
    # register -- picking an arbitrary one (e.g. the first) attests nothing
    # when only the CuTe-DSL impl serves it, and reports a false negative when
    # only the CUDA impl does, since then the CuTe-DSL launch counter
    # correctly does not move.
    contested = next(
        (
            entry["row"]
            for entry in signatures.values()
            if set(SHIPPED_IMPLS).issubset(entry["impls"])
        ),
        None,
    )
    cutedsl = specialized_gdn._load_impl("cutedsl_sm120_pdl")
    if cutedsl is None:
        results["prefers_cutedsl_attested"] = None  # DSL unavailable
    elif contested is None:
        # Nothing to attest: no shipped signature is served by both impls.
        results["prefers_cutedsl_attested"] = None
    else:
        inputs = _make_inputs(contested)
        launches_before = cutedsl.launch_count()
        gdn_fused_decode_step(**inputs)
        results["prefers_cutedsl_attested"] = (
            cutedsl.launch_count() == launches_before + 1
        )
    results["stats"] = specialized_gdn.gdn_fused_decode_stats()
    return results


def run_composable_phase(args, signatures, specialized_gdn):
    """Time the same signatures with the registry emptied in process.

    The op is then exactly its composable torch implementation -- the
    executable specification, NOT the serving chain (see the module
    docstring).  Proven, not assumed: the probe must decline every
    signature and no impl may launch during the whole phase.
    """
    import torch

    from flashinfer import (
        gdn_fused_decode_step,
        gdn_fused_decode_step_supported,
    )

    impls = {name: specialized_gdn._load_impl(name) for name in SHIPPED_IMPLS}
    results = {"phase": "composable", "registry": "emptied-in-process", "rows": []}

    with _empty_registry(specialized_gdn):
        launches_before = {
            name: impl.launch_count() for name, impl in impls.items() if impl
        }
        probe_declined = True
        for key, entry in signatures.items():
            record = _sig_fields(key)
            row = entry["row"]
            probe_declined = probe_declined and not gdn_fused_decode_step_supported(
                int(row["b"]),
                hidden_size=int(row["hidden"]),
                n_ba=int(row["n_ba"]),
                qkv_dim=int(row["qkv_dim"]),
                num_qk_heads=int(row["h_q"]),
                num_v_heads=int(row["hv"]),
                head_dim=int(row["d"]),
                conv_width=int(row["conv_width"]),
                conv_state_len=int(row["conv_state_len"]),
                conv_state_layout=str(row["conv_layout"]),
            )
            inputs = _make_inputs(row)

            def call():
                """One call through the public op with the registry emptied,
                i.e. its composable torch path."""
                gdn_fused_decode_step(**inputs)

            call()
            torch.cuda.synchronize()
            eager_ms, graph_ms = _time_call(call, args)
            record["stock_eager_ms"] = eager_ms
            record["stock_graph_ms"] = graph_ms
            results["rows"].append(record)

        # Composable-only proof 1: the routing probe declined every shipped
        # signature while the registry was empty.
        results["probe_returns_false"] = probe_declined
        # Composable-only proof 2: no specialized kernel launched during the
        # phase.  With no environment gate this launch arithmetic IS the
        # proof -- a dispatched kernel would move a counter.
        results["no_specialized_launch"] = all(
            impl.launch_count() == launches_before[name]
            for name, impl in impls.items()
            if impl
        )

    if not (results["probe_returns_false"] and results["no_specialized_launch"]):
        raise RuntimeError(
            "composable phase failed to prove the specialized backends were "
            f"unreachable with an empty registry: {results}"
        )
    return results


def _print_comparison(comparison):
    """Print the human-readable comparison table (times in microseconds)."""
    header = (
        f"{'(B, layout)':>12} | {'stock graph (us)':>17} | "
        f"{'dispatched (us)':>16} | {'dispatched x':>13} | "
        f"{'cutedsl (us)':>13} | {'cuda (us)':>11}"
    )
    print(header)
    print("-" * len(header))

    def fmt(value, scale=1.0, suffix=""):
        """Render an optional measurement; a missing one prints as "-"."""
        return "-" if value is None else f"{value * scale:.1f}{suffix}"

    for row in comparison:
        shape = f"({row['b']}, {row['conv_layout']})"
        print(
            f"{shape:>12} | {fmt(row.get('stock_graph_ms'), 1e3):>17} | "
            f"{fmt(row.get('dispatched_graph_ms'), 1e3):>16} | "
            f"{fmt(row.get('dispatched_graph_speedup'), 1.0, 'x'):>13} | "
            f"{fmt(row.get('cutedsl_sm120_pdl_graph_ms'), 1e3):>13} | "
            f"{fmt(row.get('cuda_sm120_persistent_graph_ms'), 1e3):>11}"
        )
    for name in ("dispatched",) + SHIPPED_IMPLS:
        speedups = [
            r[f"{name}_graph_speedup"]
            for r in comparison
            if r.get(f"{name}_graph_speedup")
        ]
        if speedups:
            geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            print(
                f"\ngeomean {name} graph speedup over {len(speedups)} "
                f"registered signatures: {geomean:.3f}x (cold-L2)"
            )


def run_all(args):
    """Run both phases, join them per signature and emit the comparison.

    Order matters: the composable phase runs first, while no dispatch has
    warmed a variant, so its "nothing specialized launched" proof cannot be
    confused by launch counters that moved earlier in the process.
    """
    import torch  # noqa: F401  (fail early on a CPU-only box)

    from flashinfer.gdn_kernels.experimental import (
        gdn_fused_decode_specialized as specialized_gdn,
    )

    signatures = _signatures(_shipped_registry_rows())
    # Composable arm first: it must run before any dispatch has warmed a
    # variant, so its "no specialized launch" proof cannot be confused by
    # counters that moved earlier in the process.
    stock = run_composable_phase(args, signatures, specialized_gdn)
    enabled = run_default_phase(args, signatures, specialized_gdn)

    def key(row):
        """Join key between the two phases: one registered signature."""
        return (row["b"], row["conv_layout"])

    by_sig = {key(r): r for r in stock["rows"]}
    comparison = []
    for row in enabled["rows"]:
        stock_row = by_sig[key(row)]
        merged = dict(row)
        merged["stock_eager_ms"] = stock_row["stock_eager_ms"]
        merged["stock_graph_ms"] = stock_row["stock_graph_ms"]
        for name in ("dispatched",) + SHIPPED_IMPLS:
            # A graph column is None when capture failed for that arm, so a
            # speedup exists only when both sides were captured.
            if row.get(f"{name}_graph_ms") and stock_row["stock_graph_ms"]:
                merged[f"{name}_graph_speedup"] = (
                    stock_row["stock_graph_ms"] / row[f"{name}_graph_ms"]
                )
            if row.get(f"{name}_eager_ms"):
                merged[f"{name}_eager_speedup"] = (
                    stock_row["stock_eager_ms"] / row[f"{name}_eager_ms"]
                )
        comparison.append(merged)

    print(
        "\ncomposable phase (registry emptied in process): "
        f"probe_returns_false={stock['probe_returns_false']}, "
        f"no_specialized_launch={stock['no_specialized_launch']}"
    )
    print(
        "default phase (shipped registry): prefers_cutedsl_attested="
        f"{enabled['prefers_cutedsl_attested']}\n"
    )
    _print_comparison(comparison)
    print(
        "\nNOTE: the baseline above is this op's COMPOSABLE TORCH PATH, not "
        "the multi-kernel chain a serving framework runs. These ratios are an "
        "in-repo development signal, not a serving speedup; quote the "
        "end-to-end A/B for that."
    )

    payload = {
        "baseline": (
            "composable torch implementation of gdn_fused_decode_step "
            "(the op's executable specification) -- NOT a serving chain"
        ),
        "phases": {"composable": stock, "default": enabled},
        "comparison": comparison,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nresults written to {args.output}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="write the combined results JSON here")
    parser.add_argument("--iters", type=int, default=200, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
