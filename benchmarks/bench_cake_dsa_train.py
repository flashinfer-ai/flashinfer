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

"""Benchmark the experimental Cake DSA sparse-attention training kernels (SM100 / SM103).

Times the forward, the backward and the whole training step (forward +
backward, wrapper-inclusive) of ``flashinfer.experimental.cake_dsa_train`` on
the GLM-5.2 geometry (64 query heads, 512 latent + 64 rope, 512 value
dimensions, top-k 2048, BF16) over eighteen representative rows: one causal
document of 4k .. 128k tokens, context-parallel tails (4k queries over 64k /
128k keys), packed context-parallel all-gather-KV problems (32k queries over
256k keys as 1 .. 32 equal documents, and eight skewed documents) and
CP-rank spreads (32k queries over 64k / 128k / 192k keys).  Optional
baselines when importable: FlashMLA sparse forward + cuDNN frontend sparse
attention backward (``flash_mla``, ``cudnn.DSA``) and the FA sparse-MLA
kernels (``flash_attn.cute.flash_attn_varlen_func`` with ``gather_kv_indices``).

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and
a cold L2 between iterations (per-iteration GPU span); medians over
``--steps`` iterations.  ``--accuracy`` adds the relative-L2 comparison against
the chunked FP64 reference (iid 4k x 4k and peaked-attention cases).

``--host-us`` measures the host side of the eager entry points (``forward``,
``backward``, the public ``dsa_sparse_attention`` forward / autograd backward /
step) as wall-clock microseconds per call with the GPU running asynchronously,
alternating the binding cache off / on inside one process (``--host-rounds``
rounds of ``--host-calls`` calls each), and checks that both paths produce the
same results and the same kernel-only time.

Usage::

    python benchmarks/bench_cake_dsa_train.py [--rows doc_4096 ...] [--arms cake,flashmla_cudnn,fa4]
        [--steps 20] [--accuracy] [--host-us] [--json out.json]
"""

import argparse
import json
import statistics
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.api_logging import ExperimentalWarning  # noqa: E402
from flashinfer.dsa_sparse_attention import dsa_sparse_attention  # noqa: E402
from flashinfer.experimental.cake_dsa_train import cake_backend  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.test_helpers.cake_dsa_train_reference import (  # noqa: E402
    D_LATENT,
    DEFAULT_SCALE,
    DEFAULT_TOPK,
    NUM_HEADS,
    calibrate_beta,
    make_inputs,
    reference_fp64,
    rel_l2,
    rel_l2_rows,
)

# name -> (query lengths per document, key lengths per document)
ROWS: dict[str, tuple[list, list]] = {}
for _n in (4096, 8192, 16384, 32768, 65536, 131072):
    ROWS[f"doc_{_n}"] = ([_n], [_n])
for _s in (65536, 131072):
    ROWS[f"cptail_4k_{_s}"] = ([4096], [_s])
for _N in (1, 2, 4, 8, 16, 32):
    ROWS[f"packed_N{_N}"] = ([32768 // _N] * _N, [262144 // _N] * _N)
_RATIOS = [1, 1, 2, 2, 4, 4, 8, 10]
ROWS["skewed8"] = ([1024 * r for r in _RATIOS], [8192 * r for r in _RATIOS])
for _s in (65536, 131072, 196608):
    ROWS[f"spread_32k_{_s}"] = ([32768], [_s])

ACCURACY_CASES = {
    "iid_4k": dict(seq_q=[4096], seq_k=[4096], self_including=False, target_self_weight=None),
    "peaked_053_4k": dict(seq_q=[4096], seq_k=[4096], self_including=True, target_self_weight=0.53),
    "peaked_099_4k": dict(seq_q=[4096], seq_k=[4096], self_including=True, target_self_weight=0.99),
}
SEED = 1701


def median_ms(fn, steps):
    times = bench_gpu_time(fn, dry_run_iters=3, repeat_iters=steps, enable_cupti=True, cold_l2_cache=True)
    return float(statistics.median(times))


def gib(nbytes):
    return nbytes / 2**30


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


class ArmCake:
    name = "cake"

    def __init__(self, inp):
        self.inp = inp
        self.backward_available = cake_backend.generated_program_available(inp.q_latent.device, backward=True)
        self.runner = cake_backend.prepare_dsa_train(
            inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global,
            topk_length=inp.topk_length, dout=inp.dout if self.backward_available else None,
            softmax_scale=DEFAULT_SCALE, backward=self.backward_available,
        )

    def versions(self):
        return dict(module=self.runner.module_name, abi=self.runner.abi, stages=list(self.runner.stages))

    def forward(self):
        out, lse, _ = self.runner.forward()
        return dict(out=out, lse=lse)

    def backward(self, st):
        dq_latent, dq_rope, dkv_latent, dk_rope = self.runner.backward()
        return dict(dq_latent=dq_latent, dq_rope=dq_rope, dkv_latent=dkv_latent, dk_rope=dk_rope)

    def outputs(self, st):
        return st


class ArmFlashMLACudnn:
    """FlashMLA sparse forward (packed q / kv, global indices) + cuDNN frontend DSA backward."""

    name = "flashmla_cudnn"

    def __init__(self, inp):
        import flash_mla
        from cudnn import DSA

        self.inp = inp
        self.flash_mla = flash_mla
        self.DSA = DSA
        self.q = torch.cat([inp.q_latent, inp.q_rope], dim=-1)
        self.kv = torch.cat([inp.kv_latent, inp.k_rope], dim=-1)
        self.sink = torch.full((NUM_HEADS,), float("-inf"), dtype=torch.float32, device=self.q.device)
        # cuDNN: valid slots first + topk_length; sentinels replaced by 0 (ignored past topk_length)
        self.idx_cudnn = inp.idx_global.clamp_min(0).contiguous()
        self.dq = torch.empty_like(self.q)
        self.dkv = torch.zeros_like(self.kv)

    def versions(self):
        import cudnn

        return dict(flash_mla=getattr(self.flash_mla, "__file__", "?"), cudnn_frontend=cudnn.__version__, torch_cudnn=torch.backends.cudnn.version())

    def forward(self):
        out, max_logits, lse = self.flash_mla.flash_mla_sparse_fwd(
            self.q, self.kv.unsqueeze(1), self.inp.idx_global.unsqueeze(1), DEFAULT_SCALE, D_LATENT
        )
        return dict(out=out, lse=lse, max_logits=max_logits)

    def backward(self, st):
        self.dkv.zero_()
        self.DSA.sparse_attention_backward_wrapper(
            self.q, self.kv, st["out"], self.inp.dout, st["lse"], self.sink, self.idx_cudnn,
            softmax_scale=DEFAULT_SCALE, topk_length=self.inp.topk_length, dq=self.dq, dkv=self.dkv,
        )
        return dict(
            dq_latent=self.dq[..., :D_LATENT], dq_rope=self.dq[..., D_LATENT:],
            dkv_latent=self.dkv[:, :D_LATENT], dk_rope=self.dkv[:, D_LATENT:],
        )

    def outputs(self, st):
        return dict(out=st["out"], lse=st["lse"])


class ArmFA4:
    """FA sparse-MLA kernels: varlen entry with per-document gather indices, recompute-P backward."""

    name = "fa4"

    def __init__(self, inp, token_chunk=4096):
        from flash_attn.cute import flash_attn_varlen_func

        self.inp = inp
        self.fn = flash_attn_varlen_func
        self.token_chunk = token_chunk
        self.q_rope = inp.q_rope.detach().requires_grad_()
        self.q_latent = inp.q_latent.detach().requires_grad_()
        self.k_rope = inp.k_rope.detach().unsqueeze(1).contiguous().requires_grad_()
        self.kv_latent = inp.kv_latent.detach().unsqueeze(1).contiguous().requires_grad_()

    def versions(self):
        import cutlass
        import flash_attn.cute as fc

        return dict(cutlass_dsl=cutlass.__version__, fa_file=fc.__file__)

    def forward(self):
        inp = self.inp
        out, lse = self.fn(
            self.q_rope, self.k_rope, self.kv_latent, qv=self.q_latent,
            cu_seqlens_q=inp.cu_seqlens_q, cu_seqlens_k=inp.cu_seqlens_k,
            max_seqlen_q=inp.max_seqlen_q, max_seqlen_k=inp.max_seqlen_k,
            softmax_scale=DEFAULT_SCALE, causal=True, gather_kv_indices=inp.idx_local, pack_gqa=True,
            gather_bwd_recompute_p=True, gather_bwd_token_chunk=self.token_chunk, return_lse=True,
        )
        return dict(out=out, lse=lse)

    def backward(self, st):
        dq_rope, dk_rope, dkv_latent, dq_latent = torch.autograd.grad(
            st["out"], (self.q_rope, self.k_rope, self.kv_latent, self.q_latent), self.inp.dout, retain_graph=True
        )
        return dict(dq_latent=dq_latent, dq_rope=dq_rope, dkv_latent=dkv_latent[:, 0], dk_rope=dk_rope[:, 0])

    def outputs(self, st):
        lse = st["lse"]
        if lse.shape[0] != self.inp.total_q:  # (nheads, total_q) layout
            lse = lse.transpose(0, 1)
        return dict(out=st["out"], lse=lse)


ARMS = {ArmCake.name: ArmCake, ArmFlashMLACudnn.name: ArmFlashMLACudnn, ArmFA4.name: ArmFA4}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_perf(arm, steps):
    """Median forward, backward and step milliseconds plus the peak memory above the inputs."""
    st = arm.forward()
    torch.cuda.synchronize()
    fwd_ms = median_ms(arm.forward, steps)
    bwd_ms = None
    step_ms = None
    if not isinstance(arm, ArmCake) or arm.backward_available:
        bwd_ms = median_ms(lambda: arm.backward(st), steps)

        def full():
            s = arm.forward()
            arm.backward(s)

        step_ms = median_ms(full, steps)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    s = arm.forward()
    if bwd_ms is not None:
        arm.backward(s)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return dict(fwd_ms=fwd_ms, bwd_ms=bwd_ms, step_ms=step_ms, peak_above_inputs_gib=gib(peak))


def measure_accuracy(arm, inp):
    st = arm.forward()
    grads = None
    try:
        grads = arm.backward(st)
    except NotImplementedError as exc:
        grads = dict(error=str(exc))
    torch.cuda.synchronize()
    ref = reference_fp64(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global,
        dout=inp.dout, own_key=inp.own_key,
    )
    outs = arm.outputs(st)
    valid = torch.isfinite(ref["lse"])
    record = dict(
        self_weight=ref["self_weight"],
        out_rel_l2=rel_l2(outs["out"], ref["out"]),
        lse_max_abs=float((outs["lse"].double()[valid] - ref["lse"][valid]).abs().max()) if valid.any() else 0.0,
    )
    if "error" in grads:
        record["backward"] = grads["error"]
    else:
        for name in ("dq_latent", "dq_rope", "dkv_latent", "dk_rope"):
            record[f"{name}_rel_l2"] = rel_l2(grads[name], ref[name])
        record["dq_latent_row_p99"] = float(rel_l2_rows(grads["dq_latent"], ref["dq_latent"]).quantile(0.99))
    return record


def _arm_or_error(cls, inp):
    try:
        return cls(inp), None
    except Exception as exc:  # optional dependency missing or arm unavailable on this device
        return None, f"{type(exc).__name__}: {exc}"


def run_perf(args, results):
    steps_for = lambda inp: max(args.min_steps, args.steps if inp.total_q * inp.total_k <= 2**34 else args.steps_128k)
    for row in args.rows:
        seq_q, seq_k = ROWS[row]
        inp = make_inputs(seq_q, seq_k, seed=SEED, topk=DEFAULT_TOPK, device=args.device)
        entry = dict(row=row, total_q=inp.total_q, total_k=inp.total_k, num_docs=len(seq_q), inputs_gib=gib(inp.bytes_inputs()), arms={})
        for name in args.arms:
            arm, error = _arm_or_error(ARMS[name], inp)
            if arm is None:
                entry["arms"][name] = dict(error=error)
                print(f"{row:22s} {name:15s} unavailable: {error}", flush=True)
                continue
            try:
                perf = measure_perf(arm, steps_for(inp))
                perf["versions"] = arm.versions()
            except Exception as exc:
                perf = dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            entry["arms"][name] = perf
            fmt = lambda v: "   n/a" if v is None else f"{v:8.3f}"
            if "error" in perf:
                print(f"{row:22s} {name:15s} failed: {perf['error']}", flush=True)
            else:
                print(
                    f"{row:22s} {name:15s} fwd {fmt(perf['fwd_ms'])} ms  bwd {fmt(perf['bwd_ms'])} ms  "
                    f"step {fmt(perf['step_ms'])} ms  peak {perf['peak_above_inputs_gib']:.2f} GiB",
                    flush=True,
                )
            del arm
            torch.cuda.empty_cache()
        results["rows"].append(entry)
        del inp
        torch.cuda.empty_cache()


def run_accuracy(args, results):
    for case, spec in ACCURACY_CASES.items():
        beta = 0.0
        if spec["target_self_weight"] is not None:
            beta = calibrate_beta(spec["target_self_weight"], seed=SEED, device=args.device)
        inp = make_inputs(spec["seq_q"], spec["seq_k"], seed=SEED, topk=DEFAULT_TOPK, self_including=spec["self_including"], beta=beta, device=args.device)
        entry = dict(case=case, beta=beta, arms={})
        for name in args.arms:
            arm, error = _arm_or_error(ARMS[name], inp)
            if arm is None:
                entry["arms"][name] = dict(error=error)
                continue
            try:
                entry["arms"][name] = measure_accuracy(arm, inp)
            except Exception as exc:
                entry["arms"][name] = dict(error=f"{type(exc).__name__}: {exc}")
            print(f"{case:16s} {name:15s} {json.dumps(entry['arms'][name], default=str)}", flush=True)
            del arm
            torch.cuda.empty_cache()
        results["accuracy"].append(entry)
        del inp
        torch.cuda.empty_cache()


def _host_us_per_call(fn, calls):
    """Wall-clock microseconds per call of ``fn`` (GPU asynchronous; results dropped as they come)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    return (time.perf_counter() - t0) / calls * 1e6


def measure_host_path(inp, *, calls, rounds, kernel_steps):
    """Host microseconds per call of the eager entry points with the binding cache off / on.

    Each round measures every entry point in both modes back to back (off
    first), so the two modes see the same process state; the medians over the
    rounds are reported.  Also checks that both modes give the same results
    (``out``, ``lse``, ``dq_*`` bitwise; ``dkv_*`` within the ``red.global``
    run-to-run spread) and reports the CUPTI kernel-only medians of both.
    """
    cache = cake_backend.BINDING_CACHE
    was_enabled = cache.enabled
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    backward_available = cake_backend.generated_program_available(inp.q_latent.device, backward=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        dsa_sparse_attention(*args)  # the experimental banner fires once per process
    leaves = [t.detach().clone().requires_grad_() for t in args[:4]]
    saved = cake_backend.forward(*args)
    torch.cuda.synchronize()

    def eager_forward():
        return cake_backend.forward(*args)

    def eager_backward():
        return cake_backend.backward(*args, saved[0], saved[2], saved[1], inp.dout)

    def public_forward():
        return dsa_sparse_attention(*leaves, inp.idx_global)

    graph_out = public_forward()

    def public_backward():  # autograd backward of one retained graph (the saved forward outputs are fixed)
        return torch.autograd.grad(graph_out, leaves, inp.dout, retain_graph=True)

    def autograd_step():
        out = dsa_sparse_attention(*leaves, inp.idx_global)
        return torch.autograd.grad(out, leaves, inp.dout)

    entry_points = [("forward", eager_forward), ("public_forward", public_forward)]
    if backward_available:
        entry_points += [("backward", eager_backward), ("public_backward", public_backward), ("autograd_step", autograd_step)]
    samples = {name: {"off": [], "on": []} for name, _ in entry_points}
    lookups = {name: [] for name, _ in entry_points}  # (hits, misses) of the measured cache-on calls
    try:
        for _ in range(rounds):
            for mode in ("off", "on"):
                cache.enabled = mode == "on"
                for name, fn in entry_points:
                    fn()  # the first call of a mode binds (a miss); measured calls follow
                    hits, misses = cache.hits, cache.misses
                    samples[name][mode].append(_host_us_per_call(fn, calls))
                    if mode == "on":
                        lookups[name].append((cache.hits - hits, cache.misses - misses))
        # same results from both paths
        cache.enabled = False
        off_fwd = eager_forward()
        off_bwd = eager_backward() if backward_available else None
        cache.enabled = True
        on_fwd = eager_forward()
        on_bwd = eager_backward() if backward_available else None
        torch.cuda.synchronize()
        same = dict(out=torch.equal(off_fwd[0], on_fwd[0]), lse=torch.equal(off_fwd[1], on_fwd[1]))
        if backward_available:
            same.update(
                dq_latent=torch.equal(off_bwd[0], on_bwd[0]),
                dq_rope=torch.equal(off_bwd[1], on_bwd[1]),
                dkv_latent_rel_l2=rel_l2(off_bwd[2], on_bwd[2]),
                dk_rope_rel_l2=rel_l2(off_bwd[3], on_bwd[3]),
            )
        del off_fwd, off_bwd, on_fwd, on_bwd
        # kernel-only time of both paths (CUPTI, per-iteration GPU span)
        kernel_ms = {}
        for mode in ("off", "on"):
            cache.enabled = mode == "on"
            kernel_ms[mode] = dict(forward=median_ms(eager_forward, kernel_steps))
            if backward_available:
                kernel_ms[mode]["backward"] = median_ms(eager_backward, kernel_steps)
                kernel_ms[mode]["autograd_step"] = median_ms(autograd_step, kernel_steps)
    finally:
        cache.enabled = was_enabled
    host_us = {
        name: {mode: dict(median=float(statistics.median(v)), min=float(min(v)), rounds=[float(x) for x in v]) for mode, v in modes.items()}
        for name, modes in samples.items()
    }
    for name, hm in lookups.items():
        host_us[name]["on"]["lookups"] = dict(hits=sum(h for h, _ in hm), misses=sum(m for _, m in hm))
    return dict(
        calls=calls, rounds=rounds, host_us=host_us, same_results=same, kernel_ms=kernel_ms,
        cache=dict(hits=cache.hits, misses=cache.misses, bindings=len(cache), owned_bytes=cache.owned_bytes),
    )


def run_host_path(args, results):
    for row in args.rows:
        seq_q, seq_k = ROWS[row]
        inp = make_inputs(seq_q, seq_k, seed=SEED, topk=DEFAULT_TOPK, device=args.device)
        try:
            entry = measure_host_path(inp, calls=args.host_calls, rounds=args.host_rounds, kernel_steps=max(args.min_steps, args.steps_128k))
        except Exception as exc:
            entry = dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            print(f"{row:22s} host path failed: {entry['error']}", flush=True)
        else:
            for name, modes in entry["host_us"].items():
                off, on = modes["off"]["median"], modes["on"]["median"]
                lk = modes["on"]["lookups"]
                print(
                    f"{row:22s} {name:16s} host us/call  cache off {off:8.1f}  cache on {on:8.1f}  ({off / on:5.2f}x)"
                    f"  lookups hit {lk['hits']} miss {lk['misses']}",
                    flush=True,
                )
            print(f"{row:22s} same results: {json.dumps(entry['same_results'], default=str)}", flush=True)
            for mode, ms in entry["kernel_ms"].items():
                print(f"{row:22s} kernel-only ms (cache {mode}): {json.dumps(ms)}", flush=True)
        entry["row"] = row
        results["host_path"].append(entry)
        del inp
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="*", default=list(ROWS), choices=list(ROWS))
    parser.add_argument("--arms", default="cake,flashmla_cudnn,fa4")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--steps-128k", type=int, default=13, help="iterations for the largest problems")
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--host-us", action="store_true", help="host microseconds per call, binding cache off / on")
    parser.add_argument("--host-calls", type=int, default=20)
    parser.add_argument("--host-rounds", type=int, default=3)
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    unknown = sorted(set(args.arms) - set(ARMS))
    if unknown:
        parser.error(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    torch.cuda.set_device(torch.device(args.device))
    results = dict(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        rows=[],
        accuracy=[],
        host_path=[],
    )
    if not args.no_perf:
        run_perf(args, results)
    if args.accuracy:
        run_accuracy(args, results)
    if args.host_us:
        run_host_path(args, results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
