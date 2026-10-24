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

"""Paired GPU-time comparison of prepared Hopper VSA backends on the 20 representative shapes.

Run from an editable FlashInfer checkout on H100:

    python benchmarks/bench_cake_vsa_sm90.py --output /tmp/vsa-sm90-cake.json
    python benchmarks/bench_cake_vsa_sm90.py --output ... --warm-l2                    # PR #5470 regime
    python benchmarks/bench_cake_vsa_sm90.py --output ... --gqa 4 --baseline cake_replicated          # GQA, BF16
    python benchmarks/bench_cake_vsa_sm90.py --output ... --gqa 4 --candidate cake_fp8 --baseline cake_fp8_replicated

The Cake arm is ``backend="cake"`` of the wrapper, i.e. the self-contained package
``flashinfer.cake_vsa_sm90_backend`` (its ``build.json`` names the compiled stages); the
``cake_fp8`` arm is the same wrapper planned on the Sage FP8 route
(``q_data_type=torch.float8_e4m3fn`` with the scales of ``sage_fp8_quantize_sm90``; the
quantization runs before timing and is not part of the measured call).  ``--gqa g`` (2, 4
or 8) runs every case whose head count ``H`` is a multiple of ``g`` as a grouped-query problem:
``H`` query heads over ``H / g`` kv heads, the mask and ``k`` / ``v`` drawn for ``H / g``
heads (the inputs of the ``H / g``-head case) and ``q`` for ``H`` heads.  The ``cake`` /
``cake_fp8`` arms then run the GQA kernels on the kv-head-shaped inputs; the
``cake_replicated`` / ``cake_fp8_replicated`` arms are the MHA kernels serving the same
problem on ``k`` / ``v`` / mask / scales replicated per group outside timing (the previous
Cake route); ``fa3`` is the wrapper's native GQA route.  Planning, host-to-device metadata
copies, JIT compilation, input generation, quantization, replication and output allocation
happen before timing; the same prepared ``run`` is measured for both arms in alternating
AB/BA order.  An FP8 arm is checked against the exact softmax of its dequantized inputs
(atol = rtol = 5e-2) and its error against the BF16 arm is reported, not gated;
a BF16 candidate keeps the 1e-2 / 0.03 check against the baseline.  CUPTI measures kernel
activity (a fallback to CUDA events is an error) with a cold L2 cache by default;
``--warm-l2`` reproduces the warm regime of FlashInfer PR #5470 and must never be mixed with
cold numbers.  This is not an end-to-end host latency benchmark.
"""

import argparse
import json
import math
import statistics
import warnings
from pathlib import Path

import torch

from flashinfer.sparse import VariableBlockSparseAttentionWrapper
from flashinfer.testing import bench_gpu_time

CASES = [
    {
        "uuid": "h1-m64-n64-k1-ragged0-scale0",
        "axes": {"H": 1, "M": 64, "N": 64, "K": 1, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m256-n256-k1-ragged0-scale0",
        "axes": {"H": 4, "M": 256, "N": 256, "K": 1, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m256-n256-k3-ragged0-scale0",
        "axes": {"H": 8, "M": 256, "N": 256, "K": 3, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m1024-n1024-k4-ragged0-scale0",
        "axes": {"H": 8, "M": 1024, "N": 1024, "K": 4, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m1024-n1024-k12-ragged0-scale0",
        "axes": {"H": 8, "M": 1024, "N": 1024, "K": 12, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m2048-n2048-k8-ragged0-scale0",
        "axes": {"H": 8, "M": 2048, "N": 2048, "K": 8, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m2048-n2048-k16-ragged0-scale0",
        "axes": {"H": 8, "M": 2048, "N": 2048, "K": 16, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m4096-n4096-k16-ragged0-scale0",
        "axes": {"H": 8, "M": 4096, "N": 4096, "K": 16, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m4096-n4096-k32-ragged0-scale0",
        "axes": {"H": 8, "M": 4096, "N": 4096, "K": 32, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m512-n1024-k4-ragged0-scale0",
        "axes": {"H": 4, "M": 512, "N": 1024, "K": 4, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m1024-n512-k4-ragged1-scale0",
        "axes": {"H": 4, "M": 1024, "N": 512, "K": 4, "RAGGED": 1, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m1024-n1024-k8-ragged1-scale0",
        "axes": {"H": 4, "M": 1024, "N": 1024, "K": 8, "RAGGED": 1, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m4096-n4096-k16-ragged1-scale0",
        "axes": {"H": 7, "M": 4096, "N": 4096, "K": 16, "RAGGED": 1, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m512-n512-k4-ragged0-scale1",
        "axes": {"H": 4, "M": 512, "N": 512, "K": 4, "RAGGED": 0, "SCALE_HALF": 1},
    },
    {
        "uuid": "h1-m1024-n1024-k16-ragged0-scale0",
        "axes": {"H": 1, "M": 1024, "N": 1024, "K": 16, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m16384-n16384-k32-ragged0-scale0",
        "axes": {"H": 7, "M": 16384, "N": 16384, "K": 32, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m32768-n32768-k64-ragged0-scale0",
        "axes": {"H": 7, "M": 32768, "N": 32768, "K": 64, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m65536-n65536-k64-ragged0-scale0",
        "axes": {"H": 7, "M": 65536, "N": 65536, "K": 64, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m109632-n109632-k64-ragged0-scale0",
        "axes": {
            "H": 7,
            "M": 109632,
            "N": 109632,
            "K": 64,
            "RAGGED": 0,
            "SCALE_HALF": 0,
        },
    },
    {
        "uuid": "h7-m109632-n109632-k64-ragged1-scale0",
        "axes": {
            "H": 7,
            "M": 109632,
            "N": 109632,
            "K": 64,
            "RAGGED": 1,
            "SCALE_HALF": 0,
        },
    },
]


def make_inputs(axes, seed):
    """Deterministic inputs identical to ``benchmarks/bench_vsa_sm90.py`` of PR #5470."""
    h, m, n, topk = (axes[name] for name in ("H", "M", "N", "K"))
    cpu_rng = torch.Generator(device="cpu").manual_seed(seed)
    gpu_rng = torch.Generator(device="cuda").manual_seed(seed)
    indices = (
        torch.rand((h, m // 64, n // 64), generator=cpu_rng).topk(topk, dim=-1).indices
    )
    indices = indices.sort(dim=-1).values
    if axes["RAGGED"]:
        counts = torch.randint(1, topk + 1, (h, m // 64), generator=cpu_rng)
        counts[:, 0] = topk
    else:
        counts = torch.full((h, m // 64), topk)
    mask = torch.zeros((h, m // 64, n // 64), dtype=torch.bool)
    mask.scatter_(-1, indices, torch.arange(topk) < counts.unsqueeze(-1))
    q, k, v = (
        torch.randn(
            (h, length, 128), device="cuda", dtype=torch.bfloat16, generator=gpu_rng
        )
        for length in (m, n, n)
    )
    scale = 0.5 if axes["SCALE_HALF"] else 128**-0.5
    return mask, q, k, v, scale


def make_inputs_gqa(axes, seed, gqa):
    """The GQA problem of a case: ``q`` for its ``H`` query heads (the ``H``-head case's ``q``), the mask and
    ``k`` / ``v`` for ``H / gqa`` kv heads (the ``H / gqa``-head case's); ``gqa = 1`` is ``make_inputs``."""
    if gqa == 1:
        return make_inputs(axes, seed)
    if axes["H"] % gqa:
        raise ValueError(f"H = {axes['H']} is not a multiple of the group {gqa}")
    _, q, _, _, scale = make_inputs(axes, seed)
    mask, _, k, v, _ = make_inputs({**axes, "H": axes["H"] // gqa}, seed)
    return mask, q, k, v, scale


FP8_ARM = "cake_fp8"
REPLICATED_ARMS = {"cake_replicated": "cake", "cake_fp8_replicated": FP8_ARM}


def prepare(mask, backend, scale, q=None, k=None, v=None):
    """A planned wrapper and the inputs its ``run`` takes: ``(wrapper, (q, k, v))``.  The ``cake_fp8`` arm
    quantizes the BF16 inputs with the package's quantizer (outside timing) and plans the FP8 route.  A GQA
    problem (``q`` with more heads than ``mask``) plans ``num_qo_heads = q.shape[0]`` over the mask's kv heads;
    the ``*_replicated`` arms replicate ``k`` / ``v`` / the mask per group first (outside timing) and run the
    MHA route of the arm they name."""
    num_qo_heads = q.shape[0]
    if backend in REPLICATED_ARMS:
        g = num_qo_heads // mask.shape[0]
        k, v = k.repeat_interleave(g, dim=0), v.repeat_interleave(g, dim=0)
        mask = mask.repeat_interleave(g, dim=0)
        backend = REPLICATED_ARMS[backend]
    h, mb, nb = mask.shape
    fp8 = backend == FP8_ARM
    real_backend = "cake" if fp8 else backend
    workspace_size = 0 if real_backend == "cake" else 128 * 1024 * 1024
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(workspace_size, dtype=torch.uint8, device="cuda"),
        backend=real_backend,
    )
    mask = mask.to("cuda")
    dtype_kwargs = {"q_data_type": torch.bfloat16}
    inputs = (q, k, v)
    if fp8:
        from flashinfer.cake_vsa_sm90 import sage_fp8_quantize_sm90

        q8, k8, v8, q_scale, k_scale, v_scale = sage_fp8_quantize_sm90(q, k, v)
        inputs = (q8, k8, v8)
        dtype_kwargs = {
            "q_data_type": torch.float8_e4m3fn,
            "q_scale": q_scale,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }
    wrapper.plan(
        mask,
        torch.full((h, mb), 64, dtype=torch.int32, device="cuda"),
        torch.full((h, nb), 64, dtype=torch.int32, device="cuda"),
        num_qo_heads,
        h,
        128,
        sm_scale=scale,
        non_blocking=False,
        **dtype_kwargs,
    )
    return wrapper, inputs


def dequantized_reference(q8, k8, v8, q_scale, k_scale, v_scale, mask, scale):
    """Exact FP32 softmax attention of the dequantized E4M3 inputs (reference (b)); ``k8`` /
    ``v8`` / ``k_scale`` / ``v_scale`` / ``mask`` may be per kv head (query head ``h`` reads kv head
    ``h // g``)."""
    g = q8.shape[0] // k8.shape[0]
    if g > 1:
        k8, v8 = k8.repeat_interleave(g, dim=0), v8.repeat_interleave(g, dim=0)
        k_scale, v_scale = (
            k_scale.repeat_interleave(g, dim=1),
            v_scale.repeat_interleave(g, dim=0),
        )
        mask = mask.repeat_interleave(g, dim=0)
    sk = k8.shape[1]
    k_tok = k_scale[0].repeat_interleave(16, dim=-1)[:, :sk]
    q32 = q8.float() * q_scale[0].unsqueeze(-1)
    k32 = k8.float() * k_tok.unsqueeze(-1)
    v32 = v8.float() * v_scale.unsqueeze(-2)
    # one query block at a time over its selected keys: the softmax over the selected keys equals the dense row
    # softmax with the unselected keys at -inf, without the [H, M, N] score matrix (28 GB for the largest workload)
    out = torch.empty_like(q32)
    mask = mask.to(q32.device)
    offsets = torch.arange(64, device=q32.device)
    for head in range(q32.shape[0]):
        for qb in range(mask.shape[1]):
            rows = slice(qb * 64, (qb + 1) * 64)
            blocks = mask[head, qb].nonzero().flatten()
            keys = (blocks[:, None] * 64 + offsets[None, :]).flatten()
            scores = (q32[head, rows] @ k32[head, keys].T) * float(scale)
            out[head, rows] = torch.softmax(scores, dim=-1) @ v32[head, keys]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Paired comparison of prepared Hopper VSA backends; requires H100 and CUPTI."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--case", help="Run one named workload instead of all 20")
    parser.add_argument(
        "--candidate",
        default="cake",
        help="Backend under test (default: cake; 'cake_fp8' = the Sage FP8 route of backend='cake')",
    )
    parser.add_argument(
        "--baseline",
        default="fa3",
        help="Reference backend (default: fa3; 'cake_replicated' / 'cake_fp8_replicated' = the MHA route of "
        "backend='cake' on k / v / mask replicated per group, the previous Cake route of a GQA problem)",
    )
    parser.add_argument(
        "--gqa",
        type=int,
        default=1,
        choices=(1, 2, 4, 8),
        help="Query heads per kv head (default 1 = MHA); cases whose H is not a multiple are skipped",
    )
    parser.add_argument(
        "--warm-l2",
        action="store_true",
        help="Warm-L2 regime of PR #5470 (reported separately)",
    )
    parser.add_argument("--repeat-iters", type=int, default=50)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        parser.error("Requires an SM90 GPU")
    cases = [case for case in CASES if args.case is None or case["uuid"] == args.case]
    if not cases:
        parser.error("Unknown --case")
    if args.gqa > 1:
        skipped = [case["uuid"] for case in cases if case["axes"]["H"] % args.gqa]
        cases = [case for case in cases if case["axes"]["H"] % args.gqa == 0]
        if skipped:
            print(
                f"skipping {len(skipped)} cases whose H is not a multiple of {args.gqa}: {skipped}"
            )
        if not cases:
            parser.error(f"no case has H divisible by --gqa {args.gqa}")
    if args.gqa == 1 and (
        args.candidate in REPLICATED_ARMS or args.baseline in REPLICATED_ARMS
    ):
        parser.error("the *_replicated arms are GQA baselines; give --gqa 2, 4 or 8")
    names = (args.baseline, args.candidate)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seed": args.seed,
        "gqa": args.gqa,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "timing": "prepared GPU kernel activity, CUPTI, "
        + ("warm L2" if args.warm_l2 else "cold L2")
        + ", AB/BA",
        "cases": [],
    }
    for case in cases:
        mask, q, k, v, scale = make_inputs_gqa(case["axes"], args.seed, args.gqa)
        prepared = {name: prepare(mask, name, scale, q, k, v) for name in names}
        wrappers = {name: wrapper for name, (wrapper, _) in prepared.items()}
        inputs = {name: tensors for name, (_, tensors) in prepared.items()}
        outputs = {
            name: torch.empty(
                (q.shape[0] * q.shape[1], 1, 128), dtype=torch.bfloat16, device=q.device
            )
            for name in wrappers
        }
        calls = {
            name: (
                lambda name=name: wrappers[name].run(
                    *inputs[name], out=outputs[name], enable_pdl=False
                )
            )
            for name in wrappers
        }
        for call in calls.values():
            for _ in range(10):
                call()
        torch.cuda.synchronize()
        expected, actual = outputs[args.baseline], outputs[args.candidate]
        max_error = float((actual.float() - expected.float()).abs().max())
        fp8_error = None
        if args.candidate in (FP8_ARM, "cake_fp8_replicated"):
            plan = wrappers[args.candidate]._cake_vsa_sm90_plan
            exact = dequantized_reference(
                *inputs[args.candidate],
                plan.q_scale,
                plan.k_scale,
                plan.v_scale,
                mask
                if args.candidate == FP8_ARM
                else mask.repeat_interleave(args.gqa, dim=0),
                scale,
            )
            torch.testing.assert_close(
                actual.view_as(exact).float(), exact, atol=0.05, rtol=0.05
            )
            fp8_error = float((actual.view_as(exact).float() - exact).abs().max())
        else:
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
            if max_error > 0.03:
                raise AssertionError(
                    f"{case['uuid']}: maximum absolute error {max_error} > 0.03"
                )
        samples = {name: [] for name in wrappers}
        rounds = []
        for pair in range(args.pairs * 2):
            order = list(wrappers) if pair % 2 == 0 else list(reversed(wrappers))
            measured = {}
            for name in order:
                with warnings.catch_warnings():
                    # Do not silently change the timing boundary to host launch gaps.
                    warnings.filterwarnings(
                        "error", message=".*Falling back to CUDA events.*"
                    )
                    values = bench_gpu_time(
                        calls[name],
                        enable_cupti=True,
                        use_cuda_graph=False,
                        cold_l2_cache=not args.warm_l2,
                        dry_run_iters=10,
                        repeat_iters=args.repeat_iters,
                    )
                measured[name] = statistics.median(values)
                samples[name].append(measured[name])
            rounds.append({"order": order, "time_ms": measured})
        medians = {name: statistics.median(values) for name, values in samples.items()}
        row = {
            **case,
            "num_kv_heads": mask.shape[0],
            "gqa": args.gqa,
            "stages": {
                name: getattr(wrapper._cake_vsa_sm90_plan, "stage", None)
                for name, wrapper in wrappers.items()
                if wrapper._cake_vsa_sm90_plan is not None
            },
            "max_abs_error": max_error,
            "max_abs_error_vs_dequantized": fp8_error,
            "median_ms": medians,
            "speedup": medians[args.baseline] / medians[args.candidate],
            "rounds": rounds,
        }
        report["cases"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{case['uuid']}: {args.baseline}={medians[args.baseline] * 1e3:.3f} us; "
            f"{args.candidate}={medians[args.candidate] * 1e3:.3f} us; {row['speedup']:.3f}x",
            flush=True,
        )
    report["latency_geomean_us"] = {
        backend: 1000
        * math.exp(
            statistics.mean(
                math.log(row["median_ms"][backend]) for row in report["cases"]
            )
        )
        for backend in names
    }
    report["speedup_geomean"] = math.exp(
        statistics.mean(math.log(row["speedup"]) for row in report["cases"])
    )
    report["speedup_min"] = min(row["speedup"] for row in report["cases"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"geomean speedup {args.candidate} vs {args.baseline}: {report['speedup_geomean']:.4f} (min {report['speedup_min']:.3f})"
    )


if __name__ == "__main__":
    main()
