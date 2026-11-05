# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare cuDNN preparation history with distinct per-layer operand storage.

Run the four arms with baseline and candidate Python environments, in this order:
  baseline-python this_script.py --arm parent-0 --results-dir /tmp/history-results
  candidate-python this_script.py --arm candidate-0 --results-dir /tmp/history-results
  candidate-python this_script.py --arm candidate-1 --results-dir /tmp/history-results
  baseline-python this_script.py --arm parent-1 --results-dir /tmp/history-results

Use an otherwise idle B200, the same CPU affinity and the same dependencies.
Install each source into its own environment; inspect the recorded source hashes
rather than relying on PYTHONPATH to shadow an editable install. Each arm writes
raw host samples, separate CUDA Graph timings, routes and correctness evidence.
Compare median first_ns and later_total_ns / 35 by mode/profile across both
orders. These are host submission measurements, not serving latency. The
baseline and candidate must use the same FE/provider configuration.
"""

import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import time

import cudnn
import flashinfer
import flashinfer.cudnn.prefill as low
import flashinfer.prefill as public
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--arm",
    required=True,
    choices=("parent-0", "candidate-0", "candidate-1", "parent-1"),
)
parser.add_argument("--results-dir", required=True, type=Path)
args = parser.parse_args()
arm = args.arm
candidate = arm.startswith("candidate")
source = Path(flashinfer.__file__).resolve().parent.parent
assert cudnn.__version__ == "1.32.0" and torch.cuda.get_device_capability() == (10, 0)
assert "CUDNN_FRONTEND_ENABLE_FROST_ENGINES" not in os.environ
root = args.results_dir / arm
root.mkdir(parents=True)
policy = {
    "layers": 36,
    "scope": "cuDNN preparation history: host submission and separate GPU replay; not serving",
}
cpu = lambda xs: torch.tensor(xs, dtype=torch.int32, pin_memory=True)
torch.manual_seed(98317)
profiles = []
for qs, ks in (([128], [8320]), ([64, 64, 128], [4160, 4160, 8320])):
    pages = [(n + 15) // 16 for n in ks]
    ids = torch.randperm(sum(pages), dtype=torch.int32)
    table = torch.zeros(len(qs), 1024, dtype=torch.int32)
    at = 0
    for i, n in enumerate(pages):
        table[i, :n] = ids[at : at + n]
        at += n
    profiles.append(
        dict(
            qs=qs,
            ks=ks,
            qo=cpu([0, *itertools.accumulate(qs)]),
            ip=cpu([0, *itertools.accumulate(pages)]),
            ids=ids.cuda(),
            last=cpu([(n - 1) % 16 + 1 for n in ks]),
            lens=cpu(ks),
            qlens=cpu(qs),
            table=table.cuda(),
            q=[
                torch.randn(sum(qs), 32, 128, device="cuda", dtype=torch.bfloat16)
                for _ in range(36)
            ],
        )
    )
caches = [
    torch.randn(1040, 8, 16, 256, device="cuda", dtype=torch.bfloat16)
    for _ in range(36)
]
kvs = [c.split(128, dim=-1) for c in caches]
for p in profiles:
    p["out"] = [torch.empty_like(q) for q in p["q"]]
    p["lse"] = [torch.empty(q.shape[:2], device="cuda") for q in p["q"]]
workspace = torch.empty(394 * 1024**2, device="cuda", dtype=torch.uint8)
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "HND", backend="auto")


def plan(index):
    p = profiles[index]
    w.plan(
        p["qo"],
        p["ip"],
        p["ids"],
        p["last"],
        32,
        8,
        128,
        16,
        causal=True,
        q_data_type=torch.bfloat16,
        seq_lens=p["lens"],
        seq_lens_q=p["qlens"],
        block_tables=p["table"],
        max_token_per_sequence=128,
        max_sequence_kv=16384,
    )
    assert w._backend == "cudnn"


def run(index, layer, stats=False):
    p = profiles[index]
    if stats:
        return w.run(
            p["q"][layer],
            kvs[layer],
            out=p["out"][layer],
            return_lse=True,
            lse=p["lse"][layer],
            lse_layout="NH",
            lse_base="ln",
        )
    return w.run(p["q"][layer], kvs[layer], out=p["out"][layer])


def reference(index, layer):
    p = profiles[index]
    q = p["q"][layer]
    outs = []
    stats = []
    at = 0
    for b, (nq, nk) in enumerate(zip(p["qs"], p["ks"], strict=True)):
        ids = p["table"][b, : (nk + 15) // 16].long()
        k, v = [
            x[ids].transpose(1, 2).reshape(-1, 8, 128)[:nk].double() for x in kvs[layer]
        ]
        heads = torch.arange(32, device="cuda") // 4
        scores = (
            q[at : at + nq].double().transpose(0, 1) @ k[:, heads].permute(1, 2, 0)
        ) * (128**-0.5)
        mask = (
            torch.arange(nk, device="cuda")[None]
            <= nk - nq + torch.arange(nq, device="cuda")[:, None]
        )
        scores.masked_fill_(~mask, -torch.inf)
        outs.append((scores.softmax(-1) @ v[:, heads].transpose(0, 1)).transpose(0, 1))
        stats.append(scores.logsumexp(-1).T)
        at += nq
    return torch.cat(outs), torch.cat(stats)


def route():
    g = w._cudnn_prepared.graph
    engine, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    name = g.get_plan_name_at_index(g._plan_index)
    assert int(engine) < 20000 or (
        g.selected_engine is not None
        and g.selected_engine.name == "sdpa_fwd_prefill_sm100"
    ), name
    return dict(
        engine=int(engine),
        name=name,
        knobs={str(k): int(v) for k, v in knobs.items()},
        override=w._cudnn_prepared.override_cache is not None,
        bounded_paged=w._cudnn_plan.metadata._bounded_paged,
        bounded_ragged=w._cudnn_plan.metadata._bounded_ragged,
        graph_type=str(type(g)),
    )


routes = {}
graphs = {}
checks = []
for index in (0, 1):
    plan(index)
    for layer in range(36):
        run(index, layer)
    graphs[index] = w._cudnn_prepared.graph
    routes[str(index)] = route()
    assert routes[str(index)]["override"]
    for layer in (0, 17, 35):
        expected, lse = reference(index, layer)
        run(index, layer, True)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            profiles[index]["out"][layer].double(), expected, atol=0.004, rtol=0.004
        )
        torch.testing.assert_close(
            profiles[index]["lse"][layer].double(), lse, atol=4e-4, rtol=0
        )
        checks.append(dict(profile=index, layer=layer, output_stats_fp64=True))
    for layer in range(36):
        run(index, layer)
    assert w._cudnn_prepared.graph is graphs[index]
# Both graph signatures are already warmed. Count only Python preparation in an
# untimed control, then remove the observer before all measurements.
preparations = [0]
original = low.prepare_cudnn_batch_prefill
had_alias = hasattr(public, "prepare_cudnn_batch_prefill")


def observe(*a, **kw):
    preparations[0] += 1
    return original(*a, **kw)


low.prepare_cudnn_batch_prefill = observe
if had_alias:
    public.prepare_cudnn_batch_prefill = observe
try:
    for i in range(8):
        index = i % 2
        plan(index)
        run(index, 0)
        assert w._cudnn_prepared.graph is graphs[index]
finally:
    low.prepare_cudnn_batch_prefill = original
    if had_alias:
        public.prepare_cudnn_batch_prefill = original
assert preparations[0] == (0 if candidate else 8), preparations


def sample(index):
    torch.cuda.synchronize()
    usage0 = resource.getrusage(resource.RUSAGE_THREAD)
    cpu0 = time.thread_time_ns()
    t0 = time.perf_counter_ns()
    plan(index)
    t1 = time.perf_counter_ns()
    run(index, 0)
    t2 = time.perf_counter_ns()
    for layer in range(1, 36):
        run(index, layer)
    t3 = time.perf_counter_ns()
    cpu1 = time.thread_time_ns()
    usage1 = resource.getrusage(resource.RUSAGE_THREAD)
    torch.cuda.synchronize()
    t4 = time.perf_counter_ns()
    assert w._cudnn_prepared.graph is graphs[index]
    return dict(
        profile=index,
        plan_ns=t1 - t0,
        first_ns=t2 - t1,
        later_total_ns=t3 - t2,
        submit_ns=t3 - t0,
        thread_cpu_ns=cpu1 - cpu0,
        drain_ns=t4 - t3,
        resource_delta={
            k: getattr(usage1, k) - getattr(usage0, k)
            for k in ("ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw")
        },
    )


rows = []
gc_events = []


def gc_observer(phase, info):
    gc_events.append(
        dict(phase=phase, generation=info["generation"], ns=time.perf_counter_ns())
    )


gc.callbacks.append(gc_observer)
try:
    for repetition in range(2):
        modes = (
            ("alternating", "fixed0", "fixed1")
            if repetition == 0
            else ("fixed1", "fixed0", "alternating")
        )
        for mode in modes:
            indices = [
                i % 2 if mode == "alternating" else int(mode[-1]) for i in range(8)
            ]
            for index in indices[:4]:
                plan(index)
                for layer in range(36):
                    run(index, layer)
            for index in indices:
                row = sample(index)
                row.update(repetition=repetition, mode=mode)
                rows.append(row)
            (root / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
finally:
    gc.callbacks.remove(gc_observer)
assert len(rows) == 48
# GPU replay is measured separately and is never labelled host submission.
gpu = []
hashes = {}
for index in (0, 1):
    plan(index)
    for layer in range(36):
        run(index, layer)
    assert route() == routes[str(index)]
    eager = [x.clone() for x in profiles[index]["out"]]
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for layer in range(36):
            run(index, layer)
    for x in profiles[index]["out"]:
        x.fill_(torch.nan)
    capture.replay()
    torch.cuda.synchronize()
    assert all(
        torch.equal(x, y) for x, y in zip(profiles[index]["out"], eager, strict=True)
    )
    values = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(16):
            capture.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / 16)
    gpu.append(dict(profile=index, graph_36_layers_us=values))
    outputs = [x.cpu() for x in profiles[index]["out"]]
    hashes[str(index)] = [
        hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
        for x in outputs
    ]
    path = root.parent / f"baseline-profile{index}.pt"
    if arm == "parent-0":
        torch.save(outputs, path)
    else:
        baseline = torch.load(path, weights_only=True)
        if not all(torch.equal(x, y) for x, y in zip(outputs, baseline, strict=True)):
            torch.save(outputs, root / f"mismatch-profile{index}.pt")
            raise AssertionError("All-layer output differs from parent-0")
    # Changed-data retained replay, same metadata and addresses.
    profiles[index]["q"][35].neg_()
    kvs[35][1].neg_()
    profiles[index]["out"][35].fill_(torch.nan)
    capture.replay()
    torch.cuda.synchronize()
    expected, _ = reference(index, 35)
    torch.testing.assert_close(
        profiles[index]["out"][35].double(), expected, atol=0.004, rtol=0.004
    )
    profiles[index]["q"][35].neg_()
    kvs[35][1].neg_()
    del capture
result = dict(
    device_name=torch.cuda.get_device_name(),
    sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
    loaded_cudnn=sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libcudnn" in line
        }
    ),
    arm=arm,
    scope=policy["scope"],
    rows=rows,
    gpu=gpu,
    routes=routes,
    checks=checks,
    preparation_control_count=preparations[0],
    output_hashes=hashes,
    all_36_layer_bitwise_parent=True,
    changed_input_retained_replay=True,
    gc_events=gc_events,
    cudnn=cudnn.__version__,
    backend=cudnn.backend_version(),
    torch=torch.__version__,
    capability=list(torch.cuda.get_device_capability()),
    source={
        n: hashlib.sha256((source / n).read_bytes()).hexdigest()
        for n in ("flashinfer/prefill.py", "flashinfer/cudnn/prefill.py")
    },
)
(root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print("PERF_ARM_COMPLETE", arm, len(rows), flush=True)
