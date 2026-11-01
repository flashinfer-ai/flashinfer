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

Benchmark of the Cake GDN (Gated DeltaNet) training step on one GPU: the deterministic package
(``flashinfer.experimental.cake_gdn_det_train.chunk_gated_delta_rule_det``, bitwise run to run) against the
non-deterministic one (``flashinfer.experimental.cake.cake_gdn_train.chunk_gated_delta_rule`` with
``deterministic=False``).  Both take FLA's ``chunk_gated_delta_rule`` signature; every call is a packed
variable-length batch (``B = 1``, ``cu_seqlens`` on the device and ``cu_seqlens_cpu`` on the host: the host-table
route), head dimension 128, ``use_qk_l2norm_in_kernel=True``.

Boundaries (``--boundaries``), timed with CUDA events on the current stream:

* ``fwd``: the training forward (the inputs require grad and autograd records the graph);
* ``bwd``: the backward only: an untimed forward before every sample, then ``torch.autograd.backward(o, do)``;
* ``fwdbwd``: forward + backward;
* ``ckpt``: ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` forward + backward
  (``2 x forward + backward``, the activation-checkpointing training boundary).

The input gradients are set to ``None`` before every sample (untimed), so no sample accumulates into a gradient.

Arms (``--arms``; ``nondet`` and ``--base`` are always measured), every one on the same inputs, drawn once per shape
from ``--seed``:

* ``det``: the deterministic package;
* ``nondet``: the non-deterministic package, the reference arm of every ratio;
* ``control``: the non-deterministic package again -- the identical-program control;
* ``det_dir``: another build of the deterministic package, imported from the package directory ``--det-dir`` (a
  standalone copy, for example a candidate build with prebuilt ``aot/`` libraries);
* ``tail_g2`` / ``tail_g3`` / ``unfused``: the non-deterministic package with the fixed-order tail pinned
  (``tail="tail_g2"`` / ``"tail_g3"``, value-head groups of two / three only: the arm is left out of other shapes;
  ``tail="unfused"``), the same forward as ``nondet``;
* ``tail``: the non-deterministic package with its own tail pinned (``tail="tail"``), the same forward as
  ``nondet``.

Per shape, one untimed forward + backward of ``nondet`` with the package's launches recorded prints the stages its
backward runs (``stages nondet bwd``: the tail form, the adjoint table and its scan); at the end the bench lists every
(shape, boundary) where ``nondet`` is slower than ``det`` / ``det_dir`` by more than the two spreads.

Protocol: ``--groups`` groups (at least 3).  In every group the arms run in the pattern ``[nondet, X, X, nondet]`` for
every other arm ``X``, ``--base`` first (``--base control`` puts the control in the place of ``det``).  Every call takes
``--warmup`` untimed iterations, then ``--samples`` timed ones (at least 8); every iteration follows an L2 flush (a
256 MiB buffer written) and a device synchronization.  A sample is *steady* when the iteration dispatched before it ran
the same arm: after an arm switch only the warmups are spent (with ``--warmup 0`` the first sample of a switch call is
dropped).  The figure of an arm in a group is the median of its steady samples; the figure of an arm is the median over
the groups; the spread is ``(max - min) / median`` of the group medians; p10 / p90 are taken over all steady samples.
Per boundary the report gives the ratio ``X / nondet`` of the figures (above 1: the non-deterministic package is
faster).  The raw samples of every call are kept in the JSON report.

Before a shape is timed, every arm runs one forward + backward: the output and the input gradients must be finite (a
shape that fails is not timed and the exit status is 1), and ``max|o_nondet - o_det| / max|o_det|`` is printed as
information, not as a gate.  Memory per arm is measured once outside the timing, with ``torch.cuda.memory_allocated`` /
``max_memory_allocated`` after a synchronization, relative to the baseline (the memory allocated before the call:
the inputs, the upstream gradient, the flush buffer and the pools; printed as ``baseline``): ``pool`` is the
net memory the package keeps after its first call on the shape (pooled workspaces and tables, held between calls);
``retained`` the memory after the forward with the graph alive (the output and the saved set); ``peak-temp`` the peak
during forward + backward (the input gradients included); ``ckpt-retained`` / ``ckpt-peak-temp`` the same for the
checkpointed forward.

Shapes (``--shapes``: comma-separated set names ``rows``, ``packs``, ``pairs``, ``groups`` or shape ids; ``--list``
prints the ids): ``rows`` are H = 8 / 16 packs of about 4k-token documents (and of one 16k document), ``packs`` H = 2
packs of up to 262k tokens and one H = 16 pack, ``pairs`` the five head pairs ``(H, HV)`` on the layout
``(11000, 12000, 9768)``, ``groups`` value-head groups of one, three and four on 4k-token documents.

Example::

    python benchmarks/bench_cake_gdn_train.py --shapes rows,p7074_issue_261987x3_h2 \\
        --boundaries fwd,bwd,fwdbwd,ckpt --groups 5 --samples 10 --warmup 3 --json cake_gdn_train.json
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HEAD_DIM = 128
L2_FLUSH_BYTES = 256 << 20
MIB = float(1 << 20)
ARMS = (
    "det",
    "nondet",
    "control",
    "det_dir",
    "tail_g2",
    "tail_g3",
    "unfused",
    "tail",
)
# the pinned-tail arms and their pins; a cluster tail serves its value-head group only
PIN_ARMS = {
    "tail_g2": {"tail": "tail_g2"},
    "tail_g3": {"tail": "tail_g3"},
    "unfused": {"tail": "unfused"},
    "tail": {"tail": "tail"},
}
GROUP_ARMS = {"tail_g2": 2, "tail_g3": 3}
BASES = ("det", "control")
BOUNDARIES = ("fwd", "bwd", "fwdbwd", "ckpt")

# ----------------------------------------------------------------------------------------------------------------
# Shapes: id -> (H, HV, document lengths)

ROWS: dict[str, tuple[int, int, tuple[int, ...]]] = {
    "r01_4x4096_h8": (8, 16, (4096,) * 4),
    "r02_real3x4.25k_h8": (8, 16, (5406, 3448, 3896)),
    "r03_8x4096_h8": (8, 16, (4096,) * 8),
    "r04_real7x4.3k_h8": (8, 16, (4199, 2781, 4282, 4448, 4723, 5804, 3863)),
    "r05_16x4096_h8": (8, 16, (4096,) * 16),
    "r06_real15x4.3k_h8": (
        8,
        16,
        (
            2756,
            4479,
            4053,
            4052,
            5896,
            4229,
            5238,
            3013,
            4429,
            4175,
            3743,
            4029,
            5404,
            3129,
            5875,
        ),
    ),
    "r07_1x16384_h8": (8, 16, (16384,)),
    "r08_4x4096_h16": (16, 32, (4096,) * 4),
    "r09_16x4096_h16": (16, 32, (4096,) * 16),
    "r10_1x16384_h16": (16, 32, (16384,)),
}

PACKS: dict[str, tuple[int, int, tuple[int, ...]]] = {
    "p7074_issue_261987x3_h2": (2, 6, (131071, 87531, 43385)),
    "p7074_P3_259174x2_h2": (2, 6, (128147, 131027)),
    "p7074_P1_241817x6_h2": (2, 6, (126565, 32714, 33696, 15081, 22979, 10782)),
    "p7074_64x4096_h2": (2, 6, (4096,) * 64),
    "p7074_A16_32768x3_h16": (16, 48, (11000, 12000, 9768)),
}

PAIRS: dict[str, tuple[int, int, tuple[int, ...]]] = {
    "pair_h16_hv48": (16, 48, (11000, 12000, 9768)),
    "pair_h8_hv24": (8, 24, (11000, 12000, 9768)),
    "pair_h4_hv12": (4, 12, (11000, 12000, 9768)),
    "pair_h2_hv6": (2, 6, (11000, 12000, 9768)),
    "pair_h1_hv3": (1, 3, (11000, 12000, 9768)),
}

# Value-head groups beside the rows' two and the packs' three, and the three of one key head: the classes of the
# backward-tail dispatch rule.
GROUPS: dict[str, tuple[int, int, tuple[int, ...]]] = {
    "g1_4x4096_h16": (16, 16, (4096,) * 4),
    "g1_16x4096_h8": (8, 8, (4096,) * 16),
    "g3_4x4096_h8": (8, 24, (4096,) * 4),
    "g3_16x4096_h1": (1, 3, (4096,) * 16),
    "g4_4x4096_h4": (4, 16, (4096,) * 4),
    "g4_16x4096_h4": (4, 16, (4096,) * 16),
}

SETS = {"rows": ROWS, "packs": PACKS, "pairs": PAIRS, "groups": GROUPS}
SHAPES = {**ROWS, **PACKS, **PAIRS, **GROUPS}


def parse_shapes(text: str) -> list[str]:
    """The shape ids of ``--shapes`` (set names expand to their ids), in order and without repeats."""
    ids: list[str] = []
    for token in (t.strip() for t in text.split(",")):
        if not token:
            continue
        if token in SETS:
            ids.extend(SETS[token])
        elif token in SHAPES:
            ids.append(token)
        else:
            raise ValueError(
                f"unknown shape or set {token!r} (sets: {', '.join(SETS)}; --list prints the ids)"
            )
    if not ids:
        raise ValueError("--shapes selects no shape")
    return list(dict.fromkeys(ids))


def parse_names(text: str, choices: tuple[str, ...], flag: str) -> list[str]:
    names = [t.strip() for t in text.split(",") if t.strip()]
    unknown = [n for n in names if n not in choices]
    if unknown or not names:
        raise ValueError(
            f"{flag} takes a comma-separated subset of {', '.join(choices)} (got {text!r})"
        )
    return list(dict.fromkeys(names))


# ----------------------------------------------------------------------------------------------------------------
# Arms and inputs


# The package directory of the ``det_dir`` arm (``--det-dir``).
DET_DIR: str | None = None


def _load_package_dir(path: str) -> Any:
    """Import a standalone package from its directory under the directory's name."""
    import importlib.util

    root = Path(path).resolve()
    if root.name in sys.modules:
        return sys.modules[root.name]
    spec = importlib.util.spec_from_file_location(
        root.name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"--det-dir {path}: not a package directory")
    module = importlib.util.module_from_spec(spec)
    sys.modules[root.name] = module
    spec.loader.exec_module(module)
    return module


def _entry(arm: str) -> Callable[..., Any]:
    """The GDN entry point of an arm, the package imported on first use (``--list`` needs no GPU and no package)."""
    if arm == "det":
        from flashinfer.experimental.cake_gdn_det_train import (
            chunk_gated_delta_rule_det,
        )

        return chunk_gated_delta_rule_det
    if arm == "det_dir":
        if DET_DIR is None:
            raise SystemExit("the det_dir arm needs --det-dir")
        return _load_package_dir(DET_DIR).chunk_gated_delta_rule_det
    from flashinfer.experimental.cake.cake_gdn_train import chunk_gated_delta_rule

    pins = PIN_ARMS.get(arm, {})
    return functools.partial(chunk_gated_delta_rule, deterministic=False, **pins)


class Problem(NamedTuple):
    leaves: tuple[
        torch.Tensor, ...
    ]  # q, k, v, g, beta: autograd leaves shared by every arm
    do: torch.Tensor  # the upstream gradient of o
    cu_seqlens: torch.Tensor  # int32 offsets on the device
    cu_seqlens_cpu: torch.Tensor  # the same offsets on the host


def make_problem(
    h: int, hv: int, seq_lens: tuple[int, ...], seed: int, device: torch.device
) -> Problem:
    """``q, k`` BF16 ``[1, T, H, 128]`` (L2-normalised in the kernel), ``v, do`` BF16 ``[1, T, HV, 128]``, ``g`` FP32
    ``[1, T, HV]`` (natural-log decay, ``logsigmoid`` of a normal: negative) and ``beta`` FP32 ``[1, T, HV]`` in
    (0, 1), drawn on the device from ``seed``."""
    t = sum(seq_lens)
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    q = randn(1, t, h, HEAD_DIM).to(torch.bfloat16)
    k = randn(1, t, h, HEAD_DIM).to(torch.bfloat16)
    v = randn(1, t, hv, HEAD_DIM).to(torch.bfloat16)
    g = F.logsigmoid(randn(1, t, hv))
    beta = torch.sigmoid(randn(1, t, hv))
    do = randn(1, t, hv, HEAD_DIM).to(torch.bfloat16)
    cu_cpu = torch.tensor([0, *itertools.accumulate(seq_lens)], dtype=torch.int32)
    leaves = tuple(x.requires_grad_(True) for x in (q, k, v, g, beta))
    return Problem(leaves, do, cu_cpu.to(device), cu_cpu)


def make_forward(arm: str, problem: Problem) -> Callable[..., torch.Tensor]:
    """``forward(q, k, v, g, beta) -> o`` of an arm on the problem's packed layout."""
    entry = _entry(arm)
    cu, cu_cpu = problem.cu_seqlens, problem.cu_seqlens_cpu

    def forward(q, k, v, g, beta):
        o, _ = entry(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            use_qk_l2norm_in_kernel=True,
        )
        return o

    return forward


def _clear_grads(leaves: tuple[torch.Tensor, ...]) -> None:
    for t in leaves:
        t.grad = None


def _block_bytes(t: torch.Tensor) -> int:
    """The bytes the caching allocator accounts for ``t``'s storage (sizes round up to 512 bytes)."""
    return -(-t.untyped_storage().nbytes() // 512) * 512


# ----------------------------------------------------------------------------------------------------------------
# Check and memory (outside the timing)


def check_outputs(
    forwards: dict[str, Callable[..., torch.Tensor]],
    problem: Problem,
    device: torch.device,
) -> tuple[dict[str, bool], float | None, dict[str, int]]:
    """One forward + backward per arm: whether the output and the input gradients are finite, the relative difference
    ``max|o_nondet - o_det| / max|o_det|`` (when both arms run) and the net memory each arm's first call leaves
    allocated (its pool)."""
    leaves, do = problem.leaves, problem.do
    finite: dict[str, bool] = {}
    pool: dict[str, int] = {}
    outputs: dict[str, torch.Tensor] = {}
    for arm, forward in forwards.items():
        _clear_grads(leaves)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        o = forward(*leaves)
        torch.autograd.backward(o, do)
        o = o.detach()
        finite[arm] = bool(torch.isfinite(o).all()) and all(
            t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in leaves
        )
        _clear_grads(leaves)
        torch.cuda.synchronize(device)
        pool[arm] = torch.cuda.memory_allocated(device) - before - _block_bytes(o)
        outputs[arm] = o
    diff = None
    if "det" in outputs and "nondet" in outputs:
        ref = outputs["det"].float()
        diff = float((outputs["nondet"].float() - ref).abs().max() / ref.abs().max())
    return finite, diff, pool


def record_stages(
    forward: Callable[..., torch.Tensor], problem: Problem, device: torch.device
) -> dict[str, list[str]]:
    """One untimed forward + backward of ``forward`` (the non-deterministic arm) with the package's launches recorded:
    the stages, in launch order, of its forward and its backward on this shape (the launch recorder that shows which
    tail and which adjoint table the measured rows run)."""
    from flashinfer.experimental.cake.cake_gdn_train import cake_backend

    launches: list[str] = []
    launch = cake_backend.launch_stage

    def recording(stage, *rest, **kwargs):
        launches.append(stage)
        return launch(stage, *rest, **kwargs)

    leaves, do = problem.leaves, problem.do
    _clear_grads(leaves)
    cake_backend.launch_stage = recording
    try:
        o = forward(*leaves)
        fwd = list(launches)
        launches.clear()
        torch.autograd.backward(o, do)
        torch.cuda.synchronize(device)
        bwd = list(launches)
    finally:
        cake_backend.launch_stage = launch
        _clear_grads(leaves)
    return {"fwd": fwd, "bwd": bwd}


def _stage_counts(stages: list[str]) -> str:
    counts: dict[str, int] = {}
    for stage in stages:
        counts[stage] = counts.get(stage, 0) + 1
    return ", ".join(f"{stage} x{n}" for stage, n in counts.items())


def measure_memory(
    forward: Callable[..., torch.Tensor], problem: Problem, device: torch.device
) -> dict[str, int]:
    """Bytes relative to the baseline (inputs allocated, no graph, no gradient): retained after the forward with the
    graph alive and peak during forward + backward, for the plain and the checkpointed forward."""
    leaves, do = problem.leaves, problem.do

    def plain() -> torch.Tensor:
        return forward(*leaves)

    def checkpointed() -> torch.Tensor:
        return checkpoint(forward, *leaves, use_reentrant=False)

    result: dict[str, int] = {}
    for prefix, run in (("", plain), ("ckpt_", checkpointed)):
        _clear_grads(leaves)
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        result.setdefault("baseline_bytes", baseline)
        torch.cuda.reset_peak_memory_stats(device)
        o = run()
        torch.cuda.synchronize(device)
        result[prefix + "retained_bytes"] = (
            torch.cuda.memory_allocated(device) - baseline
        )
        torch.autograd.backward(o, do)
        torch.cuda.synchronize(device)
        result[prefix + "peak_temp_bytes"] = (
            torch.cuda.max_memory_allocated(device) - baseline
        )
        del o
    _clear_grads(leaves)
    return result


# ----------------------------------------------------------------------------------------------------------------
# Timing


class Step(NamedTuple):
    prepare: Callable[
        [], Any
    ]  # untimed, before the L2 flush; its result is passed to ``timed``
    timed: Callable[
        [Any], Any
    ]  # the timed region; its result (a graph for ``fwd``) is freed after the end event


def make_step(
    forward: Callable[..., torch.Tensor], problem: Problem, boundary: str
) -> Step:
    leaves, do = problem.leaves, problem.do

    def clear() -> None:
        _clear_grads(leaves)

    def forward_untimed() -> torch.Tensor:
        _clear_grads(leaves)
        return forward(*leaves)

    def fwd(_: Any) -> torch.Tensor:
        return forward(*leaves)

    def bwd(o: torch.Tensor) -> None:
        torch.autograd.backward(o, do)

    def fwdbwd(_: Any) -> None:
        torch.autograd.backward(forward(*leaves), do)

    def ckpt(_: Any) -> None:
        torch.autograd.backward(checkpoint(forward, *leaves, use_reentrant=False), do)

    if boundary == "bwd":
        return Step(forward_untimed, bwd)
    return Step(clear, {"fwd": fwd, "fwdbwd": fwdbwd, "ckpt": ckpt}[boundary])


def _sample(
    step: Step,
    flush: torch.Tensor,
    events: tuple[torch.cuda.Event, torch.cuda.Event],
    device: torch.device,
) -> float:
    """Milliseconds of one timed iteration: prepare, flush the L2, synchronize, then time ``step.timed``."""
    state = step.prepare()
    flush.zero_()
    torch.cuda.synchronize(device)
    start, end = events
    start.record()
    out = step.timed(state)
    end.record()
    torch.cuda.synchronize(device)
    del out, state
    return start.elapsed_time(end)


def _stats(steady: list[list[float]]) -> dict[str, Any]:
    group_medians = [statistics.median(s) for s in steady]
    median = statistics.median(group_medians)
    flat = [t for s in steady for t in s]
    deciles = statistics.quantiles(flat, n=10, method="inclusive")
    return {
        "group_medians_ms": group_medians,
        "median_ms": median,
        "spread": (max(group_medians) - min(group_medians)) / median,
        "p10_ms": deciles[0],
        "p90_ms": deciles[-1],
        "samples": len(flat),
    }


def run_protocol(
    steps: dict[str, Step],
    pattern: list[str],
    args: argparse.Namespace,
    flush: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """The grouped ABBA protocol of one boundary: per arm the statistics of its steady samples and the raw calls."""
    arms = list(dict.fromkeys(pattern))
    steady: dict[str, list[list[float]]] = {
        arm: [[] for _ in range(args.groups)] for arm in arms
    }
    calls: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    events = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    last: str | None = None
    for group in range(args.groups):
        for arm in pattern:
            previous = last
            times: list[float] = []
            for i in range(args.warmup + args.samples):
                ms = _sample(steps[arm], flush, events, device)
                if i < args.warmup:
                    continue
                times.append(ms)
                if (
                    i > 0 or previous == arm
                ):  # the iteration dispatched before this one ran the same arm
                    steady[arm][group].append(ms)
            last = arm
            calls[arm].append(
                {"group": group, "previous": previous, "samples_ms": times}
            )
    return {arm: {**_stats(steady[arm]), "calls": calls[arm]} for arm in arms}


# ----------------------------------------------------------------------------------------------------------------
# Driver


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.strip().splitlines()
    return lines[0].strip() if lines else None


def environment(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": props.name,
        "capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": _driver_version(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_shape(
    shape_id: str,
    arms: list[str],
    pattern: list[str],
    args: argparse.Namespace,
    flush: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    h, hv, seq_lens = SHAPES[shape_id]
    skip = {arm for arm, group in GROUP_ARMS.items() if hv != group * h}
    if skip & set(arms):  # a cluster tail serves its value-head group only
        arms = [arm for arm in arms if arm not in skip]
        pattern = [
            arm for x in arms if x != "nondet" for arm in ("nondet", x, x, "nondet")
        ]
    record: dict[str, Any] = {
        "id": shape_id,
        "H": h,
        "HV": hv,
        "seq_lens": list(seq_lens),
        "T": sum(seq_lens),
    }
    print(
        f"\n== {shape_id}: H={h} HV={hv} docs={len(seq_lens)} T={sum(seq_lens)}",
        flush=True,
    )
    problem = make_problem(h, hv, seq_lens, args.seed, device)
    forwards = {arm: make_forward(arm, problem) for arm in arms}
    finite, diff, pool = check_outputs(forwards, problem, device)
    record["finite"] = finite
    record["rel_max_diff_o_nondet_det"] = diff
    verdicts = " ".join(
        f"{arm}={'ok' if ok else 'NOT FINITE'}" for arm, ok in finite.items()
    )
    diff_text = "n/a" if diff is None else f"{diff:.3e}"
    print(
        f"  check: {verdicts} | max|o_nondet - o_det| / max|o_det| = {diff_text}",
        flush=True,
    )
    if not all(finite.values()):
        record["skipped"] = "non-finite output or input gradient: not timed"
        return record
    if "nondet" in forwards:
        record["stages_nondet"] = record_stages(forwards["nondet"], problem, device)
        print(
            f"  stages nondet bwd: {_stage_counts(record['stages_nondet']['bwd'])}",
            flush=True,
        )

    record["memory"] = {
        arm: {**measure_memory(forwards[arm], problem, device), "pool_bytes": pool[arm]}
        for arm in arms
    }
    others = [arm for arm in arms if arm != "nondet"]
    timings: dict[str, dict[str, dict[str, Any]]] = {}
    for boundary in args.boundaries:
        steps = {arm: make_step(forwards[arm], problem, boundary) for arm in arms}
        timings[boundary] = run_protocol(steps, pattern, args, flush, device)
    record["arms"] = {
        arm: {b: timings[b][arm] for b in args.boundaries} for arm in arms
    }
    record["ratios"] = {
        b: {
            f"{x}/nondet": timings[b][x]["median_ms"]
            / timings[b]["nondet"]["median_ms"]
            for x in others
        }
        for b in args.boundaries
    }
    record["ratio_base_over_nondet"] = {
        b: record["ratios"][b][f"{args.base}/nondet"] for b in args.boundaries
    }
    _print_shape(record, arms, others, args.boundaries)
    return record


def _print_shape(
    record: dict[str, Any], arms: list[str], others: list[str], boundaries: list[str]
) -> None:
    head = [f"{arm + ' ms':>12s} {'spread':>7s}" for arm in arms] + [
        f"{x + '/nondet':>15s}" for x in others
    ]
    print(f"  {'boundary':8s} | " + " | ".join(head))
    for b in boundaries:
        cells = [
            f"{record['arms'][arm][b]['median_ms']:12.4f} {100 * record['arms'][arm][b]['spread']:6.2f}%"
            for arm in arms
        ]
        cells += [f"{record['ratios'][b][x + '/nondet']:15.4f}" for x in others]
        print(f"  {b:8s} | " + " | ".join(cells))
    print(
        f"  {'memory MiB':10s} {'baseline':>10s} {'pool':>9s} {'retained':>10s} {'peak-temp':>10s} "
        f"{'ckpt-retained':>14s} {'ckpt-peak-temp':>15s}"
    )
    for arm in arms:
        m = record["memory"][arm]
        print(
            f"  {arm:10s} {m['baseline_bytes'] / MIB:10.1f} {m['pool_bytes'] / MIB:9.1f} "
            f"{m['retained_bytes'] / MIB:10.1f} {m['peak_temp_bytes'] / MIB:10.1f} "
            f"{m['ckpt_retained_bytes'] / MIB:14.1f} {m['ckpt_peak_temp_bytes'] / MIB:15.1f}"
        )
    sys.stdout.flush()


def _list_shapes() -> None:
    for set_name, shapes in SETS.items():
        for shape_id, (h, hv, seq_lens) in shapes.items():
            print(
                f"{set_name:6s} {shape_id:26s} H={h:<3d} HV={hv:<3d} docs={len(seq_lens):<3d} T={sum(seq_lens)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cake GDN training step on one GPU: the deterministic package against the non-deterministic one."
    )
    parser.add_argument(
        "--shapes",
        default="rows,packs,pairs",
        help="comma-separated set names (rows, packs, pairs) or shape ids",
    )
    parser.add_argument(
        "--arms",
        default="det,nondet",
        help="comma-separated arms of " + ", ".join(ARMS),
    )
    parser.add_argument(
        "--det-dir",
        default=None,
        help="package directory of the det_dir arm (a deterministic build)",
    )
    parser.add_argument(
        "--base",
        default="det",
        choices=BASES,
        help="the B arm of the headline ratio base / nondet",
    )
    parser.add_argument(
        "--boundaries",
        default=",".join(BOUNDARIES),
        help="comma-separated boundaries of fwd, bwd, fwdbwd, ckpt",
    )
    parser.add_argument(
        "--groups", type=int, default=5, help="measurement groups (at least 3)"
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="timed iterations per call (at least 8)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="untimed iterations per call"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the inputs (drawn once per shape)"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="write the report here (rewritten after every shape)",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the shape ids and exit"
    )
    args = parser.parse_args()
    if args.list:
        _list_shapes()
        return 0
    try:
        shapes = parse_shapes(args.shapes)
        requested = parse_names(args.arms, ARMS, "--arms")
        args.boundaries = parse_names(args.boundaries, BOUNDARIES, "--boundaries")
    except ValueError as err:
        parser.error(str(err))
    global DET_DIR
    DET_DIR = args.det_dir
    if "det_dir" in requested and DET_DIR is None:
        parser.error("the det_dir arm needs --det-dir")
    if args.groups < 3 or args.samples < 8 or args.warmup < 0:
        parser.error(
            "the protocol takes at least 3 groups, 8 samples per call and a non-negative warmup"
        )
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")

    # nondet is the A arm of every ABBA quartet; --base is the first B arm, the other requested arms follow
    arms = list(dict.fromkeys(["nondet", args.base, *requested]))
    others = [arm for arm in arms if arm != "nondet"]
    pattern = [arm for x in others for arm in ("nondet", x, x, "nondet")]
    device = torch.device("cuda", torch.cuda.current_device())
    env = environment(device)
    print(
        f"{env['gpu']} sm_{env['capability'][0]}{env['capability'][1]} SMs={env['sm_count']} torch {env['torch']} "
        f"CUDA {env['cuda']} driver {env['driver']} | arms {arms} base {args.base} groups={args.groups} "
        f"samples={args.samples} warmup={args.warmup}",
        flush=True,
    )
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    report: dict[str, Any] = {
        "env": env,
        "args": vars(args),
        "arms": arms,
        "pattern": pattern,
        "shapes": [],
    }
    status = 0
    for shape_id in shapes:
        record = run_shape(shape_id, arms, pattern, args, flush, device)
        report["shapes"].append(record)
        if "skipped" in record:
            status = 1
        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        torch.cuda.empty_cache()
    report["regressions"] = regressions(report["shapes"], args.boundaries)
    print(
        "\nregressions (nondet slower than the arm beyond the sum of both spreads):",
        flush=True,
    )
    for line in report["regressions"] or ["none"]:
        print(f"  {line}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    return status


def regressions(records: list[dict[str, Any]], boundaries) -> list[str]:
    """The (shape, boundary, arm) cells where the non-deterministic arm is slower than the ``det`` / ``det_dir`` arm by
    more than the sum of the two arms' spreads (X_ms / nondet_ms below 1 - spread)."""
    out = []
    for rec in records:
        arms = rec.get("arms") or {}
        for arm in ("det", "det_dir"):
            if arm not in arms:
                continue
            for b in boundaries:
                x, n = arms[arm][b], arms["nondet"][b]
                ratio = x["median_ms"] / n["median_ms"]
                if ratio < 1.0 - (x["spread"] + n["spread"]):
                    out.append(
                        f"{rec['id']} {b}: {arm}/nondet = {ratio:.4f} (spreads {100 * x['spread']:.2f}% / "
                        f"{100 * n['spread']:.2f}%)"
                    )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
