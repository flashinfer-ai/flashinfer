# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""M-range sweep for the hidden_size=5120 MNNVL CuTe DSL fusion presets.

These profiles cover ``hidden_size=5120`` in bf16 at ``top_k`` 6 and 3.  This
script times the three MNNVL CuTe DSL protocols (LL / BT / HT) against each
other over a token sweep for both fusion patterns, so the M-range boundaries
of the H5120 routes in ``flashinfer/comm/mnnvl_cutedsl/presets.py`` are
measured rather than guessed.

It also times an unfused reference stack (NCCL all-reduce plus FlashInfer's
``fused_add_rmsnorm``, preceded by a torch MoE finalize for the finalize
pattern) so the fusion speedup can be quoted in absolute terms.

Timing methodology
------------------
These are collective kernels: a rank's kernel spins until every peer arrives,
so any host-side launch skew between ranks lands inside the measurement.  With
per-iteration CUPTI or CUDA-event timing that skew dominates completely -- a
4 us LL all-reduce measures at ~235 us.  Following the precedent in
``benchmarks/comm/bench_quantized_allreduce.py``, timing therefore captures
``--in-graph-iters`` back-to-back calls in one CUDA graph and divides: the
first call in a replay absorbs the residual skew and the rest run at the true
steady-state rate.  Graph replay was verified to reproduce the eager numerics
for all three protocols (the Lamport generation counter lives in device
memory, so it advances on replay exactly as it does eagerly).

Cold-L2 rotation is off: with a zero-argument closure there is nothing for
``bench_gpu_time`` to rotate, and the back-to-back steady state these presets
are dispatched for is the warm-cache one.

Usage::

    FLASHINFER_DISABLE_VERSION_CHECK=1 torchrun --nproc_per_node=4 \
        benchmarks/comm/bench_mnnvl_cutedsl_h5120.py

Options::

    --m-list 1,2,4,...    Token counts to sweep (default: a log-ish ladder)
    --top-k 6,3           top_k values to cover (default: both)
    --protocols ll,bt,ht  Protocols to time (default: all three).  Also
                          accepts bt0/bt1, which pin the BT route to a
                          single preset so the preset split is measurable.
    --patterns ar,finalize
    --no-baseline         Skip the unfused NCCL + fused_add_rmsnorm reference
    --csv FILE            Also write the raw rows to CSV (rank 0 only)
    --dry-run-iters N     Warmup replays per point (default: 10)
    --repeat-iters N      Measured replays per point (default: 30)
    --in-graph-iters N    Calls captured per graph (default: 20)
"""

import argparse
import csv
import os
import statistics
import sys

import torch
import torch.distributed as dist

from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion
from flashinfer.comm.mnnvl_cutedsl import (
    BT_ONLY_CONFIG,
    HT_ONLY_CONFIG,
    LL_ONLY_CONFIG,
)
from flashinfer.comm.mnnvl_cutedsl.config import (
    KernelTarget,
    MNNVLCuteDSLConfig,
    MRangeDispatch,
    ProtocolKind,
    StaticProfile,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
    BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_0,
    BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_1,
    BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_0,
    BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_1,
    BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_0,
    BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_1,
    BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_0,
    BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_1,
    BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_0,
    BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_1,
    BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_0,
    BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_1,
)
from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace
from flashinfer.norm import fused_add_rmsnorm
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm100a_supported

# Static shape shared by both stages.
HIDDEN_SIZE = 5120
TOP_K_STAGES = (6, 3)
RMS_EPS = 1e-6
WEIGHT_BIAS = 1.0

SHIPPED_CONFIGS = {
    "ll": LL_ONLY_CONFIG,
    "bt": BT_ONLY_CONFIG,
    "ht": HT_ONLY_CONFIG,
}

# The shipped BT route splits between two presets, so a plain
# "bt" column cannot say whether that split sits in the right place. "bt0"/"bt1"
# pin one preset across the whole sweep, which makes the split measurable. The
# public workspace takes `config=`, so this needs no library change.
_BT_FINALIZE_PRESETS = {
    (4, 6): (
        BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_0,
        BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_1,
    ),
    (4, 3): (
        BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_0,
        BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_1,
    ),
    (8, 6): (
        BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_0,
        BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_1,
    ),
    (8, 3): (
        BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_0,
        BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_1,
    ),
}
_BT_ALL_REDUCE_PRESETS = {
    4: (
        BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_0,
        BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_1,
    ),
    8: (
        BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_0,
        BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_1,
    ),
}


def _single_bt_preset_config(preset_index: int, tp_size: int, top_k: int):
    """A BT-only config pinned to one preset over every M."""

    def unbounded(preset):
        return MRangeDispatch(
            upper_bounds=(None,),
            targets=(KernelTarget(protocol=ProtocolKind.BT, preset=preset),),
        )

    return MNNVLCuteDSLConfig(
        profiles=(
            StaticProfile(
                tp_size=tp_size,
                hidden_size=HIDDEN_SIZE,
                top_k=top_k,
                dtype=torch.bfloat16,
                finalize_routes=unbounded(
                    _BT_FINALIZE_PRESETS[(tp_size, top_k)][preset_index]
                ),
                all_reduce_routes=unbounded(
                    _BT_ALL_REDUCE_PRESETS[tp_size][preset_index]
                ),
            ),
        )
    )


def _config_for(protocol: str, tp_size: int, top_k: int):
    if protocol in SHIPPED_CONFIGS:
        return SHIPPED_CONFIGS[protocol]
    if protocol in ("bt0", "bt1"):
        return _single_bt_preset_config(int(protocol[-1]), tp_size, top_k)
    raise KeyError(f"Unknown protocol {protocol!r}")


PATTERNS = {
    "ar": AllReduceFusionPattern.kARResidualRMSNorm,
    "finalize": AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
}

DEFAULT_M_LIST = (
    1,
    2,
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    64,
    80,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)


def _log(rank: int, message: str = "") -> None:
    if rank == 0:
        print(message, flush=True)


class Inputs:
    """Max-sized tensors, sliced per M so allocation stays out of the sweep."""

    def __init__(self, max_m: int, max_top_k: int, device: torch.device):
        generator = torch.Generator(device=device).manual_seed(2026)
        kw = {"dtype": torch.bfloat16, "device": device}
        self.local = torch.randn(max_m, HIDDEN_SIZE, generator=generator, **kw)
        self.residual = torch.randn(max_m, HIDDEN_SIZE, generator=generator, **kw)
        self.gamma = torch.randn(HIDDEN_SIZE, generator=generator, **kw)
        self.routed = torch.randn(
            max_m * max_top_k, HIDDEN_SIZE, generator=generator, **kw
        )
        self.weights = torch.randn(max_m, max_top_k, generator=generator, **kw)
        self.shared = torch.randn(max_m, HIDDEN_SIZE, generator=generator, **kw)
        self.residual_out = torch.empty(max_m, HIDDEN_SIZE, **kw)
        self.norm_out = torch.empty(max_m, HIDDEN_SIZE, **kw)
        # The finalize pattern reads residual_in; these profiles feed it zeros.
        self.zero_residual = torch.zeros(max_m, HIDDEN_SIZE, **kw)
        self._indices: dict[tuple[int, int], torch.Tensor] = {}
        self._device = device

    def indices(self, m: int, top_k: int) -> torch.Tensor:
        key = (m, top_k)
        if key not in self._indices:
            self._indices[key] = torch.arange(
                m * top_k, dtype=torch.int32, device=self._device
            ).reshape(m, top_k)
        return self._indices[key]


def _fused_call(inputs: Inputs, workspace, pattern_name: str, m: int, top_k: int):
    """Return a zero-arg closure running one fused call at token count ``m``."""
    residual_out = inputs.residual_out[:m]
    norm_out = inputs.norm_out[:m]
    gamma = inputs.gamma
    if pattern_name == "ar":
        local = inputs.local[:m]
        residual_in = inputs.residual[:m]

        def run_ar() -> None:
            allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=PATTERNS["ar"],
                launch_with_pdl=True,
                residual_in=residual_in,
                residual_out=residual_out,
                norm_out=norm_out,
                rms_gamma=gamma,
                rms_eps=RMS_EPS,
                weight_bias=WEIGHT_BIAS,
            )

        return run_ar

    routed = inputs.routed[: m * top_k]
    weights = inputs.weights[:m, :top_k].contiguous()
    shared = inputs.shared[:m]
    indices = inputs.indices(m, top_k)
    zero_residual = inputs.zero_residual[:m]

    def run_finalize() -> None:
        allreduce_fusion(
            input=routed,
            workspace=workspace,
            pattern=PATTERNS["finalize"],
            launch_with_pdl=True,
            residual_in=zero_residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=gamma,
            rms_eps=RMS_EPS,
            expanded_idx_to_permuted_idx=indices,
            expert_scale_factor=weights,
            shared_expert_output=shared,
            weight_bias=WEIGHT_BIAS,
        )

    return run_finalize


def _baseline_call(inputs: Inputs, group, pattern_name: str, m: int, top_k: int):
    """Unfused reference: (torch finalize +) NCCL all-reduce + fused_add_rmsnorm.

    ``fused_add_rmsnorm`` folds the residual add and the norm but has no
    ``weight_bias``, so the bias is pre-added to gamma -- mathematically the
    same normalisation the fused kernel performs.  It updates ``residual`` in
    place and overwrites its input with the normalised result, so repeated
    iterations stay bounded rather than diverging.
    """
    gamma_biased = (inputs.gamma.float() + WEIGHT_BIAS).to(torch.bfloat16)
    residual = inputs.residual[:m].clone()

    if pattern_name == "ar":
        buffer = inputs.local[:m].clone()

        def run_ar() -> None:
            dist.all_reduce(buffer, group=group)
            fused_add_rmsnorm(buffer, residual, gamma_biased, RMS_EPS)

        return run_ar

    routed = inputs.routed[: m * top_k]
    weights = inputs.weights[:m, :top_k].contiguous()
    shared = inputs.shared[:m]
    indices = inputs.indices(m, top_k).to(torch.int64)
    accumulator = torch.zeros(m, HIDDEN_SIZE, dtype=torch.float32, device=shared.device)

    def run_finalize() -> None:
        # Route-at-a-time fp32 accumulation, matching the reference finalize in
        # tests/comm/test_mnnvl_cutedsl_numerical_contract.py. Materialising the
        # whole (m, top_k, hidden) gather instead would cost far more memory
        # than any real unfused stack pays.
        accumulator.zero_()
        for route in range(top_k):
            rows = indices[:, route]
            torch.addcmul(
                accumulator,
                routed.index_select(0, rows).float(),
                weights[:, route, None].float(),
                out=accumulator,
            )
        finalized = (accumulator + shared.float()).to(torch.bfloat16)
        dist.all_reduce(finalized, group=group)
        fused_add_rmsnorm(finalized, residual, gamma_biased, RMS_EPS)

    return run_finalize


def _build_workspace(protocol: str, top_k: int, capacity_m: int, group):
    """Construct a protocol-pinned workspace, or ``(None, reason)``."""
    try:
        workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
            tp_size=dist.get_world_size(group),
            tp_rank=dist.get_rank(group),
            max_token_num=capacity_m,
            hidden_dim=HIDDEN_SIZE,
            dtype=torch.bfloat16,
            group=group,
            top_k=top_k,
            rms_eps=RMS_EPS,
            weight_bias=WEIGHT_BIAS,
            config=_config_for(protocol, dist.get_world_size(group), top_k),
        )
    except (ValueError, KeyError, RuntimeError) as error:
        return None, str(error)
    torch.cuda.synchronize()
    dist.barrier(group)
    return workspace, None


def _measure(run, group, args) -> float:
    """Median per-call GPU time in microseconds, max-aggregated across ranks.

    ``bench_gpu_time`` all-gathers each iteration and keeps the max, which is
    the right figure for a collective: the point at which every rank is done.
    """
    dist.barrier(group)
    times_ms = bench_gpu_time(
        run,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.repeat_iters,
        use_cuda_graph=True,
        num_iters_within_graph=args.in_graph_iters,
        cold_l2_cache=False,
    )
    dist.barrier(group)
    return statistics.median(times_ms) * 1e3


def _report_crossovers(rank, m_list, rows, top_k, pattern_name, active) -> None:
    """Print the fastest protocol per M and the M-range boundaries it implies."""
    if len(active) < 2:
        return
    lookup = {
        (row["backend"], row["m"]): row["us"]
        for row in rows
        if row["top_k"] == top_k and row["pattern"] == pattern_name
    }
    winners = []
    for m in m_list:
        candidates = []
        for protocol in active:
            micros = lookup.get((protocol, m))
            if micros is not None and micros == micros:  # drop NaN (failed point)
                candidates.append((micros, protocol))
        if candidates:
            winners.append((m, min(candidates)[1]))
    if not winners:
        return
    # Emit one segment per contiguous run of the same winner, labelled with the
    # swept endpoints. The upper endpoint of a run is exactly the bound to put
    # in presets.py: MRangeDispatch sends M <= bound to that run's protocol.
    segments = []
    run_start, run_winner = winners[0]
    for index, (m, winner) in enumerate(winners[1:], start=1):
        if winner != run_winner:
            run_end = winners[index - 1][0]
            segments.append(f"M {run_start}..{run_end}: {run_winner.upper()}")
            run_start, run_winner = m, winner
    segments.append(f"M {run_start}..{winners[-1][0]}: {run_winner.upper()}")
    _log(rank, f"  fastest by M -> {', '.join(segments)}")


def _sweep(rank, args, inputs, workspaces, group, top_k, pattern_name, rows) -> None:
    tp_size = dist.get_world_size(group)
    label = (
        f"top_k={top_k} pattern=finalize"
        if pattern_name == "finalize"
        else "pattern=ar (top_k independent)"
    )
    _log(rank, f"\n--- {label} --- (microseconds)")
    active = [p for p in args.protocols if workspaces[p] is not None]
    header = f"{'M':>6} | " + " | ".join(f"{p.upper():>10}" for p in active)
    if not args.no_baseline:
        header += f" | {'unfused':>10} | {'speedup':>9}"
    _log(rank, header)
    _log(rank, "-" * len(header))

    for m in m_list_for(args):
        cells = []
        measured = {}
        for protocol in active:
            run = _fused_call(inputs, workspaces[protocol], pattern_name, m, top_k)
            try:
                micros = _measure(run, group, args)
            except Exception as error:  # noqa: BLE001 - record and keep sweeping
                _log(
                    rank,
                    f"  [{protocol}] M={m} failed: {type(error).__name__}: {error}",
                )
                micros = float("nan")
            measured[protocol] = micros
            cells.append(f"{micros:>10.2f}")
            rows.append(
                {
                    "tp_size": tp_size,
                    "hidden_size": HIDDEN_SIZE,
                    "top_k": top_k,
                    "pattern": pattern_name,
                    "backend": protocol,
                    "m": m,
                    "us": micros,
                }
            )
        if not args.no_baseline:
            run = _baseline_call(inputs, group, pattern_name, m, top_k)
            try:
                base = _measure(run, group, args)
            except Exception as error:  # noqa: BLE001
                _log(rank, f"  [unfused] M={m} failed: {type(error).__name__}: {error}")
                base = float("nan")
            finite = [value for value in measured.values() if value == value]
            best = min(finite) if finite else float("nan")
            cells.append(f"{base:>10.2f}")
            cells.append(f"{base / best:>8.2f}x")
            rows.append(
                {
                    "tp_size": tp_size,
                    "hidden_size": HIDDEN_SIZE,
                    "top_k": top_k,
                    "pattern": pattern_name,
                    "backend": "unfused",
                    "m": m,
                    "us": base,
                }
            )
        _log(rank, f"{m:>6} | " + " | ".join(cells))

    _report_crossovers(rank, m_list_for(args), rows, top_k, pattern_name, active)


def m_list_for(args) -> tuple[int, ...]:
    return args.m_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-list", type=str, default=None)
    parser.add_argument("--top-k", type=str, default=",".join(map(str, TOP_K_STAGES)))
    parser.add_argument("--protocols", type=str, default="ll,bt,ht")
    parser.add_argument("--patterns", type=str, default="ar,finalize")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument("--in-graph-iters", type=int, default=20)
    args = parser.parse_args()

    args.m_values = (
        tuple(int(value) for value in args.m_list.split(","))
        if args.m_list
        else DEFAULT_M_LIST
    )
    top_ks = tuple(int(value) for value in args.top_k.split(","))
    args.protocols = tuple(value.strip() for value in args.protocols.split(","))
    patterns = tuple(value.strip() for value in args.patterns.split(","))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not is_sm100a_supported(device):
        print("SM100 or newer data-center Blackwell is required", file=sys.stderr)
        return 1

    dist.init_process_group("nccl", device_id=device)
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    tp_size = dist.get_world_size(group)
    max_m = max(args.m_values)

    _log(rank, "=" * 96)
    _log(
        rank,
        "  MNNVL CuTe DSL fusion sweep (hidden=5120) -- "
        f"{tp_size}x {torch.cuda.get_device_name(device)}",
    )
    _log(
        rank,
        f"  hidden_size={HIDDEN_SIZE} dtype=bfloat16 tp_size={tp_size} "
        f"rms_eps={RMS_EPS} weight_bias={WEIGHT_BIAS}",
    )
    _log(
        rank,
        "  bench_gpu_time(use_cuda_graph=True, num_iters_within_graph="
        f"{args.in_graph_iters}, cold_l2_cache=False); dry_run_iters="
        f"{args.dry_run_iters} repeat_iters={args.repeat_iters}",
    )
    _log(rank, "  Reported: median over replays of the max across ranks.")
    _log(rank, "=" * 96)

    inputs = Inputs(max_m, max(top_ks), device)
    rows: list[dict] = []

    for top_k in top_ks:
        workspaces = {}
        for protocol in args.protocols:
            workspace, error = _build_workspace(protocol, top_k, max_m, group)
            if workspace is None:
                _log(rank, f"\n[skip] {protocol.upper()} top_k={top_k}: {error}")
            workspaces[protocol] = workspace

        for pattern_name in patterns:
            # The all-reduce pattern ignores top_k; time it once rather than
            # twice over identical inputs.
            if pattern_name == "ar" and top_k != top_ks[0]:
                continue
            _sweep(rank, args, inputs, workspaces, group, top_k, pattern_name, rows)

        for workspace in workspaces.values():
            if workspace is not None:
                workspace.destroy()
        dist.barrier(group)

    if args.csv and rank == 0 and rows:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        _log(rank, f"\nWrote {len(rows)} rows to {args.csv}")

    dist.barrier(group)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
