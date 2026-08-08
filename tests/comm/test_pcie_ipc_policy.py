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

Properties of the PCIe IPC tuning table.

The table is a pure function, and everything else about this collective rests on
that: every rank derives its own launch configuration with no runtime agreement,
so a table that answered differently on two ranks -- or that answered with a
configuration the kernel rejects -- would hang the group rather than fail.

These are the only tests for this feature that need no GPU.
"""

import importlib.util
import pathlib

import pytest

from flashinfer.comm.pcie_ipc_policy import (
    MAX_BLOCKS,
    _is_launchable,
    get_pcie_ipc_launch_config,
)
from flashinfer.comm.pcie_ipc_topology import PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR

_SHAPES = [
    (2, 2048),
    (4, 4096),
    (8, 6144),
    (8, 8192),
]
_BATCHES = list(range(1, 257))


@pytest.mark.parametrize("profile", [PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR])
def test_every_returned_config_is_launchable(profile: str) -> None:
    """A config the kernel would reject must never leave the table.

    The C++ side hard-checks these. Reaching them means one rank raises while
    its peers are already spinning in the collective.
    """
    for world_size, hidden in _SHAPES:
        for batch in _BATCHES:
            config = get_pcie_ipc_launch_config(profile, world_size, hidden, batch)
            if config is None:
                continue
            assert _is_launchable(world_size, config, MAX_BLOCKS), (
                f"{profile} ws={world_size} hidden={hidden} batch={batch} "
                f"-> {config}, which the kernel rejects"
            )
            assert 0 < config.blocks <= MAX_BLOCKS
            assert world_size <= config.threads <= 1024


@pytest.mark.parametrize("profile", [PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR])
def test_table_is_a_pure_function(profile: str) -> None:
    """Same arguments, same answer -- no hidden state, no rank-local input."""
    for world_size, hidden in _SHAPES:
        for batch in (1, 2, 3, 8, 9, 44, 45, 128, 256):
            first = get_pcie_ipc_launch_config(profile, world_size, hidden, batch)
            for _ in range(3):
                assert (
                    get_pcie_ipc_launch_config(profile, world_size, hidden, batch)
                    == first
                )


def test_tp8_rootcplx_batch2_falls_back() -> None:
    """Batch 2 at 8 ranks is reported unsupported on purpose.

    This is a deliberate divergence from the table this was ported from, which
    runs the pack kernel here. Measured on 8x L40S, every sample of this shape
    loses to NCCL (0.67x-0.95x) and the latency is unstable, 39-61 us, while
    batch 1 -- same kernel, same 32 blocks -- holds 30.5-32.2 us. All twelve
    launchable configurations were measured; none is better.

    Pinned as a test because the natural instinct on seeing a hole in a tuning
    table is to fill it back in.
    """
    assert get_pcie_ipc_launch_config(PROFILE_ROOTCPLX, 8, 6144, 2) is None
    # Exactly one point, not a range: its neighbours stay supported.
    assert get_pcie_ipc_launch_config(PROFILE_ROOTCPLX, 8, 6144, 1) is not None
    assert get_pcie_ipc_launch_config(PROFILE_ROOTCPLX, 8, 6144, 3) is not None
    # Only the profile that was measured. The switch-paired table came from a
    # different machine and has no measurement backing a change here.
    assert get_pcie_ipc_launch_config(PROFILE_SWITCHPAIR, 8, 6144, 2) is not None


def test_untuned_shapes_report_unsupported() -> None:
    """Outside the three tuned shapes the table declines rather than guesses."""
    for profile in (PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR):
        # TP4 is tuned at exactly 4096.
        assert get_pcie_ipc_launch_config(profile, 4, 2048, 8) is None
        assert get_pcie_ipc_launch_config(profile, 4, 8192, 8) is None
        # TP2 only up to 2048.
        assert get_pcie_ipc_launch_config(profile, 2, 4096, 8) is None
        # World sizes the kernels do not implement.
        assert get_pcie_ipc_launch_config(profile, 6, 6144, 8) is None


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        get_pcie_ipc_launch_config("no-such-profile", 4, 4096, 8)


@pytest.mark.parametrize("profile", [PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR])
def test_tp8_block_kernel_always_gets_a_multiple_of_four(profile: str) -> None:
    """The block-partitioned TP8 kernel derives its chunk from ``blockIdx.x & 3``."""
    for hidden in (6144, 8192):
        for batch in _BATCHES:
            config = get_pcie_ipc_launch_config(profile, 8, hidden, batch)
            if config is None or not config.stream_mode or config.ring_push:
                continue
            assert config.blocks % 4 == 0, f"{profile} batch={batch} -> {config}"


def _load_benchmark_module():
    """Import the benchmark as a module so its pure helpers can be tested."""
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "comm"
        / "bench_pcie_ipc_all_reduce.py"
    )
    spec = importlib.util.spec_from_file_location("_bench_pcie_ipc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_ab_picks_the_right_history_per_kernel() -> None:
    """The A/B baseline must match what each kernel actually used to run.

    The kernels do not share one history: TP2, TP4 and the TP8 pack kernel
    double-buffered by per-block parity, while the two staged TP8 kernels had no
    epoch at all. Comparing against the wrong one measures a protocol that never
    shipped, which is not a performance result -- and the tuning table decides
    which kernel a shape lands on, so a table edit could silently flip a row's
    label. Pinned here for that reason.
    """
    bench = _load_benchmark_module()
    pick = bench._historical_switch

    for profile in (PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR):
        for world_size, hidden in ((2, 2048), (4, 4096), (8, 6144)):
            for batch in _BATCHES:
                config = get_pcie_ipc_launch_config(profile, world_size, hidden, batch)
                if config is None:
                    continue
                got = pick(world_size, config)
                staged_tp8 = world_size == 8 and config.stream_mode
                want = "no-block-epoch" if staged_tp8 else "per-block-epoch"
                assert got == want, (
                    f"{profile} ws={world_size} batch={batch} -> {config}: "
                    f"baseline {got}, expected {want}"
                )

    # Both branches must be reachable, or the check above is vacuous.
    reached = {
        pick(w, get_pcie_ipc_launch_config(PROFILE_ROOTCPLX, w, h, b))
        for w, h, b in ((2, 2048, 8), (4, 4096, 8), (8, 6144, 1), (8, 6144, 8))
    }
    assert reached == {"per-block-epoch", "no-block-epoch"}


def _shape_config(profile, world_size, hidden, batches):
    out = {}
    for batch in batches:
        config = get_pcie_ipc_launch_config(profile, world_size, hidden, batch)
        if config is not None:
            out[batch] = config
    return out


def test_protocol_ab_plan_partitions_shapes_by_history() -> None:
    """The history split must drive execution, not just labelling.

    Labelling a row SYNTHETIC after the fact does not undo running it, and one
    of those runs -- the fixed-half build on a kernel that always
    double-buffered -- wedges the sentinel loop, taking the whole benchmark down
    before it reports anything. So the plan each leg executes is asserted here,
    not only the label it would print.
    """
    bench = _load_benchmark_module()
    plan_of = bench._protocol_ab_plan
    batches = [1, 2, 4, 8, 16, 32, 64, 128]

    # 2 and 4 ranks never needed the fixed-half build, so it must not run there.
    for world_size, hidden in ((2, 2048), (4, 4096)):
        shapes = _shape_config(PROFILE_ROOTCPLX, world_size, hidden, batches)
        plan = plan_of(world_size, shapes, "auto")
        assert set(plan) == {"per-block-epoch"}
        assert plan["per-block-epoch"] == sorted(shapes)

    # 8 ranks straddles both histories, which is the whole reason auto exists.
    shapes = _shape_config(PROFILE_ROOTCPLX, 8, 6144, batches)
    plan = plan_of(8, shapes, "auto")
    assert set(plan) == {"per-block-epoch", "no-block-epoch"}
    pack = plan["per-block-epoch"]
    staged = plan["no-block-epoch"]
    assert pack, "the pack kernel must still be covered"
    assert staged, "the staged kernels must still be covered"
    assert not set(pack) & set(staged), "a shape may only run under its own history"
    assert sorted(pack + staged) == sorted(shapes), "every tuned shape must run once"
    for batch in pack:
        assert not shapes[batch].stream_mode
    for batch in staged:
        assert shapes[batch].stream_mode

    # The one shape deliberately handed back to NCCL runs in neither leg.
    assert get_pcie_ipc_launch_config(PROFILE_ROOTCPLX, 8, 6144, 2) is None
    assert 2 not in pack and 2 not in staged

    # An explicit switch is a request for that exact comparison, so it keeps
    # every shape and reports the mismatched ones rather than dropping them.
    explicit = plan_of(8, shapes, "per-block-epoch")
    assert explicit == {"per-block-epoch": sorted(shapes)}
    assert plan_of(8, {}, "auto") == {}


def test_protocol_ab_legs_execute_only_their_own_shapes() -> None:
    """The plan must drive execution, not sit beside it.

    An earlier version of the harness computed the split correctly and then
    ignored it, running every shape under every switch and discarding the
    mismatched rows afterwards. Discarding is too late: the kernel has already
    run, and the fixed-half build on a kernel that always double-buffered wedges
    its spin loop rather than returning a bad number. A test of the planner
    alone stayed green through all of that, so this one records what each leg is
    actually handed.
    """
    bench = _load_benchmark_module()
    batches = [1, 2, 4, 8, 16, 32, 64, 128]
    shapes = _shape_config(PROFILE_ROOTCPLX, 8, 6144, batches)
    plan = bench._protocol_ab_plan(8, shapes, "auto")

    calls = []

    def recording_sweep(broken_switch, leg_batches):
        calls.append((broken_switch, tuple(leg_batches)))
        return {b: ("timing", shapes[b], True, True) for b in leg_batches}

    legs = bench._run_ab_legs(plan, recording_sweep)

    assert set(legs) == set(plan)
    for switch, leg_batches in plan.items():
        # A-B-A: broken, shipping, broken -- all three over the same shapes.
        assert (switch, tuple(leg_batches)) in calls
        assert (None, tuple(leg_batches)) in calls
        assert len(legs[switch]) == 3
        for leg in legs[switch]:
            assert set(leg) == set(leg_batches)

    # Nothing ran outside its own plan entry.
    for broken_switch, ran in calls:
        owner = broken_switch if broken_switch is not None else None
        if owner is not None:
            assert set(ran) == set(plan[owner]), (
                f"{owner} leg ran {sorted(ran)}, owns {plan[owner]}"
            )
        else:
            assert any(set(ran) == set(v) for v in plan.values())

    # Each switch is exercised three times and no more.
    for switch, leg_batches in plan.items():
        assert calls.count((switch, tuple(leg_batches))) == 2  # the two A legs
        assert calls.count((None, tuple(leg_batches))) == 1  # the shipping leg


def test_group_correctness_uses_a_min_reduction() -> None:
    """Correctness must be an AND across ranks, which means ReduceOp.MIN.

    The cross-island race this harness rebuilds produces errors confined to one
    island, so a rank-local verdict -- or a SUM, or a MAX -- would let a clean
    rank 0 publish a cost for a baseline that was wrong on ranks 4-7. That
    happened once. The op is checked here rather than on hardware because the
    failure is a silent wrong verdict, not a crash, and no GPU is needed to see
    which reduction was asked for.
    """
    bench = _load_benchmark_module()
    calls = []

    class _FakeTensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    def fake_all_reduce(tensor, op=None, group=None):
        calls.append(op)

    real_dist = bench.dist
    real_tensor = bench.torch.tensor
    try:
        bench.dist = type(
            "D",
            (),
            {
                "all_reduce": staticmethod(fake_all_reduce),
                "ReduceOp": real_dist.ReduceOp,
            },
        )
        bench.torch = type(
            "T",
            (),
            {"tensor": staticmethod(lambda v, **kw: _FakeTensor(v[0])), "int32": None},
        )
        assert bench._group_all(True, None, None) is True
        assert bench._group_all(False, None, None) is False
    finally:
        bench.dist = real_dist
        bench.torch.tensor = real_tensor

    assert calls, "the verdict must be reduced across ranks, not decided locally"
    assert all(op is real_dist.ReduceOp.MIN for op in calls), (
        f"correctness must reduce with MIN (logical AND), got {calls}"
    )
