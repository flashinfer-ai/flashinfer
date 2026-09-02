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

Properties of the PCIe IPC launch policy.

The policy is a pure function, and everything else about this collective rests
on that: every rank derives its own launch configuration with no runtime
agreement, so a policy that answered differently on two ranks -- or that
answered with a configuration the kernel rejects -- would hang the group rather
than fail.

Two layers with different standing, tested differently. *Admission* is a
capability claim and is pinned exactly: it decides which shapes reach the
kernels at all, and both of its answers are load-bearing -- a false yes reaches
a hard check mid-collective, a false no silently routes a supported shape to
another backend. The *seed* is a default rather than a measurement, so only its
shape is asserted here; its constants belong to whatever machine measured them
and are re-measured by ``PcieIpcAllReduceWorkspace.tune``.

Nothing here needs a GPU, and nothing here may start to: no CUDA, no process
group.
"""

import importlib.util
import inspect
import pathlib

import pytest

from flashinfer.comm import pcie_ipc_tuning as tuning
from flashinfer.comm.pcie_ipc_policy import (
    MAX_BLOCKS,
    IpcLaunchConfig,
    IpcVariant,
    _admits,
    _is_launchable,
    _seed,
    get_pcie_ipc_launch_config,
)

# The launcher accepts 2-byte dtypes only (bfloat16, float16), so this is the
# only element size a caller can reach.
_ELEM_SIZE = 2
_PACK_ELEMS = 16 // _ELEM_SIZE

_WORLD_SIZES = (2, 4, 8)
_HIDDENS = (1024, 2048, 4096, 6144, 8192)
_BATCHES = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256)
_NUMELS = tuple(sorted({b * h for b in _BATCHES for h in _HIDDENS}))

# Eight packs up to a few million elements, doubling. Wide enough to contain any
# plausible crossover from either direction without naming where it sits.
_LADDER = tuple(64 << k for k in range(17))


def _config(world_size, numel, max_blocks=MAX_BLOCKS):
    return get_pcie_ipc_launch_config(world_size, numel, _ELEM_SIZE, max_blocks)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_every_returned_config_is_launchable(world_size: int) -> None:
    """A config the kernel would reject must never leave the policy.

    The C++ side hard-checks these. Reaching them means one rank raises while
    its peers are already spinning in the collective.
    """
    admitted = 0
    for numel in _NUMELS:
        for max_blocks in (MAX_BLOCKS, 8):
            config = _config(world_size, numel, max_blocks)
            if config is None:
                continue
            admitted += 1
            assert _is_launchable(world_size, config, max_blocks), (
                f"ws={world_size} numel={numel} max_blocks={max_blocks} "
                f"-> {config}, which the kernel rejects"
            )
            assert 0 < config.blocks <= max_blocks
            assert world_size <= config.threads <= 1024
    assert admitted, "the sweep admits nothing, so it checks nothing"


def test_the_policy_is_a_pure_function() -> None:
    """Same arguments, same answer -- no hidden state, no rank-local input."""
    keys = [(w, n) for w in _WORLD_SIZES for n in _NUMELS]
    first = {k: _config(*k) for k in keys}
    assert any(v is not None for v in first.values())
    # The answers are memoised, so asking again would only re-read the cache.
    # Drop it and recompute in the opposite order, which re-runs the function
    # and would also expose an answer that depended on call order.
    get_pcie_ipc_launch_config.cache_clear()
    assert {k: _config(*k) for k in reversed(keys)} == first


def test_flat_staged_is_never_selected_outside_world_size_eight() -> None:
    """It would name the same kernel as ``STAGED`` at 4 ranks, and none at 2."""
    for world_size in (2, 4):
        for numel in _NUMELS:
            # The seed, not the gated return value: the gate below already
            # refuses FLAT_STAGED here, so it would answer for the seed and the
            # loop could not fail.
            seed = _seed(world_size, numel, _ELEM_SIZE, MAX_BLOCKS)
            assert seed.variant is not IpcVariant.FLAT_STAGED
    # And the launchability check refuses it even if the seed ever returned it.
    bad = IpcLaunchConfig(1, 128, IpcVariant.FLAT_STAGED)
    assert not _is_launchable(4, bad, MAX_BLOCKS)
    assert not _is_launchable(2, bad, MAX_BLOCKS)
    assert _is_launchable(8, bad, MAX_BLOCKS)


def test_world_size_eight_staged_always_gets_a_multiple_of_four_blocks() -> None:
    """The block-partitioned TP8 kernel derives its chunk from ``blockIdx.x & 3``.

    The seed is not the only source of configurations for that kernel and need
    not reach it at all, so the tuner's candidates are checked too -- they share
    the gate, and the gate is what has to hold.
    """
    for numel in _NUMELS:
        # Same reason as the FLAT_STAGED case: read the seed, since the gate
        # would otherwise turn a bad block count into None and a green test.
        seed = _seed(8, numel, _ELEM_SIZE, MAX_BLOCKS)
        if seed.variant is IpcVariant.STAGED:
            assert seed.blocks % 4 == 0, f"numel={numel} -> {seed}"

    staged = [
        config
        for config in map(tuning.tactic_to_config, tuning.candidate_tactics(8))
        if config.variant is IpcVariant.STAGED
    ]
    assert staged, "the tuner offers this kernel nothing to run"
    for config in staged:
        assert config.blocks % 4 == 0, config

    assert not _is_launchable(8, IpcLaunchConfig(2, 256, IpcVariant.STAGED), MAX_BLOCKS)
    assert _is_launchable(8, IpcLaunchConfig(4, 256, IpcVariant.STAGED), MAX_BLOCKS)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_admission_floor_is_one_pack_per_rank(world_size: int) -> None:
    """Below one 16-byte pack per rank the reduce-scatter split degenerates.

    Refused rather than served, because the two ownership formulas in the
    kernels stop agreeing there and a payload that small belongs on another
    backend anyway.
    """
    floor = _PACK_ELEMS * world_size
    assert _config(world_size, floor) is not None
    # A whole pack below the floor, so this isolates the per-rank rule from the
    # whole-pack rule tested separately.
    assert _config(world_size, floor - _PACK_ELEMS) is None
    assert _config(world_size, _PACK_ELEMS) is None


def test_the_admission_floor_scales_with_the_world_size() -> None:
    """One shape, admitted or not depending only on how many ranks share it."""
    numel = _PACK_ELEMS * 4
    assert _config(2, numel) is not None
    assert _config(4, numel) is not None
    assert _config(8, numel) is None


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_numel_that_is_not_whole_packs_is_refused(world_size: int) -> None:
    """The kernels address the payload in 16-byte packs; the launcher agrees."""
    whole = _PACK_ELEMS * 64
    assert _config(world_size, whole) is not None
    for remainder in (1, _PACK_ELEMS // 2, _PACK_ELEMS - 1):
        assert _config(world_size, whole + remainder) is None, remainder


@pytest.mark.parametrize("world_size", (0, 1, 3, 5, 6, 7, 9, 16))
def test_world_sizes_the_kernels_do_not_implement_are_refused(world_size: int) -> None:
    for numel in (_PACK_ELEMS * 128, 6144, 65536):
        assert _config(world_size, numel) is None, numel


@pytest.mark.parametrize(
    "world_size,hidden", [(4, 2048), (4, 8192), (2, 4096), (8, 1024), (8, 2048)]
)
def test_admission_does_not_depend_on_the_hidden_size(
    world_size: int, hidden: int
) -> None:
    """Hidden size is not a term in a capability question.

    These pairs are the ones a per-hidden restriction singles out first, and the
    kernels run all of them; refusing one would route a supported shape to
    another backend for good, since a caller reads ``None`` as "use NCCL".
    """
    for batch in (1, 2, 7, 64, 256):
        config = _config(world_size, batch * hidden)
        assert config is not None, f"ws={world_size} hidden={hidden} batch={batch}"
        assert _is_launchable(world_size, config, MAX_BLOCKS)


def test_the_seed_crosses_to_a_staged_kernel_as_the_payload_grows() -> None:
    """The one crossover that ports between machines, asserted as a shape.

    One-shot moves ``(N-1)*P`` bytes per rank in a single round trip; staging
    moves ``2*(N-1)*P/N`` and adds barriers, so it wins once the payload is
    large. Where it wins is a property of the machine and is measured by
    ``tune``; *that* it wins, once and without coming back, is not, and is what
    is pinned here.
    """
    for world_size in (4, 8):
        variants = [_config(world_size, numel).variant for numel in _LADDER]
        assert variants[0] is not IpcVariant.STAGED_RING, world_size
        assert variants[-1] is IpcVariant.STAGED_RING, world_size
        staged = [v is IpcVariant.STAGED_RING for v in variants]
        assert staged == sorted(staged), (
            f"ws={world_size} leaves the staged kernel as the payload grows: "
            f"{list(zip(_LADDER, variants, strict=True))}"
        )

    # Two ranks stage exactly the bytes they would have pushed (2P/N is P at
    # N == 2), so there is no crossover to place and no second branch.
    assert len({_config(2, numel).variant for numel in _LADDER}) == 1


def test_the_answer_keys_on_the_element_count_not_on_its_factorisation() -> None:
    """A token count does not port between machines; a byte count does.

    A policy keyed on ``(hidden, batch)`` answers one payload two ways depending
    on which shape produced it. The signature is what forecloses that: with
    ``numel`` the only shape argument, no factorisation can reach the function.
    """
    params = list(inspect.signature(get_pcie_ipc_launch_config).parameters)
    assert params[:3] == ["world_size", "numel", "elem_size"]
    assert not {"batch", "hidden", "profile"} & set(params), params


@pytest.mark.parametrize("max_blocks", (1, 2, 3, 4, 7, 16, MAX_BLOCKS))
def test_a_small_max_blocks_is_never_exceeded(max_blocks: int) -> None:
    """The workspace is sized for its own ``max_blocks``, not for the default.

    A configuration over that budget indexes scratch that was never allocated,
    which the launcher rejects on the rank that asked and nowhere else.
    """
    for world_size in _WORLD_SIZES:
        admitted = 0
        for numel in _NUMELS:
            config = _config(world_size, numel, max_blocks)
            # A tighter max_blocks bounds the grid, never the supported set.
            assert (config is not None) == _admits(world_size, numel, _ELEM_SIZE)
            if config is None:
                continue
            admitted += 1
            assert 0 < config.blocks <= max_blocks, (world_size, numel, config)
            assert _is_launchable(world_size, config, max_blocks)
        assert admitted, f"ws={world_size} max_blocks={max_blocks} checks nothing"


def test_every_dispatchable_variant_is_reachable_from_the_tuner() -> None:
    """A variant the dispatch can launch but nothing can select is invisible.

    This is the direction ``test_every_returned_config_is_launchable`` does not
    cover. A kernel that nothing selects is either dead code or an unexplored
    region of the launch space, and the two are indistinguishable from outside
    -- which is how the flat-staged kernel stayed unreachable for a whole tuning
    round. The seed deliberately reaches only some of them, so the tuner's
    candidate list is where this has to hold.
    """
    expected = {
        2: {IpcVariant.UNSTAGED, IpcVariant.STAGED},
        4: {IpcVariant.UNSTAGED, IpcVariant.STAGED, IpcVariant.STAGED_RING},
        8: {
            IpcVariant.UNSTAGED,
            IpcVariant.STAGED,
            IpcVariant.STAGED_RING,
            IpcVariant.FLAT_STAGED,
        },
    }
    seen = {
        world_size: {
            tuning.tactic_to_config(t).variant
            for t in tuning.candidate_tactics(world_size)
        }
        for world_size in _WORLD_SIZES
    }
    assert seen == expected


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


def _one_config_per_variant(world_size):
    """One launchable configuration per variant the dispatch reaches."""
    out = {}
    for tactic in tuning.candidate_tactics(world_size):
        config = tuning.tactic_to_config(tactic)
        out.setdefault(config.variant, config)
    return out


def _expected_switch(world_size, config):
    """The protocol this kernel actually used to run.

    TP2, TP4, the TP8 pack kernel and the flat-staged kernel double-buffered by
    per-block parity. The two topology-staged TP8 kernels had no epoch at all.
    """
    topo_staged_tp8 = world_size == 8 and config.variant in (
        IpcVariant.STAGED,
        IpcVariant.STAGED_RING,
    )
    return "no-block-epoch" if topo_staged_tp8 else "per-block-epoch"


def test_protocol_ab_picks_the_right_history_per_kernel() -> None:
    """The A/B baseline must match what each kernel actually used to run.

    Comparing against the wrong one measures a protocol that never shipped,
    which is not a performance result. Checked over every kernel the dispatch
    reaches rather than only the ones some shape currently lands on, since which
    kernel a shape gets is a default that is expected to move.
    """
    bench = _load_benchmark_module()
    pick = bench._historical_switch

    reached = set()
    for world_size in _WORLD_SIZES:
        for variant, config in _one_config_per_variant(world_size).items():
            want = _expected_switch(world_size, config)
            reached.add(want)
            assert pick(world_size, config) == want, (
                f"ws={world_size} {variant.name} -> {pick(world_size, config)}, "
                f"expected {want}"
            )
    assert reached == {"per-block-epoch", "no-block-epoch"}

    # And the shapes the shipping path actually produces land on the same side.
    for world_size, hidden in ((2, 2048), (4, 4096), (8, 6144)):
        for batch in (1, 2, 4, 8, 16, 32, 64, 128):
            config = _config(world_size, batch * hidden)
            if config is None:
                continue
            assert pick(world_size, config) == _expected_switch(world_size, config), (
                f"ws={world_size} hidden={hidden} batch={batch} -> {config}"
            )


def _shape_config(world_size, hidden, batches):
    out = {}
    for batch in batches:
        config = _config(world_size, batch * hidden)
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
        shapes = _shape_config(world_size, hidden, batches)
        assert shapes, f"ws={world_size} hidden={hidden} admits nothing"
        plan = plan_of(world_size, shapes, "auto")
        assert set(plan) == {"per-block-epoch"}
        assert plan["per-block-epoch"] == sorted(shapes)

    # 8 ranks straddles both histories, which is the whole reason auto exists.
    shapes = _shape_config(8, 6144, batches)
    history = {batch: _expected_switch(8, c) for batch, c in shapes.items()}
    assert set(history.values()) == {"per-block-epoch", "no-block-epoch"}, (
        "this sweep no longer covers both histories, so the partition below "
        f"would hold for want of anything to separate: {history}"
    )

    plan = plan_of(8, shapes, "auto")
    assert set(plan) == {"per-block-epoch", "no-block-epoch"}
    per_block = plan["per-block-epoch"]
    no_epoch = plan["no-block-epoch"]
    assert not set(per_block) & set(no_epoch), (
        "a shape may only run under its own history"
    )
    assert sorted(per_block + no_epoch) == sorted(shapes), (
        "every admitted shape must run once"
    )
    for switch, leg_batches in plan.items():
        for batch in leg_batches:
            assert history[batch] == switch, (
                f"batch {batch} ({shapes[batch]}) ran under {switch}, "
                f"but its history is {history[batch]}"
            )

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
    shapes = _shape_config(8, 6144, batches)
    plan = bench._protocol_ab_plan(8, shapes, "auto")
    assert len(plan) == 2, f"one leg only; nothing to keep apart: {plan}"

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
        if broken_switch is not None:
            assert set(ran) == set(plan[broken_switch]), (
                f"{broken_switch} leg ran {sorted(ran)}, owns {plan[broken_switch]}"
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
    real_torch = bench.torch
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
        bench.torch = real_torch

    assert calls, "the verdict must be reduced across ranks, not decided locally"
    assert all(op is real_dist.ReduceOp.MIN for op in calls), (
        f"correctness must reduce with MIN (logical AND), got {calls}"
    )
