"""Tests for distributed-aware autotuning.

Covers the real risks introduced by ``set_autotune_process_group``. Because the
production ``_profile_single_kernel`` needs CUDA (and NCCL) it cannot run under
CI, so the coverage is split into three complementary, GPU-free checks:

1. **Source-level (AST) guards on the production code** — assert that the real
   ``AutoTuner._profile_single_kernel`` keeps its all-reduce on every path
   (including when ``pure_profile`` raises), and that ``AutoTuner.choose_one``
   does not unconditionally re-raise OOM when a tune group is set. These guard
   the actual functions, not a copy.
2. **Runtime (gloo) check of the reduce pattern** — a standalone 2-process
   ``gloo`` reimplementation of the try/except→``inf``→all-reduce pattern,
   confirming that one rank failing does not block peers on the collective.
   This validates gloo's behavior for the pattern, not the production call path
   (which #1 guards).
3. **Runtime (gloo) check of preparation OOM** — two processes call the real
   ``AutoTuner.choose_one`` control flow and confirm that one rank's preparation
   OOM produces the same fallback on every rank.

Uses ``gloo`` (CPU) to avoid depending on CUDA or NCCL. No GPU required.
"""

import ast
import inspect
import os
import subprocess
import sys
import textwrap
import time

import pytest

from flashinfer.autotuner import AutoTuner


def test_profile_single_kernel_preserves_collective_cardinality_on_exception():
    """Source-level guard: an exception inside ``pure_profile`` must NOT skip the all-reduce.

    If ``pure_profile`` raises (OOM, ptxas mismatch, illegal instruction
    in a bad tactic, etc.) and the rank that failed exits
    ``_profile_single_kernel`` *without* participating in the
    all-reduce, peer ranks are left blocked waiting for it. The next
    tactic's reduce then has off-by-one cardinality and deadlocks —
    the exact class of hang this API was added to prevent. We assert
    the source still has the required shape:

    * ``pure_profile(...)`` is called inside a ``try/except`` whose
      handler sets ``avg_time = float("inf")``.
    * ``dist.all_reduce`` appears AFTER that ``try/except`` (so it
      runs on both success and caught-exception paths).
    * A ``raise`` follows, so ``choose_one``'s outer handler still
      sees the original failure for logging / OOM fallback.

    AST-based, not substring-based, so comments / formatting / variable
    renames don't produce spurious failures.
    """
    src = textwrap.dedent(inspect.getsource(AutoTuner._profile_single_kernel))
    func = ast.parse(src).body[0]
    assert isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))

    # Find the Try whose body contains a call to ``pure_profile``.
    wrapping_try = None
    for node in ast.walk(func):
        if not isinstance(node, ast.Try):
            continue
        for sub in ast.walk(ast.Module(body=list(node.body), type_ignores=[])):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "pure_profile"
            ):
                wrapping_try = node
                break
        if wrapping_try is not None:
            break
    assert wrapping_try is not None, (
        "pure_profile(...) must be called inside a try/except so an "
        "exception on one rank does not skip the cross-rank all-reduce"
    )

    def _is_inf_assign(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "avg_time" for t in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "float"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "inf"
        )

    handler_ok = any(
        _is_inf_assign(n)
        for h in wrapping_try.handlers
        for n in ast.walk(ast.Module(body=list(h.body), type_ignores=[]))
    )
    assert handler_ok, (
        "the except handler around pure_profile must set "
        "avg_time = float('inf') so the reduced tactic becomes inf "
        "on every rank"
    )

    # The reduce + reraise must follow the wrapping Try in the function body.
    try_index = func.body.index(wrapping_try)
    post_try = func.body[try_index + 1 :]
    has_all_reduce = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "all_reduce"
        for stmt in post_try
        for n in ast.walk(stmt)
    )
    assert has_all_reduce, (
        "dist.all_reduce must appear AFTER the try/except so every "
        "rank reaches exactly one reduce per _profile_single_kernel "
        "call (collective-cardinality invariant)"
    )

    has_reraise = any(
        isinstance(n, ast.Raise) for stmt in post_try for n in ast.walk(stmt)
    )
    assert has_reraise, (
        "the caught exception must be re-raised after the reduce so "
        "choose_one's outer handler still records the failure"
    )


def test_choose_one_oom_preserves_cardinality_under_group():
    """Source guard: ``choose_one`` must not unconditionally re-raise OOM when a
    tune group is set.

    A per-rank ``OutOfMemoryError`` inside the tactic loop must be handled like
    any other failed tactic when ``_tune_process_group`` is set (disqualify with
    ``inf`` and keep looping), so this rank stays in lockstep on every
    subsequent tactic's all-reduce. If it re-raised unconditionally,
    ``choose_one``'s outer OOM handler would early-return ``runners[0], -1`` and
    this rank would leave the tuning loop while peers keep reducing -> deadlock.

    AST-based, so comments / formatting don't cause spurious failures. We locate
    the ``try`` that wraps ``self._profile_single_kernel(...)`` and assert its
    ``except ...OutOfMemoryError`` handler both gates on ``_tune_process_group``
    and disqualifies the tactic with ``float("inf")`` (rather than *only*
    re-raising).
    """
    src = textwrap.dedent(inspect.getsource(AutoTuner.choose_one))
    func = ast.parse(src).body[0]
    assert isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))

    # The per-tactic try is the *innermost* try wrapping
    # self._profile_single_kernel (distinct from the outer prep/OOM try, which
    # transitively contains the same call and early-returns on OOM).
    def _wraps_profile(t: ast.Try) -> bool:
        return any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "_profile_single_kernel"
            for c in ast.walk(ast.Module(body=list(t.body), type_ignores=[]))
        )

    def _has_nested_wrapping_try(t: ast.Try) -> bool:
        return any(
            isinstance(sub, ast.Try) and sub is not t and _wraps_profile(sub)
            for sub in ast.walk(ast.Module(body=list(t.body), type_ignores=[]))
        )

    candidates = [
        t
        for t in ast.walk(func)
        if isinstance(t, ast.Try)
        and _wraps_profile(t)
        and not _has_nested_wrapping_try(t)
    ]
    assert candidates, (
        "could not find the innermost try/except wrapping "
        "self._profile_single_kernel(...)"
    )
    prof_try = candidates[0]

    def _is_oom_handler(h: ast.ExceptHandler) -> bool:
        t = h.type
        candidates = t.elts if isinstance(t, ast.Tuple) else [t]
        return any(
            isinstance(x, ast.Attribute) and x.attr == "OutOfMemoryError"
            for x in candidates
            if x is not None
        )

    oom_handlers = [h for h in prof_try.handlers if _is_oom_handler(h)]
    assert oom_handlers, (
        "expected an `except torch.cuda.OutOfMemoryError` handler around "
        "_profile_single_kernel"
    )
    handler_body = ast.Module(body=list(oom_handlers[0].body), type_ignores=[])

    gates_on_group = any(
        isinstance(n, ast.Name) and n.id == "_tune_process_group"
        for n in ast.walk(handler_body)
    )
    assert gates_on_group, (
        "the per-tactic OOM handler must gate on `_tune_process_group` so a "
        "distributed OOM is disqualified (inf) instead of unconditionally "
        "re-raising (which desyncs the per-tactic all-reduce)"
    )

    disqualifies_with_inf = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "float"
        and len(n.args) == 1
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "inf"
        for n in ast.walk(handler_body)
    )
    assert disqualifies_with_inf, (
        "the distributed OOM path must disqualify the tactic with "
        "`float('inf')` and continue, preserving collective cardinality"
    )


# Standalone worker for the runtime gloo check. Run as ``python -c`` in its own
# subprocess so it imports ONLY torch — never this test module — which makes it
# independent of how the harness imported the tests (a ``spawn`` worker that
# re-imports the test module dies with ``ModuleNotFoundError: No module named
# 'tests'`` when the package isn't importable in the fresh interpreter). Uses a
# FileStore rendezvous (``file://``) so there is no TCP-port race, and a bounded
# ``init_process_group`` timeout so a failed rendezvous errors instead of hanging.
# It reimplements ``_profile_single_kernel``'s try/except→``inf``→all-reduce shape
# (the production structure itself is guarded by the AST tests above).
_GLOO_WORKER_SRC = r"""
import datetime, os, sys
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
store_file = os.environ["GLOO_STORE_FILE"]

dist.init_process_group(
    backend="gloo",
    init_method="file://" + store_file,
    rank=rank,
    world_size=world_size,
    timeout=datetime.timedelta(seconds=60),
)
try:
    group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    profile_exc = None
    try:
        if rank == 0:
            raise RuntimeError("simulated tactic failure on rank 0")
        avg_time = 2.0
    except Exception as e:
        avg_time = float("inf")
        profile_exc = e

    t = torch.tensor([avg_time], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    reduced = t.item() / dist.get_world_size(group)

    expect_raised = rank == 0
    if (profile_exc is not None) != expect_raised:
        sys.exit("rank %d: unexpected raised=%s" % (rank, profile_exc is not None))
    if reduced != float("inf"):
        sys.exit("rank %d: reduced=%r != inf" % (rank, reduced))
    print("rank %d: OK reduced=inf" % rank)
finally:
    dist.destroy_process_group()
"""


def test_exception_path_does_not_deadlock_the_reduce(tmp_path):
    """Empirical: one rank raising must NOT block peers on the cross-rank reduce.

    Runs the reduce *pattern* (see module docstring) on a real 2-rank gloo group,
    each rank in its own subprocess importing only torch — independent of how the
    harness imported this module — with a FileStore rendezvous (no TCP-port race).
    The production ``_profile_single_kernel`` / ``choose_one`` structure is guarded
    by the AST tests above. Rank 0 fails (``inf``) while rank 1 succeeds with
    ``avg_time=2``; both must still reach the all-reduce and observe the mean
    ``(inf + 2) / 2 == inf``. A hung reduce is caught by the subprocess timeout.
    """
    store_file = str(tmp_path / "gloo_rendezvous")
    world_size = 2
    procs = []
    deadline = time.monotonic() + 180
    try:
        for rank in range(world_size):
            env = dict(os.environ)
            env.update(
                RANK=str(rank), WORLD_SIZE=str(world_size), GLOO_STORE_FILE=store_file
            )
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", _GLOO_WORKER_SRC],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        outputs = {}
        for rank, p in enumerate(procs):
            try:
                out, _ = p.communicate(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
                out, _ = p.communicate()
                raise AssertionError(
                    f"rank {rank} did not finish (possible reduce deadlock):\n{out}"
                ) from None
            outputs[rank] = out
            assert p.returncode == 0, (
                f"rank {rank} worker failed (exit {p.returncode}):\n{out}"
            )
    finally:
        for process in procs:
            if process.poll() is None:
                process.kill()
        for process in procs:
            process.communicate()

    # Both ranks reaching "OK reduced=inf" proves the failing rank did not skip
    # the reduce (else the peer would have hung and hit the timeout above).
    for rank in range(world_size):
        assert "OK reduced=inf" in outputs[rank], (
            f"rank {rank} unexpected output:\n{outputs[rank]}"
        )


_PREPARATION_OOM_WORKER_SRC = r"""
import datetime, os, sys
import torch
import torch.distributed as dist

from flashinfer.autotuner import (
    AutoTuner,
    TunableRunner,
    TuningConfig,
    autotune,
    set_autotune_process_group,
)

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
store_file = os.environ["GLOO_STORE_FILE"]
failure_phase = os.environ["FAILURE_PHASE"]


class PreparationRunner(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        if failure_phase == "nested_empty" and rank == 1:
            return ()
        return (0,)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        if do_preparation and rank == 0 and failure_phase != "nested_input":
            raise MemoryError("simulated workspace allocation failure")
        return inputs[0]


class NestedRunner(TunableRunner):
    from flashinfer.fused_moe.runners import _CutlassRunnerBase

    get_valid_tactics = _CutlassRunnerBase.get_valid_tactics
    _num_top_tactics_per_stage = 2
    backend_key = "cutlass"
    _device_arch = 90

    def __init__(self):
        self._inner = PreparationRunner()

    def _require_built(self):
        pass

    def _validate_input_count(self, inputs):
        pass

    def forward(self, inputs, tactic=-1, **kwargs):
        return inputs[0]


def profile(self, *args, **kwargs):
    elapsed = torch.tensor([1.0], dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.SUM)
    return elapsed.item() / world_size


prepare_inputs = AutoTuner._prepare_input_tensors
preparation_calls = 0


def prepare(self, *args, **kwargs):
    global preparation_calls
    preparation_calls += 1
    if failure_phase == "nested_input" and rank == 0 and preparation_calls == 2:
        raise MemoryError("simulated nested input allocation failure")
    return prepare_inputs(self, *args, **kwargs)


dist.init_process_group(
    backend="gloo",
    init_method="file://" + store_file,
    rank=rank,
    world_size=world_size,
    timeout=datetime.timedelta(seconds=60),
)
try:
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda.empty_cache = lambda: None
    AutoTuner._profile_single_kernel = profile
    AutoTuner._prepare_input_tensors = prepare
    set_autotune_process_group(dist.group.WORLD)

    with autotune(True):
        _, tactic = AutoTuner.get().choose_one(
            "distributed_preparation_oom",
            [PreparationRunner() if failure_phase == "runner" else NestedRunner()],
            TuningConfig(),
            [torch.empty((8, 8)) for _ in range(6)],
        )

    if tactic != -1:
        sys.exit("rank %d: tactic=%r" % (rank, tactic))
    print("rank %d: fallback" % rank)
finally:
    set_autotune_process_group(None)
    dist.destroy_process_group()
"""


@pytest.mark.parametrize(
    "failure_phase", ("runner", "nested_input", "nested_runner", "nested_empty")
)
def test_preparation_memory_error_falls_back_on_every_rank(tmp_path, failure_phase):
    """A rank-local preparation OOM produces one group-wide fallback."""
    store_file = str(tmp_path / "preparation_oom_rendezvous")
    procs = []
    deadline = time.monotonic() + 180
    try:
        for rank in range(2):
            env = dict(os.environ)
            env.update(
                RANK=str(rank),
                WORLD_SIZE="2",
                GLOO_STORE_FILE=store_file,
                FAILURE_PHASE=failure_phase,
                TORCH_DISTRIBUTED_DEBUG="DETAIL",
            )
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", _PREPARATION_OOM_WORKER_SRC],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        outputs = {}
        for rank, process in enumerate(procs):
            try:
                output, _ = process.communicate(
                    timeout=max(0, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
                raise AssertionError(
                    f"rank {rank} timed out during preparation OOM synchronization:\n{output}"
                ) from None
            outputs[rank] = output
            assert process.returncode == 0, (
                f"rank {rank} worker failed (exit {process.returncode}):\n{output}"
            )
    finally:
        for process in procs:
            if process.poll() is None:
                process.kill()
        for process in procs:
            process.communicate()

    for rank in range(2):
        assert f"rank {rank}: fallback" in outputs[rank], (
            f"rank {rank} unexpected output:\n{outputs[rank]}"
        )


_CACHE_HIT_WORKER_SRC = r"""
import datetime, os
from pathlib import Path
import torch
import torch.distributed as dist
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig, autotune, set_autotune_process_group
from flashinfer.autotune_cache import autotune_v2, autotune_v2_reload, ManagedAutotuneCache

rank = int(os.environ["RANK"])
case = os.environ["CACHE_CASE"]
root = Path(os.environ["CASE_ROOT"])

class RunnerA(TunableRunner):
    def get_cache_key_extras(self, inputs):
        return (0 if case == "concurrent_publish" else rank,)
    def get_valid_tactics(self, inputs, profile):
        return [0, 1]
    def forward(self, inputs, tactic=-1):
        return inputs[0]

class RunnerB(RunnerA):
    pass

config = TuningConfig()
inputs = [torch.zeros(8, 8)]
runners = [RunnerA(), RunnerB()]
op = "test::distributed_cache_hit"
tuner = AutoTuner.get()
tuner.clear_cache()
calls = []

def profile(self, runner, inputs, tactic, tuning_config, **kwargs):
    calls.append((type(runner).__name__, tactic))
    elapsed = torch.tensor([1.0 if isinstance(runner, RunnerB) and tactic == 1 else 5.0])
    dist.all_reduce(elapsed)
    if case == "failed_profile":
        return float("inf")
    return elapsed.item() / 2

AutoTuner._profile_single_kernel = profile
# CPU tests replace only CUDA measurement/capture queries; cache lookup,
# invalidation, publication, choose_one and distributed collectives are real.
torch.cuda.is_current_stream_capturing = lambda: False
dist.init_process_group("gloo", init_method="file://" + str(root / "rendezvous"),
                        rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30))
try:
    context = (autotune(True) if case in ("partial_memory", "partial_file")
               else autotune_v2(cache_root=root / "cache"))
    with context:
        key = tuner._get_cache_key(op, runners[0], (inputs[0].shape,), config,
                                   runners[0].get_cache_key_extras(inputs))
        policy = tuner._profiling_policy(config)
        store = tuner._active_managed_store
        if case == "all_hit" or (rank == 0 and case not in ("all_miss", "concurrent_publish")):
            if case == "partial_memory":
                tuner._winner_cache()[key] = (0, None)
            elif case == "partial_file":
                tuner._file_configs[key.file_key] = ("RunnerA", 0)
            else:
                store.publish(key.file_key, "RunnerA", 0, key_fields=key.key_fields,
                              profiling_policy=policy)
        dist.barrier()
        search = tuner.search_cache
        first = True
        def ordered_search(*args, **kwargs):
            global first
            if not first:
                return search(*args, **kwargs)
            first = False
            if case != "concurrent_publish":
                result = search(*args, **kwargs)
                expected_hit = case == "all_hit" or (rank == 0 and case != "all_miss")
                assert result[0] == expected_hit, (rank, case, result)
                return result
            if rank == 0:
                result = search(*args, **kwargs)
                assert not result[0]
                dist.barrier()
                return result
            dist.barrier()
            # Model another deployment publishing between the two lookups.
            writer = ManagedAutotuneCache(store.manifest, root=store.root)
            writer.publish(key.file_key, "RunnerA", 0, key_fields=key.key_fields,
                           profiling_policy=policy)
            result = search(*args, **kwargs)
            assert result[0]
            return result
        tuner.search_cache = ordered_search
        set_autotune_process_group(dist.group.WORLD)
        # A nested rank_tactics shortlist has its own process-local cache.
        ranked_key = tuner._get_cache_key("test::ranked", runners[0],
            (inputs[0].shape,), config, runners[0].get_cache_key_extras(inputs))
        if rank == 0:
            tuner._ranked_tactics_cache[ranked_key] = (policy, (1, 0))
        ranked = tuner.rank_tactics("test::ranked", [runners[0]], config, inputs, k=2)
        assert ranked == ([-1] if case == "failed_profile" else [0, 1])
        calls.clear()
        expected = ("RunnerA", 0) if case == "all_hit" else (
            ("RunnerA", -1) if case == "failed_profile" else ("RunnerB", 1))
        chosen, tactic = tuner.choose_one(op, runners, config, inputs)
        assert (type(chosen).__name__, tactic) == expected
        assert bool(calls) == (case != "all_hit")
        count = len(calls)
        chosen, tactic = tuner.choose_one(op, runners, config, inputs)
        assert (type(chosen).__name__, tactic) == expected
        if case == "failed_profile":
            assert len(calls) == 2 * count, "failed profiling must remain retryable"
        else:
            assert len(calls) == count, "second call must reuse the coordinated winner"
        count = len(calls)
        set_autotune_process_group(None)
        tuner.search_cache = search
    dist.barrier()
    if store is not None:
        autotune_v2_reload()
    # Serving remains collective-free even if the caller has left the group set.
    set_autotune_process_group(dist.group.WORLD)
    original_reduce = dist.all_reduce
    def unexpected_reduce(*args, **kwargs):
        raise AssertionError("serving must not synchronize")
    dist.all_reduce = unexpected_reduce
    chosen, tactic = tuner.choose_one(op, runners, config, inputs)
    assert (type(chosen).__name__, tactic) == expected, "reload revived an old runner"
    assert len(calls) == count
    # Even warm distributed tuning cannot perform a host collective in capture.
    torch.cuda.is_current_stream_capturing = lambda: True
    with autotune(True):
        try:
            tuner.choose_one(op, runners, config, inputs)
        except RuntimeError as error:
            assert "CUDA Graph capture" in str(error)
        else:
            raise AssertionError("distributed tuning in capture must fail")
    dist.all_reduce = original_reduce
    print("cache agreement OK", rank, case, count, flush=True)
finally:
    set_autotune_process_group(None)
    dist.destroy_process_group()
"""


@pytest.mark.parametrize(
    "case",
    [
        "all_hit",
        "all_miss",
        "partial_memory",
        "partial_file",
        "partial_managed",
        "concurrent_publish",
        "failed_profile",
    ],
)
def test_cache_hit_disagreement_reprofiles_together(tmp_path, case):
    """Real two-rank lookup/collectives converge despite partial or racing caches."""
    processes = []
    deadline = time.monotonic() + 120
    try:
        for rank in range(2):
            env = dict(
                os.environ,
                RANK=str(rank),
                CACHE_CASE=case,
                CASE_ROOT=str(tmp_path),
                TORCH_DISTRIBUTED_DEBUG="DETAIL",
            )
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", _CACHE_HIT_WORKER_SRC],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        for process in processes:
            output, _ = process.communicate(timeout=max(0, deadline - time.monotonic()))
            assert process.returncode == 0, output
            assert "cache agreement OK" in output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate()
