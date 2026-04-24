"""Tests for distributed-aware autotuning.

Covers ``set_autotune_process_group`` / ``get_autotune_process_group``
and the cross-rank timing sync inside ``_profile_single_kernel``.

Uses ``gloo`` (CPU backend) to avoid depending on CUDA or NCCL. No GPU
required.
"""

import ast
import inspect
import multiprocessing as mp
import os
import socket
import textwrap

import pytest
import torch

from flashinfer.autotuner import (
    AutoTuner,
    get_autotune_process_group,
    set_autotune_process_group,
)

from .utils import reset_autotuner


def _find_free_port() -> int:
    """Ask the OS for an unused TCP port.

    Hardcoded ports race under parallel test runners (pytest-xdist); let
    the kernel pick a free one instead.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_set_and_get_autotune_process_group_roundtrip():
    """Setter should update the module-level state and getter should return it."""
    reset_autotuner()
    set_autotune_process_group(None)
    assert get_autotune_process_group() is None

    # The setter does not validate its argument — using a sentinel keeps
    # this test GPU-free. If the API ever tightens to require a real
    # ``ProcessGroup`` instance, replace the sentinel with an actual
    # ``dist.new_group(backend="gloo")`` call.
    sentinel = object()
    set_autotune_process_group(sentinel)
    try:
        assert get_autotune_process_group() is sentinel
    finally:
        set_autotune_process_group(None)

    assert get_autotune_process_group() is None


def test_set_autotune_process_group_defaults_to_none():
    """Module default must be ``None`` so autotuning stays local by default."""
    reset_autotuner()
    set_autotune_process_group(None)
    assert get_autotune_process_group() is None


def _gloo_worker_all_reduce_mean(
    rank: int, world_size: int, port: int, queue: "mp.Queue"
) -> None:
    """Gloo worker: init PG, set autotune group, run the exact reduction
    the production code uses, put the reduced value on the queue."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        set_autotune_process_group(group)
        try:
            local_time = float(rank + 1)  # 1.0, 2.0 on 2 ranks
            # Mirror the exact reduction logic from _profile_single_kernel
            # so the test is self-contained and does not require a real
            # profiling run (which would need CUDA).
            t = torch.tensor([local_time], dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
            mean_time = t.item() / dist.get_world_size(group)
            queue.put((rank, mean_time))
        finally:
            set_autotune_process_group(None)
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(60)
def test_all_reduce_mean_agrees_across_ranks():
    """With the process group set, both ranks compute the same mean time.

    This exercises the same reduction logic that
    ``_profile_single_kernel`` runs after ``pure_profile``. If ranks
    measured ``1.0`` and ``2.0`` respectively, both ranks must see
    ``1.5`` so their ``argmin`` agrees and the chosen tactic is stable.
    """
    reset_autotuner()
    world_size = 2
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    queue: "mp.Queue" = ctx.Queue()
    procs = [
        ctx.Process(
            target=_gloo_worker_all_reduce_mean,
            args=(rank, world_size, port, queue),
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=45)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    # Fixed cardinality read — robust against platform-specific
    # ``queue.empty()`` races between ``join`` and ``get``.
    results = {}
    for _ in range(world_size):
        rank, mean = queue.get(timeout=10)
        results[rank] = mean

    assert set(results.keys()) == {0, 1}
    # rank 0 measured 1.0, rank 1 measured 2.0 -> mean 1.5 on both.
    assert results[0] == pytest.approx(1.5)
    assert results[1] == pytest.approx(1.5)


def test_profile_single_kernel_preserves_collective_cardinality_on_exception():
    """Regression guard: exceptions inside ``pure_profile`` must NOT skip the all-reduce.

    If an exception in ``pure_profile`` (OOM, ptxas mismatch, illegal
    instruction in a bad tactic, etc.) causes the rank that failed to
    exit ``_profile_single_kernel`` *without* participating in the
    all-reduce, peer ranks are left blocked waiting for it. The next
    tactic's reduce then gets an off-by-one collective and deadlocks.

    This is exactly the class of collective-desync hang this API was
    added to prevent, so we guard the source shape explicitly:

    * ``pure_profile(...)`` is called inside a ``try/except`` that sets
      ``avg_time = float("inf")`` on failure.
    * The ``if _tune_process_group is not None: ... dist.all_reduce(...)``
      block appears AFTER that try/except (i.e. on the success path
      AND on the caught-exception path).
    * The original exception is re-raised after the reduce so the outer
      handler in ``choose_one`` (logging / stats / OOM fallback) still
      runs.

    AST-based instead of substring so comments, whitespace and variable
    renames don't produce spurious failures.
    """
    src = textwrap.dedent(inspect.getsource(AutoTuner._profile_single_kernel))
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))

    # Locate the Try node whose body contains the call to ``pure_profile``.
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

    # The handler must assign ``avg_time = float("inf")`` so the
    # reduction marks the tactic infeasible for every rank.
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

    # After the Try, find a call to ``dist.all_reduce`` — ensures the
    # reduce is reachable on both the success and caught-exception
    # paths.
    try_index = func.body.index(wrapping_try)
    post_try_body = func.body[try_index + 1 :]
    has_all_reduce = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "all_reduce"
        for stmt in post_try_body
        for n in ast.walk(stmt)
    )
    assert has_all_reduce, (
        "dist.all_reduce must appear AFTER the try/except so every "
        "rank reaches exactly one reduce per _profile_single_kernel "
        "call (collective cardinality invariant)"
    )

    # A ``raise`` statement must follow the reduce so the caught
    # exception is re-raised and the outer handler in ``choose_one``
    # still runs (logging, stats, OOM fallback).
    has_reraise = any(
        isinstance(n, ast.Raise) for stmt in post_try_body for n in ast.walk(stmt)
    )
    assert has_reraise, (
        "the caught exception must be re-raised after the reduce so "
        "choose_one's outer handler still records the failure"
    )


def _gloo_worker_exception_path(
    rank: int,
    world_size: int,
    port: int,
    queue: "mp.Queue",
) -> None:
    """Worker that simulates one rank raising inside ``pure_profile``.

    Patches ``_profile_single_kernel`` to a minimal reimplementation
    that shares the same try/except + reduce shape the production code
    has. Rank 0 raises; rank 1 returns a finite time. Both ranks must
    reach the all-reduce exactly once (cardinality preserved).
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        set_autotune_process_group(group)
        try:
            # Mirror production shape:
            #   try: avg_time = pure_profile(...)
            #   except: avg_time = inf; exc = e
            #   if _tune_process_group is not None: all-reduce
            #   if exc is not None: raise exc
            profile_exc = None
            try:
                if rank == 0:
                    raise RuntimeError("simulated tactic failure on rank 0")
                avg_time = 2.0
            except Exception as e:
                avg_time = float("inf")
                profile_exc = e

            # Both ranks reach this line, so the all-reduce completes.
            t = torch.tensor([avg_time], dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
            reduced = t.item() / dist.get_world_size(group)

            # If we get here with the expected reduced value, the
            # collective did NOT deadlock.
            queue.put((rank, reduced, profile_exc is not None))
        finally:
            set_autotune_process_group(None)
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(60)
def test_exception_path_does_not_deadlock_the_reduce():
    """One rank raising must NOT block peers on the cross-rank reduce.

    Rank 0 simulates a failing tactic; rank 1 succeeds with avg_time=2.
    Both ranks must still reach the all-reduce and both must see the
    reduced value (``inf + 2``) / 2 == ``inf``. If the failing rank
    skipped the reduce, rank 1 would time out and the test would fail
    via pytest-timeout rather than via an assertion.
    """
    reset_autotuner()
    world_size = 2
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    queue: "mp.Queue" = ctx.Queue()
    procs = [
        ctx.Process(
            target=_gloo_worker_exception_path,
            args=(rank, world_size, port, queue),
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=45)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    results = {}
    for _ in range(world_size):
        rank, reduced, raised = queue.get(timeout=10)
        results[rank] = (reduced, raised)

    assert set(results.keys()) == {0, 1}
    # Both ranks see the reduced value (inf because rank 0 failed).
    # The exact check is that the mean is inf on BOTH ranks, which
    # correctly disqualifies the tactic everywhere.
    assert results[0][0] == float("inf")
    assert results[1][0] == float("inf")
    # Rank 0 raised; rank 1 did not.
    assert results[0][1] is True
    assert results[1][1] is False
