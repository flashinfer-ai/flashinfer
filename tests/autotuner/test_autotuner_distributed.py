"""Tests for distributed-aware autotuning.

Covers the two real risks introduced by ``set_autotune_process_group``:

1. The all-reduce inside ``_profile_single_kernel`` must run on every
   ``_profile_single_kernel`` call — including the path where
   ``pure_profile`` raises — or per-tactic collective cardinality
   desyncs and peers deadlock waiting for a matching reduce.
2. With one rank simulating a tactic failure, peers must still reach
   the reduce and observe the failure (``inf`` propagates) instead of
   blocking.

Uses ``gloo`` (CPU) to avoid depending on CUDA or NCCL. No GPU required.
"""

import ast
import inspect
import multiprocessing as mp
import os
import socket
import textwrap

import pytest
import torch

from flashinfer.autotuner import AutoTuner, set_autotune_process_group


def _find_free_port() -> int:
    """Ask the OS for an unused TCP port (avoids races under pytest-xdist)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def _gloo_worker_exception_path(
    rank: int,
    world_size: int,
    port: int,
    queue: "mp.Queue",
) -> None:
    """Worker that reproduces ``_profile_single_kernel``'s exception path.

    Rank 0 raises; rank 1 succeeds. Both ranks must still reach the
    all-reduce (cardinality preserved) and observe the reduced ``inf``.
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

            queue.put((rank, reduced, profile_exc is not None))
        finally:
            set_autotune_process_group(None)
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(60)
def test_exception_path_does_not_deadlock_the_reduce():
    """Empirical: one rank raising must NOT block peers on the cross-rank reduce.

    Rank 0 simulates a failing tactic; rank 1 succeeds with ``avg_time=2``.
    Both ranks must still reach the all-reduce and observe the mean
    ``(inf + 2) / 2 == inf``. If the failing rank skipped the reduce,
    rank 1 would time out and the test would fail via pytest-timeout.
    """
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
    assert results[0][0] == float("inf")
    assert results[1][0] == float("inf")
    assert results[0][1] is True
    assert results[1][1] is False
