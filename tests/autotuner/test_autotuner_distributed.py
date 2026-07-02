"""Tests for distributed-aware autotuning.

Covers the real risks introduced by ``set_autotune_process_group``. Because the
production ``_profile_single_kernel`` needs CUDA (and NCCL) it cannot run under
CI, so the coverage is split into two complementary, GPU-free checks:

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


def _gloo_worker_exception_path(
    rank: int,
    world_size: int,
    port: int,
    queue: "mp.Queue",
) -> None:
    """Standalone gloo reimplementation of the exception-path reduce pattern.

    This does NOT call the production ``_profile_single_kernel`` (which needs
    CUDA); it re-creates the same try/except→``inf``→all-reduce shape so the
    gloo runtime behavior can be exercised on CPU. The production function's
    structure is guarded separately by the AST test above. Rank 0 raises; rank
    1 succeeds. Both ranks must still reach the all-reduce (cardinality
    preserved) and observe the reduced ``inf``.
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

    Exercises the reduce *pattern* on gloo (see ``_gloo_worker_exception_path``),
    not the production ``_profile_single_kernel`` itself (guarded by the AST test
    above). Rank 0 simulates a failing tactic; rank 1 succeeds with
    ``avg_time=2``. Both ranks must still reach the all-reduce and observe the
    mean ``(inf + 2) / 2 == inf``. If the failing rank skipped the reduce, rank 1
    would time out and the test would fail via pytest-timeout.
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
