"""Warm Prims-TS attention compilation for selected tests in isolated processes.

Workers execute fixture-free cases using real kernels and numerical checks,
persisting every plan, including later replans. Tests then execute normally
in the parent, which owns their authoritative pytest result. This follows the
tests' own dispatch and avoids maintaining a second shape/plan table.
"""

from concurrent.futures import ProcessPoolExecutor
import functools
import importlib.util
import inspect
import multiprocessing
import os
from pathlib import Path
import sys
from unittest.mock import patch


@functools.cache
def load_test_module(path):
    """Import test setup once per worker, without invoking pytest collection."""
    name = f"prims_ts_prebuild_{Path(path).stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initialize_worker():
    """Limit CPU tensor work and apply the same DSL compatibility setup as pytest."""
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from tests.conftest import _patch_cutlass_dsl_operand_major_mode

    _patch_cutlass_dsl_operand_major_mode()


def precompile_case(case):
    """Warm every plan of a case; return counts and defer failures to normal pytest.

    A per-specialization lock deduplicates concurrent requests. Compilation
    runs outside the module lock; the existing cache writer locks only while
    publishing objects and metadata. Worker execution discovers later replans
    that would otherwise leave a serial compilation tail in the parent.
    """
    import pytest
    from filelock import FileLock
    from flashinfer.jit.core import get_tmpdir
    from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl

    path, function_name, parameters = case
    built = 0

    def build_and_load(spec):
        """Persist this specialization and load the real callable for worker execution."""
        nonlocal built
        lock = get_tmpdir() / f"{spec.module_dir_name}-{spec.kernel_name}.prebuild.lock"
        with FileLock(lock, thread_local=False):
            if not spec.is_compiled:
                spec.compile_and_persist()
                if not spec.is_compiled:
                    raise RuntimeError(f"Could not persist {spec.name}")
                built += 1
        return spec.load()

    try:
        module = load_test_module(path)
        with patch.object(JitSpecCuteDsl, "build_and_load", build_and_load):
            getattr(module, function_name)(**parameters)
    except pytest.skip.Exception:
        pass
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        # Prebuilding is optional. The unmodified parent test owns the real
        # result, including pytest's BaseException-derived assertion outcomes.
        return built, f"{function_name}: {type(error).__name__}: {error}"
    return built, None


def precompile_cases(cases, workers):
    """Use spawn rather than forking CUDA or sharing DSL compiler global state."""
    built, errors = 0, []
    # Set these before spawn/import: torch.set_num_threads does not control
    # NumPy's BLAS pool. Otherwise each compiler worker can create dozens of
    # additional CPU threads. Restore the parent's environment afterward.
    with (
        patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        ),
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
        ) as pool,
    ):
        for count, error in pool.map(precompile_case, cases):
            built += count
            if error is not None:
                errors.append(error)
    return built, errors


def selected_cases(items):
    """Keep unskipped tests whose arguments come entirely from parametrization."""
    cases = []
    for item in items:
        if item.get_closest_marker("skip") is not None:
            continue
        # These attention files use Boolean architecture guards. Unknown string
        # guards are left to normal pytest rather than evaluated here.
        if any(
            any(condition for condition in mark.args)
            or mark.kwargs.get("condition", False)
            for mark in item.iter_markers("skipif")
        ):
            continue
        parameters = dict(getattr(getattr(item, "callspec", None), "params", {}))
        required = {
            name
            for name, parameter in inspect.signature(item.function).parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        if not required <= parameters.keys():
            continue
        cases.append((str(item.path), item.originalname, parameters))
    return cases


def prebuild_workers():
    """Respect the runner's compiler budget and cap per-session process count."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    # Bound compiler CPU/memory contention; higher concurrency increased
    # cold-suite wall time in local MLA profiling.
    limits = [16, cpus]
    for name in ("FLASHINFER_JIT_PREBUILD_MAX_JOBS", "MAX_JOBS"):
        value = os.environ.get(name)
        if value is not None:
            limits.append(max(1, int(value)))
    return min(limits)


def prebuild_selected_attention_tests(items):
    """Warm attention artifacts while preserving cache-disabled and single-worker modes."""
    from flashinfer.jit.cute_dsl_core import cute_dsl_cache_disabled

    if cute_dsl_cache_disabled() or os.environ.get("FLASHINFER_DISABLE_JIT"):
        return 0, [], 0
    workers = prebuild_workers()
    cases = selected_cases(items)
    if workers < 2 or len(cases) < 2:
        return 0, [], 0
    # Dense MLA puts its longest multi-plan integration cases at the end.
    # Start those chains first, leaving short cases to fill the tail. Other
    # suites retain their collected order.
    dense = [
        case
        for case in cases
        if Path(case[0]).name == "test_attention_ts_mla_decode.py"
    ]
    other_cases = [
        case
        for case in cases
        if Path(case[0]).name != "test_attention_ts_mla_decode.py"
    ]
    cases = list(reversed(dense)) + other_cases
    built, errors = precompile_cases(cases, min(workers, len(cases)))
    return built, errors, len(cases)
