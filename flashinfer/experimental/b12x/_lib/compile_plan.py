"""Plan production compilation by real cache identity, without compiling it.

Compile-only factories may return deferred kernels during discovery. They must
return their compilation carriers, and host launcher closures explicitly retain
those carriers through ``attach_programs``. No closure/source introspection or
alternative persistent cache format is involved.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProgramKey:
    dialect: str
    key: str
    name: str = field(default="", compare=False)


@dataclass(frozen=True)
class ProgramBundle:
    programs: tuple[ProgramKey, ...]

    @property
    def __b12x_programs__(self):
        return self.programs


@dataclass(eq=False)
class _RetainedPrograms:
    cache: dict[object, Any] = field(default_factory=dict)
    owners: dict[int, Any] = field(default_factory=dict)


_PLANNING = ContextVar("b12x_compile_planning", default=None)
_OBSERVERS = ContextVar("b12x_program_observers", default=())
_SERIALIZE_TRITON = ContextVar("b12x_triton_compile_locks", default=False)
_COMPILE_ONLY_LAUNCHES = ContextVar("b12x_compile_only_launches", default=False)
_FORBID_LOWERING = ContextVar("b12x_forbid_triton_lowering", default=False)
_RETAINED_PROGRAMS = ContextVar("b12x_retained_programs", default=None)
_TRITON_HOOK = None
_RESIDENT_PROGRAMS = weakref.WeakValueDictionary()
_LIVE_RETAINED_PROGRAMS = weakref.WeakSet()
_NATIVE_JITS = weakref.WeakSet()


def retained_program_keys():
    return frozenset(
        program for retained in tuple(_LIVE_RETAINED_PROGRAMS)
        for owner in retained.owners.values() for program in program_keys(owner)
    )


def _planning_artifacts(value: Any, seen: set[int] | None = None) -> Iterator[ProgramKey]:
    """The programs of every deferred kernel inside a memoized value that was never compiled."""
    seen = set() if seen is None else seen
    if id(value) in seen or value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    seen.add(id(value))
    if isinstance(value, (DeferredCuTeKernel, DeferredTritonKernel)):
        if value._resolved is None:
            yield value.__b12x_programs__[0]
        return
    if isinstance(value, Mapping):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    else:
        children = getattr(value, "__b12x_dependencies__", None) or ()
    for child in children:
        yield from _planning_artifacts(child, seen)


def evict_planning_artifacts(programs: Iterable[ProgramKey] | None = None) -> int:
    """Drop the memoized programs that compile planning produced but never compiled.

    Planning installs deferred programs in the families' kernel memos and in
    Triton's per-function caches; a later compile in the same process would
    be handed those entries instead of compiling. Called before an in-process
    compile of `programs` so the factories lower them for real; with no
    program set, every uncompiled deferred program is dropped.
    """
    from .program_cache import _CACHES, _MAPPING_CACHES
    targets = None if programs is None else frozenset(programs)

    def stale(value) -> bool:
        return any(targets is None or program in targets for program in _planning_artifacts(value))

    removed = 0
    caches = [(cache._values, (), cache._lock) for cache in tuple(_CACHES)]
    for cache, mirrors, lock in (*caches, *_MAPPING_CACHES):
        with lock if lock is not None else nullcontext():
            obsolete = [key for key, value in cache.items() if stale(value)]
            for key in obsolete:
                del cache[key]
                for mirror in mirrors:
                    mirror.pop(key, None)
            removed += len(obsolete)
    for jit in tuple(_NATIVE_JITS):
        for kernel_cache, key_cache, *_ in jit.device_caches.values():
            obsolete = {key for key, kernel in kernel_cache.items() if stale(kernel)}
            for key in obsolete:
                del kernel_cache[key]
            for key, value in tuple(key_cache.items()):
                if value in obsolete:
                    del key_cache[key]
            removed += len(obsolete)
    return removed


def evict_unretained_triton(keep):
    removed = 0
    for jit in tuple(_NATIVE_JITS):
        for kernel_cache, key_cache, *_ in jit.device_caches.values():
            obsolete = {key for key, kernel in kernel_cache.items()
                        if not frozenset(program_keys(kernel)) <= keep}
            for key in obsolete:
                del kernel_cache[key]
            for key, value in tuple(key_cache.items()):
                if value in obsolete:
                    del key_cache[key]
            removed += len(obsolete)
    return removed


def compiled_program_available(program: ProgramKey) -> bool:
    """Check an executable or actual artifact, never an unresolved factory."""
    if program in _RESIDENT_PROGRAMS:
        return True
    if program.dialect == "cute":
        from .compiler import _valid_cute_compile_cache
        return _valid_cute_compile_cache(program.key)
    if program.dialect == "triton":
        from triton.compiler.compiler import get_cache_manager
        if not program.name:
            raise ValueError("Triton availability requires its current descriptor name")
        group = get_cache_manager(program.key).get_group(f"{program.name[:150]}.json")
        return bool(group) and all(Path(path).is_file() for path in group.values())
    raise ValueError(f"unsupported compiler dialect {program.dialect!r}")


def compile_only_launches_enabled() -> bool:
    return _COMPILE_ONLY_LAUNCHES.get()


@contextmanager
def compile_only_launches():
    """Resolve ``compiler.launch`` programs without executing their GPU calls."""
    token = _COMPILE_ONLY_LAUNCHES.set(True)
    try:
        with observe_programs() as programs:
            yield programs
    finally:
        _COMPILE_ONLY_LAUNCHES.reset(token)


@contextmanager
def forbid_lowering():
    """Resolve existing Triton objects, rejecting misses before lowering."""
    _ensure_triton_hook()
    token = _FORBID_LOWERING.set(True)
    try:
        yield
    finally:
        _FORBID_LOWERING.reset(token)


def _load_triton_only(source, target=None, options=None, _env_vars=None):
    from triton.compiler.compiler import CompiledKernel, get_cache_manager

    program = _triton_program(source, target, options, _env_vars)
    group = get_cache_manager(program.key).get_group(f"{source.name[:150]}.json")
    if not group:
        raise RuntimeError(f"no-compilation phase encountered an uncached Triton program: {program.name} {program.key}")
    record_program(program)
    return CompiledKernel(source, group, program.key)


def launch_triton(kernel, grid, *args, **kwargs):
    """One production invocation, or its exact warmup in compile-only mode."""
    if compile_only_launches_enabled():
        from triton.runtime.jit import MockTensor

        args = tuple(
            MockTensor(arg.dtype, tuple(arg.shape))
            if getattr(getattr(arg, "device", None), "type", None) == "meta"
            or getattr(arg, "fake_mode", None) is not None else arg
            for arg in args
        )
        compiled = kernel.warmup(*args, grid=grid, **kwargs)
        for program in program_keys(compiled):
            record_program(program)
        return compiled
    return kernel[grid](*args, **kwargs)


def planning() -> bool:
    return _PLANNING.get() is not None


def record_program(program: ProgramKey, owner: Any = None) -> None:
    captured = _PLANNING.get()
    if captured is not None:
        captured.add(program)
    for observed in _OBSERVERS.get():
        observed.add(program)
    if owner is not None:
        retained = _RETAINED_PROGRAMS.get()
        if retained is not None:
            retained.owners[id(owner)] = owner


def program_keys(value: Any) -> tuple[ProgramKey, ...]:
    if value is None or isinstance(value, (str, int, float, bool)) or type(value).__module__ == "torch":
        return ()
    if isinstance(value, Mapping):
        value = tuple(value.values())
    if isinstance(value, (tuple, list)):
        return tuple(dict.fromkeys(key for item in value for key in program_keys(item)))
    keys = getattr(value, "__b12x_programs__", None)
    if keys is not None:
        return tuple(keys)
    from triton.compiler.compiler import CompiledKernel
    if isinstance(value, CompiledKernel):
        return (ProgramKey("triton", value.hash, value.name),)
    raise TypeError(f"compile factory returned an unannotated {type(value).__name__}")


def attach_programs(value: Any, *dependencies: Any) -> Any:
    """Retain exact compiler keys on a host launch closure or compile plan."""
    keys = tuple(dict.fromkeys(key for item in dependencies for key in program_keys(item)))
    object.__setattr__(value, "__b12x_programs__", keys)
    object.__setattr__(value, "__b12x_dependencies__", tuple(dependencies))
    return value


class CompiledCuTeProgram:
    """A vendor executable with an immutable production-program identity."""

    def __init__(self, executable, program):
        self.__b12x_programs__ = (program,)
        self._executable = executable
        _RESIDENT_PROGRAMS[program] = self

    def __call__(self, *args, **kwargs):
        record_program(self.__b12x_programs__[0], self)
        return self._executable(*args, **kwargs)

    def __getattr__(self, name):
        record_program(self.__b12x_programs__[0], self)
        return getattr(self._executable, name)


def tag_compiled(value: Any, program: ProgramKey) -> Any:
    record_program(program)
    if isinstance(value, CompiledCuTeProgram):
        if value.__b12x_programs__ != (program,):
            raise ValueError("compiled executable has a conflicting program identity")
        return value
    return CompiledCuTeProgram(value, program)


@contextmanager
def observe_programs():
    observed = set()
    token = _OBSERVERS.set((*_OBSERVERS.get(), observed))
    try:
        yield observed
    finally:
        _OBSERVERS.reset(token)


@contextmanager
def retain_compiled_programs():
    """Own warmed programs until captured graphs release their CUDA functions."""
    _ensure_triton_hook()
    retained = _RETAINED_PROGRAMS.get()
    if retained is not None:
        yield retained
        return
    retained = _RetainedPrograms()
    _LIVE_RETAINED_PROGRAMS.add(retained)
    token = _RETAINED_PROGRAMS.set(retained)
    try:
        yield retained
    finally:
        _RETAINED_PROGRAMS.reset(token)


class DeferredCuTeKernel:
    def __init__(self, program: ProgramKey, memory_key):
        self.__b12x_programs__ = (program,)
        self._memory_key = memory_key
        self._resolved = None

    def _load(self):
        if planning():
            raise RuntimeError("compile planning attempted kernel execution or resource inspection")
        record_program(self.__b12x_programs__[0])
        if self._resolved is None:
            from . import compiler
            from .runtime_control import raise_if_kernel_resolution_frozen
            program = self.__b12x_programs__[0]
            value = compiler._memory_cache_get(self._memory_key)
            if value is None:
                raise_if_kernel_resolution_frozen("planned CuTe object load", cache_key=program.key)
                value = compiler._load_cute_compile_from_disk(program.key)
                if value is None:
                    raise RuntimeError(f"planned CuTe program has not been compiled: {program.name} {program.key}")
                value = tag_compiled(value, program)
                compiler._memory_cache_put(self._memory_key, value)
            self._resolved = value
        return self._resolved

    def __call__(self, *args, **kwargs):
        record_program(self.__b12x_programs__[0])
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)


class DeferredTritonKernel:
    def __init__(self, program: ProgramKey, source):
        self.__b12x_programs__ = (program,)
        self._source = source
        self._resolved = None

    def _load(self):
        if planning():
            raise RuntimeError("compile planning attempted Triton execution or resource inspection")
        if self._resolved is None:
            from triton.compiler.compiler import CompiledKernel, get_cache_manager
            from .runtime_control import raise_if_kernel_resolution_frozen
            program = self.__b12x_programs__[0]
            raise_if_kernel_resolution_frozen("planned Triton object load", cache_key=program.key)
            group = get_cache_manager(program.key).get_group(f"{self._source.name[:150]}.json")
            if not group:
                raise RuntimeError(f"planned Triton program has not been compiled: {program.name} {program.key}")
            self._resolved = CompiledKernel(self._source, group, program.key)
            _RESIDENT_PROGRAMS[program] = self._resolved
        record_program(self.__b12x_programs__[0], self._resolved)
        return self._resolved

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __getitem__(self, grid):
        return self._load()[grid]


def load_programs(value: Any) -> Any:
    """Load explicit executable dependencies before priming or graph capture.

    This follows declared carrier references only; it does not inspect closures
    or infer readiness from metadata keys. Loading never compiles missing code.
    """
    if planning() or compile_only_launches_enabled():
        raise RuntimeError("executable loading is not a metadata-only operation")
    from triton.compiler.compiler import CompiledKernel
    from .runtime_control import raise_if_kernel_resolution_frozen

    visited = set()

    def load(item):
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(item, Mapping):
            for child in item.values():
                load(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                load(child)
        elif isinstance(item, (DeferredCuTeKernel, DeferredTritonKernel)):
            load(item._load())
        elif isinstance(item, CompiledCuTeProgram):
            record_program(item.__b12x_programs__[0], item)
        elif isinstance(item, CompiledKernel):
            if item.module is None:
                raise_if_kernel_resolution_frozen("prepared Triton module load", cache_key=item.hash)
                item._init_handles()
            for program in program_keys(item):
                record_program(program, item)
        else:
            dependencies = getattr(item, "__b12x_dependencies__", None)
            if dependencies is None:
                raise TypeError(f"executable carrier has no explicit dependencies: {type(item).__name__}")
            for child in dependencies:
                load(child)

    load(value)
    return value


def _triton_program(source, target=None, options=None, _env_vars=None):
    # These are the exact Triton 3.7 compiler cache-key operations, including its
    # source/options/environment key and optional instrumentation contribution.
    from triton import knobs
    from triton.compiler.compiler import (
        ASTSource, driver, get_cache_invalidating_env_vars, get_cache_key, make_backend,
    )
    if not isinstance(source, ASTSource):
        raise TypeError("startup planning requires a Triton AST compilation request")
    target = driver.active.get_current_target() if target is None else target
    backend = make_backend(target)
    parsed = backend.parse_options(dict(options or {}, **source.parse_options()))
    environment = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
    key = get_cache_key(source, backend, parsed, env_vars=environment)
    if knobs.runtime.add_stages_inspection_hook is not None:
        instrumentation_key, _ = knobs.runtime.add_stages_inspection_hook()
        key += instrumentation_key
    return ProgramKey("triton", hashlib.sha256(key.encode()).hexdigest(), source.name)


def _plan_triton(source, target=None, options=None, _env_vars=None):
    program = _triton_program(source, target, options, _env_vars)
    record_program(program)
    return DeferredTritonKernel(program, source)


@contextmanager
def serialized_compilations():
    """Single-flight Triton lowering by its real persistent cache key."""
    _ensure_triton_hook()
    token = _SERIALIZE_TRITON.set(True)
    try:
        yield
    finally:
        _SERIALIZE_TRITON.reset(token)


def _locked_triton_compile(compile_fn, source, target=None, options=None, _env_vars=None):
    import fcntl
    from pathlib import Path
    from triton.compiler.compiler import get_cache_manager

    program = _triton_program(source, target, options, _env_vars)
    cache = get_cache_manager(program.key)
    directory = getattr(cache, "cache_dir", None)
    if directory is None:
        raise RuntimeError("parallel Triton compilation requires its local persistent cache")
    Path(directory).mkdir(parents=True, exist_ok=True)
    with (Path(directory) / ".b12x-compile.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            result = compile_fn(source, target=target, options=options, _env_vars=_env_vars)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    record_program(program)
    return result


def _ensure_triton_hook():
    global _TRITON_HOOK
    if _TRITON_HOOK is not None:
        return
    from triton.runtime.jit import JITFunction
    from triton.compiler.compiler import CompiledKernel
    original = JITFunction._do_compile
    original_run = JITFunction.run
    original_init = CompiledKernel._init_handles
    original_getitem = CompiledKernel.__getitem__

    def init_handles(self):
        fn = getattr(getattr(getattr(self, "src", None), "fn", None), "fn", None)
        module = getattr(fn, "__module__", "")
        native = module == "b12x" or module.startswith("b12x.")
        if native and self.module is None:
            from .runtime_control import raise_if_kernel_resolution_frozen
            raise_if_kernel_resolution_frozen("Triton compiled module load", target=fn, cache_key=self.hash)
        original_init(self)
        if native:
            if not hasattr(self, "__b12x_programs__"):
                self.__b12x_programs__ = (ProgramKey("triton", self.hash, self.name),)
            program = self.__b12x_programs__[0]
            if _RESIDENT_PROGRAMS.get(program) is not self:
                _RESIDENT_PROGRAMS[program] = self

    def getitem_observed(self, grid):
        runner = original_getitem(self, grid)
        if _OBSERVERS.get() or _RETAINED_PROGRAMS.get() is not None:
            for program in program_keys(self):
                record_program(program, self)
        return runner

    def run_observed(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        module = getattr(self.fn, "__module__", "")
        if module == "b12x" or module.startswith("b12x."):
            _NATIVE_JITS.add(self)
        if result is not None and (_OBSERVERS.get() or planning() or _RETAINED_PROGRAMS.get() is not None):
            for program in program_keys(result):
                record_program(program, result)
        return result

    def compile_or_plan(self, key, signature, device, constexprs, options, attrs, warmup):
        module = getattr(self.fn, "__module__", "")
        if module == "b12x" or module.startswith("b12x."):
            from .runtime_control import raise_if_kernel_resolution_frozen
            raise_if_kernel_resolution_frozen("Triton JIT miss", target=self.fn, cache_key=key)
        if not planning() and not _SERIALIZE_TRITON.get() and not _FORBID_LOWERING.get():
            return original(self, key, signature, device, constexprs, options, attrs, warmup)
        if planning() and not warmup:
            raise RuntimeError("compile-plan factories must use Triton warmup, never launch")
        previous = self.compile
        if planning():
            self.compile = _plan_triton
        elif _FORBID_LOWERING.get():
            self.compile = _load_triton_only
        else:
            from functools import partial
            self.compile = partial(_locked_triton_compile, previous)
        try:
            return original(self, key, signature, device, constexprs, options, attrs, warmup)
        finally:
            self.compile = previous

    JITFunction._do_compile = compile_or_plan
    JITFunction.run = run_observed
    CompiledKernel._init_handles = init_handles
    CompiledKernel.__getitem__ = getitem_observed
    _TRITON_HOOK = compile_or_plan


@contextmanager
def plan_compilations():
    """Capture the real requests of one metadata-only production factory."""
    _ensure_triton_hook()
    captured = set()
    token = _PLANNING.set(captured)
    try:
        yield captured
    finally:
        _PLANNING.reset(token)


__all__ = ["DeferredCuTeKernel", "ProgramKey", "ProgramBundle", "attach_programs",
           "compile_only_launches", "compile_only_launches_enabled", "observe_programs",
           "launch_triton", "plan_compilations", "planning", "program_keys", "record_program",
           "retain_compiled_programs", "serialized_compilations", "tag_compiled"]
