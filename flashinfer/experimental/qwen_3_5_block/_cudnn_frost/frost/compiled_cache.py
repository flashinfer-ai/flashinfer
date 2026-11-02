# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent cache of compiled FROST kernels, across processes.

``cute.compile`` runs the whole backend (NVVM, ptxas, host stub) once per
process for every distinct kernel -- 0.7 s for a GEMM, 2.6 s for an SDPA
prefill on SM100 -- and the DSL's own file cache stores only MLIR bytecode, so
the heavy part is paid again at every start-up. A serving process that warms
tens of plans pays minutes. This module keeps the exported object of every
kernel compiled with ``--enable-tvm-ffi`` and reloads it (a few milliseconds)
instead.

Layout, and the rules that keep a stale artifact from ever being reused:

    <root>/v2/<env_hash>/manifest.json           the environment identity, human-readable
    <root>/v2/<env_hash>/<entry_hash>/kernel.o   the exported tvm-ffi object
    <root>/v2/<env_hash>/<entry_hash>/entry.json the key and calling convention it was stored under (commit marker)

- ``env_hash`` is the content hash of :func:`environment_manifest`: frontend
  version and a digest of the whole ``cudnn`` package's source (the shared
  helpers a kernel imports are not in its own key; an edited checkout must
  not hit a wheel's entries), cutlass-dsl and tvm-ffi versions, the CUDA
  driver, and the device's name,
  compute capability, SM count and L2 size (FROST bakes the last two into
  kernels). Any change lands in a different directory; nothing is ever
  "matched approximately".
- ``entry_hash`` is the content hash of the caller's key -- for the FROST GEMM
  engine the digest of the generated source, which already carries the tile
  config, dtypes, fusion chain and ``--gpu-arch`` -- plus the exported symbol
  and the compile options. ``entry.json`` embeds the full key and is compared
  on load, so a hash collision or a foreign file is a miss.
- Missing, unreadable, mismatched: a miss, never an error. The object is
  written to a temporary name and renamed into place; ``entry.json`` is written
  after it, so a crash never leaves a loadable entry without its key.
- Bump ``_SCHEMA`` on any incompatible change to this layout.

``compile_cached`` is a drop-in for ``cute.compile`` at a kernel's compile
site. A reloaded tvm-ffi function is positional-only, so it is wrapped with a
kwargs wrapper built from the kernel function's Python signature -- the same
thing the in-process object does -- and the caller cannot tell a hit from a
miss. Kernels whose in-process object converts raw pointer arguments are not
cached (the reload path has no converter); everything else is.

In front of the files sits an in-process memo: the object ``compile_cached``
handed out for a ``(target device, live device, cache_key, symbol, options)`` is
handed out again for the same key in the same process -- no trace, no compile, no
file touched, whatever the on-disk cache's state (the two devices: the one the
build bakes its constants for and the CUDA context it is issued under,
:func:`_memo_device`; they agree unless a handle scoped the build to another
GPU). A graph that builds several plans over one kernel, or a test suite that
builds one block instance per case, otherwise pays a full JIT per plan:
``cute.compile`` has no memo of its own. The compiled
object is a stateless callable (a plan already shares one across every
execute), so one per key per process is what every caller wants; a kernel
without a key is never memoised. ``clear_memo()`` forgets the objects;
``stats()["memo_hits"]`` counts the hits.

Location: ``CUDNN_FRONTEND_COMPILED_CACHE`` (a directory), else
``$XDG_CACHE_HOME/cudnn_frontend/compiled_plans`` (``~/.cache`` when unset).
``CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`` turns the cache off;
``CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES`` caps the root (4 GiB by default, 0 = never
prune): a process's first write removes whole dead environment directories,
oldest first, until the root fits. ``CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO=0``
turns the in-process memo off (a test of the reload path wants every call to
reach the files; an A/B of the memo itself). A caller
that manages its own workspace (FlashInfer) points :func:`set_cache_dir` at
it once per process.

Plan serialization (``cudnn._plan_serdes``) carries entries between processes:
:func:`record_entries` collects the entries a build used, :func:`read_entry`
reads one with its manifest, and :func:`seed_entry` installs it only under an
identical manifest.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

_SCHEMA = "v2"  # v2: the record carries the runtime wrapper spec, not a Python signature
_ENV_DIR = "CUDNN_FRONTEND_COMPILED_CACHE"
_ENV_DISABLE = "CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"
_ENV_MAX_BYTES = "CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES"
_ENV_MEMO = "CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO"
_DEFAULT_MAX_BYTES = 4 * 1024**3  # every environment the root has seen, together
_OBJECT = "kernel.o"
_ENTRY = "entry.json"
_MANIFEST = "manifest.json"
_SCHEMA_DIR = re.compile(r"v[0-9]+")  # the shape of a schema directory name ...
_DIGEST_DIR = re.compile(r"[0-9a-f]{24}")  # ... and of an environment / entry directory name (see _digest)

_LOG = logging.getLogger("cudnn.frost.compiled_cache")
_LOCK = threading.Lock()
_dir_override: Optional[Path] = None
_stats = {"hits": 0, "misses": 0, "bypassed": 0, "invalid": 0, "export_failed": 0, "pruned": 0, "memo_hits": 0, "seeded": 0, "seed_rejected": 0}
_pruned_this_process = False
# The in-process memo: (target device, live device, cache_key, symbol, options) -> the object compile_cached handed out
# for it in THIS process, whatever the files did (a reloaded artifact or the in-process object); the two devices are
# _memo_device()'s. Guarded by _LOCK like the counters.
# Deliberately UNBOUNDED. It holds one entry per distinct kernel the process ever compiled, and each entry is the very
# object the execution plan that asked for it keeps alive for its whole life, so the memo's footprint is the set of distinct
# kernels, not the number of plans; only once a caller drops a plan does the memo hold anything the process would not
# otherwise hold (a compiled object: its cubin and host stub, hundreds of KB), until clear_memo() or exit. An LRU cap
# would need a reuse pattern to evict by and would reintroduce the full recompile the memo exists to avoid; a caller
# that mints ever-new keys (a sweep over shapes) pays a compile per key anyway and has clear_memo() as the release valve.
_MEMO: Dict[tuple, Any] = {}
# Per-thread stack of lists that record_entries() hands out; compile_cached appends to the innermost.
_RECORDING = threading.local()
# Per-thread depth of refuse_compiles() scopes; nonzero makes a cache miss raise instead of compiling.
_REFUSING = threading.local()


# ---------------------------------------------------------------------------
# Location and switches
# ---------------------------------------------------------------------------


def enabled() -> bool:
    """Whether compiled kernels are persisted and reloaded at all."""
    return os.environ.get(_ENV_DISABLE, "0").strip().lower() not in ("1", "true", "yes", "on")


def memo_enabled() -> bool:
    """Whether a kernel compiled once in this process is handed out again for the same key
    (the default); ``CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO=0`` turns that off."""
    return os.environ.get(_ENV_MEMO, "1").strip().lower() not in ("0", "false", "no", "off")


def clear_memo() -> None:
    """Forget every compiled object the in-process memo holds. The objects callers already
    hold stay valid; the next ``compile_cached`` of a forgotten key reaches the files (or the
    compiler) again."""
    with _LOCK:
        _MEMO.clear()


def set_cache_dir(path) -> None:
    """Point the cache at ``path`` for this process (``None`` restores the default)."""
    global _dir_override
    with _LOCK:
        _dir_override = None if path is None else Path(path)


def get_cache_dir() -> Path:
    """The cache root: :func:`set_cache_dir`, else ``$CUDNN_FRONTEND_COMPILED_CACHE``,
    else ``$XDG_CACHE_HOME/cudnn_frontend/compiled_plans``. Falls back to a
    per-user directory under the system temp dir when that is not writable."""
    with _LOCK:
        if _dir_override is not None:
            return _dir_override
    base = os.environ.get(_ENV_DIR)
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "cudnn_frontend", "compiled_plans")
    root = Path(base)
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not os.access(root, os.W_OK):
            raise PermissionError(f"{root} is not writable")
        return root
    except OSError as exc:
        suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
        fallback = Path(tempfile.gettempdir()) / f"cudnn_frontend_compiled_plans{suffix}"
        fallback.mkdir(parents=True, exist_ok=True)
        _LOG.warning("compiled-plan cache %s is unusable (%s); using %s. Set %s to choose another location.", root, exc, fallback, _ENV_DIR)
        return fallback


def stats() -> Dict[str, int]:
    """Counters for this process: hits, misses, bypassed (not cacheable), invalid (entry rejected), export_failed, pruned (environment
    directories removed), memo_hits (served from the in-process memo: neither the files nor the compiler were reached), seeded /
    seed_rejected (:func:`seed_entry` installed an entry / refused one from another environment)."""
    with _LOCK:
        return dict(_stats)


def reset_stats() -> None:
    with _LOCK:
        for k in _stats:
            _stats[k] = 0


def _count(name: str, n: int = 1) -> None:
    with _LOCK:
        _stats[name] += n


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 -- the package may be vendored or absent
        return "unknown"


_SOURCE_DIGEST: Optional[str] = None


def source_tree_digest(root: Path) -> str:
    """One digest over every ``.py`` under ``root`` (relative path + bytes, sorted).

    A kernel is compiled from more than its own template: the shared frost
    helpers, the DSL adapters and the generator it imports all shape the traced
    code, and none of them is in a template's key. The version string covers a
    wheel; it does not cover a checkout someone is editing, and a stale hit
    there is a wrong kernel with no error. So the manifest carries the digest
    of the whole package instead of a commit hash: an uncommitted edit counts
    too, and an installed wheel hashes the same every time. Computed once per
    process (a few tens of milliseconds for the tree).
    """
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _cudnn_source_digest() -> str:
    global _SOURCE_DIGEST
    if _SOURCE_DIGEST is None:
        try:
            import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

            _SOURCE_DIGEST = source_tree_digest(Path(cudnn.__file__).resolve().parent)
        except Exception:  # noqa: BLE001 -- a package that cannot be read shares nothing
            _SOURCE_DIGEST = "unknown"
    return _SOURCE_DIGEST


def environment_manifest(device: Optional[int] = None) -> Dict[str, str]:
    """Everything a compiled object depends on, as deterministic strings.

    A field that cannot be read becomes ``"unknown"``, and :func:`compile_cached`
    then persists nothing: an environment that cannot identify itself must not
    share an entry with one that happens to carry the same unknowns. Never
    loosen a field to widen hits: a wrong hit is a wrong kernel.
    """
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    manifest: Dict[str, str] = {"schema": _SCHEMA, "cudnn_frontend": str(getattr(cudnn, "__version__", "unknown")), "cudnn_source": _cudnn_source_digest()}
    try:
        import cutlass

        manifest["cutlass_dsl"] = str(getattr(cutlass, "__version__", _dist_version("nvidia-cutlass-dsl")))
    except Exception:  # noqa: BLE001
        manifest["cutlass_dsl"] = "unknown"
    manifest["tvm_ffi"] = _dist_version("apache-tvm-ffi")
    try:
        from cuda.bindings import driver as _cu

        err, ver = _cu.cuDriverGetVersion()
        manifest["cuda_driver"] = str(ver) if int(err) == 0 else "unknown"
    except Exception:  # noqa: BLE001
        manifest["cuda_driver"] = "unknown"
    try:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import device as _dev

        d = _dev.resolve_device(device)
        cc = _dev.compute_capability(d)
        manifest["device_name"] = str(_dev.device_name(d))
        manifest["compute_capability"] = f"{cc[0]}.{cc[1]}"
        manifest["sm_count"] = str(_dev.multiprocessor_count(d))
        manifest["l2_bytes"] = str(_dev.l2_cache_bytes(d))
    except Exception:  # noqa: BLE001 -- no device: the manifest still names that fact
        for k in ("device_name", "compute_capability", "sm_count", "l2_bytes"):
            manifest[k] = "unknown"
    return manifest


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _entry_dir(root: Path, manifest: Dict[str, str], key: str) -> Path:
    return root / _SCHEMA / _digest(_canonical(manifest)) / _digest(key)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=f".{os.getpid()}.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 -- unreadable is a miss
        return None


_PLAIN = (int, float, bool, str, type(None))


def _plain(value: Any) -> bool:
    """Whether ``repr(value)`` names the same thing in another process: plain
    scalars, tuples/lists of them, dataclass instances whose fields are (a
    template's params record travels into compile() this way), and types
    (``cutlass.BFloat16`` and friends; a class repr is its qualified name)."""
    if isinstance(value, _PLAIN) or isinstance(value, type):
        return True
    if isinstance(value, (tuple, list)):
        return all(_plain(v) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return all(_plain(getattr(value, f.name)) for f in dataclasses.fields(value))
    return False


def template_key(module_globals: Dict[str, Any], arguments: Dict[str, Any], function: str = "compile") -> Optional[str]:
    """The cache key of one ``compile()`` call of a kernel template.

    A template specializes twice: at import, from the file and its
    ``FROST_TEMPLATE_PARAMS`` (``template_loader`` records that as
    ``FROST_SOURCE_DIGEST``), and at ``compile()``, from the call's arguments
    (shapes, strides, flags), which pin the traced code as much as the params
    do. The key joins the two. Call it as the FIRST statement of the function,
    with ``locals()``, so the arguments are exactly the parameters. None -- not
    cacheable -- when the module carries no digest or an argument is not a
    plain value (a tensor, a device, a stream): its ``repr`` need not name the
    same kernel in another process.
    """
    digest = module_globals.get("FROST_SOURCE_DIGEST")
    if not digest:
        return None
    items = sorted(arguments.items())
    if not all(_plain(v) for _, v in items):
        return None
    return f"{digest}|{function}|{items!r}"


def _json_plain(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _wrapper_spec_of(compiled: Any, args: tuple, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The calling convention of the in-process object, as JSON.

    The DSL calls a kernel through a kwargs wrapper generated from its
    ORIGINAL signature minus the parameters that do not exist at run time:
    ``cutlass.Constexpr`` arguments are baked into the kernel and an
    env-stream argument is read from the environment. A wrapper built from the
    Python signature instead would shift every argument after a constexpr one.
    So the spec is taken the way the DSL takes it: ``execution_args`` names the
    parameters and their defaults; the in-process wrapper's own signature says
    which of them survived. None when it cannot be reproduced exactly -- no
    wrapper, a default that JSON cannot carry, or a runtime dataclass argument
    (the DSL unpacks those through a hook the record does not describe).
    A constexpr dataclass is absent from the runtime signature and needs no
    conversion when the artifact is reloaded.
    """
    wrapper = getattr(compiled, "_kwargs_wrapper", None)
    execution_args = getattr(compiled, "execution_args", None)
    if wrapper is None or execution_args is None or not hasattr(execution_args, "get_kwargs_wrapper_spec"):
        return None
    if any(dataclasses.is_dataclass(a) and not isinstance(a, type) for a in list(args) + list(kwargs.values())):
        try:
            # Let the DSL remove its own constexpr arguments. Guessing from
            # annotations or positional offsets can silently shift the ABI.
            signature = execution_args.original_signature
            function_kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
            runtime_args = execution_args.get_rectified_args_from_original_args(args, function_kwargs)
        except Exception:  # noqa: BLE001 -- older executors cannot establish this contract
            return None
        if any(dataclasses.is_dataclass(a) and not isinstance(a, type) for a in runtime_args):
            return None
    try:
        survived = set(inspect.signature(wrapper).parameters)
        full = execution_args.get_kwargs_wrapper_spec(())
        excluded = [n for n in list(full.arg_names) + list(full.kwonly_names) if n not in survived]
        spec = execution_args.get_kwargs_wrapper_spec(excluded) if excluded else full
    except Exception:  # noqa: BLE001 -- a DSL whose executor does not expose the spec
        return None
    arg_defaults, kwonly_defaults = list(spec.arg_defaults), dict(spec.kwonly_defaults)
    if not all(_json_plain(v) for v in arg_defaults + list(kwonly_defaults.values())):
        return None
    return {"arg_names": list(spec.arg_names), "arg_defaults": arg_defaults, "kwonly_names": list(spec.kwonly_names), "kwonly_defaults": kwonly_defaults}


def _rebuild_wrapper(raw: Callable, spec: Dict[str, Any]) -> Optional[Callable]:
    """The kwargs wrapper the in-process object had, over the reloaded function."""
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    if not isinstance(spec, dict) or not all(k in spec for k in ("arg_names", "arg_defaults", "kwonly_names", "kwonly_defaults")):
        return None
    return make_kwargs_wrapper(
        raw,
        arg_names=list(spec["arg_names"]),
        arg_defaults=tuple(spec["arg_defaults"]),
        kwonly_names=list(spec["kwonly_names"]),
        kwonly_defaults=dict(spec["kwonly_defaults"]),
    )


def max_bytes() -> int:
    """The size the whole root may grow to before old environments are removed:
    ``CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES`` (0 disables pruning), else 4 GiB."""
    raw = os.environ.get(_ENV_MAX_BYTES)
    if raw is None or raw == "":
        return _DEFAULT_MAX_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MAX_BYTES


def _dir_bytes_and_mtime(path: Path):
    total, newest = 0, 0.0
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                st = os.stat(os.path.join(dirpath, name), follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue  # a symlinked file would count its target's size
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return total, newest


def prune(root: Optional[Path] = None, limit: Optional[int] = None, keep: Optional[Path] = None) -> int:
    """Remove whole ENVIRONMENT directories, oldest first, until the root is under
    ``limit`` bytes. Returns the number removed.

    The manifest hashes the package's source, so every edited checkout -- and
    every CI commit -- lands in a new environment directory that will never be
    hit again; on a runner with a persistent home that is a few hundred MB per
    commit, forever. Whole directories go, never single entries: an
    environment is either current (``keep``, this process's own) or dead.
    Other schema versions' roots are dead outright and go first. A directory
    another live process is still reading only costs that process misses.
    Only directories this cache made are candidates -- a ``v<N>`` schema
    directory, under it an environment named by our digest whose manifest
    parses and names that schema and a cudnn_frontend version; a caller who
    points the root at a shared directory keeps everything else, uncounted.
    """
    root = Path(root) if root is not None else get_cache_dir()
    limit = max_bytes() if limit is None else limit
    if limit <= 0 or not root.is_dir():
        return 0
    resolved_root = root.resolve()
    envs = []
    # Symlinks are never followed and nothing outside the resolved root is ever
    # a deletion target: a link planted in the cache must not redirect rmtree.
    for schema_dir in root.iterdir():
        if schema_dir.is_symlink() or not schema_dir.is_dir() or not _SCHEMA_DIR.fullmatch(schema_dir.name):
            continue
        for env in schema_dir.iterdir():
            if env.is_symlink() or not env.is_dir() or not _DIGEST_DIR.fullmatch(env.name):
                continue
            if not env.resolve().is_relative_to(resolved_root) or not _is_our_environment(env, schema_dir.name):
                continue
            size, mtime = _dir_bytes_and_mtime(env)
            envs.append((schema_dir.name != _SCHEMA, mtime, size, env))
    total = sum(e[2] for e in envs)
    removed = 0
    # dead schemas first, then oldest first; the current environment is never a candidate
    for _dead_schema, _mtime, size, env in sorted(envs, key=lambda e: (not e[0], e[1])):
        if total <= limit:
            break
        if keep is not None and env.resolve() == Path(keep).resolve():
            continue
        try:
            shutil.rmtree(env)
        except OSError as exc:
            _LOG.warning("compiled-plan cache: could not remove %s (%s)", env, exc)
            continue
        total -= size
        removed += 1
    if removed:
        _count("pruned", removed)
    return removed


def _is_our_environment(env: Path, schema: str) -> bool:
    """An environment directory this cache wrote: its manifest parses and names
    this schema and a cudnn_frontend version. A same-shaped directory another
    tool made (or a half-written one) is not ours and is never removed."""
    manifest = _read_json(env / _MANIFEST)
    if not isinstance(manifest, dict) or manifest.get("schema") != schema:
        return False
    version = manifest.get("cudnn_frontend")
    return isinstance(version, str) and bool(version.strip())


def _prune_once(root: Path, current_env: Path) -> None:
    """Run :func:`prune` the first time this process writes an entry."""
    global _pruned_this_process
    with _LOCK:
        if _pruned_this_process:
            return
        _pruned_this_process = True
    try:
        prune(root, keep=current_env)
    except Exception as exc:  # noqa: BLE001 -- housekeeping never fails a compile
        _LOG.warning("compiled-plan cache: prune failed (%s)", exc)


# ---------------------------------------------------------------------------
# The drop-in
# ---------------------------------------------------------------------------


def _try_load(entry: Path, key: str, symbol: str):
    """The reloaded kernel behind the in-process calling convention, or None
    (missing, mismatched, unloadable)."""
    record = _read_json(entry / _ENTRY)
    if not isinstance(record, dict):
        return None
    if record.get("schema") != _SCHEMA or record.get("key") != key or record.get("symbol") != symbol:
        _count("invalid")
        _LOG.warning("compiled-plan cache: ignoring %s (embedded key does not match); treating as a miss", entry)
        return None
    obj = entry / _OBJECT
    if not obj.is_file():
        return None
    try:
        import cutlass

        module = cutlass.runtime.load_module(str(obj), enable_tvm_ffi=True)
        raw = getattr(module, symbol)
        wrapper = _rebuild_wrapper(raw, record.get("wrapper"))
        if wrapper is None:
            return None
        wrapper._compiled_cache_module = module  # keep the engine alive as long as the callable
        wrapper._compiled_cache_entry = str(entry)
        wrapper._compiled_cache_raw = raw  # positional-only tvm_ffi.Function, for a prepared launch
        return wrapper
    except Exception as exc:  # noqa: BLE001 -- a stale or foreign artifact is a miss
        _count("invalid")
        _LOG.warning("compiled-plan cache: could not reload %s (%s); recompiling", obj, exc)
        return None


def _export(entry: Path, compiled: Any, key: str, symbol: str, manifest: Dict[str, str], wrapper: Dict[str, Any]) -> None:
    """Publish ``compiled`` under ``entry``: object first, record last (the commit marker)."""
    entry.mkdir(parents=True, exist_ok=True)
    manifest_path = entry.parent / _MANIFEST
    if not manifest_path.exists():
        _write_atomic(manifest_path, json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8"))
    # export_to_c writes the file itself, so it gets a unique temp name and is
    # renamed into place; a reader sees the whole object or nothing.
    tmp = entry / f"{_OBJECT}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    try:
        compiled.export_to_c(str(tmp), function_name=symbol)
        os.replace(tmp, entry / _OBJECT)
    finally:
        if tmp.exists():
            tmp.unlink()
    record = {"schema": _SCHEMA, "key": key, "symbol": symbol, "wrapper": wrapper}
    _write_atomic(entry / _ENTRY, json.dumps(record, indent=1, sort_keys=True).encode("utf-8"))


def positional_entry(compiled: Any) -> Optional[Any]:
    """The positional ``tvm_ffi.Function`` behind a ``compile_cached`` result, or None.

    A reloaded artifact carries it as ``_compiled_cache_raw``; an in-process
    ``--enable-tvm-ffi`` object exposes it through ``__tvm_ffi_object__``. Pointer
    parameters take plain integer addresses, ``Tuple[int, ...]`` parameters take
    tuples; the caller keeps ``compiled`` alive for as long as it uses the entry."""
    raw = getattr(compiled, "_compiled_cache_raw", None)
    if raw is not None:
        return raw
    get = getattr(compiled, "__tvm_ffi_object__", None)
    if get is None:
        return None
    try:
        return get()
    except Exception:  # noqa: BLE001 -- not a tvm-ffi object after all
        return None


def _memo_device() -> Tuple[Optional[int], Optional[int]]:
    """The two CUDA ordinals a compile issued now is for, both part of the memo key.

    First the device whose constants the object bakes (``current_device()``: a ``build_device()``
    scope, else the live device) -- SM count, L2 size, SMEM budget -- which is why the on-disk
    manifest names the device too. Then the live device itself (``ambient_device()``: the CUDA
    context the compile is issued under, scope or no scope). The DSL documents a compiled
    object's executor as tied to the device context it was created in, and an engine that
    builds under a handle's device scope launches under whatever context is current at execute
    (gemm, conv): a per-build compile never shared one object between a build issued under
    context 0 and one issued under context 1, so neither may the memo -- a build scoped to
    device 1 under context 0 and the same build under context 1 are two objects. For an
    unscoped build the two ordinals agree. ``None`` for an ordinal no CUDA device answers: one
    slot, like the manifest's "unknown"."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import device as _dev

    def ordinal(query) -> Optional[int]:
        try:
            return int(query())
        except Exception:  # noqa: BLE001 -- no CUDA device visible
            return None

    return ordinal(_dev.current_device), ordinal(_dev.ambient_device)


def compile_cached(fn: Callable, *args: Any, cache_key: Optional[str], symbol: str = "kernel", **kwargs: Any) -> Any:
    """``cute.compile(fn, *args, **kwargs)`` with a persistent object behind it.

    ``cache_key`` names the kernel exactly (for a generated kernel: the digest
    of its source); ``None`` means "not cacheable" and compiles as before. The
    compile options must include ``--enable-tvm-ffi`` -- that is what makes the
    object exportable and reloadable; other kernels compile as before.

    A second call with the same ``(cache_key, symbol, options)`` for the same
    device -- the one the build targets AND the live CUDA device it is issued
    under, :func:`_memo_device` -- in the same process returns the object the
    first call returned --
    whatever that was (the reloaded artifact or the in-process object) and
    whatever the on-disk cache's state -- without re-tracing and without
    touching the files (``memo_enabled()``); a key of ``None`` is never memoised.
    """
    import cutlass.cute as cute

    options = str(kwargs.get("options") or "")
    if cache_key is None:
        _refuse_compile(symbol)
        _count("bypassed")
        return cute.compile(fn, *args, **kwargs)
    memo_key = (*_memo_device(), cache_key, symbol, options) if memo_enabled() else None
    if memo_key is not None:
        with _LOCK:
            memo = _MEMO.get(memo_key)
        if memo is not None:
            _count("memo_hits")
            _record(cache_key, symbol, options)
            return memo
    compiled = _compile_persistent(fn, args, kwargs, cache_key=cache_key, symbol=symbol, options=options)
    if memo_key is not None:
        with _LOCK:
            # Threads that raced on an empty slot each compiled once; every one of them leaves with the
            # FIRST object stored, so one key means one object for the rest of the process.
            compiled = _MEMO.setdefault(memo_key, compiled)
    _record(cache_key, symbol, options)
    return compiled


def _compile_persistent(fn: Callable, args: tuple, kwargs: Dict[str, Any], *, cache_key: str, symbol: str, options: str) -> Any:
    """The files behind :func:`compile_cached`: reload the entry for the key, or compile, export and reload it."""
    import cutlass.cute as cute

    if not enabled() or "--enable-tvm-ffi" not in options:
        _refuse_compile(symbol)
        _count("bypassed")
        return cute.compile(fn, *args, **kwargs)
    manifest = environment_manifest()
    if any(v == "unknown" for v in manifest.values()):
        # An environment that cannot identify itself shares nothing: two
        # incompatible stacks with the same unknowns would otherwise hash alike.
        _refuse_compile(symbol)
        _count("bypassed")
        return cute.compile(fn, *args, **kwargs)
    root = get_cache_dir()  # one root for the export and the prune that follows it
    entry = _entry_dir(root, manifest, f"{cache_key}|{symbol}|{options}")
    loaded = _try_load(entry, cache_key, symbol)
    if loaded is not None:
        _count("hits")
        return loaded
    with _entry_lock(root, entry):
        # Another process may have published this exact artifact while we
        # waited. Warm loads above need no lock, and different keys can compile
        # concurrently. Keep the lock through export and reload.
        loaded = _try_load(entry, cache_key, symbol)
        if loaded is not None:
            _count("hits")
            return loaded
        _refuse_compile(symbol)
        return _compile_and_export(fn, args, kwargs, root, entry, cache_key, symbol, manifest)


@contextmanager
def _entry_lock(root: Path, entry: Path):
    """Best-effort process lock, released by the OS if the compiler dies.

    Lock files live outside the prunable environment directories and are never
    unlinked: replacing an inode while a waiter holds it would create two locks
    for the same artifact. They contain no kernel data.
    """
    lock_file = None
    try:
        import fcntl

        lock_dir = root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = (lock_dir / f"{entry.parent.name}-{entry.name}.lock").open("a+b")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    except (ImportError, OSError) as exc:
        if lock_file is not None:
            lock_file.close()
            lock_file = None
        # Persistence has always been optional; an unavailable lock must not
        # prevent an otherwise valid compilation, or hide a compiler failure.
        _LOG.warning("compiled-plan cache: cannot lock %s (%s); compiling without a process lock", entry, exc)
    except BaseException:
        if lock_file is not None:
            lock_file.close()
        raise
    try:
        yield
    finally:
        if lock_file is not None:
            lock_file.close()


def _compile_and_export(fn, args, kwargs, root, entry, cache_key, symbol, manifest):
    """Compile a miss and return its reloaded artifact when export is supported."""
    import cutlass.cute as cute

    compiled = cute.compile(fn, *args, **kwargs)
    _count("misses")
    execution_args = getattr(compiled, "execution_args", None)
    if getattr(execution_args, "has_pointer_address_arg_specs", False) or not hasattr(compiled, "export_to_c"):
        return compiled  # the in-process object converts raw pointers; a reloaded one could not
    wrapper = _wrapper_spec_of(compiled, args, kwargs)
    if wrapper is None:
        return compiled  # a calling convention the record cannot reproduce exactly
    try:
        _export(entry, compiled, cache_key, symbol, manifest, wrapper)
    except Exception as exc:  # noqa: BLE001 -- persistence is best-effort; the kernel is compiled either way
        _count("export_failed")
        _LOG.warning("compiled-plan cache: could not persist %s (%s); the kernel will be recompiled next process", entry, exc)
        return compiled
    _prune_once(root, entry.parent)  # this process's first write: retire dead environments
    # Hand back the artifact rather than the in-process object, so a hit and a
    # miss run the same thing and a bad artifact fails here, not next start-up.
    loaded = _try_load(entry, cache_key, symbol)
    return loaded if loaded is not None else compiled


# ---------------------------------------------------------------------------
# Carrying entries to another process (plan serialization)
# ---------------------------------------------------------------------------


class CompileRefused(RuntimeError):
    """A kernel had to compile inside :func:`refuse_compiles`."""


def _refuse_compile(symbol: str) -> None:
    if getattr(_REFUSING, "depth", 0):
        raise CompileRefused(f"{symbol} is not in the compiled cache and compiling is refused here")


@contextlib.contextmanager
def refuse_compiles():
    """Make a ``compile_cached`` miss in this thread raise :class:`CompileRefused` instead of compiling."""
    _REFUSING.depth = getattr(_REFUSING, "depth", 0) + 1
    try:
        yield
    finally:
        _REFUSING.depth -= 1


@contextlib.contextmanager
def record_entries():
    """Collect the on-disk entry of every ``compile_cached`` call this thread makes inside the block.

    Yields a list of entry directories, filled as kernels are compiled, reloaded or served from the
    memo. A kernel that was not persisted (uncacheable, cache disabled, export failed) leaves nothing.
    """
    stack = getattr(_RECORDING, "stack", None)
    if stack is None:
        stack = _RECORDING.stack = []
    entries: list = []
    stack.append(entries)
    try:
        yield entries
    finally:
        stack.pop()


def _record(cache_key: str, symbol: str, options: str) -> None:
    stack = getattr(_RECORDING, "stack", None)
    if not stack or not enabled() or "--enable-tvm-ffi" not in options:
        return
    manifest = environment_manifest()
    if any(v == "unknown" for v in manifest.values()):
        return
    entry = _entry_dir(get_cache_dir(), manifest, f"{cache_key}|{symbol}|{options}")
    if (entry / _ENTRY).is_file() and (entry / _OBJECT).is_file() and entry not in stack[-1]:
        stack[-1].append(entry)


def read_entry(entry: Path) -> Optional[Tuple[Dict[str, Any], bytes]]:
    """``(description, object bytes)`` of a persisted entry, or None if it is gone or unreadable.

    The description carries the environment manifest the entry was compiled under, which
    :func:`seed_entry` compares against its own before installing anything."""
    manifest = _read_json(entry.parent / _MANIFEST)
    record = _read_json(entry / _ENTRY)
    try:
        data = (entry / _OBJECT).read_bytes()
    except OSError:
        return None
    if not isinstance(manifest, dict) or not isinstance(record, dict):
        return None
    return {"environment": manifest, "entry": entry.name, "record": record}, data


def seed_entry(description: Dict[str, Any], data: bytes, device: Optional[int] = None) -> bool:
    """Install an entry produced by :func:`read_entry` in another process; True when it is (or already was) present.

    Installed only when its environment manifest equals this process's exactly -- the same rule that
    keeps a stale artifact out of the cache, so a seeded entry is never one this process could not
    have compiled itself. ``device`` is the one the plan will build on (default: the current device). Anything else is refused and the kernel compiles as usual. The entry is
    written like an export (object first, record last), and loading it still goes through the key
    check in ``compile_cached``.
    """
    if not enabled():
        return False
    manifest = environment_manifest(device)
    record = description.get("record")
    name = str(description.get("entry", ""))
    if (
        description.get("environment") != manifest
        or any(v == "unknown" for v in manifest.values())
        or not _DIGEST_DIR.fullmatch(name)
        or not isinstance(record, dict)
        or record.get("schema") != _SCHEMA
    ):
        _count("seed_rejected")
        return False
    entry = get_cache_dir() / _SCHEMA / _digest(_canonical(manifest)) / name
    if (entry / _ENTRY).is_file() and (entry / _OBJECT).is_file():
        return True
    entry.mkdir(parents=True, exist_ok=True)
    manifest_path = entry.parent / _MANIFEST
    if not manifest_path.exists():
        _write_atomic(manifest_path, json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8"))
    _write_atomic(entry / _OBJECT, data)
    _write_atomic(entry / _ENTRY, json.dumps(record, indent=1, sort_keys=True).encode("utf-8"))
    _count("seeded")
    return True
