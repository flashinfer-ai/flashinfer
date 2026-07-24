"""Managed on-disk autotune cache (v2, experimental).

This module owns *persistence only*: given an already-canonical lookup key
(the same ``str((custom_op, runner_class_name, profile, extras))`` string the
:class:`~flashinfer.autotuner.AutoTuner` uses for its JSON config files), it
stores one winner per key as one small JSON file under an
environment-hashed directory:

.. code-block:: text

    <root>/v2/<environment_hash>/
        manifest.json               # canonical environment, human-readable
        entries/<operation_hash>.json

Design rules (kept deliberately minimal for the first version):

* The environment hash isolates incompatible environments (FlashInfer,
  CUDA, cuBLAS, cuDNN backend/frontend versions, GPU). A version bump
  simply produces a new directory; old directories are left untouched.
* One entry file per tuned operation, published with a same-directory
  temporary file + atomic ``os.replace``. Concurrent writers may do
  redundant work; the last valid atomic write wins. No locks.
* A missing, malformed, or key-mismatched entry is a cache miss, never an
  error. Cache failures must not affect correctness.
* This module does not interpret tactics. Callers pass JSON-safe tactic
  values in and get them back out unchanged
  (see ``flashinfer.autotuner._tactic_to_json`` / ``_json_to_tactic``).

The cache root defaults to ``FLASHINFER_CACHE_DIR/autotune`` and can be
overridden with the ``FLASHINFER_AUTOTUNE_CACHE_DIR`` environment variable
or ``autotune_v2(root=...)`` (always a directory, never a schema-bearing
filename: the location is placement only, so no choice of root can mix
incompatible entries).
"""

import contextlib
import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any, Dict, Optional, Set, Tuple, Union

from .jit.core import logger

# Bump the schema directory (v2 -> v3) on any incompatible change to the
# directory layout or entry format; do not add per-entry format versions.
_SCHEMA = "v2"


@dataclasses.dataclass(frozen=True)
class MeasurementPolicy:
    """How tuning measurements are taken, matched to the deployment.

    The one decision is whether per-call HOST cost counts: under
    CUDA-graph serving it is paid once at capture (excluding it is
    correct); under eager serving it is paid on every call, and excluding
    it mis-ranks host-heavy candidates (a measured cuDNN GEMM reads ~8 us
    host-excluded but ~330 us host-included).  ``execution_mode`` states
    the deployment; capture behavior and timing implementation are
    derived from it.

    The policy is part of the store's environment identity (manifest):
    entries tuned under different policies never overwrite each other,
    and a consuming process must attach with the same policy
    (``autotune_v2(mode="replay", measure=...)``).

    Attributes:
        execution_mode:
            * ``"cuda_graph"`` — profile under real graph capture +
              replay, host cost excluded.  Capture-unsafe tactics fail
              profiling and are skipped.
            * ``"eager"`` — profile eagerly with per-call host cost
              included (no delay kernel).
            * ``"auto"`` (default) — the library decides; currently the
              legacy behavior (capture inherited from each op's
              ``TuningConfig.use_cuda_graph``, host-excluded event
              timing).  Frameworks that know their serving mode should
              pass it explicitly.
        cold_l2: Force cold-/hot-L2 profiling; ``None`` (default)
            inherits each op's ``TuningConfig.use_cold_l2_cache``.
            Orthogonal to ``execution_mode``.
    """

    execution_mode: str = "auto"
    cold_l2: Optional[bool] = None
    # Internal/diagnostic override pinning the timing implementation
    # ("events" | "events_no_delay" | "cupti"); "auto" derives it from
    # execution_mode.  Used by the accuracy harness to A/B
    # implementations; explicit values are identity-bearing (manifest) so
    # diagnostic runs never overwrite default entries.
    _timer: str = "auto"

    def __post_init__(self):
        if self.execution_mode not in ("auto", "cuda_graph", "eager"):
            raise ValueError(
                f"MeasurementPolicy.execution_mode must be 'cuda_graph', "
                f"'eager', or 'auto', got {self.execution_mode!r}"
            )
        if self._timer not in ("auto", "cupti", "events", "events_no_delay"):
            raise ValueError(
                f"MeasurementPolicy._timer must be 'auto', 'cupti', 'events', "
                f"or 'events_no_delay', got {self._timer!r}"
            )
        if self.execution_mode == "eager" and self._timer == "cupti":
            raise ValueError(
                "MeasurementPolicy(execution_mode='eager') rejects "
                "_timer='cupti': CUPTI spans exclude per-call host cost, "
                "which eager serving pays"
            )

    @property
    def cuda_graph(self) -> Optional[bool]:
        """Derived from ``execution_mode``; ``None`` (mode ``"auto"``)
        inherits each op's ``TuningConfig.use_cuda_graph``."""
        if self.execution_mode == "cuda_graph":
            return True
        if self.execution_mode == "eager":
            return False
        return None

    @property
    def timer(self) -> str:
        """Effective timing implementation: ``eager`` -> events_no_delay
        (host cost measured); otherwise events, unless pinned by
        ``_timer``."""
        if self._timer != "auto":
            return self._timer
        if self.execution_mode == "eager":
            return "events_no_delay"
        return "events"

    def manifest_fields(self) -> Dict[str, str]:
        """Manifest entries for the non-inherited fields (empty if all default)."""
        fields: Dict[str, str] = {}
        if self.execution_mode != "auto":
            fields["measure_execution_mode"] = self.execution_mode
        if self.cold_l2 is not None:
            fields["measure_cold_l2"] = str(self.cold_l2)
        if self._timer != "auto":
            fields["measure_timer"] = self._timer
        return fields


def get_default_cache_root() -> pathlib.Path:
    """Return the managed autotune cache root (env override or default)."""
    override = os.getenv("FLASHINFER_AUTOTUNE_CACHE_DIR")
    if override:
        return pathlib.Path(override)
    from .jit import env as jit_env

    return jit_env.FLASHINFER_CACHE_DIR / "autotune"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _atomic_write_json(path: pathlib.Path, obj: Any) -> None:
    """Write JSON to *path* via a same-directory temp file + atomic rename."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".autotune_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


class ManagedAutotuneCache:
    """Per-entry, environment-hashed persistence for autotune winners.

    Activated by :func:`autotune_v2`. The
    :class:`~flashinfer.autotuner.AutoTuner` consults :meth:`lookup` on an
    in-memory cache miss and calls :meth:`publish` for each newly tuned
    winner; both are best-effort and never raise.
    """

    def __init__(
        self,
        manifest: Dict[str, str],
        root: Union[str, os.PathLike, None] = None,
    ):
        self.manifest = dict(manifest)
        self.manifest["cache_schema"] = _SCHEMA
        self.env_hash = hashlib.sha256(
            _canonical_json(self.manifest).encode()
        ).hexdigest()[:16]
        root = pathlib.Path(root) if root is not None else get_default_cache_root()
        self.root = root
        self.env_dir = root / _SCHEMA / self.env_hash
        self.entries_dir = self.env_dir / "entries"
        # Positive/negative lookup memos: at most one filesystem probe per
        # key per process.  A concurrent process's publish becomes visible
        # on the next attach.
        self._hits: Dict[str, Tuple[str, Any]] = {}
        self._missing: Set[str] = set()

    def _entry_path(self, file_key: str) -> pathlib.Path:
        op_hash = hashlib.sha256(file_key.encode()).hexdigest()[:24]
        return self.entries_dir / f"{op_hash}.json"

    def _ensure_dirs(self) -> None:
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.env_dir / "manifest.json"
        # Content-addressed by env_hash: an existing manifest is identical.
        if not manifest_path.exists():
            _atomic_write_json(manifest_path, self.manifest)

    def lookup(self, file_key: str) -> Optional[Tuple[str, Any]]:
        """Return ``(runner_class_name, json_tactic)`` for *file_key*, or None.

        Any failure (missing file, malformed JSON, embedded-key mismatch)
        is a cache miss.
        """
        hit = self._hits.get(file_key)
        if hit is not None:
            return hit
        if file_key in self._missing:
            return None
        path = self._entry_path(file_key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            # Guard against hash collisions and foreign/corrupt files: the
            # entry must embed the exact canonical key it was stored under.
            if entry["key"] != file_key:
                raise ValueError(f"embedded key mismatch (found {entry['key']!r})")
            hit = (entry["runner"], entry["tactic"])
            self._hits[file_key] = hit
            return hit
        except FileNotFoundError:
            self._missing.add(file_key)
            return None
        except Exception as e:
            logger.warning(
                f"[Autotuner]: Ignoring invalid managed cache entry {path} "
                f"for key {file_key}: {e}. Treating as a cache miss."
            )
            self._missing.add(file_key)
            return None

    def publish(self, file_key: str, runner_name: str, json_tactic: Any) -> None:
        """Atomically persist the tuned winner for *file_key* (best-effort)."""
        try:
            self._ensure_dirs()
            entry = {
                "key": file_key,
                "runner": runner_name,
                "tactic": json_tactic,
            }
            _atomic_write_json(self._entry_path(file_key), entry)
            self._hits[file_key] = (runner_name, json_tactic)
            self._missing.discard(file_key)
        except Exception as e:
            logger.warning(
                f"[Autotuner]: Failed to persist managed cache entry for "
                f"{file_key} under {self.entries_dir}: {e}. Tuning result "
                f"remains available in memory for this process."
            )

    def clear_memo(self) -> None:
        """Forget memoized lookups (e.g. after AutoTuner.clear_cache)."""
        self._hits.clear()
        self._missing.clear()

    def __repr__(self) -> str:
        return (
            f"ManagedAutotuneCache(env_dir={str(self.env_dir)!r}, "
            f"hits={len(self._hits)}, misses={len(self._missing)})"
        )


def _resolve_managed_store(tuner, cache_root, manifest: Dict[str, str]):
    """Return the managed store for ``(cache_root, manifest)`` and set it as
    the process ambient (last-wins).

    A persistent autotune_v2 context always attaches its store as the ambient
    the moment it opens, so bare, context-free serving after the context
    exits resolves to it.  autotune_v2 does not nest, so there is no scoped
    variant to preserve an outer ambient for; sequential contexts are simply
    last-wins.  The store object (and its memos) is reused across identity
    switches via the registry.
    """
    resolved = (
        pathlib.Path(cache_root) if cache_root is not None else get_default_cache_root()
    )
    candidate = dict(manifest)
    candidate["cache_schema"] = _SCHEMA
    reg_key = (str(resolved), _canonical_json(candidate))
    with tuner._lock:
        # Reuse the store object (and its hit/miss memos) for a given
        # identity instead of rebuilding it on every switch.
        store = tuner._managed_stores.get(reg_key)
        if store is None:
            store = ManagedAutotuneCache(manifest=manifest, root=resolved)
            tuner._managed_stores[reg_key] = store
        if store is not tuner._managed_cache:
            if tuner._managed_cache is not None:
                logger.info(
                    f"[Autotuner]: Re-attaching managed autotune cache at "
                    f"{resolved} (was {tuner._managed_cache.root}, "
                    f"env {tuner._managed_cache.env_hash})."
                )
            tuner._managed_cache = store
            # Deliberately NOT clearing _managed_decoded: it is keyed by
            # (root, env_hash, file_key), so entries decoded under other
            # identities coexist safely and are never served cross-identity.
            # Clearing here would force alternating graph/eager regions to
            # re-read disk on every switch, breaking the "post-hydrate
            # lookups are pure memory" contract.  Invalidation is explicit
            # (autotune_v2_reload / clear_cache).
            logger.info(
                f"[Autotuner]: Managed autotune cache attached for this "
                f"process: {store!r}"
            )
        return store


def autotune_v2_reload() -> None:
    """Rank-consistency finalize step: drop in-process winners so later
    lookups re-read the (shared) managed store.

    Homogeneous ranks tuning concurrently into one store may each hold a
    locally-measured winner in memory that differs from the store's final
    (last valid publish wins) entry.  Calling this on every rank AFTER a
    barrier that follows tuning makes all ranks serve byte-identical
    tactics — the divergence class behind NCCL symmetric-memory deadlocks
    (issue #3186; composes with ``set_autotune_process_group``, which fixes
    the in-session window).

    Only meaningful when tuning published to a store: winners that were
    never published (e.g. ``persistent_cache=False`` segments) fall back to
    heuristics after the reload.
    """
    from .autotuner import AutoTuner

    tuner = AutoTuner.get()
    with tuner._lock:
        tuner.profiling_cache.clear()
        tuner._winner_partitions.clear()
        tuner._managed_decoded.clear()
        store = tuner._managed_cache
        if store is not None:
            store.clear_memo()
    logger.info(
        "[Autotuner]: autotune_v2_reload(): in-memory winners dropped; "
        "lookups now re-read the managed store."
    )


@contextlib.contextmanager
def autotune_v2(
    mode: str = "tune",
    persistent_cache: bool = True,
    cache_root: Union[str, os.PathLike, None] = None,
    tuning_buckets: Optional[Tuple[int, ...]] = None,
    round_up: Optional[bool] = None,
    skip_ops: Optional[Union[str, Set[str]]] = None,
    measure: Optional[MeasurementPolicy] = None,
):
    """Autotune with FlashInfer-managed on-disk persistence (experimental v2).

    A standalone entry point, deliberately disjoint from
    :func:`flashinfer.autotune`'s legacy ``cache=<json path>`` persistence so
    the v2 design can iterate without inheriting v1 semantics.  Tuning
    behavior (runner registration, bucketing, profiling, the in-memory
    ``profiling_cache``) is shared with :func:`flashinfer.autotune`; only
    persistence differs.

    ``mode`` names what the context does (a positive action, not a
    negated flag); ``persistent_cache`` is the orthogonal disk toggle:

    ===============  =====================  ==================================
    ``mode``         ``persistent_cache``   meaning
    ===============  =====================  ==================================
    ``"tune"``       True (default)         tune misses, publish to disk
    ``"tune"``       False                  tune in-memory only (no disk I/O)
    ``"replay"``     True                   serve from the on-disk cache
                                            (no profiling)
    ``"replay"``     False                  memory-only replay (see note)
    ===============  =====================  ==================================

    ``mode="replay"`` is the serving path: it replays previously tuned
    winners without paying any profiling cost (it does **not** discard
    tuning results — that is what the negated ``enable_tuning=False`` it
    replaced read like).  With ``persistent_cache=False`` it is
    *memory-only replay*: ``persistent_cache=False`` forbids disk for the
    context, so an on-disk store hydrated earlier is **not** consulted; it
    serves only what is already in the in-memory winner cache (e.g. tuned
    in-memory earlier this process).

    **Attach semantics**: ``persistent_cache=True`` attaches the managed
    store for the remainder of the process (like v1's ``load_configs``),
    so inference *after* the context exits keeps reusing on-disk entries.
    Publishing only happens while tuning inside a context.  See the module
    docstring for the store's layout and failure rules.

    Args:
        mode: ``"tune"`` (default) profiles uncovered shapes and publishes
            winners; ``"replay"`` only serves already-tuned winners (no
            profiling).
        persistent_cache: If ``True`` (default), attach the managed
            on-disk store (read + publish).  If ``False``, do not touch
            the filesystem -- for environments where disk access is
            unwanted.  Note: a store attached earlier in the process
            remains attached (detach is not supported yet); a process
            that never wants disk I/O should always pass ``False``.
        cache_root: Optional cache root **directory** (placement only --
            the schema and environment namespaces live below it, so no
            choice of root can mix incompatible entries).  Defaults to
            ``FLASHINFER_CACHE_DIR/autotune``, overridable with the
            ``FLASHINFER_AUTOTUNE_CACHE_DIR`` environment variable.
        tuning_buckets: Forwarded to :func:`flashinfer.autotune`.
        round_up: Forwarded to :func:`flashinfer.autotune`.  Context-
            scoped: post-context lookups use each op's default bucket
            mapper, so leave ``None`` unless tuning and serving share one
            context.
        skip_ops: Forwarded to :func:`flashinfer.autotune`; quarantines
            specific ``custom_op`` names (e.g. an op whose tuning
            crashes) without losing autotune for everything else.
        measure: Optional :class:`MeasurementPolicy` -- measure tactics
            the way the deployment runs:
            ``MeasurementPolicy(execution_mode="cuda_graph")`` or
            ``MeasurementPolicy(execution_mode="eager")``.  Part of the
            store identity; consumers must attach with the same policy.

    Examples::

        # Tune once; entries are published as they are tuned.
        with autotune_v2():
            model(inputs)

        # Any later process in the same environment: hydrate + serve.
        with autotune_v2(mode="replay"):
            pass
        model(inputs)  # reuses tuned winners, no context needed
    """
    from .autotuner import AutoTuner, _collect_metadata, autotune

    if mode not in ("tune", "replay"):
        raise ValueError(
            f"mode must be 'tune' or 'replay', got {mode!r}; use 'replay' to "
            f"serve already-tuned winners without profiling."
        )
    if not isinstance(persistent_cache, bool):
        # Catch autotune_v2(persistent_cache="/some/path") early: any truthy
        # object would silently enable persistence at the DEFAULT root.
        raise TypeError(
            f"persistent_cache must be a bool, got "
            f"{type(persistent_cache).__name__!r}; to place the store, pass "
            f"cache_root=<directory> instead."
        )
    enable_tuning = mode == "tune"

    tuner = AutoTuner.get()
    # autotune_v2 does not nest: a v2-inside-v2 context would have ambiguous
    # store/policy targeting (which store does the outer's later work publish
    # to?).  Fail fast instead of silently last-wins.  Nesting a plain v1
    # autotune() inside or around autotune_v2 is fine and stays supported.
    if tuner._v2_local.active:
        raise RuntimeError(
            "nested autotune_v2 contexts are unsupported: an autotune_v2 "
            "context is already open on this thread.  Use one context per "
            "warmup/serving region; nesting a plain autotune() is fine."
        )
    # Enter the delegated context first: if its argument validation fails,
    # this context has no lasting side effects (no attach, no policy set).
    with autotune(
        enable_tuning,
        tuning_buckets=tuning_buckets,
        round_up=round_up,
        skip_ops=skip_ops,
    ):
        if persistent_cache:
            manifest = _collect_metadata()
            if measure is not None:
                # The policy is part of the environment identity.
                manifest.update(measure.manifest_fields())
            # A persistent context always attaches its store as the process
            # ambient (last-wins), so bare serving after it exits resolves
            # to it.
            store = _resolve_managed_store(tuner, cache_root, manifest)
        else:
            # Disk forbidden for this context: lookups and publishes skip the
            # managed store even if one is attached as ambient (the ambient
            # remains for bare serving outside this context).
            store = None
            if tuner._managed_cache is not None:
                logger.info(
                    "[Autotuner]: autotune_v2(persistent_cache=False): disk "
                    "access disabled inside this context; the previously "
                    "attached store remains active for serving outside it."
                )
        local = tuner._v2_local
        local.active = True
        local.store = store
        local.measure = measure
        try:
            yield
        finally:
            local.active = False
            local.store = None
            local.measure = None
