"""Opt-in worker processes that build cuDNN execution plans in parallel.

cuDNN plan building holds the GIL and cuDNN refuses its own multithreaded
builds, so autotuning builds every candidate plan of a graph one at a time
(flashinfer-ai/flashinfer#9409). With ``FLASHINFER_CUDNN_PLAN_BUILD_WORKERS=N``
the candidates of one graph are split across N worker processes, each of which
loads ``_cudnn_plan_build.py`` without torch, builds its share, and returns
serialized single-plan graphs.

Any failure (spawn, crash, timeout, error, or workers disagreeing on the
candidate list) returns ``None`` so the caller builds in-process as before, and
turns the pool off for the rest of the process.
"""

import atexit
import contextlib
import logging
import os
import pathlib
import pickle
import selectors
import subprocess
import sys
import threading
import time

from ._cudnn_plan_build import (
    _FRAME_HEADER,
    Bf16OverrideGraphSpec,
    spec_to_wire,
    write_frame,
)

logger = logging.getLogger(__name__)

ENV_VAR = "FLASHINFER_CUDNN_PLAN_BUILD_WORKERS"
_WORKER_SCRIPT = pathlib.Path(__file__).with_name("_cudnn_plan_build.py")
# Libraries the worker's cuDNN must resolve the same way the parent's did.
_LIB_PREFIXES = ("libcudnn", "libcublas", "libcudart", "libnvrtc", "libnvJitLink")


class _WorkerFailure(Exception):
    pass


def _loaded_library_dirs() -> list:
    dirs = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.rsplit(None, 1)[-1]
                if not path.startswith("/"):
                    continue
                name = os.path.basename(path)
                directory = os.path.dirname(path)
                if name.startswith(_LIB_PREFIXES) and directory not in dirs:
                    dirs.append(directory)
    except OSError:
        pass
    return dirs


def _visible_device_entry(device_index: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(device_index)
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if device_index >= len(entries):
        raise _WorkerFailure(
            f"device {device_index} is not in CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return entries[device_index]


def _read_frame(proc: subprocess.Popen, deadline: float):
    """Read one frame from ``proc.stdout`` before ``deadline`` (monotonic)."""
    fd = proc.stdout.fileno()
    buf = b""
    need = _FRAME_HEADER.size
    size = None
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            if len(buf) >= need:
                if size is None:
                    (size,) = _FRAME_HEADER.unpack(buf[:need])
                    need += size
                    continue
                return pickle.loads(buf[_FRAME_HEADER.size : need])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _WorkerFailure("timed out")
            if not selector.select(remaining):
                raise _WorkerFailure("timed out")
            chunk = os.read(fd, max(need - len(buf), 65536))
            if not chunk:
                raise _WorkerFailure(f"worker exited (code {proc.poll()})")
            buf += chunk


class CudnnPlanBuildPool:
    """A fixed number of plan-build workers per CUDA device, started lazily."""

    def __init__(
        self,
        num_workers: int,
        timeout_s: float = 120.0,
        worker_cmd=None,
        idle_s: float = 30.0,
    ):
        self.num_workers = num_workers
        self.timeout_s = timeout_s
        self._worker_cmd = worker_cmd or [sys.executable, "-I", str(_WORKER_SCRIPT)]
        self._workers: dict = {}  # device_index -> list[Popen]
        self._lock = threading.Lock()
        self._broken = False
        self.last_responses: list = []
        # Each worker holds ~800 MB and a CUDA context; tuning comes in bursts,
        # so release them once idle and respawn on the next build.
        self.idle_s = idle_s
        self._idle_timer: threading.Timer | None = None
        self._last_use = 0.0

    @property
    def broken(self) -> bool:
        return self._broken

    def _all_processes(self) -> list:
        return [p for procs in self._workers.values() for p in procs]

    def _spawn(self, device_index: int) -> list:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = _visible_device_entry(device_index)
        lib_dirs = _loaded_library_dirs()
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                lib_dirs + [p for p in [env.get("LD_LIBRARY_PATH")] if p]
            )
        return [
            subprocess.Popen(
                self._worker_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            for _ in range(self.num_workers)
        ]

    def build(self, spec: Bf16OverrideGraphSpec, device_index: int, heur_modes):
        """Return ``[(tactic, serialized_plan), ...]`` in candidate order, or ``None``."""
        with self._lock:
            if self._broken:
                return None
            try:
                result = self._build(spec, device_index, list(heur_modes))
            except Exception as exc:  # noqa: BLE001 - never fail a tuning run
                self._fail(exc)
                return None
            self._last_use = time.monotonic()
            self._schedule_idle_stop()
            return result

    def disable(self, reason) -> None:
        """Stop the workers and build in-process from now on (e.g. a result failed to load)."""
        with self._lock:
            if not self._broken:
                self._fail(reason)

    def _schedule_idle_stop(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(self.idle_s, self._idle_stop)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_stop(self) -> None:
        with self._lock:
            if self._broken or time.monotonic() - self._last_use < self.idle_s:
                return
            self._stop(kill=False)
            self._workers.clear()

    def _build(self, spec, device_index, heur_modes):
        procs = self._workers.get(device_index)
        if procs is None:
            procs = self._workers[device_index] = self._spawn(device_index)
        wire = spec_to_wire(spec)
        for shard_index, proc in enumerate(procs):
            write_frame(
                proc.stdin,
                {
                    "device_index": device_index,
                    "spec": wire,
                    "heur_modes": heur_modes,
                    "shard": (shard_index, len(procs)),
                },
            )
        deadline = time.monotonic() + self.timeout_s
        responses = [_read_frame(proc, deadline) for proc in procs]
        self.last_responses = responses

        candidates = responses[0]["candidates"]
        by_index = {}
        for response in responses:
            if response.get("error"):
                raise _WorkerFailure(response["error"])
            if response["device_index"] != device_index:
                raise _WorkerFailure("worker answered for another device")
            if response["candidates"] != candidates:
                raise _WorkerFailure("workers disagree on the candidate plans")
            for index, tactic, blob in response["plans"]:
                if not 0 <= index < len(candidates) or candidates[index] != tactic:
                    raise _WorkerFailure("plan does not match its candidate")
                by_index[index] = (tactic, blob)
        # A missing plan would silently narrow the tuned tactic set; the
        # in-process build is the reference, so fall back to it instead.
        if len(by_index) != len(candidates):
            raise _WorkerFailure(
                f"built {len(by_index)} of {len(candidates)} candidate plans"
            )
        return [by_index[index] for index in sorted(by_index)]

    def _fail(self, exc) -> None:
        self._broken = True
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        logger.warning(
            "[cuDNN plan workers]: %s; building cuDNN plans in-process for the "
            "rest of this process",
            exc,
        )
        self._stop(kill=True)

    def _stop(self, kill: bool) -> None:
        for proc in self._all_processes():
            try:
                if kill:
                    proc.kill()
                else:
                    proc.stdin.close()
            except OSError:
                pass
        for proc in self._all_processes():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            for stream in (proc.stdin, proc.stdout):
                with contextlib.suppress(OSError):
                    stream.close()

    def shutdown(self) -> None:
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._stop(kill=False)
            self._workers.clear()


_pool = None
_pool_resolved = False


def get_cudnn_plan_build_pool():
    """The process-wide pool, or ``None`` when ``FLASHINFER_CUDNN_PLAN_BUILD_WORKERS`` is 0/unset."""
    global _pool, _pool_resolved
    if _pool_resolved:
        return _pool
    _pool_resolved = True
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        workers = int(raw)
    except ValueError:
        workers = -1
    if workers < 0:
        logger.warning("Ignoring %s=%r: expected a non-negative integer", ENV_VAR, raw)
        return None
    if workers == 0:
        return None
    cpus = os.cpu_count()
    if cpus and workers > cpus:
        # Workers are per tuned device and ~800 MB each; more than the CPUs
        # cannot build faster.
        logger.warning("Capping %s=%d at %d (CPU count)", ENV_VAR, workers, cpus)
        workers = cpus
    _pool = CudnnPlanBuildPool(workers)
    atexit.register(_pool.shutdown)
    return _pool
