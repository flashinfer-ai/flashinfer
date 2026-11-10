# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Parallel compilation of CuTe DSL kernels into the disk cache.

Autotuning compiles every candidate of a bucket before profiling it, and
``cute.compile`` cannot run on two threads of one process (its MLIR context
is thread-local). A ``CompilePool`` keeps a few worker subprocesses that
compile the uncached candidates in parallel and export them to the disk
cache; the caller then loads them from there. The workers never touch the
GPU (the compile target comes from ``CUTE_DSL_ARCH``) and exit once the pool
has been idle for ``IDLE_SECONDS``. A worker that does not answer a job
within ``JOB_TIMEOUT_SECONDS`` is killed, and the caller compiles those
kernels in process.

Worker protocol: one JSON job per stdin line, answered by one stdout line:
``null`` or the error text. The worker module calls ``serve(kernel_spec)``,
where ``kernel_spec(*job)`` returns ``(module, name, compile_fn, key_files)``.
"""

import contextlib
import json
import os
import queue
import select
import subprocess
import sys
import threading
from concurrent.futures import Future

from .core import logger

# The autotuner prepares the buckets of a shape back to back; keep the
# workers (and their imports) alive across them.
IDLE_SECONDS = 60.0
# One compile takes seconds; a worker silent for this long is hung.
JOB_TIMEOUT_SECONDS = 300.0


def workers_from_env(value):
    """``value`` as a worker count, else half the usable CPUs up to 8."""
    if value is not None:
        return max(1, int(value))
    return max(1, min(8, len(os.sched_getaffinity(0)) // 2))


class CompilePool:
    """Worker subprocesses running ``python -m worker_module``.

    ``num_workers()`` is read when the workers start; 1 means the caller
    compiles in process.
    """

    def __init__(self, worker_module, num_workers, label):
        self.argv = (sys.executable, "-m", worker_module)
        self.num_workers = num_workers
        self.label = label
        self._workers = None
        self._disabled = False
        self._lock = threading.Lock()

    def submit(self, jobs, arch, start=True):
        """Queue ``jobs`` on the worker subprocesses, in order.

        Returns one ``Future`` per job whose result is ``None`` or the error
        text, or ``None`` when no workers are running and ``start`` is false
        or the workers cannot be started.
        """
        with self._lock:
            if self._disabled or (self._workers is None and not start):
                return None
            if self._workers is None:
                try:
                    self._workers = _Workers(self, arch)
                except OSError as e:
                    logger.warning(
                        f"{self.label}: no compile workers ({e}); compiling serially"
                    )
                    self._disabled = True
                    return None
            workers = self._workers
            if workers.timer is not None:
                workers.timer.cancel()
                workers.timer = None
            workers.pending += len(jobs)
            futures = []
            for job in jobs:
                fut: Future = Future()
                workers.jobs.put((job, fut))
                futures.append(fut)
        return futures

    def shutdown(self):
        """Stop the workers now, dropping queued jobs."""
        with self._lock:
            workers, self._workers = self._workers, None
            if workers is not None:
                if workers.timer is not None:
                    workers.timer.cancel()
                workers.close()

    def _release(self, workers):
        with self._lock:
            if self._workers is workers and workers.pending == 0:
                self._workers = None
                workers.close()


class _Workers:
    def __init__(self, pool, arch):
        env = dict(os.environ, CUTE_DSL_ARCH=arch)
        self.procs = [
            subprocess.Popen(
                pool.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
            )
            for _ in range(pool.num_workers())
        ]
        self.pool = pool
        self.jobs = queue.Queue()
        self.pending = 0
        self.timer = None
        for proc in self.procs:
            threading.Thread(target=self._serve, args=(proc,), daemon=True).start()

    def _serve(self, proc):
        alive = True
        while True:
            job, fut = self.jobs.get()
            if job is None:
                return
            err = "compile worker exited"
            if alive:
                try:
                    proc.stdin.write(json.dumps(job) + "\n")
                    if not select.select([proc.stdout], [], [], JOB_TIMEOUT_SECONDS)[0]:
                        proc.kill()
                        raise TimeoutError(f"no reply in {JOB_TIMEOUT_SECONDS:.0f} s")
                    reply = proc.stdout.readline()
                    alive = bool(reply)
                    if alive:
                        err = json.loads(reply)
                except Exception as e:  # noqa: BLE001 -- reported to the caller
                    alive = False
                    err = f"{type(e).__name__}: {e}"
            with self.pool._lock:
                self.pending -= 1
                if self.pending == 0:
                    self.timer = threading.Timer(
                        IDLE_SECONDS, self.pool._release, (self,)
                    )
                    self.timer.daemon = True
                    self.timer.start()
            fut.set_result(err)

    def close(self):
        """Drop the queued jobs and let the workers exit after their current one."""
        while True:
            try:
                job, fut = self.jobs.get_nowait()
            except queue.Empty:
                break
            self.pending -= 1
            fut.set_result("compile pool closed")
        for _ in self.procs:
            self.jobs.put((None, None))
        for proc in self.procs:
            with contextlib.suppress(OSError):
                proc.stdin.close()


def serve(kernel_spec):
    """Worker main loop: compile each job's kernel into the disk cache."""
    # Replies own the real stdout; anything else printed goes to stderr.
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    from .cute_dsl_core import JitSpecCuteDsl, _hash_source_files

    for line in sys.stdin:
        try:
            module, name, compile_fn, key_files = kernel_spec(*json.loads(line))
            spec = JitSpecCuteDsl(
                module, name, compile_fn, _hash_source_files(key_files)
            )
            if not spec.is_compiled:
                spec.compile_and_persist()
            err = None
        except Exception as e:  # noqa: BLE001 -- reported to the parent
            err = f"{type(e).__name__}: {e}"
        replies.write(json.dumps(err) + "\n")
