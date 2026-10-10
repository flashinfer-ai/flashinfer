# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Parallel compilation of SM12x MXFP8 tactics into the CuTe DSL disk cache.

Autotuning compiles every candidate of a bucket before profiling it, and
``cute.compile`` cannot run on two threads of one process (its MLIR context
is thread-local). A few worker subprocesses compile the
uncached candidates of a bucket in parallel and export them to the disk
cache; this process then loads them from there. The workers never touch the
GPU (the compile target comes from ``CUTE_DSL_ARCH``) and exit once the pool
has been idle for ``IDLE_SECONDS``.

Worker protocol: one JSON job ``[n, k, tactic, sms, l2_bytes, out_f16]`` per
stdin line, answered by one stdout line: ``null`` or the error text.
"""

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import Future

from ....jit.core import logger

# The autotuner prepares the buckets of a shape back to back; keep the
# workers (and their imports) alive across them.
IDLE_SECONDS = 60.0

_pool = None
_lock = threading.Lock()
_disabled = False


def num_workers():
    """``FLASHINFER_SM12X_MXFP8_COMPILE_WORKERS``, else half the usable CPUs
    up to 8. 1 compiles in this process."""
    env = os.environ.get("FLASHINFER_SM12X_MXFP8_COMPILE_WORKERS")
    if env is not None:
        return max(1, int(env))
    return max(1, min(8, len(os.sched_getaffinity(0)) // 2))


class _Pool:
    def __init__(self, workers, arch):
        env = dict(os.environ, CUTE_DSL_ARCH=arch)
        self.procs = [
            subprocess.Popen(
                [sys.executable, "-m", __name__],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
            )
            for _ in range(workers)
        ]
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
                    reply = proc.stdout.readline()
                    alive = bool(reply)
                    if alive:
                        err = json.loads(reply)
                except Exception as e:  # noqa: BLE001 -- reported to the caller
                    alive = False
                    err = f"{type(e).__name__}: {e}"
            with _lock:
                self.pending -= 1
                if self.pending == 0:
                    self.timer = threading.Timer(IDLE_SECONDS, _release, (self,))
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


def _release(pool):
    global _pool
    with _lock:
        if _pool is pool and pool.pending == 0:
            _pool = None
            pool.close()


def shutdown():
    """Stop the workers now, dropping queued jobs."""
    global _pool
    with _lock:
        pool, _pool = _pool, None
        if pool is not None:
            if pool.timer is not None:
                pool.timer.cancel()
            pool.close()


def submit(jobs, arch, start=True):
    """Queue ``jobs`` on the worker subprocesses, in order.

    Returns one ``Future`` per job whose result is ``None`` or the error
    text, or ``None`` when no pool is running and ``start`` is false or the
    workers cannot be started.
    """
    global _pool, _disabled
    with _lock:
        if _disabled or (_pool is None and not start):
            return None
        if _pool is None:
            try:
                _pool = _Pool(num_workers(), arch)
            except OSError as e:
                logger.warning(
                    f"SM12x mxfp8: no compile workers ({e}); compiling serially"
                )
                _disabled = True
                return None
        pool = _pool
        if pool.timer is not None:
            pool.timer.cancel()
            pool.timer = None
        pool.pending += len(jobs)
        futures = []
        for job in jobs:
            fut: Future = Future()
            pool.jobs.put((job, fut))
            futures.append(fut)
    return futures


def _main():
    # Replies own the real stdout; anything else printed goes to stderr.
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    from ....jit.cute_dsl_core import JitSpecCuteDsl, _hash_source_files
    from .policy import Device
    from .runner import _kernel_spec

    for line in sys.stdin:
        n, k, tactic, sms, l2_bytes, out_f16 = json.loads(line)
        try:
            module, name, compile_fn, key_files = _kernel_spec(
                n, k, tuple(tactic), Device(sms, l2_bytes), out_f16
            )
            spec = JitSpecCuteDsl(
                module, name, compile_fn, _hash_source_files(key_files)
            )
            if not spec.is_compiled:
                spec.compile_and_persist()
            err = None
        except Exception as e:  # noqa: BLE001 -- reported to the parent
            err = f"{type(e).__name__}: {e}"
        replies.write(json.dumps(err) + "\n")


if __name__ == "__main__":
    _main()
