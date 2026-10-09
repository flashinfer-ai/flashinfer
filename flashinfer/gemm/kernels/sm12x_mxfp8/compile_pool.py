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
        self.busy = 0
        self.timer = None
        for proc in self.procs:
            threading.Thread(target=self._serve, args=(proc,), daemon=True).start()

    def _serve(self, proc):
        alive = True
        while True:
            job, done = self.jobs.get()
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
            done.put((job, err))

    def close(self):
        for _ in self.procs:
            self.jobs.put((None, None))
        for proc in self.procs:
            with contextlib.suppress(OSError):
                proc.stdin.close()


def _release():
    global _pool
    with _lock:
        pool = _pool
        if pool is not None and pool.busy == 0:
            _pool = None
            pool.close()


def compile_all(jobs, arch, on_done):
    """Compile ``jobs`` in worker subprocesses; ``on_done(job, error)`` runs in
    this thread as each one finishes. Returns ``False`` (nothing done) when the
    pool is unavailable."""
    global _pool, _disabled
    with _lock:
        if _disabled:
            return False
        if _pool is None:
            try:
                _pool = _Pool(num_workers(), arch)
            except OSError as e:
                logger.warning(
                    f"SM12x mxfp8: no compile workers ({e}); compiling serially"
                )
                _disabled = True
                return False
        pool = _pool
        pool.busy += 1
        if pool.timer is not None:
            pool.timer.cancel()
    try:
        done = queue.Queue()
        for job in jobs:
            pool.jobs.put((job, done))
        for _ in jobs:
            on_done(*done.get())
    finally:
        with _lock:
            pool.busy -= 1
            if pool.busy == 0:
                pool.timer = threading.Timer(IDLE_SECONDS, _release)
                pool.timer.daemon = True
                pool.timer.start()
    return True


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
