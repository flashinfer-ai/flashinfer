"""Opt-in wall timing for preparation work and its external driver."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


class PreparationTiming:
    def __init__(self, role, **context):
        root = os.environ.get("B12X_PREPARATION_TRACE_DIR")
        self.path = None
        self.context = context
        self.started = self.last_report = time.perf_counter()
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)
        if root:
            directory = Path(root)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"{role}-{os.getpid()}.jsonl"
            self.record("begin")

    def add(self, name, seconds):
        if self.path is not None:
            self.seconds[name] += seconds
            self.counts[name] += 1

    @contextmanager
    def span(self, name):
        if self.path is None:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def record(self, event, *, periodic=False, **fields):
        if self.path is None:
            return
        now = time.perf_counter()
        if periodic and now - self.last_report < 5.0:
            return
        self.last_report = now
        payload = dict(
            event=event, wall_time=time.time(), elapsed_s=now - self.started,
            seconds=dict(self.seconds), counts=dict(self.counts),
            **self.context, **fields,
        )
        with self.path.open("a") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
