"""Bit-exact CUDA-graph replay checks for b12x kernels under perturbation.

A scenario supplies ``forward`` (launches the kernels under test on the current
stream from fixed inputs), the ``outputs`` it writes, and optionally ``reset``
(restores anything ``forward`` mutates, such as accumulators or state; it is
recorded in the same graph ahead of ``forward``). ``ReplayCheck`` captures
both, takes the first replay as the reference, requires quiet replays to
reproduce it (so nondeterministic kernels are reported as such rather than as
races), and then counts replays whose outputs differ in any bit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch


class NondeterministicScenario(AssertionError):
    """Quiet replays of a scenario disagree, so bit-exact replay checks cannot apply."""


@dataclass
class RaceReport:
    replays: int
    bad: int
    mismatched_bytes: dict[str, int]
    first: list[dict] = field(default_factory=list)
    noise_kernels: int | None = None

    def __str__(self) -> str:
        noise = (
            ""
            if self.noise_kernels is None
            else f", {self.noise_kernels} noise kernels"
        )
        worst = {k: v for k, v in self.mismatched_bytes.items() if v}
        return (
            f"{self.bad} of {self.replays} replays differ from the reference{noise}"
            + (f"; mismatched bytes {worst}; first {self.first}" if self.bad else "")
        )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


class ReplayCheck:
    def __init__(
        self,
        forward: Callable[[], None],
        outputs: Mapping[str, torch.Tensor],
        *,
        reset: Callable[[], None] | None = None,
        device: torch.device | None = None,
        quiet_replays: int = 64,
    ):
        for name, t in outputs.items():
            if not t.is_contiguous():
                raise ValueError(f"output {name} must be contiguous")
        self.device = device or next(iter(outputs.values())).device
        self.outputs = dict(outputs)
        self.names = list(self.outputs)
        self.stream = torch.cuda.Stream(device=self.device)

        def step():
            if reset is not None:
                reset()
            forward()

        with torch.cuda.stream(self.stream):
            step()  # compile and warm outside capture
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            step()
        self._replay()
        self.stream.synchronize()
        self.refs = {n: t.clone() for n, t in self.outputs.items()}

        self.counts = torch.zeros(
            len(self.names), dtype=torch.int64, device=self.device
        )
        self.host = torch.zeros(len(self.names), dtype=torch.int64, pin_memory=True)
        with torch.cuda.stream(self.stream):
            self._compare_ops()
        self.stream.synchronize()
        self.compare = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.compare, stream=self.stream):
            self._compare_ops()
            self.host.copy_(self.counts, non_blocking=True)

        quiet = self.run(replays=quiet_replays)
        if quiet.bad:
            raise NondeterministicScenario(f"quiet replays: {quiet}")

    def _replay(self):
        with torch.cuda.stream(self.stream):
            self.graph.replay()

    def _compare_ops(self):
        torch.stack(
            [
                (_bytes(self.outputs[n]) != _bytes(self.refs[n])).sum()
                for n in self.names
            ],
            out=self.counts,
        )

    def _describe(self, replay: int, row: list[int]) -> dict:
        record = {"replay": replay}
        for n, count in zip(self.names, row):
            if not count:
                continue
            cur, ref = self.outputs[n], self.refs[n]
            entry = {"bytes": int(count)}
            if cur.dim() >= 2 and cur.is_floating_point():
                rows = (
                    (cur != ref).reshape(cur.shape[0], -1).any(dim=1).nonzero().view(-1)
                )
                entry["rows"] = rows[:16].tolist()
            record[n] = entry
        return record

    def run(
        self,
        *,
        replays: int,
        seconds: float | None = None,
        contention=None,
        keep_first: int = 4,
    ) -> RaceReport:
        deadline = None if seconds is None else time.monotonic() + seconds
        bad = done = 0
        totals = [0] * len(self.names)
        first = []
        start_noise = None if contention is None else contention.launched
        while done < replays and (deadline is None or time.monotonic() < deadline):
            with torch.cuda.stream(self.stream):
                self.graph.replay()
                self.compare.replay()
            self.stream.synchronize()
            done += 1
            row = self.host.tolist()
            if any(row):
                bad += 1
                totals = [a + b for a, b in zip(totals, row)]
                if len(first) < keep_first:
                    first.append(self._describe(done, row))
        return RaceReport(
            replays=done,
            bad=bad,
            mismatched_bytes=dict(zip(self.names, totals)),
            first=first,
            noise_kernels=None
            if contention is None
            else contention.launched - start_noise,
        )
