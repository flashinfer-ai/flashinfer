"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Collective counting for the context-parallel (KCP) training tests: how many ``torch.distributed`` collectives a
checkpointed layer issues, and how many waits on their work handles it makes, in each phase -- the checkpointed
(original) forward, torch's recompute of it inside ``.backward()``, and the rest of the backward::

    with CollectiveCounter() as cc:
        out = checkpoint(fn, *args, use_reentrant=False,
                         context_fn=lambda: (contextlib.nullcontext(), RecomputePhase(cc)))
        cc.phase = "bwd"
        out.backward(dout)
    cc.record()  # {"calls": {"orig": n, "recompute": n, "bwd": n}, "waits": {...}}

The shims replace the ``torch.distributed`` module attributes while the counter is entered, so code that calls
``dist.<name>(...)`` at run time is counted; tensor contents are never read.  :class:`RecomputePhase` can be entered
again (torch enters the recompute context once per recompute, e.g. a second backward with ``retain_graph``).
"""

from __future__ import annotations

import torch.distributed as dist


class CollectiveCounter:
    """Counts the ``torch.distributed`` collectives issued and the waits on their work handles, by phase: ``"orig"``
    (the checkpointed forward), ``"recompute"`` (torch's recompute, marked by the checkpoint's ``context_fn``) and
    ``"bwd"`` (the rest of the backward).  The shims replace the module attributes the package calls
    (``dist.all_to_all_single`` ...) while the counter is entered; tensor contents are never read."""

    NAMES = (
        "all_to_all_single",
        "batch_isend_irecv",
        "all_gather_object",
        "all_gather",
        "all_gather_into_tensor",
        "all_reduce",
        "broadcast",
        "reduce_scatter_tensor",
    )
    # ``isend`` / ``irecv`` are not shimmed: ring exchanges pass those functions themselves as ``P2POp`` operations,
    # which torch checks by identity; a point-to-point batch counts once, as its ``batch_isend_irecv`` call

    def __init__(self) -> None:
        self.phase = "orig"
        self.calls: dict[str, int] = {}
        self.waits: dict[str, int] = {}
        self._saved: dict[str, object] = {}

    def _bump(self, table: dict[str, int]) -> None:
        table[self.phase] = table.get(self.phase, 0) + 1

    def _work(self, w):
        counter = self

        class _Work:
            def __init__(self, inner) -> None:
                self._inner = inner

            def wait(self, *a, **kw):
                counter._bump(counter.waits)
                return self._inner.wait(*a, **kw)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        return _Work(w) if hasattr(w, "wait") else w

    def __enter__(self) -> "CollectiveCounter":
        for name in self.NAMES:
            fn = getattr(dist, name, None)
            if fn is None:
                continue
            self._saved[name] = fn

            def shim(*a, _fn=fn, **kw):
                self._bump(self.calls)
                r = _fn(*a, **kw)
                if isinstance(r, list):
                    return [self._work(x) for x in r]
                return self._work(r)

            setattr(dist, name, shim)
        return self

    def __exit__(self, *exc) -> None:
        for name, fn in self._saved.items():
            setattr(dist, name, fn)
        self._saved.clear()

    def record(self) -> dict:
        return {"calls": dict(self.calls), "waits": dict(self.waits)}


class RecomputePhase:
    """The recompute context of ``checkpoint(..., context_fn=...)``: sets the counter's phase to ``"recompute"`` while
    torch recomputes the region, and restores it afterwards; re-entrant across recomputes."""

    def __init__(self, counter: CollectiveCounter) -> None:
        self.counter = counter
        self.prev: list[str] = []

    def __enter__(self) -> "RecomputePhase":
        self.prev.append(self.counter.phase)
        self.counter.phase = "recompute"
        return self

    def __exit__(self, *exc) -> None:
        self.counter.phase = self.prev.pop()


__all__ = ["CollectiveCounter", "RecomputePhase"]
