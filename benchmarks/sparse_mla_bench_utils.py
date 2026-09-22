# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cold-L2 CUDA-Graph timing with explicit event boundaries and no fallback."""

from collections.abc import Callable
import weakref

import torch
from cuda.bindings import runtime as cudart

from flashinfer.testing.utils import get_l2_cache_size


def _cuda_value(result):
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA benchmark operation failed: {result[0]}")
    return result[1] if len(result) == 2 else None


class _ExternalEvent:
    """Explicit CUDA event nodes, including on Torch versions without external=."""

    def __init__(self):
        self.handle = _cuda_value(cudart.cudaEventCreateWithFlags(0))
        self._finalizer = weakref.finalize(self, cudart.cudaEventDestroy, self.handle)

    def record(self, stream):
        _cuda_value(
            cudart.cudaEventRecordWithFlags(
                self.handle,
                stream.cuda_stream,
                cudart.cudaEventRecordExternal,
            )
        )

    def elapsed_time(self, other):
        return _cuda_value(cudart.cudaEventElapsedTime(self.handle, other.handle))


class ColdL2GraphBenchmark:
    """Capture eviction -> start -> complete invocation -> stop for each sample.

    The callable must bind preallocated buffers and must not mutate its inputs.
    All GPU work required per invocation belongs in the callable, including
    split reductions and counter resets. External event nodes keep eviction
    outside each measured span. No buffer rotation or eager fallback is used.

    Eviction reads a separate, nonconstant buffer, sized to at least 4x L2.
    A reduction makes those reads observable without leaving the entire L2
    dirty and charging its writebacks to the timed attention. Qualification
    requires cache-counter checks and a larger-buffer stability check;
    allocated pool size is not evidence.
    """

    def __init__(
        self,
        fn: Callable[[], object],
        *,
        device: torch.device,
        eviction_bytes: int | None = None,
        samples_per_replay: int = 8,
    ):
        if samples_per_replay <= 0:
            raise ValueError("samples_per_replay must be positive")
        # DSL graphs capture raw pointers, so retain the callable's closure
        # (and its input/plan/workspace owners) for every subsequent replay.
        self.fn = fn
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.l2_bytes = get_l2_cache_size(self.device)
            minimum_bytes = 4 * self.l2_bytes
            if eviction_bytes is None:
                eviction_bytes = minimum_bytes
            if eviction_bytes < minimum_bytes:
                raise ValueError(
                    "eviction buffer must be at least four times device L2"
                )
            self.eviction_bytes = (eviction_bytes + 3) // 4 * 4
            self.eviction = torch.randint(
                0,
                2**30,
                (self.eviction_bytes // 4,),
                dtype=torch.int32,
                device=self.device,
            )
            self.eviction_checksum = torch.empty(
                (), dtype=torch.int64, device=self.device
            )
            self.stream = torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    fn()
            self.stream.synchronize()
            self.starts = [_ExternalEvent() for _ in range(samples_per_replay)]
            self.stops = [_ExternalEvent() for _ in range(samples_per_replay)]
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                for start, stop in zip(self.starts, self.stops, strict=True):
                    torch.sum(
                        self.eviction,
                        dim=0,
                        dtype=torch.int64,
                        out=self.eviction_checksum,
                    )
                    start.record(self.stream)
                    fn()
                    stop.record(self.stream)
            self.stream.synchronize()

    def sample(self) -> list[float]:
        """Return microseconds for each invocation, excluding eviction.

        Call benchmarks in an alternating order to collect paired samples on
        the same immutable fixture. Synchronization happens outside GPU spans.
        """
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.graph.replay()
        self.stream.synchronize()
        return [
            a.elapsed_time(b) * 1000
            for a, b in zip(self.starts, self.stops, strict=True)
        ]
