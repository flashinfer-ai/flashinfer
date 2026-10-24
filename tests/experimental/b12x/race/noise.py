"""SM contention for race tests: highest-priority hog and skew kernels on a side stream.

Races between CTAs, warps, or the generic and async proxies rarely show on a
quiet GPU because every CTA starts in the same order and runs at the same
pace. ``SMContention`` disturbs that while the kernel under test replays:

- hog kernels take random CTA counts (1-2x the SM count) for random durations,
  holding either all of an SM's shared memory (so test CTAs wait for SMs to
  drain and start unevenly) or a light footprint that co-resides with test
  CTAs; half of them stream random loads over a buffer to disturb L2 and TLB;
- skew kernels of 1-3 warps co-reside with test CTAs and load the SM's
  scheduler sub-partitions unevenly, so sibling warps of one test CTA drift;
- idle gaps of random length vary the phase against the kernel under test.

Launches are recorded once into a few random schedule graphs. A host thread
replays them on a highest-priority stream, so hog CTAs are scheduled ahead of
pending test CTAs as SMs drain. The kernels are CuTe DSL; no nvcc is needed.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile

_GOLDEN = -7046029254386353131  # 0x9E3779B97F4A7C15 as int64
_LCG_MUL = 6364136223846793005
_LCG_ADD = 1442695040888963407


def _words_storage(words: int):
    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "words": cute.struct.Align[cute.struct.MemRange[cutlass.Int64, words], 128]
    }
    return cute.struct(SharedStorage)


class _Hog:
    """Spin for ``ns`` on %globaltimer holding ``smem_words`` of shared memory;
    with ``touch``, stream random 8-byte loads over ``buf``."""

    def __init__(self, threads: int, smem_words: int):
        self.threads, self.smem_words = threads, smem_words

    @cute.jit
    def __call__(
        self,
        buf: cute.Tensor,
        ctas: Int32,
        ns: Int64,
        seed: Int64,
        touch: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(buf, ns, seed, touch).launch(
            grid=(ctas, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, buf: cute.Tensor, ns: Int64, seed: Int64, touch: Int32):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(_words_storage(self.smem_words))
        held = storage.words.get_tensor(cute.make_layout((self.smem_words,)))
        x = seed ^ (Int64(cta) * Int64(_GOLDEN)) ^ Int64(tid)
        held[tid % self.smem_words] = x
        words = Int32(cute.size(buf))
        acc = Int64(0)
        start = cute.arch.globaltimer()
        while cute.arch.globaltimer() - start < ns:
            if touch != Int32(0):
                for _ in cutlass.range_constexpr(4):
                    x = x * Int64(_LCG_MUL) + Int64(_LCG_ADD)
                    acc += buf[Int32((x >> 17) & Int64(0x7FFFFFFF)) % words]
        if acc == Int64(0x5555555555555555):
            buf[0] = acc + held[(tid ^ 1) % self.smem_words]


class _Skew:
    """Issue-slot hog: spin on dependent FMAs for ``ns`` with no shared memory."""

    def __init__(self, threads: int):
        self.threads = threads

    @cute.jit
    def __call__(
        self, sink: cute.Tensor, ctas: Int32, ns: Int64, stream: cuda.CUstream
    ):
        self.kernel(sink, ns).launch(
            grid=(ctas, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, sink: cute.Tensor, ns: Int64):
        tid, _, _ = cute.arch.thread_idx()
        a = Float32(tid) * Float32(1e-3)
        start = cute.arch.globaltimer()
        while cute.arch.globaltimer() - start < ns:
            for _ in cutlass.range_constexpr(64):
                a = a * Float32(1.0001) + Float32(1e-7)
        if a == Float32(1234.5):
            sink[0] = a


@dataclass(frozen=True)
class _Launch:
    kind: str  # "hog", "skew", or "gap"
    variant: tuple
    ctas: int
    ns: int
    seed: int = 0
    touch: int = 0


class SMContention:
    """Context manager that keeps highest-priority noise running on ``device``.

    ``launched`` counts noise kernels issued so far; tests report it to show the
    noise ran. The default schedule mirrors the perturbation that exposed the
    b12x dense GEMM stage-reuse race (local-inference-lab/flashinfer#16).
    """

    def __init__(
        self,
        device: torch.device,
        *,
        seed: int = 12345,
        buffer_bytes: int = 1 << 30,
        max_us: int = 400,
        max_gap_us: int = 300,
        touch_percent: int = 50,
        skew_percent: int = 25,
        schedules: int = 6,
        launches_per_schedule: int = 48,
    ):
        self.device = device
        props = torch.cuda.get_device_properties(device)
        self.sms = props.multi_processor_count
        # Leave room for the compiler's static reservation.
        full_words = (props.shared_memory_per_block_optin - 1024) // 8
        self.buffer = torch.full(
            (buffer_bytes // 8,), 0x3C3C3C3C3C3C3C3C, dtype=torch.int64, device=device
        )
        self.sink = torch.zeros(1, dtype=torch.float32, device=device)
        self.stream = torch.cuda.Stream(device=device, priority=-100)
        self._buf = from_dlpack(self.buffer, assumed_align=16)
        self._sink = from_dlpack(self.sink, assumed_align=4)
        self._kernels = {}
        for threads in (256, 1024):
            for words in (full_words, threads):
                self._kernels[("hog", threads, words)] = self._compile_hog(
                    threads, words
                )
        for threads in (32, 64, 96):
            self._kernels[("skew", threads)] = self._compile_skew(threads)

        rng = random.Random(seed)
        self._graphs = []
        for _ in range(schedules):
            plan = [
                self._random_launch(
                    rng, full_words, max_us, max_gap_us, touch_percent, skew_percent
                )
                for _ in range(launches_per_schedule)
            ]
            self._graphs.append(
                (self._capture(plan), sum(1 for p in plan if p.kind != "gap"))
            )
        self._rng = random.Random(seed ^ 0x5EED)
        self.launched = 0
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    def _compile_hog(self, threads, words):
        return b12x_compile(
            _Hog(threads, words),
            self._buf,
            Int32(1),
            Int64(1),
            Int64(1),
            Int32(0),
            self._cu_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "test.race.hog", 1, (threads, words)
            ),
        )

    def _compile_skew(self, threads):
        return b12x_compile(
            _Skew(threads),
            self._sink,
            Int32(1),
            Int64(1),
            self._cu_stream(),
            compile_spec=KernelCompileSpec.from_key("test.race.skew", 1, (threads,)),
        )

    def _cu_stream(self):
        return cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)

    def _random_launch(
        self, rng, full_words, max_us, max_gap_us, touch_percent, skew_percent
    ):
        roll = rng.randrange(100)
        if roll < 15:
            return _Launch("gap", ("skew", 32), 1, 1000 * rng.randrange(max_gap_us + 1))
        if roll < 15 + skew_percent:
            threads = rng.choice((32, 64, 96))
            return _Launch(
                "skew",
                ("skew", threads),
                1 + rng.randrange(6 * self.sms),
                1000 * (5 + rng.randrange(max_us)),
            )
        threads = rng.choice((256, 1024))
        words = full_words if rng.randrange(3) else threads
        return _Launch(
            "hog",
            ("hog", threads, words),
            1 + rng.randrange(2 * self.sms),
            1000 * (2 + rng.randrange(max_us)),
            rng.getrandbits(62),
            int(rng.randrange(100) < touch_percent),
        )

    def _issue(self, launch):
        kernel = self._kernels[launch.variant]
        if launch.variant[0] == "hog":
            kernel(
                self._buf,
                Int32(launch.ctas),
                Int64(launch.ns),
                Int64(launch.seed),
                Int32(launch.touch),
                self._cu_stream(),
            )
        else:
            kernel(self._sink, Int32(launch.ctas), Int64(launch.ns), self._cu_stream())

    def _capture(self, plan):
        with torch.cuda.stream(self.stream):
            for launch in plan[:2]:
                self._issue(launch)
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            for launch in plan:
                self._issue(launch)
        return graph

    def _run(self):
        try:
            with torch.cuda.device(self.device):
                while not self._stop.is_set():
                    graph, kernels = self._graphs[
                        self._rng.randrange(len(self._graphs))
                    ]
                    with torch.cuda.stream(self.stream):
                        graph.replay()
                    self.stream.synchronize()
                    self.launched += kernels
        except BaseException as error:  # noqa: BLE001 - surfaced by __exit__
            self._error = error

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="b12x-sm-contention", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.stream.synchronize()
        if self._error is not None:
            raise RuntimeError("SM contention thread failed") from self._error
        return False
