"""The Cake Ulysses all-to-all package: its portfolio through the public communicator, and the package route.

The backend assertions import the package only through :func:`tests.test_helpers.cake_package.cake_package`
(flashinfer-ai/flashinfer#6553); the multi-GPU portfolio runs through the public communicator so the public
contract (caller-owned outputs, NCCL-equal results) is what is checked.
"""

import ctypes
import gc
import hashlib
import re
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.comm.cake_ulysses_a2a_fixture import (
    NVLINK_KERNELS,
    PreparedCase,
    cake_direct_available,
    cake_route_available,
    prepare,
    shapes,
)
from tests.comm.test_ulysses_a2a import multi_process_parallel
from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_ulysses")
cake_build = cake_package("cake_ulysses", "cake_build")
loader = cake_package("cake_ulysses", "cake_jit")
backend = cake_package("cake_ulysses", "cake_backend")


# ---- the portfolio through the public communicator ---------------------------------


def _correctness_worker(world_size, rank, port, nvlink_kernel):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        for shape in shapes(world_size):
            case = prepare(shape, backend="nvlink", nvlink_kernel=nvlink_kernel)
            try:
                assert case.nvlink_kernel == nvlink_kernel
                case.check()
            finally:
                case.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("nvlink_kernel", NVLINK_KERNELS)
@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_generated_ulysses_portfolio(world_size, nvlink_kernel):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    if nvlink_kernel == "cake":
        if (
            torch.cuda.get_device_capability(0)
            not in pkg.SUPPORTED_COMPUTE_CAPABILITIES
        ):
            pytest.skip("the Cake portfolio targets Blackwell SM100/SM103")
        if not cake_route_available(world_size):
            pytest.skip(
                "build.json delivers no library for this device yet (the export run fills it)"
            )
    multi_process_parallel(world_size, _correctness_worker, (nvlink_kernel,))


def _refresh(case, step):
    """New input values and poisoned outputs, so a call that writes nothing or the wrong output fails."""
    generator = torch.Generator(device="cuda").manual_seed(
        7919 * step + case.communicator.rank
    )
    for source in case.inputs:
        source.copy_(
            torch.randn(
                source.shape, dtype=source.dtype, device="cuda", generator=generator
            )
        )
    for output in case.outputs:
        output.fill_(float("nan"))


def _registered_worker(world_size, rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        for index, shape in enumerate(shapes(world_size)):
            case = prepare(
                shape, backend="nvlink", nvlink_kernel="cake", registered=True
            )
            communicator = case.communicator
            try:
                assert case.nvlink_kernel == "cake" and case.registered
                assert {output.data_ptr() for output in case.outputs} == set(
                    communicator._nvlink_outputs
                )
                _refresh(case, 0)
                case.check()
                # caller-owned outputs on the same communicator take the staged
                # route; the two routes alternate on disjoint words of one
                # signal allocation
                staged = PreparedCase(
                    communicator,
                    case.inputs,
                    tuple(torch.empty_like(output) for output in case.outputs),
                    None,
                    "cake",
                )
                for step in range(1, 3):
                    _refresh(staged, step)
                    staged.check()
                    _refresh(case, step)
                    case.check()
                if index == 0:
                    # the direct route under CUDA graph capture and replay
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        case.run()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        case.run()
                    for step in range(3, 6):
                        _refresh(case, step)
                        graph.replay()
                        case.check(run=False)
                    del graph
                    torch.cuda.synchronize()
                    dist.barrier()
                    # an output registered under inference mode keeps no base
                    # view; releasing it after close() must stay safe
                    with torch.inference_mode():
                        late = communicator.allocate_output(
                            case.inputs[0], "scatter_heads"
                        )
                    assert late.data_ptr() in communicator._nvlink_outputs
                _check_zone_tail(case)
            finally:
                case.close()
            # close releases every registered allocation and peer mapping
            assert not communicator._nvlink_outputs
            assert not communicator._exports and not communicator._imports
            if index == 0:
                del late
                gc.collect()
        # the portable kernel registers nothing: allocate_output stays a plain allocation
        portable = prepare(
            shapes(world_size)[0],
            backend="nvlink",
            nvlink_kernel="portable",
            registered=True,
        )
        try:
            assert not portable.communicator._nvlink_outputs
            portable.check()
        finally:
            portable.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_registered_outputs_take_the_direct_route(world_size):
    """Registered outputs: the portfolio bit-exact, alternation with caller-owned outputs, graph replay, close."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    if torch.cuda.get_device_capability(0) not in pkg.SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("the Cake portfolio targets Blackwell SM100/SM103")
    if not cake_direct_available(world_size):
        pytest.skip("build.json delivers no direct-route library for this device yet")
    multi_process_parallel(world_size, _registered_worker, ())


def _offset_input(dimensions, dtype, offset):
    """A contiguous CUDA tensor that starts ``offset`` elements into its storage."""
    numel = 1
    for extent in dimensions:
        numel *= extent
    storage = torch.empty(numel + offset, dtype=dtype, device="cuda")
    return storage[offset:].view(dimensions)


def _unaligned_rank_worker(world_size, rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        for shape in shapes(world_size):
            if shape["performance"]:
                continue
            dtype = getattr(torch, shape["dtype"])
            batch, sequence, heads, head_dim = (
                shape[key] for key in ("B", "S_local", "H", "D")
            )
            local_shape = (batch, sequence, heads, head_dim)
            global_shape = (batch, sequence * world_size, heads // world_size, head_dim)
            offset = 1 if rank == 1 else 0
            inputs = tuple(
                _offset_input(dimensions, dtype, offset)
                for dimensions in (local_shape, local_shape, local_shape, global_shape)
            )
            if rank == 1:
                assert all(source.data_ptr() % 16 for source in inputs)
            case = prepare(
                shape,
                backend="nvlink",
                nvlink_kernel="cake",
                inputs=inputs,
                registered=True,
            )
            try:
                assert case.nvlink_kernel == "cake" and case.registered
                for step in range(2):
                    _refresh(case, step)
                    case.check()
                _check_zone_tail(case)
            finally:
                case.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_registered_outputs_with_one_unaligned_rank(world_size):
    """Rank 1 passes every input one element into its storage, so its streams take the
    element path while its peers' take the 16-byte vector path; every correctness row
    of the world completes bit-exact on the direct route."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    if torch.cuda.get_device_capability(0) not in pkg.SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("the Cake portfolio targets Blackwell SM100/SM103")
    if not cake_direct_available(world_size):
        pytest.skip("build.json delivers no direct-route library for this device yet")
    multi_process_parallel(world_size, _unaligned_rank_worker, ())


# ---- the zone of the world-8 bf16 direct scatter and gather (eight GPUs) ------------
#
# The world-8 bf16 scatter and gather programs of the direct route (the zone
# programs) let a stream whose destination has not entered the launch yet
# store its first 16-byte units into a zone at the tail of the destination's
# signal allocation; the destination copies them into ``out`` before its
# launch ends. No other program writes the zone. CTA c serves the same
# destination in both programs and both advance the same per-CTA epoch, so
# the two share one communicator's zone. The checks read this rank's u32
# signal words: EPOCHS + c (the epoch of CTA c's last launch), ZR + c (the
# epoch of the last launch whose incoming stream c used the zone) and the
# zone's units, unit ((c * 2 + epoch % 2) * ZONE_SLOTS + slot) * 480 + lane
# from word ZONE_DATA.

# (dtype, world size, collective) of every direct program that uses the zone
_ZONE_PROGRAMS = frozenset({("bf16", 8, "scatter"), ("bf16", 8, "gather")})
_MODE = {"scatter": 0, "gather": 1}  # the record's mode of each collective
_OPERAND = {"scatter": 0, "gather": 3}  # its source and output in a case
_PARTNER = {"scatter": "gather", "gather": "scatter"}


def _zone_program(dtype, world_size, op):
    """Whether the direct ``op`` program of this dtype and world size has the zone."""
    return (backend.dtype_name(dtype), world_size, op) in _ZONE_PROGRAMS


_GRID = 148
_EPOCHS = 324
_ZONE_WORD = 1024
_ZONE_DATA = 2048
_ZONE_SLOTS = 2
_ZONE_LANES = 480
_CHUNK_UNITS = 7680
_ZONE_POISON = -0x21524111  # 0xDEADBEEF as an int32 word
# the late rank's delay before its launch (several milliseconds): its peers
# start first and find it not entered yet
_SKEW_CYCLES = 10_000_000
# The late rank's entry poll (world 8): before its delay and its launch, a
# kernel on its stream reads (ld.acquire.sys) its own ENTRY64 word of the
# destination of every CTA c until the word differs from LAST64[c], the token
# CTA c recorded in its previous launch: u64 words 236 + t and 244 + c of the
# world-8 direct programs, CTA c serving rank + offset + 1 (_zone_destination)
# in the scatter and the gather program alike, which keep LAST64[c] for that
# one destination, so the poll covers a following launch of either program.
# The launch that follows on the same stream reads those words after the poll,
# so each of its streams finds its destination's token at its one read (route
# D with the entry token already seen). The poll writes no signal word; a CTA
# that sees no new token within _ENTRY_POLL_TIMEOUT_NS stops waiting and fails
# the trial.
_ENTRY64 = 236
_LAST64 = 244
_ENTRY_POLL_TIMEOUT_NS = 120_000_000_000
_ENTRY_POLL_CUDA = r"""
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

__global__ void cake_zone_entry_poll(const unsigned long long* words,
                                     const int* dest, int grid,
                                     long long timeout_ns, int* status) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= grid) return;
  const int d = dest[c];
  const unsigned long long* entry = words + ENTRY64 + d;
  unsigned long long last, token, start, now;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
               : "=l"(last) : "l"(words + LAST64 + c) : "memory");
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  bool waited = false;
  for (;;) {
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(token) : "l"(entry) : "memory");
    if (token != last) break;
    waited = true;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (static_cast<long long>(now - start) > timeout_ns) {
      atomicAdd(status, 1);
      atomicOr(status + 1, 1 << d);
      break;
    }
    __nanosleep(100);
  }
  if (waited) atomicAdd(status + 2, 1);
}

void entry_poll(torch::Tensor words, torch::Tensor dest, torch::Tensor status,
                int64_t timeout_ns) {
  TORCH_CHECK(words.is_cuda() && dest.is_cuda() && status.is_cuda(),
              "entry_poll: CUDA tensors");
  TORCH_CHECK(dest.scalar_type() == at::kInt && status.scalar_type() == at::kInt &&
                  status.numel() >= 3,
              "entry_poll: int32 dest and status");
  TORCH_CHECK(dest.numel() >= 1 && dest.numel() <= 160, "entry_poll: 1 .. 160 CTAs");
  TORCH_CHECK(words.numel() * words.element_size() >= (LAST64 + dest.numel()) * 8,
              "entry_poll: short signal words");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(words.data_ptr()) % 8 == 0,
              "entry_poll: unaligned signal words");
  cake_zone_entry_poll<<<1, 160, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const unsigned long long*>(words.data_ptr()),
      dest.data_ptr<int>(), static_cast<int>(dest.numel()),
      static_cast<long long>(timeout_ns), status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
# (B, S_local, H, D) at world 8: one 16-byte unit a row for H = 8, D = 8, so
# the stream sizes sit on the zone slot (480 units) and chunk (7680 units)
# boundaries, up to every CTA of a destination holding a stripe (21 / 22);
# rows with B > 1 put batch boundaries inside a zone slot (the scatter's
# output places each batch apart); the last two rows are not whole 16-byte
# units and take the elements path
_ZONE_ROWS = [
    (1, 1, 8, 8),
    (1, 5, 8, 8),
    (3, 1, 8, 8),
    (7, 3, 8, 8),
    (1, 479, 8, 8),
    (1, 480, 8, 8),
    (1, 481, 8, 8),
    (1, 7679, 8, 8),
    (1, 7680, 8, 8),
    (1, 7681, 8, 8),
    (2, 7681, 8, 8),
    (1, 15359, 8, 8),
    (1, 15360, 8, 8),
    (1, 15361, 8, 8),
    (1, 161279, 8, 8),
    (1, 161281, 8, 8),
    (1, 168959, 8, 8),
    (1, 168961, 8, 8),
    (1, 3, 16, 64),
    (2, 16, 24, 64),
    (1, 4095, 40, 128),
    (1, 5, 24, 3),
    (1, 7681, 8, 4),
]


def _zone_shape(row, dtype="bfloat16"):
    batch, sequence, heads, head_dim = row
    return {
        "label": f"zone_w8_{dtype}_{batch}x{sequence}x{heads}x{head_dim}",
        "world_size": 8,
        "B": batch,
        "S_local": sequence,
        "H": heads,
        "D": head_dim,
        "dtype": dtype,
        "performance": False,
    }


def _stream_units(shape):
    """16-byte units of one stream of a zone program (bf16, scatter or gather
    alike) on the vector path; 0 for another dtype or when rows are not whole
    units."""
    row = shape["H"] // shape["world_size"] * shape["D"]
    if shape["dtype"] != "bfloat16" or row % 8:
        return 0
    return shape["B"] * shape["S_local"] * row // 8


def _zone_stream(cta):
    """CTA ``cta``'s peer offset (it serves rank + offset + 1) and index among that peer's CTAs."""
    if cta < 22:
        return 0, cta
    return 1 + (cta - 22) // 21, (cta - 22) % 21


def _zone_source(receiver, cta):
    """The rank whose CTA ``cta`` streams into ``receiver``."""
    return (receiver + 7 - _zone_stream(cta)[0]) % 8


def _zone_destination(rank, cta):
    """The rank CTA ``cta`` of ``rank`` streams into, in either collective."""
    return (rank + _zone_stream(cta)[0] + 1) % 8


_entry_poll_module = None


def _entry_poll_extension():
    """The entry poll kernel, built once per process (torch extension cache)."""
    global _entry_poll_module
    if _entry_poll_module is None:
        from torch.utils.cpp_extension import load_inline

        _entry_poll_module = load_inline(
            name="cake_zone_entry_poll",
            cpp_sources="void entry_poll(torch::Tensor words, torch::Tensor dest, "
            "torch::Tensor status, int64_t timeout_ns);",
            cuda_sources=_ENTRY_POLL_CUDA,
            functions=["entry_poll"],
            extra_cuda_cflags=["-O2", f"-DENTRY64={_ENTRY64}", f"-DLAST64={_LAST64}"],
        )
    return _entry_poll_module


class _EntryPoll:
    """The late rank's entry poll on a world-8 Cake communicator, before a
    launch of its scatter or its gather."""

    def __init__(self, case):
        communicator = case.communicator
        assert communicator.world_size == 8
        rank = communicator.rank
        self.words = communicator._cake.signals_self.view(torch.int32)
        self.dest = torch.tensor(
            [_zone_destination(rank, c) for c in range(_GRID)],
            dtype=torch.int32,
            device="cuda",
        )
        self.status = torch.zeros(3, dtype=torch.int32, device="cuda")
        self.module = _entry_poll_extension()
        self.launches = self.waited = self.timeouts = 0

    def enqueue(self):
        """On the current stream, before this rank's delay and launch."""
        self.module.entry_poll(
            self.words, self.dest, self.status, _ENTRY_POLL_TIMEOUT_NS
        )
        self.launches += 1

    def errors(self):
        """After a synchronize: a timed-out poll fails the trial."""
        timed_out, mask, waited = self.status.tolist()
        self.status.zero_()
        self.waited += waited
        self.timeouts += timed_out
        if not timed_out:
            return []
        return [
            f"entry poll: {timed_out} CTAs saw no new ENTRY64 token in "
            f"{_ENTRY_POLL_TIMEOUT_NS / 1e9:.0f} s (destinations mask {mask:#x})"
        ]


class _ZoneProbe:
    """This rank's signal words of a Cake communicator, as int32."""

    def __init__(self, case):
        self.words = case.communicator._cake.signals_self.view(torch.int32)
        assert self.words.numel() == loader.SIGNAL_WORDS
        self.case = case
        self.poll = None

    def entry_poll(self):
        """The late rank's entry poll (world 8 only; None otherwise)."""
        if self.case.communicator.world_size != 8:
            return None
        if self.poll is None:
            self.poll = _EntryPoll(self.case)
        return self.poll

    def tail_is_zero(self):
        return int(self.words[_ZONE_WORD:].count_nonzero()) == 0

    def poison(self):
        self.words[_ZONE_DATA:].fill_(_ZONE_POISON)

    def read(self):
        """(EPOCHS, ZR, written units) after this rank's work completes."""
        torch.cuda.synchronize()
        epochs = [
            word & 0xFFFFFFFF for word in self.words[_EPOCHS : _EPOCHS + _GRID].tolist()
        ]
        tags = [
            word & 0xFFFFFFFF
            for word in self.words[_ZONE_WORD : _ZONE_WORD + _GRID].tolist()
        ]
        written = (
            self.words[_ZONE_DATA:]
            .view(_GRID, 2, _ZONE_SLOTS, _ZONE_LANES, 4)
            .ne(_ZONE_POISON)
            .any(dim=-1)
            .cpu()
        )
        return epochs, tags, written


def _raise_if_any(errors, what):
    """Fail every rank together: a rank failing alone would leave its peers waiting."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, errors)
    failures = [f"rank {r}: {e}" for r, found in enumerate(gathered) for e in found]
    if failures:
        raise AssertionError(f"{what}: " + "; ".join(failures[:8]))


def _zone_audit(
    before, after, *, rank, launches, units, adopted, poisoned, direct_sources=()
):
    """Epoch and zone audit of this rank over ``launches`` direct launches.

    Every CTA advanced its epoch by ``launches``. A ZR word that changed holds
    an epoch of these launches, under a zone program only, and never for a
    stream from one of ``direct_sources`` (an unaligned rank, or a late rank
    whose destinations had all entered: it sends straight into ``out`` after
    the barrier that follows its read of the entry token). With ``poisoned``
    (one launch, the zone poisoned before it), exactly the streams whose ZR
    word changed wrote their parity's slot 0, the units of their first trip,
    and nothing else of the zone changed. Returns the errors and the number of
    streams that used the zone.
    """
    epochs0, tags0, _ = before
    epochs1, tags1, written = after
    errors = []
    advanced = [c for c in range(_GRID) if epochs1[c] != epochs0[c] + launches]
    if advanced:
        c = advanced[0]
        errors.append(f"epoch of CTA {c}: {epochs0[c]} -> {epochs1[c]} in {launches}")
    changed = [c for c in range(_GRID) if tags1[c] != tags0[c]]
    for c in changed:
        if not epochs0[c] < tags1[c] <= epochs1[c]:
            errors.append(f"ZR + {c} = {tags1[c]}, epochs ({epochs0[c]}, {epochs1[c]}]")
        if not adopted:
            errors.append(f"ZR + {c} changed under a program without the zone")
        if _zone_source(rank, c) in direct_sources:
            errors.append(
                f"ZR + {c} changed for a stream from rank {_zone_source(rank, c)}"
            )
    if poisoned:
        for c in range(_GRID):
            expected = torch.zeros(2, _ZONE_SLOTS, _ZONE_LANES, dtype=torch.bool)
            if c in changed:
                count = min(
                    max(units - _zone_stream(c)[1] * _CHUNK_UNITS, 0), _ZONE_LANES
                )
                if count == 0:
                    errors.append(f"stream {c} used the zone without a stripe")
                expected[epochs1[c] % 2, 0, :count] = True
            if not torch.equal(written[c], expected):
                errors.append(
                    f"zone of CTA {c}: {int(written[c].sum())} units written, "
                    f"{int(expected.sum())} expected"
                )
    return errors, len(changed)


def _check_zone_tail(case):
    """No program but the zone programs writes the zone: on a Cake
    communicator that runs none of them its words stay zero, and on one that
    does every ZR word holds 0 or an epoch its CTA has completed."""
    communicator = case.communicator
    if communicator._cake is None:
        return
    probe = _ZoneProbe(case)
    errors = []
    dtype, world_size = case.inputs[0].dtype, communicator.world_size
    if any(_zone_program(dtype, world_size, op) for op in _MODE):
        epochs, tags, _ = probe.read()
        errors += [
            f"ZR + {c} = {tags[c]} beyond epoch {epochs[c]}"
            for c in range(_GRID)
            if tags[c] > epochs[c]
        ]
    elif not probe.tail_is_zero():
        errors.append("a program without the zone wrote the zone words")
    _raise_if_any(errors, "zone words")


def _expected(case, index, source=None):
    """The all-gather reference of operand ``index`` (0-2 scatter, 3 gather)."""
    communicator = case.communicator
    world, rank = communicator.world_size, communicator.rank
    source = case.inputs[index] if source is None else source
    peers = [torch.empty_like(source) for _ in range(world)]
    dist.all_gather(peers, source, group=communicator.group)
    if index < 3:
        heads = source.shape[2] // world
        return torch.cat(
            [peer[:, :, rank * heads : (rank + 1) * heads] for peer in peers], dim=1
        )
    sequence = source.shape[1] // world
    return torch.cat(
        [peer[:, rank * sequence : (rank + 1) * sequence] for peer in peers], dim=2
    )


def _launch(case, op):
    """``op`` on the case's operand of that collective, into its output."""
    index = _OPERAND[op]
    source, out = case.inputs[index], case.outputs[index]
    if op == "gather":
        case.communicator.gather_heads(source, out=out)
    else:
        case.communicator.scatter_heads(source, out=out)


def _launch_audit(rows, *, rank, direct_sources=()):
    """Epoch and zone audit of each launch of a back-to-back sequence.

    ``rows[k]`` and ``rows[k + 1]`` hold this rank's (EPOCHS, ZR) words before
    and after launch k: every CTA advanced its epoch by one, and a ZR word
    kept its value or took the launch's epoch, never for a stream from one of
    ``direct_sources``. Returns the errors and, per launch, the number of
    streams that used the zone.
    """
    errors, routed = [], []
    for k in range(len(rows) - 1):
        epochs0, tags0 = ([word & 0xFFFFFFFF for word in row] for row in rows[k])
        epochs1, tags1 = ([word & 0xFFFFFFFF for word in row] for row in rows[k + 1])
        stale = [c for c in range(_GRID) if epochs1[c] != epochs0[c] + 1]
        if stale:
            c = stale[0]
            errors.append(f"launch {k}: CTA {c} epoch {epochs0[c]} -> {epochs1[c]}")
        changed = [c for c in range(_GRID) if tags1[c] != tags0[c]]
        for c in changed:
            source = _zone_source(rank, c)
            if tags1[c] != epochs1[c]:
                errors.append(f"launch {k}: ZR + {c} = {tags1[c]}, epoch {epochs1[c]}")
            if source in direct_sources:
                errors.append(f"launch {k}: ZR + {c} changed for rank {source}")
        routed.append(len(changed))
    return errors, routed


def _zone_trial(case, probe, op, late, step, units, unaligned=()):
    """One direct collective with rank ``late`` entering late (``None``: no delay).

    The late rank refills its output after its delay, right before its launch
    (the entry probe: a store into its output before it entered is lost). Its
    entry poll (world 8) precedes the delay, so its own streams find every
    destination entered (route D with the entry token already seen) and none
    of them uses the zone. The output is checked
    bit-exact by a copy stream-ordered after this rank's launch (kernel_end)
    and after a barrier (post_barrier); the epoch and zone audits follow, the
    zone allowed only to a zone program. Returns every rank's count of
    streams that used the zone.
    """
    communicator = case.communicator
    rank = communicator.rank
    index = _OPERAND[op]
    source, out = case.inputs[index], case.outputs[index]
    _refresh(case, step)
    probe.poison()
    before = probe.read()
    dist.barrier()
    poll = probe.entry_poll() if rank == late else None
    if rank == late:
        if poll is not None:
            poll.enqueue()
        torch.cuda._sleep(_SKEW_CYCLES)
        out.fill_(float("nan"))
    _launch(case, op)
    at_end = out.clone()
    torch.cuda.synchronize()
    dist.barrier()
    after = probe.read()
    expected = _expected(case, index)
    errors = [] if poll is None else poll.errors()
    if not torch.equal(at_end, expected):
        errors.append(f"{op} kernel_end mismatch (late rank {late})")
    if not torch.equal(out, expected):
        errors.append(f"{op} post_barrier mismatch (late rank {late})")
    found, routed = _zone_audit(
        before,
        after,
        rank=rank,
        launches=1,
        units=units,
        adopted=_zone_program(source.dtype, communicator.world_size, op),
        poisoned=True,
        direct_sources=tuple(unaligned)
        + ((late,) if late is not None and late != rank else ()),
    )
    errors += found
    counts = [None] * communicator.world_size
    dist.all_gather_object(counts, routed)
    _raise_if_any(errors, f"{case.inputs[0].shape} {op} step {step}")
    return counts


def _require_zone_route(counts, late, what):
    assert counts[late] > 0, (
        f"VOID {what}: no stream into the late rank {late} used the zone ({counts})"
    )


def _zone_worker(world_size, rank, port, op, other_dtypes):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        partner = _PARTNER[op]
        step = 0
        for index, row in enumerate(_ZONE_ROWS):
            shape = _zone_shape(row)
            units = _stream_units(shape)
            case = prepare(
                shape, backend="nvlink", nvlink_kernel="cake", registered=True
            )
            try:
                probe = _ZoneProbe(case)
                # a new communicator starts on a zeroed zone, and its first
                # launch takes the zone
                _raise_if_any(
                    [] if probe.tail_is_zero() else ["zone words not zero"],
                    f"{row} init",
                )
                late = index % world_size
                plan = [
                    (op, late),
                    (op, None),
                    (partner, late),
                    (op, (late + 3) % world_size),
                    (op, late),
                    (op, late),
                    (op, late),
                    (partner, None),
                ]
                if row == (1, 7681, 8, 8):
                    plan += [
                        (trial_op, other)
                        for other in range(world_size)
                        for trial_op in (op, partner)
                    ]
                for trial_op, late_rank in plan:
                    step += 1
                    counts = _zone_trial(case, probe, trial_op, late_rank, step, units)
                    zoned = _zone_program(torch.bfloat16, world_size, trial_op)
                    if zoned and late_rank is not None and units:
                        _require_zone_route(
                            counts, late_rank, f"{row} {trial_op} step {step}"
                        )
                    if not units:
                        assert not any(counts), f"elements path used the zone {counts}"
            finally:
                case.close()
        if not other_dtypes:
            return
        # the f16 and f32 programs, late rank or not, never use the zone
        for dtype in ("float16", "float32"):
            for index, row in enumerate(((1, 7681, 8, 8), (1, 4095, 40, 128))):
                case = prepare(
                    _zone_shape(row, dtype),
                    backend="nvlink",
                    nvlink_kernel="cake",
                    registered=True,
                )
                try:
                    probe = _ZoneProbe(case)
                    _raise_if_any(
                        [] if probe.tail_is_zero() else ["zone words not zero"],
                        f"{dtype} {row} init",
                    )
                    for late in (None, index, index + 4):
                        for trial_op in ("gather", "scatter"):
                            step += 1
                            counts = _zone_trial(case, probe, trial_op, late, step, 0)
                            assert not any(counts), f"{dtype} used the zone {counts}"
                finally:
                    case.close()
    finally:
        dist.destroy_process_group()


def _zone_graph_worker(world_size, rank, port, op):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        partner = _PARTNER[op]
        step = 1000
        for row in ((1, 7681, 8, 8), (1, 4095, 40, 128)):
            shape = _zone_shape(row)
            units = _stream_units(shape)
            case = prepare(
                shape, backend="nvlink", nvlink_kernel="cake", registered=True
            )
            communicator = case.communicator
            try:
                probe = _ZoneProbe(case)
                late, launches, replays = 5, 4, 3
                source, out = case.inputs[_OPERAND[op]], case.outputs[_OPERAND[op]]
                partner_source = case.inputs[_OPERAND[partner]]
                partner_out = case.outputs[_OPERAND[partner]]
                generator = torch.Generator(device="cuda").manual_seed(4099 + rank)

                def stash(like):
                    return [
                        torch.randn(
                            like.shape,
                            dtype=like.dtype,
                            device="cuda",
                            generator=generator,
                        )
                        for _ in range(launches)
                    ]

                sources, partner_sources = stash(source), stash(partner_source)
                poll = probe.entry_poll() if rank == late else None
                results = [torch.empty_like(out) for _ in range(launches)]
                partner_results = [
                    torch.empty_like(partner_out) for _ in range(launches)
                ]

                def body(mixed):
                    # distinct inputs a launch: a stale zone parity read differs
                    for i in range(launches):
                        if mixed:
                            partner_source.copy_(partner_sources[i])
                            if rank == late:
                                # no delay: its peers' streams into it may
                                # still use the zone, its own find every
                                # destination entered
                                poll.enqueue()
                            _launch(case, partner)
                            partner_results[i].copy_(partner_out)
                        source.copy_(sources[i])
                        if rank == late:
                            poll.enqueue()
                            torch.cuda._sleep(_SKEW_CYCLES)
                            out.fill_(float("nan"))
                        _launch(case, op)
                        results[i].copy_(out)

                for mixed in (False, True):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        body(mixed)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        body(mixed)
                    before = probe.read()
                    dist.barrier()
                    for _ in range(replays):
                        graph.replay()  # back to back, nothing between replays
                    torch.cuda.synchronize()
                    dist.barrier()
                    after = probe.read()
                    errors = [] if poll is None else poll.errors()
                    for i in range(launches):
                        expected = _expected(case, _OPERAND[op], sources[i])
                        if not torch.equal(results[i], expected):
                            errors.append(f"replayed {op} {i} mismatch")
                        if mixed and not torch.equal(
                            partner_results[i],
                            _expected(case, _OPERAND[partner], partner_sources[i]),
                        ):
                            errors.append(f"replayed {partner} {i} mismatch")
                    found, routed = _zone_audit(
                        before,
                        after,
                        rank=rank,
                        launches=replays * launches * (2 if mixed else 1),
                        units=units,
                        adopted=True,
                        poisoned=False,
                        direct_sources=(late,) if rank != late else (),
                    )
                    errors += found
                    counts = [None] * world_size
                    dist.all_gather_object(counts, routed)
                    _raise_if_any(errors, f"{row} graph (mixed {mixed})")
                    _require_zone_route(counts, late, f"{row} graph (mixed {mixed})")
                    del graph
                    torch.cuda.synchronize()
                    dist.barrier()
                # the staged route (caller-owned outputs) alternates with the
                # direct collective on the same signal allocation and leaves
                # the epochs, ZR words and zone alone
                staged = PreparedCase(
                    communicator,
                    case.inputs,
                    tuple(torch.empty_like(output) for output in case.outputs),
                    None,
                    "cake",
                )
                for _ in range(2):
                    step += 1
                    _refresh(staged, step)
                    probe.poison()
                    before = probe.read()
                    dist.barrier()
                    staged.check()
                    after = probe.read()
                    found, _ = _zone_audit(
                        before,
                        after,
                        rank=rank,
                        launches=0,
                        units=units,
                        adopted=False,
                        poisoned=True,
                    )
                    _raise_if_any(found, f"{row} staged step {step}")
                    step += 1
                    counts = _zone_trial(case, probe, op, late, step, units)
                    _require_zone_route(counts, late, f"{row} {op} after staged")
                _check_zone_tail(case)
            finally:
                case.close()
    finally:
        dist.destroy_process_group()


def _zone_unaligned_worker(world_size, rank, port, op):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        unaligned, step = 1, 2000
        for row in ((1, 7681, 8, 8), (1, 4095, 40, 128)):
            shape = _zone_shape(row)
            units = _stream_units(shape)
            batch, sequence, heads, head_dim = row
            local_shape = (batch, sequence, heads, head_dim)
            global_shape = (batch, sequence * world_size, heads // world_size, head_dim)
            offset = 1 if rank == unaligned else 0
            inputs = tuple(
                _offset_input(dimensions, torch.bfloat16, offset)
                for dimensions in (local_shape, local_shape, local_shape, global_shape)
            )
            case = prepare(
                shape,
                backend="nvlink",
                nvlink_kernel="cake",
                inputs=inputs,
                registered=True,
            )
            try:
                probe = _ZoneProbe(case)
                # late = the unaligned rank: its aligned sources take the zone
                # and it copies their units in the block's 16-byte geometry;
                # late = its successor: the unaligned rank's stream into it
                # takes the elements path and never the zone
                for late in (unaligned, (unaligned + 1) % world_size):
                    for trial_op in (op, op, op, _PARTNER[op]):
                        step += 1
                        counts = _zone_trial(
                            case,
                            probe,
                            trial_op,
                            late,
                            step,
                            units,
                            unaligned=(unaligned,),
                        )
                        if _zone_program(torch.bfloat16, world_size, trial_op):
                            _require_zone_route(
                                counts, late, f"{row} unaligned {trial_op}"
                            )
            finally:
                case.close()
    finally:
        dist.destroy_process_group()


# Scatters and gathers back to back on one communicator, with every transition
# between the two collectives: scatter -> gather, gather -> gather,
# gather -> scatter and scatter -> scatter.
_SHARED_OPS = (
    "scatter",
    "gather",
    "gather",
    "scatter",
    "scatter",
    "gather",
    "scatter",
    "gather",
)
_SHARED_ROWS = ((1, 7681, 8, 8), (2, 16, 24, 64), (1, 4095, 40, 128))


def _zone_shared(case, probe, late, step, units, graph):
    """``_SHARED_OPS`` back to back on one communicator: run once (eager), or
    captured in a CUDA graph and replayed back to back (``graph``).

    Each launch takes distinct inputs, so a stale zone parity read differs.
    Before each of its launches the late rank polls, waits and refills that
    output: in every launch its peers' streams into it may use the zone, and
    its own streams find every destination entered. Each launch's output and
    this rank's EPOCHS and ZR words are copied stream-ordered after it. The
    outputs are checked bit-exact, each launch is audited
    (:func:`_launch_audit`), and so is the whole run (:func:`_zone_audit`).
    Each zone program must use the zone into the late rank (VOID otherwise).
    """
    communicator = case.communicator
    rank, world_size = communicator.rank, communicator.world_size
    launches, replays = len(_SHARED_OPS), 3
    generator = torch.Generator(device="cuda").manual_seed(6151 * step + rank)
    sources = [
        torch.randn(
            case.inputs[_OPERAND[op]].shape,
            dtype=case.inputs[_OPERAND[op]].dtype,
            device="cuda",
            generator=generator,
        )
        for op in _SHARED_OPS
    ]
    results = [torch.empty_like(case.outputs[_OPERAND[op]]) for op in _SHARED_OPS]
    # this rank's (EPOCHS, ZR) words before the first launch and after each
    words = torch.zeros(launches + 1, 2, _GRID, dtype=torch.int32, device="cuda")
    poll = probe.entry_poll() if rank == late else None

    def record(k):
        words[k, 0].copy_(probe.words[_EPOCHS : _EPOCHS + _GRID])
        words[k, 1].copy_(probe.words[_ZONE_WORD : _ZONE_WORD + _GRID])

    def body():
        record(0)
        for k, op in enumerate(_SHARED_OPS):
            index = _OPERAND[op]
            case.inputs[index].copy_(sources[k])
            if poll is not None:
                poll.enqueue()
                torch.cuda._sleep(_SKEW_CYCLES)
                case.outputs[index].fill_(float("nan"))
            _launch(case, op)
            results[k].copy_(case.outputs[index])
            record(k + 1)

    captured = None
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            body()
    before = probe.read()
    dist.barrier()
    if captured is None:
        body()
    else:
        for _ in range(replays):
            captured.replay()  # back to back, nothing between replays
    torch.cuda.synchronize()
    dist.barrier()
    after = probe.read()
    direct_sources = (late,) if rank != late else ()
    errors = [] if poll is None else poll.errors()
    for k, op in enumerate(_SHARED_OPS):
        if not torch.equal(results[k], _expected(case, _OPERAND[op], sources[k])):
            errors.append(f"{op} {k} mismatch")
    found, routed = _launch_audit(
        words.tolist(), rank=rank, direct_sources=direct_sources
    )
    errors += found
    found, _ = _zone_audit(
        before,
        after,
        rank=rank,
        launches=launches * (1 if captured is None else replays),
        units=units,
        adopted=True,
        poisoned=False,
        direct_sources=direct_sources,
    )
    errors += found
    by_op = {
        op: sum(n for n, o in zip(routed, _SHARED_OPS, strict=True) if o == op)
        for op in _MODE
    }
    counts = [None] * world_size
    dist.all_gather_object(counts, by_op)
    what = f"{case.inputs[0].shape} {'graph' if graph else 'eager'} step {step}"
    _raise_if_any(errors, what)
    dtype = case.inputs[0].dtype
    for op in _MODE:
        if _zone_program(dtype, world_size, op):
            _require_zone_route([count[op] for count in counts], late, f"{what} {op}")
    if captured is not None:
        del captured
        torch.cuda.synchronize()
        dist.barrier()


def _zone_sharing_worker(world_size, rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        step = 3000
        # each sequence on its own communicator; the last one only opens one
        sequences = [(row, graph) for row in _SHARED_ROWS for graph in (False, True)]
        sequences.append((_SHARED_ROWS[0], None))
        for index, (row, graph) in enumerate(sequences):
            shape = _zone_shape(row)
            units = _stream_units(shape)
            late = (3 * index + 2) % world_size
            case = prepare(
                shape, backend="nvlink", nvlink_kernel="cake", registered=True
            )
            try:
                probe = _ZoneProbe(case)
                # the first launch on a fresh zone after each sequence: the
                # new communicator's zone is zero, and its first launch (the
                # scatter and the gather in turn) takes the zone
                _raise_if_any(
                    [] if probe.tail_is_zero() else ["zone words not zero"],
                    f"{row} fresh zone",
                )
                first = ("scatter", "gather")[index % 2]
                second = _PARTNER[first]
                step += 1
                counts = _zone_trial(case, probe, first, late, step, units)
                _require_zone_route(counts, late, f"{row} first {first}, fresh zone")
                if graph is None:
                    _check_zone_tail(case)
                    continue
                if not graph:
                    # the two alternate with a barrier between launches, the
                    # zone poisoned before each and audited per program
                    other = (late + 4) % world_size
                    plan = [
                        (second, late),
                        (first, late),
                        (first, other),
                        (second, other),
                        (second, late),
                        (first, None),
                        (second, None),
                    ]
                    for op, late_rank in plan:
                        step += 1
                        counts = _zone_trial(case, probe, op, late_rank, step, units)
                        if late_rank is not None:
                            _require_zone_route(
                                counts, late_rank, f"{row} {op} step {step}"
                            )
                step += 1
                _zone_shared(case, probe, late, step, units, graph)
                # each program again on the zone the sequence left
                for op in (first, second):
                    step += 1
                    counts = _zone_trial(case, probe, op, late, step, units)
                    _require_zone_route(counts, late, f"{row} {op} after the sequence")
                _check_zone_tail(case)
            finally:
                case.close()
    finally:
        dist.destroy_process_group()


def _skip_without_zone_route():
    if torch.cuda.device_count() < 8:
        pytest.skip("requires 8 GPUs")
    if torch.cuda.get_device_capability(0) not in pkg.SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("the Cake portfolio targets Blackwell SM100/SM103")
    if not cake_direct_available(8):
        pytest.skip("build.json delivers no world-8 direct-route library yet")


def test_direct_gather_zone_world8():
    """The world-8 bf16 direct gather, with scatters between, plain and with
    one rank entering late (entry probe, kernel_end, post_barrier), on stream
    sizes at the zone slot and chunk boundaries and on stripe-less CTAs:
    bit-exact, the epoch and zone audits hold, a late rank's incoming streams
    use the zone in either collective (none doing so is VOID), and each new
    communicator starts on a zeroed zone; the f16 and f32 programs, checked
    alike in both collectives, never write the zone."""
    _skip_without_zone_route()
    multi_process_parallel(8, _zone_worker, ("gather", True))


def test_direct_gather_zone_graph_replays_world8():
    """Back-to-back CUDA graph replays of gathers with a late rank, alone and
    mixed with scatters, then the staged route alternating with the gather."""
    _skip_without_zone_route()
    multi_process_parallel(8, _zone_graph_worker, ("gather",))


def test_direct_gather_zone_with_one_unaligned_rank_world8():
    """One rank's inputs unaligned, as the late receiver and as a source into
    the late rank: bit-exact, and its own streams never use the zone."""
    _skip_without_zone_route()
    multi_process_parallel(8, _zone_unaligned_worker, ("gather",))


def test_direct_scatter_zone_world8():
    """The world-8 bf16 direct scatter, with gathers between, checked as the
    gather is: plain and with one rank entering late, on the same stream
    sizes, which include several batches in one zone slot (each batch lands
    apart in the scatter's output). The f16 and f32 programs are checked in
    the gather test."""
    _skip_without_zone_route()
    multi_process_parallel(8, _zone_worker, ("scatter", False))


def test_direct_scatter_zone_graph_replays_world8():
    """Back-to-back CUDA graph replays of scatters with a late rank, alone and
    mixed with gathers, then the staged route alternating with the scatter."""
    _skip_without_zone_route()
    multi_process_parallel(8, _zone_graph_worker, ("scatter",))


def test_direct_scatter_zone_with_one_unaligned_rank_world8():
    """One rank's inputs unaligned, as the late receiver of scatters and as a
    source into the late rank: bit-exact, and its own streams never use the
    zone."""
    _skip_without_zone_route()
    multi_process_parallel(8, _zone_unaligned_worker, ("scatter",))


def test_direct_zone_shared_by_scatter_and_gather_world8():
    """The world-8 bf16 scatter and gather share one communicator's zone:
    they alternate with a barrier between launches (the zone poisoned before
    each), back to back eagerly, and in back-to-back replays of one CUDA graph
    holding both, with a late rank throughout; then each runs again on the
    zone the sequence left, and after each sequence a new communicator's first
    launch takes its fresh zone. Every launch is bit-exact and passes the
    epoch and zone audits, and both programs use the zone into the late rank."""
    _skip_without_zone_route()
    multi_process_parallel(8, _zone_sharing_worker, ())


# ---- the package route (CPU) -------------------------------------------------------


@pytest.mark.parametrize(
    "capability,arch",
    [
        ((10, 0), "sm_100a"),
        ((10, 3), "sm_103a"),
        ((9, 0), None),
        ((12, 0), None),
    ],
)
def test_arch_for_capability_names_the_exported_targets(capability, arch):
    assert loader.arch_for_capability(capability) == arch
    assert pkg.arch_for_capability(capability) == arch
    if arch is None:
        assert not loader.route_applies(
            device_capability=capability, dtype="bf16", world_size=2
        )
    else:
        assert arch in loader.ARCHES
        assert capability in pkg.SUPPORTED_COMPUTE_CAPABILITIES


def test_records_cover_the_portfolio_once():
    """One record per (route, dtype, world size, mode), each carrying the host plan of its collective."""
    expected = {
        loader.record_name(dtype, world, mode, direct=direct)
        for direct in (False, True)
        for dtype in loader.DTYPES
        for world in loader.WORLD_SIZES
        for mode in loader.MODES
    }
    assert set(loader.MODULES) == expected
    assert len(loader.MODULES) == 48
    libraries = set()
    for name, record in loader.MODULES.items():
        direct = name.startswith("direct_")
        direction = "scatter" if record["mode"] == 0 else "gather"
        assert name == (
            f"{'direct_' if direct else ''}{direction}_{record['dtype']}_w{record['world_size']}"
        )
        assert (
            loader.record_for(
                record["dtype"], record["world_size"], record["mode"], direct=direct
            )
            is record
        )
        assert record["library"] == record["kernel"]
        assert re.fullmatch(r"cake_ulysses_a2a_[0-9a-f]{20}", record["library"])
        assert record["kernel_symbol"] == f"kernel_{record['library']}"
        libraries.add(record["library"])
        assert list(record["arches"]) == list(loader.ARCHES)
        assert record["ffi_entry"] == "run"
        launch = record["launch"]
        assert tuple(launch["block"]) == (512, 1, 1)
        assert tuple(launch["cluster"]) == (1, 1, 1)
        assert launch["dynamic_smem_bytes"] == 0
        assert launch["use_pdl"] is False and launch["cooperative"] is False
        assert loader.launch_grid(record) == (148, 1, 1)
        plan = [tuple(pair) for pair in record["arg_plan"]]
        assert plan[-3:] == [("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]
        kinds = {kind for kind, _ in plan[:-3]}
        assert kinds <= {"buffer", "parameter", "peer_table"}
        logical = record["logical"]
        assert {key for _, key in plan[:-3]} == set(logical)
        vocabulary = (
            set(loader.LOGICAL_TENSORS)
            | set(loader.LOGICAL_PEER_TABLES)
            | set(loader.LOGICAL_SCALARS)
        )
        assert set(logical.values()) <= vocabulary
        for kind, key in plan[:-3]:
            if kind == "peer_table":
                assert logical[key] in loader.LOGICAL_PEER_TABLES
            elif kind == "buffer":
                assert logical[key] in loader.LOGICAL_TENSORS
            else:
                assert logical[key] in loader.LOGICAL_SCALARS
        # the staged route reads the staging tables, the direct route the output's
        tables = [logical[key] for kind, key in plan if kind == "peer_table"]
        assert tables == [
            "out_peers" if direct else "staging_peers",
            "signals_peers",
        ]
    # every program is one (route, dtype, world, mode): no two records share a library
    assert len(libraries) == 48
    with pytest.raises(ValueError):
        loader.record_name("f64", 2, 0)
    with pytest.raises(ValueError):
        loader.record_name("bf16", 3, 0)
    with pytest.raises(ValueError):
        loader.record_name("bf16", 2, 2)


def test_plan_arguments_follow_the_record():
    record = loader.record_for("bf16", 4, 1)
    values = {
        "inp": "INP",
        "out": "OUT",
        "staging_peers": "STAGING",
        "signals_peers": "SIGNALS",
        "signals_self": "SELF",
        "rank": 3,
        "B": 1,
        "S_local": 5,
        "H_local": 6,
        "D": 128,
    }
    args = loader.plan_arguments(record, values)
    assert args == [
        "INP",
        "OUT",
        "STAGING",
        "SIGNALS",
        "SELF",
        3,
        1,
        5,
        6,
        128,
        148,
        1,
        1,
    ]
    with pytest.raises(ValueError, match="needs 'out'"):
        loader.plan_arguments(
            record, {key: value for key, value in values.items() if key != "out"}
        )
    direct = loader.record_for("bf16", 4, 1, direct=True)
    with pytest.raises(ValueError, match="needs 'out_peers'"):
        loader.plan_arguments(direct, values)
    args = loader.plan_arguments(direct, {**values, "out_peers": "OUTPEERS"})
    assert args == [
        "INP",
        "OUT",
        "OUTPEERS",
        "SIGNALS",
        "SELF",
        3,
        1,
        5,
        6,
        128,
        148,
        1,
        1,
    ]


def test_peer_tables_and_signal_constants():
    table = backend.peer_table([11, 22])
    assert table.dtype == torch.int64 and table.device.type == "cpu"
    assert table.is_contiguous() and table.numel() == loader.PEER_TABLE_CAPACITY == 8
    assert table.tolist() == [11, 22, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        backend.peer_table([11, 0])
    with pytest.raises(ValueError):
        backend.peer_table(list(range(1, 10)))
    assert loader.SIGNAL_WORDS == 1138688 and loader.SIGNAL_BYTES == 4554752
    assert pkg.SIGNAL_BYTES == loader.SIGNAL_BYTES
    # the zone (ZR words, then two parities x ZONE_SLOTS slots of 480 16-byte
    # units a CTA) ends the signal allocation
    assert _ZONE_WORD + _GRID <= _ZONE_DATA
    assert _ZONE_DATA + _GRID * 2 * _ZONE_SLOTS * _ZONE_LANES * 4 == loader.SIGNAL_WORDS
    assert [_zone_stream(c)[0] for c in range(_GRID)].count(0) == 22
    assert {_zone_source(0, c) for c in range(_GRID)} == set(range(1, 8))
    # CTA c of a source streams into the receiver whose incoming stream c it
    # is, in the scatter and the gather alike: the late rank's entry poll waits
    # on the destinations of its streams of either program
    assert all(
        _zone_destination(_zone_source(receiver, c), c) == receiver
        for receiver in range(8)
        for c in range(_GRID)
    )
    # the zone programs are direct programs of the portfolio, bf16 at world 8
    for dtype, world_size, op in _ZONE_PROGRAMS:
        assert dtype in loader.DTYPES and world_size in loader.WORLD_SIZES
        assert _MODE[op] in loader.MODES
    assert {op for _, _, op in _ZONE_PROGRAMS} == set(_MODE)
    assert all(_zone_program(torch.bfloat16, 8, op) for op in _MODE)
    assert not any(
        _zone_program(dtype, world_size, op)
        for dtype in (torch.bfloat16, torch.float16, torch.float32)
        for world_size in (2, 4, 6)
        for op in _MODE
    )
    assert not any(
        _zone_program(dtype, 8, op)
        for dtype in (torch.float16, torch.float32)
        for op in _MODE
    )
    assert backend.dtype_name(torch.bfloat16) == "bf16"
    assert backend.dtype_name(torch.float16) == "f16"
    assert backend.dtype_name(torch.float32) == "f32"
    with pytest.raises(ValueError):
        backend.dtype_name(torch.int32)


def test_route_requires_delivered_libraries(monkeypatch):
    """A route exists only where the manifest delivers both the scatter and the gather library."""
    scatter = loader.library_name(loader.record_name("bf16", 2, 0), "sm_100a")
    gather = loader.library_name(loader.record_name("bf16", 2, 1), "sm_100a")
    monkeypatch.setattr(cake_build, "libraries_by_name", lambda: {})
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2
    )
    monkeypatch.setattr(cake_build, "libraries_by_name", lambda: {scatter: {}})
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2
    )
    monkeypatch.setattr(
        cake_build, "libraries_by_name", lambda: {scatter: {}, gather: {}}
    )
    assert loader.route_applies(device_capability=(10, 0), dtype="bf16", world_size=2)
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=4
    )
    assert not loader.route_applies(
        device_capability=(10, 3), dtype="bf16", world_size=2
    )
    assert not loader.route_applies(
        device_capability=(9, 0), dtype="bf16", world_size=2
    )
    with pytest.raises(RuntimeError):
        loader.library_name(loader.record_name("bf16", 2, 0), "sm_90a")
    # the direct route needs its own two libraries
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2, direct=True
    )
    direct = {
        loader.library_name(
            loader.record_name("bf16", 2, mode, direct=True), "sm_100a"
        ): {}
        for mode in loader.MODES
    }
    monkeypatch.setattr(cake_build, "libraries_by_name", lambda: direct)
    assert loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2, direct=True
    )
    assert not loader.route_applies(
        device_capability=(10, 0), dtype="bf16", world_size=2
    )


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only.
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES)
    package = Path(cake_build.__file__).parent
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
    libraries = cake_build.libraries_by_name()
    if not libraries:
        pytest.skip("build.json delivers no library yet (the export run fills it)")
    referenced = set()
    for name, record in loader.MODULES.items():
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)


def test_the_zone_program_pairs_with_the_signal_allocation():
    """The zone programs, the world-8 bf16 direct scatter and gather, are the
    programs that use the zone: each one's kernel source stores zone units
    from word ZONE_DATA and ZR words from ZONE_WORD, and no other kernel
    source does. The manifest delivers the sources it was built from (digests
    checked for every entry), and the package's signal allocation
    (SIGNAL_WORDS) ends at the zone's end. No record launches with
    programmatic dependent launch or cooperatively, no kernel waits on a grid
    dependency and every binding launches without launch attributes, so
    consecutive launches on a stream never overlap."""
    package = Path(cake_build.__file__).parent
    libraries = cake_build.libraries_by_name()
    if not libraries:
        pytest.skip("build.json delivers no library yet (the export run fills it)")
    zones = {
        loader.record_name(dtype, world_size, _MODE[op], direct=True)
        for dtype, world_size, op in _ZONE_PROGRAMS
    }
    assert zones == {"direct_scatter_bf16_w8", "direct_gather_bf16_w8"}
    assert zones <= set(loader.MODULES)
    marks = (
        f"peer_signals + ({_ZONE_DATA} + ",
        f"signals_self + ({_ZONE_DATA} + ",
        f"peer_signals) + ({_ZONE_WORD} + bid_1)",
        f"signals_self + ({_ZONE_WORD} + bid_1)",
    )
    for name, record in loader.MODULES.items():
        kernel = (package / "csrc" / f"{record['kernel']}_kernel.cu").read_text()
        binding = (package / "csrc" / f"{record['kernel']}_binding.cu").read_text()
        found = [mark for mark in marks if mark in kernel]
        assert found == (list(marks) if name in zones else []), (name, found)
        assert "griddepcontrol" not in kernel, name
        assert "cudaLaunchKernelExC" in binding, name
        assert "attrs" not in binding and "numAttrs" not in binding, name
        launch = record["launch"]
        assert launch["use_pdl"] is False and launch["cooperative"] is False, name
        for arch in record["arches"]:
            entry = libraries[loader.library_name(name, arch)]
            for source in entry["sources"]:
                digest = hashlib.sha256((package / source).read_bytes()).hexdigest()
                assert entry["sha256"][source] == digest, (name, arch, source)
    assert loader.SIGNAL_WORDS == 1138688
    assert loader.SIGNAL_WORDS == (
        _ZONE_DATA + _GRID * 2 * _ZONE_SLOTS * _ZONE_LANES * 4
    )
    assert pkg.SIGNAL_BYTES == loader.SIGNAL_BYTES == 4 * loader.SIGNAL_WORDS


# ---- the package's view of the signal allocation (one GPU) --------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_signal_view_aliases_the_allocation_without_a_copy():
    device_index = torch.cuda.current_device()
    buffer = torch.zeros(loader.SIGNAL_WORDS, dtype=torch.uint32, device="cuda")
    view = backend.signal_view(buffer.data_ptr(), device_index)
    assert view.dtype == torch.uint32 and view.shape == (loader.SIGNAL_WORDS,)
    assert view.is_contiguous() and view.device == buffer.device
    assert view.data_ptr() == buffer.data_ptr()
    view.fill_(7)
    torch.cuda.synchronize()
    assert torch.equal(buffer, view)
    view.zero_()
    torch.cuda.synchronize()
    assert int(buffer.sum()) == 0
    del view
    with pytest.raises(ValueError):
        backend.signal_view(0, device_index)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_device_view_aliases_an_output_allocation(dtype):
    device_index = torch.cuda.current_device()
    buffer = torch.randn(1000, device="cuda").to(dtype)
    view = backend.device_view(buffer.data_ptr(), device_index, dtype, 640)
    assert view.dtype == dtype and view.shape == (640,) and view.is_contiguous()
    assert view.data_ptr() == buffer.data_ptr() and view.device == buffer.device
    assert torch.equal(view, buffer[:640])
    view.fill_(3)
    torch.cuda.synchronize()
    assert torch.equal(buffer[:640], torch.full_like(view, 3))
    with pytest.raises(ValueError):
        backend.device_view(buffer.data_ptr(), device_index, torch.int64, 8)
    with pytest.raises(ValueError):
        backend.device_view(buffer.data_ptr(), device_index, dtype, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_device_view_structures_outlive_a_base_dropped_under_inference_mode():
    """A view narrowed under inference mode records no base: the DLPack structures live with the storage."""
    device_index = torch.cuda.current_device()
    buffer = torch.zeros(1024, device="cuda", dtype=torch.bfloat16)

    def narrowed():
        with torch.inference_mode():
            view = backend.device_view(
                buffer.data_ptr(), device_index, torch.bfloat16, 1024
            )
            return view.narrow(0, 0, 512).view(16, 32)

    out = narrowed()
    gc.collect()
    # refill freed small Python allocations with nonzero bytes: structures freed
    # with the base would now hold a garbage deleter
    churn = [(ctypes.c_ubyte * 64).from_buffer_copy(b"\xff" * 64) for _ in range(4096)]
    with torch.inference_mode():
        out.fill_(5)
    torch.cuda.synchronize()
    assert int((buffer[:512] == 5).sum()) == 512
    assert int(buffer[512:].count_nonzero()) == 0
    del out
    gc.collect()  # releases the storage through the view's deleter
    del churn
    assert int((buffer[:512] == 5).sum()) == 512
