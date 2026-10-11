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

Cake backend of :class:`flashinfer.comm.QkvNormRopeExchangeOverlap`: the fused
QKV projection + per-head RMSNorm + RoPE producer that stores every head band
straight into the destination rank's symmetric receive slot (SM90 WGMMA, SM100
/ SM103 tcgen05), the one-CTA readiness wait kernel and the two-phase NVLink
barrier, driven from Python one launcher call per stage.

Per call (current stream = producer stream):

1. wait for the slot's previous exchange (``exchange_done[slot]``; eager only),
2. barrier kernel (phase alternating per call) and a ``barrier_done`` record,
3. producer kernel (persistent grid ``min(num_tiles, sm_count - 2)``), which
   publishes ``ready_target`` into every destination's readiness pad per chunk,
4. consumer stream: wait ``barrier_done``, wait kernel over this call's
   ``W * num_words`` words, record ``exchange_done[slot]``,
5. the current stream waits ``exchange_done[slot]`` so the outputs are
   consumable in stream order.

Eager calls publish a monotonic epoch; a call issued while the current stream
is capturing publishes the sentinel ``0xFFFFFFFF``, clears its slot's words
with one stream memset after the wait kernel, uses capture-local events and
never waits on an eager predecessor (the capturing caller orders its replays).
Nothing allocates, synchronises, builds or runs a collective inside a capture:
the session prepares its symmetric slots, readiness pad, barrier flags, JIT
programs and the tile plan of the sample rows eagerly, and refuses a row count
it has not seen (or a row count above ``max_tokens``) inside a capture.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import tvm_ffi

from ..jit import cake_qkv_norm_rope_exchange_overlap as loader
from . import cake_qkv_exchange_plan as plans

CAPTURE_READY_TARGET = 0xFFFFFFFF
EAGER_READY_TARGET_MAX = 0xFFFFFFFE
# SMs the persistent producer leaves free: one for the wait kernel, one slack.
RESERVED_SMS = 2
# torch's default CUDA symmetric-memory backend provides the peer buffer
# pointers, ``memset32_`` and the rendezvous this backend needs.
SUPPORTED_SYMMETRIC_MEMORY_BACKENDS = ("CUDA",)


def symmetric_memory_backend(device: torch.device) -> str:
    """Upper-case name of torch's symmetric-memory backend for ``device`` (empty when none)."""

    name = symm_mem.get_backend(device)
    return "" if name is None else str(name).upper()


def supports_symmetric_memory_backend(device: torch.device) -> bool:
    """Whether torch's symmetric-memory backend for ``device`` can serve this backend (unset is selectable)."""

    return symmetric_memory_backend(device) in (
        "",
        *SUPPORTED_SYMMETRIC_MEMORY_BACKENDS,
    )


def device_arch(device: torch.device) -> Optional[str]:
    """Exported architecture of ``device`` or ``None`` when the Cake programs do not target it."""

    index = device.index if device.index is not None else torch.cuda.current_device()
    return loader.device_facts(int(index)).arch


def _group_name(group: dist.ProcessGroup) -> str:
    name = getattr(group, "group_name", None)
    if not isinstance(name, str) or not name:
        raise RuntimeError("the process group does not expose a non-empty group_name")
    return name


def _peer_pointers(
    handle: Any, world_size: int, numel: int, dtype: torch.dtype
) -> list[int]:
    return [
        int(handle.get_buffer(peer, (int(numel),), dtype, 0).data_ptr())
        for peer in range(int(world_size))
    ]


def _rendezvous(
    tensor: torch.Tensor, *, group_name: str, rank: int, world_size: int, what: str
):
    handle = symm_mem.rendezvous(tensor, group=group_name)
    if int(handle.rank) != rank or int(handle.world_size) != world_size:
        raise RuntimeError(
            f"{what}: symmetric-memory topology does not match the process group"
        )
    if int(handle.buffer_size) < tensor.numel() * tensor.element_size():
        raise RuntimeError(
            f"{what}: symmetric-memory rendezvous returned an undersized handle"
        )
    return handle


@dataclass
class _RowPlan:
    policy: plans.Policy
    plan: plans.TilePlan
    tiles: torch.Tensor
    grid: int
    program: str


class CakeQkvExchangeSession:
    """One prepared session on one NCCL process group (collective constructor; see the module docstring)."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        device: torch.device,
        hidden: int,
        num_heads: int,
        head_dim: int,
        layout: str,
        max_tokens: int,
        num_slots: int,
        sample_batch: Optional[int] = None,
        sample_s_local: Optional[int] = None,
    ) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the fused QKV exchange session is collective and cannot be created inside a capture"
            )
        if str(dist.get_backend(group)).lower() != "nccl":
            raise ValueError("group must use the NCCL backend")
        if device.type != "cuda":
            raise ValueError(f"the Cake backend needs a CUDA device, got {device}")
        self.device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        self.device_index = int(self.device.index)
        arch = device_arch(self.device)
        if arch is None:
            raise ValueError(
                f"device {self.device} (capability {loader.device_facts(self.device_index).capability}) is not an "
                f"architecture the Cake programs target ({sorted(loader.CAPABILITY_ARCH.values())})"
            )
        if not supports_symmetric_memory_backend(self.device):
            raise RuntimeError(
                f"torch symmetric-memory backend {symmetric_memory_backend(self.device)!r} is not supported; the Cake "
                f"backend needs one of {SUPPORTED_SYMMETRIC_MEMORY_BACKENDS}"
            )
        self.arch = arch
        self.group = group
        self.group_name = _group_name(group)
        self.rank = int(dist.get_rank(group=group))
        self.world_size = int(dist.get_world_size(group=group))
        if not 0 <= self.rank < self.world_size:
            raise RuntimeError(
                f"process group reports rank {self.rank} outside [0, {self.world_size})"
            )
        if self.world_size not in loader.SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"world size must be one of {loader.SUPPORTED_WORLD_SIZES}, got {self.world_size}"
            )
        if layout not in plans.LAYOUTS:
            raise ValueError(f"layout must be one of {plans.LAYOUTS}, got {layout!r}")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
        if (
            isinstance(num_slots, bool)
            or not isinstance(num_slots, int)
            or num_slots < 2
        ):
            raise ValueError(
                f"num_slots must be an int of at least 2, got {num_slots!r}"
            )
        if int(num_heads) * int(head_dim) != int(hidden):
            raise ValueError(
                f"hidden={hidden} must equal num_heads * head_dim = {num_heads} * {head_dim}"
            )
        # Any row count up to max_tokens shares the band width; the geometry of max_tokens rows sizes the slots.
        capacity = plans.Geometry(
            world_size=self.world_size,
            batch=1,
            s_local=int(max_tokens),
            hidden=int(hidden),
            head_dim=int(head_dim),
        )
        self.hidden, self.num_heads, self.head_dim = (
            int(hidden),
            int(num_heads),
            int(head_dim),
        )
        self.layout = layout
        self.max_tokens = int(max_tokens)
        self.num_slots = int(num_slots)
        self.bn = plans.block_n_for(capacity)
        self.slot_stride = capacity.slot_elems
        self.words_cap = plans.words_capacity_for_rows(self.max_tokens)
        self.sm_count = loader.device_facts(self.device_index).sm_count
        if symmetric_memory_backend(self.device) == "":
            symm_mem.set_backend("CUDA")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            symm_mem.enable_symm_mem_for_group(self.group_name)

        # Programs (JIT build is collective-safe only outside the barrier: every rank builds before any launch).
        self.pair_available = {
            bn: loader.has_producer(self.arch, bn, True) for bn in plans.BLOCK_N_CHOICES
        }
        self._barrier = tuple(
            loader.load(loader.barrier_program(self.arch, phase), self.arch)
            for phase in (0, 1)
        )
        self._barrier_names = tuple(
            loader.barrier_program(self.arch, phase) for phase in (0, 1)
        )
        self._wait_name = loader.wait_program(self.arch)
        self._wait = loader.load(self._wait_name, self.arch)
        self._producers: dict[str, Any] = {}
        for pair in self._reachable_pairs():
            name = loader.producer_program(self.arch, self.bn, pair)
            self._producers[name] = loader.load(name, self.arch)

        # Symmetric memory: receive slots, readiness pad, barrier flags (rendezvous = collective).
        words = self.num_slots * self.world_size * self.words_cap
        flag_words = loader.barrier_flag_words(self.world_size)
        self._recv = symm_mem.empty(
            self.num_slots * self.slot_stride, dtype=torch.bfloat16, device=self.device
        )
        self._ready = symm_mem.empty(words, dtype=torch.uint32, device=self.device)
        self._flags = symm_mem.empty(flag_words, dtype=torch.uint32, device=self.device)
        self._recv.zero_()
        self._ready.zero_()
        self._flags.zero_()

        def rendezvous(tensor: torch.Tensor, what: str):
            return _rendezvous(
                tensor,
                group_name=self.group_name,
                rank=self.rank,
                world_size=self.world_size,
                what=what,
            )

        self._recv_handle = rendezvous(self._recv, "receive slots")
        self._ready_handle = rendezvous(self._ready, "readiness pad")
        self._flags_handle = rendezvous(self._flags, "barrier flags")
        self.dst_ptrs = torch.tensor(
            _peer_pointers(
                self._recv_handle, self.world_size, self._recv.numel(), torch.bfloat16
            ),
            dtype=torch.int64,
            device=self.device,
        )
        self.ready_ptrs = torch.tensor(
            _peer_pointers(self._ready_handle, self.world_size, words, torch.uint32),
            dtype=torch.int64,
            device=self.device,
        )
        self.flag_peers = tvm_ffi.Shape(
            _peer_pointers(
                self._flags_handle, self.world_size, flag_words, torch.uint32
            )
        )
        self.counters = torch.zeros(
            self.world_size * self.words_cap, dtype=torch.uint32, device=self.device
        )
        # The captured-call clear needs ``memset32_`` on the dedicated pad; probe it once with a rank-uniform verdict.
        self.memset32_probe = self._probe_memset32(words)
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=group)

        self._consumer = torch.cuda.Stream(device=self.device)
        self._barrier_done = torch.cuda.Event()
        self._exchange_done = [torch.cuda.Event() for _ in range(self.num_slots)]
        self._capture_done: dict[int, torch.cuda.Event] = {}
        self._lock = threading.Lock()
        self._phase = 0
        self._epoch = 0
        self._next_slot = 0
        self._calls = 0
        self._captured_calls = 0
        self._poisoned = False
        self._closed = False
        self._plans: dict[tuple[int, int], _RowPlan] = {}
        self._last: Optional[tuple[int, _RowPlan]] = None
        if sample_batch is not None and sample_s_local is not None:
            self.prepare_rows(int(sample_batch), int(sample_s_local))

    # ------------------------------------------------------------------ preparation

    def _reachable_pairs(self) -> tuple[bool, ...]:
        """Pair flags the policy can select for row counts up to ``max_tokens`` (programs to load eagerly)."""

        if self.arch not in plans.PAIR_ARCHES:
            return (False,)
        if self.bn == plans.PAIR_BN:
            return (True,) if self.pair_available[self.bn] else (False,)
        pairs = {False} if self.max_tokens < plans.PAIR_MIN_M else set()
        if self.max_tokens >= plans.PAIR_MIN_M:
            pairs.add(bool(self.pair_available[self.bn]))
        return tuple(sorted(pairs))

    def _probe_memset32(self, words: int) -> dict[str, Any]:
        local_error: Optional[str] = None
        try:
            torch.ops.symm_mem.memset32_(self._ready, 0, 0, int(words))
        except Exception as error:  # noqa: BLE001 - the verdict is published to every rank below
            local_error = f"{type(error).__name__}: {error}"[:500]
        verdicts: list[Any] = [None] * self.world_size
        dist.all_gather_object(
            verdicts, {"rank": self.rank, "error": local_error}, group=self.group
        )
        rejecting = {
            int(v["rank"]): v["error"] for v in verdicts if v.get("error") is not None
        }
        if rejecting:
            raise RuntimeError(
                "torch.ops.symm_mem.memset32_ rejected the dedicated readiness pad on ranks "
                f"{sorted(rejecting)}: {rejecting[sorted(rejecting)[0]]}; the captured-call clear needs it"
            )
        return {"accepted": True, "words": int(words)}

    def geometry(self, batch: int, s_local: int) -> plans.Geometry:
        return plans.Geometry(
            world_size=self.world_size,
            batch=int(batch),
            s_local=int(s_local),
            hidden=self.hidden,
            head_dim=self.head_dim,
        )

    def prepare_rows(self, batch: int, s_local: int) -> _RowPlan:
        """Build and upload the tile plan of ``batch * s_local`` rows (eager; cached per ``(batch, s_local)``)."""

        key = (int(batch), int(s_local))
        row = self._plans.get(key)
        if row is not None:
            return row
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"the fused QKV exchange session has no tile plan for batch={batch}, s_local={s_local}; a row count "
                "first seen inside a CUDA-graph capture would allocate -- call the session eagerly once with that shape"
            )
        geometry = self.geometry(batch, s_local)
        if geometry.rows > self.max_tokens:
            raise ValueError(
                f"batch * s_local = {geometry.rows} exceeds the session's max_tokens={self.max_tokens}"
            )
        if geometry.slot_elems > self.slot_stride:
            raise ValueError(
                "row geometry exceeds the receive slot capacity of the session"
            )
        policy, plan = plans.plan_for(
            geometry,
            self.layout,
            self.rank,
            arch=self.arch,
            pair_available=self.pair_available[self.bn],
            words_capacity_per_dest=self.words_cap,
        )
        if plan.bn != self.bn or plan.num_words > self.words_cap:
            raise RuntimeError("tile plan disagrees with the session capacity")
        program = loader.producer_program(self.arch, plan.bn, plan.pair)
        if program not in self._producers:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"program {program!r} is not loaded; a JIT build cannot run inside a capture"
                )
            self._producers[program] = loader.load(program, self.arch)
        row = _RowPlan(
            policy=policy,
            plan=plan,
            tiles=plan.tiles.to(self.device),
            grid=plans.producer_grid(
                plan.num_tiles,
                sm_count=self.sm_count,
                reserved_sms=RESERVED_SMS,
                lockstep=plan.lockstep,
            ),
            program=program,
        )
        self._plans[key] = row
        return row

    def bound_programs(self, batch: int, s_local: int) -> dict[str, str]:
        """Program names the stages of a ``(batch, s_local)`` call launch (prepares the row eagerly)."""

        row = self.prepare_rows(batch, s_local)
        return {
            "barrier_p0": self._barrier_names[0],
            "barrier_p1": self._barrier_names[1],
            "producer": row.program,
            "wait": self._wait_name,
        }

    # ------------------------------------------------------------------ calls

    def _validate(self, x, w_qkv, q_scale, k_scale, cos_sin) -> tuple[int, int]:
        for name, tensor in (
            ("x", x),
            ("w_qkv", w_qkv),
            ("q_norm_weight", q_scale),
            ("k_norm_weight", k_scale),
            ("cos_sin", cos_sin),
        ):
            if not isinstance(tensor, torch.Tensor) or tensor.device != self.device:
                raise ValueError(f"{name} must be a tensor on {self.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if x.dtype != torch.bfloat16 or x.dim() != 3 or int(x.shape[2]) != self.hidden:
            raise ValueError(
                f"x must be BF16 [B, S_local, {self.hidden}], got {tuple(x.shape)} {x.dtype}"
            )
        batch, s_local = int(x.shape[0]), int(x.shape[1])
        if batch <= 0 or s_local <= 0:
            raise ValueError("x must have positive batch and sequence dimensions")
        if w_qkv.dtype != torch.bfloat16 or tuple(w_qkv.shape) != (
            3 * self.hidden,
            self.hidden,
        ):
            raise ValueError(
                f"w_qkv must be BF16 [{3 * self.hidden}, {self.hidden}], got {tuple(w_qkv.shape)} {w_qkv.dtype}"
            )
        for name, scale in (("q_norm_weight", q_scale), ("k_norm_weight", k_scale)):
            if scale.dtype != torch.bfloat16 or tuple(scale.shape) != (self.head_dim,):
                raise ValueError(
                    f"{name} must be BF16 [{self.head_dim}], got {tuple(scale.shape)} {scale.dtype}"
                )
        rows = batch * s_local
        if cos_sin.dtype != torch.float32 or tuple(cos_sin.shape) != (
            rows,
            self.head_dim,
        ):
            raise ValueError(
                f"cos_sin must be fp32 [{rows}, {self.head_dim}], got {tuple(cos_sin.shape)} {cos_sin.dtype}"
            )
        if rows > self.max_tokens:
            raise ValueError(
                f"batch * s_local = {rows} exceeds the session's max_tokens={self.max_tokens}"
            )
        return batch, s_local

    def __call__(self, x, w_qkv, q_scale, k_scale, cos_sin) -> int:
        """Run one exchange; returns the slot the outputs live in (see :meth:`outputs`)."""

        if self._closed:
            raise RuntimeError("the fused QKV exchange session is closed")
        batch, s_local = self._validate(x, w_qkv, q_scale, k_scale, cos_sin)
        capturing = bool(torch.cuda.is_current_stream_capturing())
        row = self._plans.get((batch, s_local))
        if row is None:
            row = self.prepare_rows(batch, s_local)  # raises inside a capture
        plan = row.plan
        with self._lock:
            if self._poisoned:
                raise RuntimeError(
                    "the fused QKV exchange session is poisoned by a prior failed collective"
                )
            slot = self._next_slot
            if capturing:
                if slot in self._capture_done:
                    raise RuntimeError(
                        f"at most num_slots={self.num_slots} calls of this session can be captured into one graph "
                        "without an eager call in between (the slot's capture-local event would be reused)"
                    )
                ready_target = CAPTURE_READY_TARGET
            else:
                if self._epoch >= EAGER_READY_TARGET_MAX:
                    raise RuntimeError(
                        "the eager readiness epoch exhausted the uint32 range"
                    )
                ready_target = self._epoch + 1
            phase = self._phase
            stream = torch.cuda.current_stream(self.device)
            consumer = self._consumer
            # Readiness words of one call: ``world_size`` contiguous pads of ``plan.num_words`` words
            # (the wait kernel scans ``words_total`` words from ``slot_base``).  Slots are spaced by the
            # session capacity, not by this call's ``num_words``: consecutive calls with different row
            # shapes may be in flight on different slots, and a shape-dependent stride would let a
            # smaller call's slot overlap a larger call's words.
            words_total = self.world_size * plan.num_words
            slot_base = slot * self.world_size * self.words_cap
            pad_base = slot_base + self.rank * plan.num_words
            try:
                if capturing:
                    barrier_done = torch.cuda.Event()
                    done = torch.cuda.Event()
                else:
                    stream.wait_event(self._exchange_done[slot])
                    barrier_done = self._barrier_done
                    done = self._exchange_done[slot]
                with tvm_ffi.use_torch_stream():
                    self._barrier[phase].run(
                        self.device_index,
                        self.world_size,
                        self.rank,
                        self.flag_peers,
                        1,
                        1,
                        1,
                    )
                    barrier_done.record(stream)
                    self._producers[row.program].run(
                        x.view(plan.M, plan.K),
                        w_qkv,
                        cos_sin,
                        q_scale,
                        k_scale,
                        row.tiles,
                        self.dst_ptrs,
                        self.ready_ptrs,
                        self.counters,
                        plan.num_tiles,
                        plan.M,
                        plan.K,
                        s_local,
                        plan.head_stride,
                        plan.batch_stride,
                        slot * self.slot_stride,
                        pad_base,
                        plan.num_words,
                        self.rank,
                        self.world_size,
                        ready_target,
                        row.grid,
                        1,
                        1,
                    )
                    consumer.wait_event(barrier_done)
                    with torch.cuda.stream(consumer):
                        self._wait.run(
                            self._ready, words_total, slot_base, ready_target, 1, 1, 1
                        )
                        if capturing:
                            # A replay re-issues the sentinel: clear this call's words once the wait kernel consumed them.
                            torch.ops.symm_mem.memset32_(
                                self._ready, slot_base, 0, words_total
                            )
                        done.record(consumer)
                stream.wait_event(done)
            except Exception:
                # A partially submitted collective cannot be retried with either barrier phase or the slot.
                self._poisoned = True
                raise
            self._phase = 1 - phase
            self._next_slot = (slot + 1) % self.num_slots
            self._calls += 1
            if capturing:
                self._capture_done[slot] = done
                self._captured_calls += 1
            else:
                self._epoch = ready_target
                self._capture_done.clear()
            self._last = (slot, row)
            return slot

    # ------------------------------------------------------------------ outputs and state

    @property
    def recv_local(self) -> torch.Tensor:
        return self._recv

    def outputs(self, slot: int, row: _RowPlan):
        """Views of ``slot`` for the row geometry: ``(q, k, v)`` for ``"bshd"``, one payload for ``"fused"``."""

        plan = row.plan
        base = int(slot) * self.slot_stride
        region = self._recv[base : base + plan.slot_elems]
        if self.layout == "bshd":
            shape = (plan.batch, plan.s_global, plan.h_local, plan.head_dim)
            return tuple(
                region[index * plan.plane : (index + 1) * plan.plane].view(shape)
                for index in range(3)
            )
        return region.view(plan.batch, plan.s_global, plan.h_local, 3 * plan.head_dim)

    def last_outputs(self):
        if self._last is None:
            raise RuntimeError("no call has run on this session yet")
        slot, row = self._last
        return self.outputs(slot, row)

    def poison_receive_slots(self) -> None:
        """Fill this rank's whole receive region with ``0xFF`` bytes (eager only; proves the next call rewrites it)."""

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("poison_receive_slots must not be captured")
        self._recv.view(torch.int16).fill_(-1)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "nvlink",
            "arch": self.arch,
            "world_size": self.world_size,
            "rank": self.rank,
            "layout": self.layout,
            "hidden": self.hidden,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "block_n": self.bn,
            "pair_available": dict(self.pair_available),
            "max_tokens": self.max_tokens,
            "num_slots": self.num_slots,
            "slot_stride_elems": self.slot_stride,
            "words_capacity_per_dest": self.words_cap,
            "reserved_sms": RESERVED_SMS,
            "sm_count": self.sm_count,
            "ready_mode": "dedicated_symmetric_u32",
            "memset32_probe": dict(self.memset32_probe),
            "programs": {
                "barrier_p0": self._barrier_names[0],
                "barrier_p1": self._barrier_names[1],
                "wait": self._wait_name,
                "producers": sorted(self._producers),
            },
            "prepared_rows": sorted(
                f"{batch}x{s_local}" for batch, s_local in self._plans
            ),
            "calls": self._calls,
            "captured_calls": self._captured_calls,
        }

    def close(self) -> None:
        """Release the session (collective: every rank calls it at the same point, never inside a capture)."""

        if self._closed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("close() must not be captured")
        self._closed = True
        torch.cuda.current_stream(self.device).synchronize()
        self._consumer.synchronize()
        dist.barrier(group=self.group)
        self._plans.clear()
        self._last = None
        self._producers.clear()
        self._capture_done.clear()
        self._recv = self._ready = self._flags = None  # type: ignore[assignment]
        self._recv_handle = self._ready_handle = self._flags_handle = None
        self.dst_ptrs = self.ready_ptrs = self.counters = None  # type: ignore[assignment]


__all__ = [
    "CAPTURE_READY_TARGET",
    "EAGER_READY_TARGET_MAX",
    "RESERVED_SMS",
    "SUPPORTED_SYMMETRIC_MEMORY_BACKENDS",
    "CakeQkvExchangeSession",
    "device_arch",
    "supports_symmetric_memory_backend",
    "symmetric_memory_backend",
]
