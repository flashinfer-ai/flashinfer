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

SM90 push-based MegaMoE (single-node NVLink expert parallelism).
"""

from __future__ import annotations

import atexit
import contextlib
import weakref
from dataclasses import dataclass
from enum import Enum

import torch

from .jit import gen_sm90_push_a2a_module

__all__ = [
    "Sm90PushPayload",
    "Sm90PushCombine",
    "Sm90PushConfig",
    "Sm90PushPipe",
]


class Sm90PushPayload(Enum):
    """Dispatch payload dtype: FP8 quantizes 1x128 at the source; BF16 is the debug anchor."""

    FP8 = "fp8"
    BF16 = "bf16"


class Sm90PushCombine(Enum):
    """Combine partial dtype: FP8 halves combine ingress; BF16 is the debug anchor."""

    FP8 = "fp8"
    BF16 = "bf16"


@dataclass(frozen=True)
class Sm90PushConfig:
    """Construction-time knobs (frozen; the hot path takes no strings)."""

    payload_dtype: Sm90PushPayload = Sm90PushPayload.FP8
    combine_dtype: Sm90PushCombine = Sm90PushCombine.FP8
    fuse_act: bool = True
    capacity_factor: float = 1.0
    dedup_dispatch: bool = False
    grouped_combine: bool = False
    fuse_fc1_epilogue: bool = False


# Window views carry C++ deleters that must not run during interpreter
# finalization: atexit drains them early; the window itself is left to the
# OS since a peer may still map it at exit.
_LIVE_PIPES: list = []

# Deliberate process-lifetime keepalive (~100 B/view): pack_strided_memory's
# ctypes DLManagedTensor wrapper must outlive every derived view's storage,
# or torch calls the deleter through a freed struct at storage destruction.
_CAPSULE_KEEPALIVE: list = []


def _drain_live_pipes() -> None:
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()
    for ref in _LIVE_PIPES:
        pipe = ref()
        if pipe is not None:
            pipe._release_window_views()
    _LIVE_PIPES.clear()


atexit.register(_drain_live_pipes)


def _record_stage(name: str, enabled: bool):
    """Host-side stage range (a few us/round when enabled) for profiling."""
    return torch.profiler.record_function(name) if enabled else contextlib.nullcontext()


def _align(x: int, a: int = 128) -> int:
    return (x + a - 1) // a * a


class _SingleRankBackend:
    """Trivial comm backend for ep_size == 1 (no MPI / torch.distributed)."""

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def allgather(self, data):
        return [data]

    def bcast(self, data, root: int = 0):
        return data

    def barrier(self) -> None:
        pass

    def Split(self, color: int, key: int):
        return self


def _default_comm_backend(ep_size: int):
    if ep_size == 1:
        return _SingleRankBackend()
    import torch.distributed as dist

    if dist.is_initialized():
        from .....comm.mnnvl import TorchDistBackend

        return TorchDistBackend()
    from .....comm.mnnvl import MPIBackend

    return MPIBackend()


def _run_guarded_phase(comm, rank: int, name: str, fn):
    """Run fn locally, then unconditionally allgather per-rank (error, payload) reports."""
    err, payload = None, None
    try:
        payload = fn()
    except Exception as exc:
        err = f"rank {rank}: {type(exc).__name__}: {exc}"
    reports = comm.allgather((err, payload))
    failures = [e for e, _ in reports if e is not None]
    if failures:
        raise RuntimeError(
            f"sm90_push init phase '{name}' failed on "
            f"{len(failures)}/{len(reports)} rank(s); all ranks abort "
            f"together: " + " | ".join(failures)
        )
    return [p for _, p in reports]


def _run_phase0_handshake(
    comm, reported_world: int, reported_rank: int, fingerprint, validate=None
):
    comm_rank = comm.Get_rank()

    def _probe():
        if validate is not None:
            validate()
        return reported_world, reported_rank, fingerprint

    reports = _run_guarded_phase(comm, comm_rank, "validate", _probe)
    comm_world = len(reports)
    bad = [
        i
        for i, (world, rank, peer_fingerprint) in enumerate(reports)
        if world != comm_world or rank != i or peer_fingerprint != fingerprint
    ]
    if bad:
        raise RuntimeError(
            f"sm90_push topology/fingerprint mismatch at rank(s) {bad}: "
            f"communicator size={comm_world}, local reported rank/world="
            f"({reported_rank}, {reported_world})"
        )
    return reports


class Sm90PushPipe:
    """Symmetric-window owner + device round protocol for the SM90 push path."""

    def __init__(
        self,
        *,
        ep_size: int,
        rank: int,
        num_local_experts: int,
        hidden_size: int,
        top_k: int,
        token_capacity: int,
        device_index: int,
        config: Sm90PushConfig | None = None,
        comm_backend=None,
        out_dtype: torch.dtype = torch.float32,
        allow_unverified_p2p: bool = False,
    ):
        if config is None:
            config = Sm90PushConfig()
        if ep_size < 1 or ep_size > 32:
            raise ValueError(f"ep_size must be in [1, 32], got {ep_size}")
        comm = (
            comm_backend if comm_backend is not None else _default_comm_backend(ep_size)
        )
        comm_size, comm_rank = comm.Get_size(), comm.Get_rank()

        def _validate_arguments():
            if not isinstance(config, Sm90PushConfig):
                raise ValueError(
                    f"config must be Sm90PushConfig, got {type(config).__name__}"
                )
            if not isinstance(config.payload_dtype, Sm90PushPayload):
                raise ValueError(
                    f"payload_dtype must be Sm90PushPayload, got {config.payload_dtype!r}"
                )
            if not isinstance(config.combine_dtype, Sm90PushCombine):
                raise ValueError(
                    f"combine_dtype must be Sm90PushCombine, got {config.combine_dtype!r}"
                )
            if out_dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(
                    f"out_dtype must be torch.float32 or torch.bfloat16, got {out_dtype}"
                )
            if ep_size < 1 or ep_size > 32:
                raise ValueError(f"ep_size must be in [1, 32], got {ep_size}")
            if num_local_experts < 1:
                raise ValueError(
                    f"num_local_experts must be >= 1, got {num_local_experts}"
                )
            if hidden_size < 128 or hidden_size % 128 != 0:
                raise ValueError(
                    f"hidden_size must be a positive multiple of 128, got {hidden_size}"
                )
            if top_k not in (1, 2, 4, 6, 8):
                raise ValueError(f"top_k must be one of (1, 2, 4, 6, 8), got {top_k}")
            if token_capacity < 1:
                raise ValueError(f"token_capacity must be >= 1, got {token_capacity}")
            if not (0.0 < config.capacity_factor <= 1.0):
                raise ValueError(
                    f"capacity_factor must be in (0, 1], got {config.capacity_factor}"
                )
            if config.grouped_combine and config.combine_dtype != Sm90PushCombine.FP8:
                raise ValueError(
                    "grouped_combine requires combine_dtype=Sm90PushCombine.FP8"
                )
            if config.fuse_fc1_epilogue and not config.fuse_act:
                raise ValueError("fuse_fc1_epilogue=True requires fuse_act=True")
            return None

        argument_fingerprint = (
            num_local_experts,
            hidden_size,
            top_k,
            token_capacity,
            str(out_dtype),
            repr(getattr(config, "payload_dtype", None)),
            repr(getattr(config, "combine_dtype", None)),
            getattr(config, "fuse_act", None),
            getattr(config, "capacity_factor", None),
            getattr(config, "dedup_dispatch", None),
            getattr(config, "grouped_combine", None),
            getattr(config, "fuse_fc1_epilogue", None),
            bool(allow_unverified_p2p),
        )
        _run_phase0_handshake(
            comm,
            ep_size,
            rank,
            argument_fingerprint,
            _validate_arguments,
        )
        ep_size, rank = comm_size, comm_rank

        from .....comm import pack_strided_memory
        from .....comm.mnnvl import SymmDeviceMemory

        self.ep, self.rank = ep_size, rank
        self.E, self.H, self.K, self.token_capacity = (
            num_local_experts,
            hidden_size,
            top_k,
            token_capacity,
        )
        self.config = config
        self.device_index = device_index
        self.out_dtype = out_dtype

        max_recv_routes = ep_size * token_capacity * top_k
        if max_recv_routes > 2**31 - 1:
            raise ValueError(
                f"ep_size * token_capacity * top_k = {max_recv_routes} overflows the "
                "packed reservation counter"
            )
        dedup = config.dedup_dispatch
        self.meta_rows = max(int(config.capacity_factor * max_recv_routes), 1)
        self.pool_rows = (
            max(int(config.capacity_factor * ep_size * token_capacity), 1)
            if dedup
            else self.meta_rows
        )
        self.m_ws = max_recv_routes
        # compute-row capacity: GEMM rows == meta records this rank can hold
        self.m_cap = self.meta_rows
        fp8_payload = config.payload_dtype == Sm90PushPayload.FP8
        self.bytes_per_row = hidden_size if fp8_payload else 2 * hidden_size
        nkb = hidden_size // 128
        E, eps = num_local_experts, ep_size

        off = 0
        self.pool_offset = off
        off = _align(off + self.pool_rows * self.bytes_per_row)
        self.pool_sc_offset = off
        off = _align(off + (self.pool_rows * nkb * 4 if fp8_payload else 0))
        self.pool_meta_offset = off
        off = _align(off + self.meta_rows * 16)
        self.pool_head_offset = off
        pool_head_padding_end = _align(off + 8)
        if off + 16 > pool_head_padding_end:
            raise AssertionError("pool-head padding must contain one uint64 abort cell")
        off = pool_head_padding_end
        self.base_cells_offset = off
        off = _align(off + eps * 8)
        self.count_cells_offset = off
        off = _align(off + E * eps * 8)
        self.cdone_cells_offset = off
        off = _align(off + eps * 8)
        self.ack_cells_offset = off
        off = _align(off + eps * 8)
        fp8_combine = config.combine_dtype == Sm90PushCombine.FP8
        self.combine_slots = ep_size if config.grouped_combine else top_k
        cslots = self.combine_slots
        self.combine_offset = off
        off = _align(
            off + (0 if fp8_combine else token_capacity * top_k * hidden_size * 2)
        )
        self.cfp8_offset = off
        off = _align(
            off + (token_capacity * cslots * hidden_size if fp8_combine else 0)
        )
        self.csc_offset = off
        off = _align(off + (token_capacity * cslots * nkb * 4 if fp8_combine else 0))
        self.total_bytes = off

        self._comm = comm

        def _phase(name, fn):
            return _run_guarded_phase(comm, rank, name, fn)

        def _phase_a_probe():
            if not torch.cuda.is_available():
                raise RuntimeError("sm90_push requires CUDA")
            major, _minor = torch.cuda.get_device_capability(device_index)
            if major != 9:
                raise RuntimeError(
                    f"sm90_push requires an SM90 (Hopper) device, got SM{major}x"
                )
            props = torch.cuda.get_device_properties(device_index)
            uuid_str = str(getattr(props, "uuid", ""))
            if "MIG" in props.name or uuid_str.startswith("MIG"):
                raise RuntimeError(
                    "sm90_push does not support MIG slices (the protocol "
                    f"needs whole-GPU NVLink peer mapping); got {props.name}"
                )
            return None

        _phase("validate-device", _phase_a_probe)

        def _phase_b_jit():
            import socket

            self.module = gen_sm90_push_a2a_module().build_and_load()
            props = torch.cuda.get_device_properties(device_index)
            return (socket.gethostname(), device_index, str(getattr(props, "uuid", "")))

        topo = _phase("a2a-jit", _phase_b_jit)

        def _phase_c_peers():
            import warnings

            hosts = {h for h, _, _ in topo}
            if len(hosts) > 1:
                raise RuntimeError(
                    f"sm90_push is single-node only; EP group spans {hosts}"
                )
            my_uuid = topo[rank][2]
            for peer_rank, (_, peer_dev, peer_uuid) in enumerate(topo):
                if peer_rank == rank:
                    continue
                if peer_uuid and peer_uuid == my_uuid:
                    raise RuntimeError(
                        f"rank {rank} and peer rank {peer_rank} report the SAME "
                        f"physical GPU ({peer_uuid}); one GPU cannot host two "
                        "EP ranks of the push protocol"
                    )
                probeable = (
                    peer_dev != device_index and peer_dev < torch.cuda.device_count()
                )
                if not probeable:
                    msg = (
                        f"cannot probe P2P capability between rank {rank} "
                        f"(device {device_index}, {my_uuid or 'uuid?'}) and peer "
                        f"rank {peer_rank} ({peer_uuid or 'uuid?'}): the peer's "
                        f"device index {peer_dev} does not name that GPU in this "
                        "process (per-rank CUDA_VISIBLE_DEVICES masking). "
                        "Unverified: cudaDeviceCanAccessPeer and NVLink-native "
                        "system-scope atomics."
                    )
                    if not allow_unverified_p2p:
                        raise RuntimeError(
                            msg + " Expose all EP GPUs to every rank, or opt in "
                            "explicitly with allow_unverified_p2p=True "
                            "(set allow_unverified_p2p on the backend config)."
                        )
                    warnings.warn(
                        "sm90_push: proceeding with UNVERIFIED peer-to-peer "
                        "capability -- " + msg,
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    continue
                if not torch.cuda.can_device_access_peer(device_index, peer_dev):
                    raise RuntimeError(
                        f"device {device_index} cannot P2P-access peer rank "
                        f"{peer_rank}'s device {peer_dev}"
                    )
                if not self.module.sm90_push_p2p_native_atomics(device_index, peer_dev):
                    raise RuntimeError(
                        f"no NVLink-native system-scope atomics between device "
                        f"{device_index} and peer device {peer_dev} (PCIe-only "
                        "P2P cannot run the push protocol)"
                    )
            return None

        _phase("peer-topology", _phase_c_peers)

        def _phase_d_window():
            self.symm = SymmDeviceMemory(
                buf_size=self.total_bytes,
                group_size=eps,
                group_rank=rank,
                device_idx=device_index,  # SymmDeviceMemory's own kwarg name
                comm_backend_for_handle_transfer=comm,
                enable_multicast=False,
                allocate_signal_pads=False,  # the protocol never uses signal pads
            )
            self.peer_bases = self.symm.get_buffer_ptrs_dev()
            wrapper = getattr(self.peer_bases, "_capsule_wrapper", None)
            if wrapper is not None:  # same keepalive hazard as view()
                _CAPSULE_KEEPALIVE.append(wrapper)
            base = self.symm.get_unicast_ptr(rank)
            dv = torch.device("cuda", device_index)
            self.device = dv
            self._abort_stream = torch.cuda.Stream(device=dv, priority=-1)

            def view(offset: int, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
                t = pack_strided_memory(
                    base + offset, nbytes, nbytes, 1, dtype, device_index
                )
                wrapper = getattr(t, "_capsule_wrapper", None)
                if wrapper is not None:  # see _CAPSULE_KEEPALIVE
                    _CAPSULE_KEEPALIVE.append(wrapper)
                t = t.reshape(-1)
                # dlpack labels bf16 as fp16 (bits are correct); re-label
                if t.dtype != dtype:
                    t = t.view(dtype)
                return t

            if not fp8_combine:
                self.combine_t = view(
                    self.combine_offset,
                    token_capacity * top_k * hidden_size * 2,
                    torch.bfloat16,
                ).reshape(token_capacity, top_k, hidden_size)
            else:
                self.csc_t = view(
                    self.csc_offset, token_capacity * cslots * nkb * 4, torch.float32
                )

            n_total_experts = E * eps
            self._round = torch.zeros(1, dtype=torch.int32, device=dv)
            if dedup:
                self._lc = torch.zeros(
                    n_total_experts + eps, dtype=torch.int32, device=dv
                )
                self._pc = self._lc[n_total_experts:]
                self._ploff = torch.empty(
                    token_capacity * top_k, dtype=torch.int32, device=dv
                )
                self._pkeybase = torch.empty(eps, dtype=torch.int32, device=dv)
            else:
                self._lc = torch.zeros(n_total_experts, dtype=torch.int32, device=dv)
            self._done = torch.zeros(n_total_experts, dtype=torch.int32, device=dv)
            self._keybase = torch.empty(n_total_experts, dtype=torch.int32, device=dv)
            self._loff = torch.empty(
                token_capacity * top_k, dtype=torch.int32, device=dv
            )
            self._rows_per_src = torch.empty(eps, dtype=torch.int32, device=dv)
            self._cdone_local = torch.empty(eps, dtype=torch.int32, device=dv)
            if fp8_combine and config.grouped_combine:
                self._grp_cnt = torch.empty(
                    eps * token_capacity, dtype=torch.int32, device=dv
                )
                self._grp_rows = torch.empty(
                    eps * token_capacity * top_k, dtype=torch.int32, device=dv
                )
                self._grp_list = torch.empty(
                    eps * token_capacity, dtype=torch.int32, device=dv
                )
                self._n_groups = torch.zeros(1, dtype=torch.int32, device=dv)
                self._groups_per_src = torch.empty(eps, dtype=torch.int32, device=dv)
            self._offsets = torch.empty(E + 1, dtype=torch.int64, device=dv)
            self._seg_src_base = torch.empty(
                n_total_experts, dtype=torch.int32, device=dv
            )
            self._seg_out_base = torch.empty(
                n_total_experts + 1, dtype=torch.int32, device=dv
            )
            self._pad_base = torch.empty(E, dtype=torch.int32, device=dv)
            self._m_dev = torch.zeros(1, dtype=torch.int32, device=dv)
            self._p_dev = torch.zeros(1, dtype=torch.int32, device=dv)
            self._next_row = torch.zeros(1, dtype=torch.int32, device=dv)

            whole = pack_strided_memory(
                base, self.total_bytes, self.total_bytes, 1, torch.uint8, device_index
            )
            whole.zero_()
            torch.cuda.synchronize()
            return None

        _phase("window+scratch", _phase_d_window)
        self._round_open = False  # host-side single-inflight misuse guard
        self._poisoned = False
        self._destroyed = False
        _LIVE_PIPES.append(weakref.ref(self))

    def _release_window_views(self) -> None:
        """Drop the dlpack-wrapped window views (atexit; see _drain_live_pipes)."""
        self.combine_t = None
        self.csc_t = None
        self.peer_bases = None

    def destroy(self) -> None:
        """Quiesce local CUDA work before releasing this rank's window views."""
        if self._destroyed:
            return
        torch.cuda.synchronize(self.device)
        self._release_window_views()
        self.symm = None
        self._destroyed = True

    # 21 layout scalars forwarded to every kernel launch (mirrors LAYOUT_PARAMS)
    def _layout_args(self):
        return (
            self.peer_bases,
            self.ep,
            self.rank,
            self.E,
            self.pool_rows,
            self.meta_rows,
            self.bytes_per_row,
            self.H,
            self.K,
            self.token_capacity,
            self.pool_offset,
            self.pool_sc_offset,
            self.pool_meta_offset,
            self.pool_head_offset,
            self.base_cells_offset,
            self.count_cells_offset,
            self.cdone_cells_offset,
            self.ack_cells_offset,
            self.combine_offset,
            self.cfp8_offset,
            self.csc_offset,
        )

    def proto_begin_round(self) -> None:
        if self._destroyed:
            raise RuntimeError("sm90_push pipe has been destroyed")
        if self._poisoned:
            raise RuntimeError(
                "sm90_push pipe is poisoned by an earlier mid-round failure"
            )
        # single-inflight: an unacked prior round means misuse
        if self._round_open:
            raise RuntimeError(
                "sm90_push pipe: previous round was never acked "
                "(proto_ack); the pipe runs ONE round at a time"
            )
        self._round_open = True
        if self.config.combine_dtype == Sm90PushCombine.FP8:
            self.csc_t.zero_()
        else:
            self.combine_t.zero_()
        self.module.sm90_push_bump_tag(self._round)
        self.module.sm90_push_wait_acks(*self._layout_args(), self._round)

    def proto_dispatch(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor
    ) -> None:
        fp8 = self.config.payload_dtype == Sm90PushPayload.FP8
        if self.config.dedup_dispatch:
            fn = (
                self.module.sm90_push_dispatch_dedup_fp8
                if fp8
                else self.module.sm90_push_dispatch_dedup
            )
            fn(
                x,
                topk_ids,
                topk_w,
                *self._layout_args(),
                self._lc,
                self._pc,
                self._loff,
                self._ploff,
                self._done,
                self._keybase,
                self._pkeybase,
                self._round,
            )
            return
        fn = (
            self.module.sm90_push_dispatch_fp8
            if fp8
            else self.module.sm90_push_dispatch
        )
        fn(
            x,
            topk_ids,
            topk_w,
            *self._layout_args(),
            self._lc,
            self._loff,
            self._done,
            self._keybase,
            self._round,
        )

    def proto_wait_prefix(self) -> None:
        self.module.sm90_push_wait_prefix(
            *self._layout_args(),
            self._round,
            self._rows_per_src,
            self._offsets,
            self._seg_src_base,
            self._seg_out_base,
            self._pad_base,
            self._m_dev,
            self._p_dev,
            self._next_row,
            self.m_cap,
        )

    def proto_compact(
        self,
        a_fp8: torch.Tensor,
        sfa: torch.Tensor,
        meta: torch.Tensor,
        row_expert: torch.Tensor,
    ) -> None:
        self.module.sm90_push_compact(
            a_fp8,
            sfa,
            meta,
            row_expert,
            *self._layout_args(),
            self._offsets,
            self._seg_src_base,
            self._seg_out_base,
            self._pad_base,
            self._m_dev,
            self._p_dev,
            self._next_row,
        )

    def proto_silu_mul_quant(
        self,
        h: torch.Tensor,
        a_fp8: torch.Tensor,
        sfa: torch.Tensor,
        row_expert: torch.Tensor,
    ) -> None:
        self.module.sm90_silu_mul_quant_grouped(
            a_fp8,
            sfa,
            h,
            self._offsets,
            self._pad_base,
            self._m_dev,
            self._p_dev,
            row_expert,
            h.shape[0],
        )

    def proto_combine(self, y: torch.Tensor, meta: torch.Tensor) -> None:
        if (
            self.config.combine_dtype == Sm90PushCombine.FP8
            and self.config.grouped_combine
        ):
            self.module.sm90_push_combine_fp8_grouped(
                y,
                meta,
                *self._layout_args(),
                self._m_dev,
                self._grp_cnt,
                self._grp_rows,
                self._grp_list,
                self._n_groups,
                self._groups_per_src,
                self._cdone_local,
                self._round,
            )
            return
        fn = (
            self.module.sm90_push_combine_fp8
            if self.config.combine_dtype == Sm90PushCombine.FP8
            else self.module.sm90_push_combine
        )
        fn(
            y,
            meta,
            *self._layout_args(),
            self._m_dev,
            self._rows_per_src,
            self._cdone_local,
            self._round,
        )

    def proto_wait_combine(self) -> None:
        self.module.sm90_push_wait_combine(*self._layout_args(), self._round)

    def proto_reduce(self, output: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Reduce this rank's combine inbox directly into the supplied output."""
        if self.config.combine_dtype == Sm90PushCombine.FP8:
            fn = (
                self.module.sm90_combine_reduce_fp8_grouped
                if self.config.grouped_combine
                else self.module.sm90_combine_reduce_fp8
            )
        else:
            fn = self.module.sm90_combine_reduce
        fn(output, *self._layout_args(), num_tokens)
        return output

    def proto_ack(self) -> None:
        """End the round: reset pool head + scratch, release the ack cells."""
        self.module.sm90_push_ack(
            *self._layout_args(), self._round, self._lc, self._done
        )
        self._round_open = False

    def proto_abort(self) -> None:
        """Publish a round abort when CUDA still accepts work, then poison this pipe."""
        if self._poisoned:
            return
        self._poisoned = True
        self._round_open = False
        try:
            with torch.cuda.stream(self._abort_stream):
                self.module.sm90_push_publish_abort(*self._layout_args(), self._round)
            self._abort_stream.synchronize()
        except Exception:  # noqa: BLE001 - sticky CUDA errors make publication best-effort
            pass
