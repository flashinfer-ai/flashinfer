# SPDX-License-Identifier: Apache-2.0
# Ported from the locally validated MiniMax-H3 operator overlay.
"""Pinned SM100 distributed-Q/O implementation; created only via collective preflight."""

from __future__ import annotations
from contextlib import nullcontext
from typing import Optional
import cuda.bindings.driver as drv
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from flash_attn.cute.flash_fwd_sm100_distributed_qo import DistArgs
from flash_attn.cute.interface import _flash_attn_fwd


def _nvtx_range(name: str, enabled: bool):
    """Return an NVTX context without perturbing normal benchmark runs."""

    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


def _normalize_cuda_device(device: Optional[torch.device]) -> torch.device:
    resolved = torch.device(
        device
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if resolved.type != "cuda":
        raise ValueError(f"attention adapter requires a CUDA device, got {resolved}")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def distributed_qo_local_alignment(q_stage: int = 2) -> int:
    """Return the minimum peer-Q alignment supported by FA4 on SM100.

    H3's long dense-Q path has q_stage=2: 2CTA consumes 512 rows per
    persistent work item, while 1CTA consumes 256. The interface automatically
    downgrades only the distributed call to 1CTA when a 512-row work item would
    cross a peer boundary.
    """

    if q_stage not in (1, 2):
        raise ValueError(f"q_stage must be 1 or 2, got {q_stage}")
    return q_stage * 128


def _supports_peer_q_layout(s_local: int, q_stage: int = 2) -> bool:
    alignment = distributed_qo_local_alignment(q_stage)
    # Preserve the validated tiny 128/256-row cases where the entire shard is
    # a divisor of one work item. Long production shards must be a multiple.
    return s_local % alignment == 0 or alignment % s_local == 0


def aligned_global_seqlen(seq_len: int, world: int, tile_m: int = 256) -> int:
    """Round a global physical length before SP sharding.

    Padding must be appended globally, before the contiguous rank shards are
    taken.  Appending rows to each local shard would insert padding in the middle
    of the global real-token prefix.
    """

    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if world <= 0 or tile_m <= 0:
        raise ValueError(f"world and tile_m must be positive, got {world}, {tile_m}")
    alignment = world * tile_m
    return ((seq_len + alignment - 1) // alignment) * alignment


def _check_cuda(ret: tuple) -> None:
    if int(ret[0]) != 0:
        raise RuntimeError(f"CUDA: {drv.cuGetErrorName(ret[0])[1]}")


def _memcpy3d(
    src_ptr: int,
    dst_ptr: int,
    width_bytes: int,
    height: int,
    src_pitch: int,
    dst_pitch: int,
) -> drv.CUDA_MEMCPY3D:
    desc = drv.CUDA_MEMCPY3D()
    dev_mem = drv.CUmemorytype.CU_MEMORYTYPE_DEVICE
    desc.srcMemoryType = dev_mem
    desc.srcDevice = int(src_ptr)
    desc.srcPitch = int(src_pitch)
    desc.srcHeight = int(height)
    desc.dstMemoryType = dev_mem
    desc.dstDevice = int(dst_ptr)
    desc.dstPitch = int(dst_pitch)
    desc.dstHeight = int(height)
    desc.WidthInBytes = int(width_bytes)
    desc.Height = int(height)
    desc.Depth = 1
    return desc


class DistributedFA4Runner:
    """H3-specialized distributed-Q/O fast path.

    Inputs are post-QKNorm/post-RoPE Q/K and raw projected V.  The returned tensor
    aliases a reusable symmetric workspace and is overwritten by the next call.
    One instance can be shared serially by all 50 H3 layers.
    """

    def __init__(
        self,
        rank: int,
        world: int,
        s_local: int,
        *,
        nheads: int = 56,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        tile_n: int = 128,
        softmax_scale: Optional[float] = None,
        group: Optional[dist.ProcessGroup] = None,
        batch: int = 1,
        o_tma: bool = False,
        mem_backend: str = "nvshmem",
        enable_nvtx: bool = False,
        q_stage: Optional[int] = None,
        sched_used_q: Optional[int] = None,
    ) -> None:
        if world < 2 or world > 32:
            raise ValueError(
                "distributed-Q/O fusion requires 2 <= world <= 32; use "
                "the framework's ordinary Ulysses path instead"
            )
        if nheads % world:
            raise ValueError(f"nheads={nheads} must be divisible by world={world}")
        if batch != 1:
            raise ValueError(
                "H3 packed-prefix fused path currently supports physical batch B=1"
            )
        if s_local % 128:
            raise ValueError(
                f"s_local={s_local} must be a multiple of 128; pad global S to world*128 before sharding"
            )
        if q_stage not in (None, 1, 2):
            raise ValueError(f"q_stage must be None, 1, or 2, got {q_stage}")
        effective_q_stage = 2 if q_stage is None else q_stage
        physical_q = s_local * world
        if sched_used_q is not None:
            if isinstance(sched_used_q, bool) or not isinstance(sched_used_q, int):
                raise TypeError(
                    "sched_used_q must be an integer or None, got "
                    f"{type(sched_used_q).__name__}"
                )
            if not 0 < sched_used_q <= physical_q:
                raise ValueError(
                    f"sched_used_q must be in [1,{physical_q}], got {sched_used_q}"
                )
            minimum_work_rows = distributed_qo_local_alignment(effective_q_stage)
            if sched_used_q % minimum_work_rows:
                raise ValueError(
                    "sched_used_q must align to at least one q_stage Q work item: "
                    f"sched_used_q={sched_used_q}, work_rows={minimum_work_rows}. "
                    "The FA4 interface additionally validates the selected 1CTA/2CTA "
                    "work boundary"
                )
        if not _supports_peer_q_layout(s_local, effective_q_stage):
            q_work_rows = distributed_qo_local_alignment(effective_q_stage)
            raise ValueError(
                f"s_local={s_local} splits a distributed peer-Q work item; "
                f"this FA4 configuration requires local alignment {q_work_rows}. "
                f"Pad global S to world*{q_work_rows} before sharding"
            )
        if s_local % tile_n:
            raise ValueError(f"s_local={s_local} must be a multiple of tile_n={tile_n}")
        if mem_backend not in ("nccl", "nvshmem"):
            raise ValueError(f"unsupported symmetric-memory backend: {mem_backend}")

        self.rank, self.world = rank, world
        self.B, self.s_local, self.S = batch, s_local, s_local * world
        self.H, self.D, self.h_local = nheads, head_dim, nheads // world
        self.inner = nheads * head_dim
        self.hl_d = self.h_local * head_dim
        self.dtype = dtype
        self.element_bytes = torch.finfo(dtype).bits // 8
        self.device = _normalize_cuda_device(device)
        if self.device.index != torch.cuda.current_device():
            raise ValueError(
                f"current CUDA device is cuda:{torch.cuda.current_device()}, "
                f"but adapter device is {self.device}"
            )
        self.tile_mn = (128, tile_n)
        self.scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
        self.group = group if group is not None else dist.group.WORLD
        if world != dist.get_world_size(self.group):
            raise ValueError(
                f"world={world} does not match process-group size "
                f"{dist.get_world_size(self.group)}"
            )
        if rank != dist.get_rank(self.group):
            raise ValueError("rank must be the process-group-local rank")
        self.group_name = self.group.group_name
        self.mem_backend = mem_backend
        self.enable_nvtx = enable_nvtx
        self.q_stage = q_stage
        self.sched_used_q = sched_used_q

        # Entry/exit barriers are indexed by the physical persistent grid, so a
        # scheduler extent mismatch across ranks could deadlock.  This constructor
        # is already collective (symmetric-window rendezvous below); validate the
        # optional value collectively before allocating those windows.  -1 encodes
        # the default full-physical-Q path.
        local_sched_config = torch.tensor(
            [-1 if sched_used_q is None else sched_used_q],
            dtype=torch.int64,
            device=self.device,
        )
        gathered_sched_config = [
            torch.empty_like(local_sched_config) for _ in range(world)
        ]
        dist.all_gather(gathered_sched_config, local_sched_config, group=self.group)
        sched_values = [int(value.item()) for value in gathered_sched_config]
        if len(set(sched_values)) != 1:
            raise ValueError(
                "sched_used_q must be identical on every distributed rank; "
                f"gathered values={sched_values} (-1 means None)"
            )
        if mem_backend == "nvshmem":
            symm_mem.enable_symm_mem_for_group(self.group_name)

        B, s, S, H, D, inner = (
            self.B,
            self.s_local,
            self.S,
            self.H,
            self.D,
            self.inner,
        )
        self.preK = torch.empty(B, s, inner, device=self.device, dtype=dtype)
        self.preV = torch.empty(B, s, inner, device=self.device, dtype=dtype)

        def make_window(*shape: int):
            tensor = symm_mem.empty(*shape, dtype=dtype, device=self.device)
            tensor.zero_()
            handle = symm_mem.rendezvous(tensor, group=self.group_name)
            return tensor, handle, [int(ptr) for ptr in handle.buffer_ptrs]

        self.Qwin, self._hQ, q_peers = make_window(B, s, inner)
        self.postK, self._hK, k_peers = make_window(B, S, self.h_local, D)
        self.postV, self._hV, v_peers = make_window(B, S, self.h_local, D)
        self.X, self._hX, x_peers = make_window(B, s, inner)
        self._x_heads = self.X.view(B, s, H, D)

        self._opsK = self._make_push_ops(self.preK.data_ptr(), k_peers)
        self._opsV = self._make_push_ops(self.preV.data_ptr(), v_peers)
        self._gK = self.postK.view(B, S, self.h_local, D)
        self._gV = self.postV.view(B, S, self.h_local, D)
        self._kself = self.preK.view(B, s, H, D)
        self._vself = self.preV.view(B, s, H, D)
        # The interface requires a nominal dense tensor, but distributed calls
        # replace it with gK/gV.  Reusing gK avoids a large dead workspace.
        self._dummy = self._gK

        nblocks = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._sync = symm_mem.empty(
            2 * nblocks * world, dtype=torch.int32, device=self.device
        )
        self._sync.zero_()
        sync_handle = symm_mem.rendezvous(self._sync, group=self.group_name)
        sync_peers = [int(ptr) for ptr in sync_handle.buffer_ptrs]
        self._hSync = sync_handle

        q_peer_wrap, q_stride = self._detect_peer_layout(q_peers, "Q")
        x_peer_wrap, x_stride = self._detect_peer_layout(x_peers, "X")
        sync_peer_wrap, sync_stride = self._detect_peer_layout(sync_peers, "sync")
        if len({q_peer_wrap, x_peer_wrap, sync_peer_wrap}) != 1:
            raise RuntimeError(
                "Q/X/sync symmetric windows use different peer layouts: "
                f"Q={'rotated' if q_peer_wrap else 'linear'}, "
                f"X={'rotated' if x_peer_wrap else 'linear'}, "
                f"sync={'rotated' if sync_peer_wrap else 'linear'}"
            )
        peer_wrap = q_peer_wrap

        self._dist_args = DistArgs(
            gK=self._gK,
            gV=self._gV,
            q_base=q_peers[rank],
            o_base=x_peers[rank],
            q_pstride=q_stride,
            o_pstride=x_stride,
            rank=rank,
            world=world,
            s_local=s,
            hl_d=self.hl_d,
            dim=inner,
            batch=B,
            o_tma=o_tma,
            q_stage=q_stage,
            sched_used_q=sched_used_q,
            sync_base=sync_peers[rank],
            sync_pstride=sync_stride,
            sync_nblk=nblocks,
            peer_wrap=peer_wrap,
        )
        # Retained only as small host-side metadata for benchmark diagnostics.
        self.debug_peer_info = {
            "q": q_peers,
            "k": k_peers,
            "v": v_peers,
            "x": x_peers,
            "sync": sync_peers,
            "q_stride": q_stride,
            "o_stride": x_stride,
            "sync_stride": sync_stride,
            "q_peer_wrap": q_peer_wrap,
            "x_peer_wrap": x_peer_wrap,
            "sync_peer_wrap": sync_peer_wrap,
            "peer_wrap": peer_wrap,
            "sched_used_q": sched_used_q,
        }

        self._seqused_k = torch.empty(B, dtype=torch.int32, device=self.device)
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._copy_stream_handle = drv.CUstream(int(self._copy_stream.cuda_stream))
        self._v_ready = torch.cuda.Event()
        self._k_ready = torch.cuda.Event()
        self._copies_done = torch.cuda.Event()
        self._bound_stream_id: Optional[int] = None
        self._setup_done = torch.cuda.Event()
        self._setup_done.record(torch.cuda.current_stream(self.device))

    def _detect_peer_layout(self, peers: list[int], label: str) -> tuple[bool, int]:
        """Validate one symmetric window's stride/layout on every rank."""

        if self.world == 1:
            return False, 0
        ordered = sorted(peers)
        gaps = [ordered[i + 1] - ordered[i] for i in range(self.world - 1)]
        uniform = len(set(gaps)) == 1
        stride = gaps[0] if uniform else 0
        linear = uniform and all(
            peers[p] == peers[0] + p * stride for p in range(self.world)
        )
        rotated = uniform and all(
            peers[p] == peers[self.rank] + ((p - self.rank) % self.world) * stride
            for p in range(self.world)
        )
        flags = torch.tensor(
            [int(uniform), int(linear), int(rotated)], device=self.device
        )
        dist.all_reduce(flags, op=dist.ReduceOp.SUM, group=self.group)
        uniform_all, linear_all, rotated_all = [int(v) for v in flags.tolist()]
        if uniform_all != self.world:
            raise RuntimeError(
                f"{label} symmetric peer virtual addresses are not uniformly strided"
            )
        if linear_all == self.world:
            return False, stride
        if rotated_all == self.world:
            return True, stride
        raise RuntimeError(
            f"{label} symmetric peer layout is neither consistently linear nor rotated"
        )

    def _make_push_ops(self, pre_ptr: int, peer_post: list[int]) -> list:
        s, hlocal, D, H, world, rank, eb, B, S = (
            self.s_local,
            self.h_local,
            self.D,
            self.H,
            self.world,
            self.rank,
            self.element_bytes,
            self.B,
            self.S,
        )
        operations = []
        for batch_idx in range(B):
            src_batch = batch_idx * s * H * D * eb
            dst_batch = batch_idx * S * hlocal * D * eb
            for step in range(1, world):
                peer = (rank + step) % world
                src = pre_ptr + src_batch + peer * hlocal * D * eb
                dst = peer_post[peer] + dst_batch + rank * s * hlocal * D * eb
                width = hlocal * D * eb
                operations.append(_memcpy3d(src, dst, width, s, H * D * eb, width))
        return operations

    def valid_local_rows(self, used_seqlen: int) -> int:
        """Number of semantically valid prefix rows owned by this rank."""

        global_start = self.rank * self.s_local
        return max(0, min(self.s_local, used_seqlen - global_start))

    def rebind_stream(
        self, stream: torch.cuda.Stream, *, synchronize: bool = False
    ) -> None:
        """Bind this non-reentrant workspace to one CUDA stream.

        Same-stream ordering protects Q/K/V staging and X consumption between
        successive layers. Migrating to another stream is allowed only after an
        explicit device synchronization, for example before CUDA Graph capture.
        """

        stream_id = int(stream.cuda_stream)
        if self._bound_stream_id not in (None, stream_id):
            if not synchronize:
                raise RuntimeError(
                    "H3 fused workspace is already bound to another CUDA stream; "
                    "call rebind_stream(new_stream, synchronize=True) only at a "
                    "known quiescent boundary"
                )
            torch.cuda.synchronize(self.device)
        self._bound_stream_id = stream_id

    def forward_prepared(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        used_seqlen: int,
        stream: Optional[torch.cuda.Stream] = None,
        discard_tail_output: bool = False,
    ) -> torch.Tensor:
        """Run H3 prepared Q/K/V and return a reusable ``[B,s,H,D]`` view.

        For padded inputs the caller must explicitly acknowledge that padding
        rows are disposable.  Only the first ``valid_local_rows(used)`` rows on
        each rank match H3's packed-varlen semantics.  When ``sched_used_q`` is
        enabled, query rows in ``[sched_used_q:S]`` are not scheduled and their
        output storage may retain old values; therefore ``used_seqlen`` must not
        exceed that extent and all tail rows must still be discarded.
        """

        expected = (self.B, self.s_local, self.H, self.D)
        if (
            tuple(q.shape) != expected
            or tuple(k.shape) != expected
            or tuple(v.shape) != expected
        ):
            raise ValueError(f"q/k/v must all have shape {expected}")
        if q.dtype != self.dtype or k.dtype != self.dtype or v.dtype != self.dtype:
            raise ValueError(f"q/k/v must all use {self.dtype}")
        if (
            q.device != self.device
            or k.device != self.device
            or v.device != self.device
        ):
            raise ValueError(f"q/k/v must all be on adapter device {self.device}")
        if not 0 < used_seqlen <= self.S:
            raise ValueError(f"used_seqlen must be in [1,{self.S}], got {used_seqlen}")
        if self.sched_used_q is not None and used_seqlen > self.sched_used_q:
            raise ValueError(
                "used_seqlen exceeds the compiled distributed scheduler extent; "
                f"used_seqlen={used_seqlen}, sched_used_q={self.sched_used_q}. "
                "Use the framework's ordinary exact-varlen attention path"
            )
        if used_seqlen < self.S and not discard_tail_output:
            raise ValueError(
                "fused prefix mode changes padding-query outputs; pass "
                "discard_tail_output=True only when global rows [used:S] are discarded"
            )

        stream = stream or torch.cuda.current_stream(self.device)
        self.rebind_stream(stream)
        with _nvtx_range("H3_FUSED", self.enable_nvtx):
            with (
                torch.cuda.stream(stream),
                _nvtx_range("H3_FUSED_QKV_STAGING", self.enable_nvtx),
            ):
                stream.wait_event(self._setup_done)
                self.preV.copy_(v.reshape_as(self.preV))
                self._v_ready.record(stream)
                self.preK.copy_(k.reshape_as(self.preK))
                self._k_ready.record(stream)
                self.Qwin.copy_(q.reshape_as(self.Qwin))
                self._seqused_k.fill_(used_seqlen)

            with _nvtx_range("H3_FUSED_KV_PEER_PUSH", self.enable_nvtx):
                self._copy_stream.wait_event(self._v_ready)
                for operation in self._opsV:
                    _check_cuda(
                        drv.cuMemcpy3DAsync(operation, self._copy_stream_handle)
                    )

                self._copy_stream.wait_event(self._k_ready)
                for operation in self._opsK:
                    _check_cuda(
                        drv.cuMemcpy3DAsync(operation, self._copy_stream_handle)
                    )
                self._copies_done.record(self._copy_stream)

            with (
                torch.cuda.stream(stream),
                _nvtx_range("H3_FUSED_FA", self.enable_nvtx),
            ):
                stream.wait_event(self._copies_done)
                _flash_attn_fwd(
                    self._dummy,
                    self._kself,
                    self._vself,
                    seqused_k=self._seqused_k,
                    softmax_scale=self.scale,
                    causal=False,
                    out=self._dummy,
                    tile_mn=self.tile_mn,
                    dist_args=self._dist_args,
                )
        return self._x_heads
