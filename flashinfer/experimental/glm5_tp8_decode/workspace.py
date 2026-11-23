# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Caller-owned eight-rank state, with explicit collective lifetime."""

from __future__ import annotations

import socket
import warnings
import weakref
from typing import Any

import torch

from .geometry import (
    CACHE_PAD,
    GRID_SIZE,
    HIDDEN_SIZE,
    DEFAULT_MAX_SEQ_LEN,
    MOE_Q4_COUNTER_OFFSET,
    MOE_STATE_BYTES,
    Q1_STATE_BYTES,
    WORLD_SIZE,
    check_capacity,
    check_rank,
    communication_layout,
)


class DecodeWorkspace:
    """A serialized sequence of collective calls on one NVLink-connected node.

    Creation, warmup, reset_request and destroy are collective. Plans retain this
    object; callers must also retain it while captured graphs can be replayed.
    Independent workspaces may be interleaved, but persistent peer-waiting grids
    must not overlap on the same GPUs.
    """

    def __init__(
        self,
        *,
        rank: int,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        group=None,
        device=None,
        comm_backend=None,
        operators=("attention", "moe"),
    ):
        from ...comm.comm_backend import TorchDistBackend
        from ...comm.mnnvl import SymmDeviceMemory, create_tensor_from_cuda_memory

        self._destroyed = False
        self._constructed = False
        self._memory = None
        self._views: dict[str, torch.Tensor] = {}
        self._last_stream = None
        self._modules: dict[str, Any] = {}
        self._plans: weakref.WeakSet[Any] = weakref.WeakSet()
        self._moe: dict[str, torch.Tensor] = {}
        self._attention: dict[str, torch.Tensor] = {}
        self.comm = (
            comm_backend if comm_backend is not None else TorchDistBackend(group)
        )
        self.rank = rank
        self.max_seq_len = max_seq_len
        self.operators = ()
        self.device = None

        def validate():
            check_rank(rank)
            check_capacity(max_seq_len)
            self.operators = tuple(sorted(set(operators)))
            self.device = (
                torch.device("cuda", torch.cuda.current_device())
                if device is None
                else torch.device(device)
            )
            if self.device.type == "cuda" and self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if self.comm.Get_size() != WORLD_SIZE or self.comm.Get_rank() != rank:
                raise ValueError(
                    "workspace requires an eight-rank group and matching logical rank"
                )
            if self.device.type != "cuda":
                raise ValueError("workspace device must be CUDA")
            communication_layout(self.operators)
            prop = torch.cuda.get_device_properties(self.device)
            if (prop.major, prop.minor) != (
                10,
                3,
            ) or prop.multi_processor_count != GRID_SIZE:
                raise ValueError(
                    "this implementation is qualified for B300 GPUs with 148 SMs"
                )
            return (socket.gethostname(), str(prop.uuid), max_seq_len, self.operators)

        config = self.collective_check(validate)
        if (
            len({x[0] for x in config}) != 1
            or len({x[1] for x in config}) != WORLD_SIZE
        ):
            raise ValueError("workspace needs eight distinct GPUs on one node")
        if any(x[2:] != config[0][2:] for x in config):
            raise ValueError("all ranks must use the same capacity and operator set")
        self._group_uuids = [x[1] for x in config]
        self.logical_bytes, self._offsets = communication_layout(self.operators)
        try:
            with torch.cuda.device(self.device):
                self.warmup()

                def map_memory():
                    self._memory = SymmDeviceMemory(
                        self.logical_bytes,
                        WORLD_SIZE,
                        rank,
                        self.device.index,
                        comm_backend_for_handle_transfer=self.comm,
                        enable_multicast=False,
                        allocate_signal_pads=False,
                    )
                    self.peer_bases = [
                        int(p) for p in self._memory.get_buffer_ptrs_host()
                    ]
                    self._views["storage"] = create_tensor_from_cuda_memory(
                        self.peer_bases[rank],
                        (self.logical_bytes,),
                        torch.uint8,
                        self.device.index,
                    )
                    self._views["storage"].zero_()

                self.collective_check(map_memory)

                def allocate_local():
                    if "moe" in self.operators:
                        self._allocate_moe()
                    if "attention" in self.operators:
                        self._allocate_attention()
                    torch.cuda.synchronize(self.device)

                self.collective_check(allocate_local)
            self.comm.barrier()
            self._constructed = True
        except Exception:
            with torch.cuda.device(self.device):
                try:
                    torch.cuda.synchronize(self.device)
                finally:
                    self._release_local()
            self._destroyed = True
            raise

    def collective_check(self, function):
        """Propagate rank-local preparation errors before any peer-waiting launch."""
        value = None
        error = None
        try:
            value = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        results = self.comm.allgather((error, value))
        errors = [
            f"rank {r}: {entry[0]}" for r, entry in enumerate(results) if entry[0]
        ]
        if errors:
            raise ValueError("collective preparation failed: " + "; ".join(errors))
        return [entry[1] for entry in results]

    def warmup(self):
        """Compile and load each requested kernel before the first exchange/capture."""
        with torch.cuda.device(self.device):
            return self._warmup_current_device()

    def _warmup_current_device(self):
        from .jit import load_module

        if self._destroyed:
            raise RuntimeError("workspace is destroyed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("workspace warmup must be outside CUDA graph capture")
        for family in self.operators:
            if family in self._modules:
                continue

            def load():
                module = load_module(family)
                for q_len in (1, 4):
                    module.initialize(q_len, self.rank, self.device.index)
                self._modules[family] = module

            self.collective_check(load)
        self.collective_check(self._check_native_atomics)
        self.comm.barrier()

    def _check_native_atomics(self):
        from ...utils import has_cuda_cudart

        if has_cuda_cudart():
            import cuda.cudart as cudart
        else:
            from cuda.bindings import runtime as cudart

        visible = {
            str(torch.cuda.get_device_properties(i).uuid): i
            for i in range(torch.cuda.device_count())
        }
        if all(uuid in visible for uuid in self._group_uuids):
            for uuid in self._group_uuids:
                peer = visible[uuid]
                if peer == self.device.index:
                    continue
                error, supported = cudart.cudaDeviceGetP2PAttribute(
                    cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported,
                    self.device.index,
                    peer,
                )
                if int(error) != 0 or not supported:
                    raise ValueError(
                        "native P2P atomics are required between every peer pair"
                    )
            return
        # CUDA ordinals can be remapped/masked per process. NVML queries use the
        # physical UUIDs gathered from the actual participating devices.
        import pynvml

        pynvml.nvmlInit()
        try:
            own_uuid = str(torch.cuda.get_device_properties(self.device).uuid)
            own = pynvml.nvmlDeviceGetHandleByUUID(own_uuid)
            for uuid in self._group_uuids:
                if uuid == own_uuid:
                    continue
                other = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                status = pynvml.nvmlDeviceGetP2PStatus(
                    own, other, pynvml.NVML_P2P_CAPS_INDEX_ATOMICS
                )
                if status != pynvml.NVML_P2P_STATUS_OK:
                    raise ValueError(
                        "native P2P atomics are required between every peer pair"
                    )
        finally:
            pynvml.nvmlShutdown()

    def _empty(self, shape, dtype, *, zero=False):
        return (torch.zeros if zero else torch.empty)(
            shape, dtype=dtype, device=self.device
        )

    def region(self, name: str) -> list[int]:
        return [p + self._offsets[name] for p in self.peer_bases]

    def record_view(self, name: str, size: int):
        offset = self._offsets[name]
        return self._views["storage"][offset : offset + size]

    def _allocate_moe(self):
        self._moe_states = {
            1: self._empty(Q1_STATE_BYTES, torch.uint8, zero=True),
            4: self._empty(MOE_STATE_BYTES, torch.uint8, zero=True),
        }
        # q4 launch counters start at one; see csrc/moe_api.h.
        offset = MOE_Q4_COUNTER_OFFSET
        counters = self._moe_states[4][offset : offset + GRID_SIZE * 4]
        counters.view(torch.int32).fill_(1)
        self._moe = {
            "scores": self._empty((128, 32), torch.float32),
            "mid": self._empty((4, 9, 256), torch.bfloat16),
            "norm": self._empty((4, HIDDEN_SIZE), torch.bfloat16),
            "probs": self._empty((4, 8), torch.float32),
            "ids": self._empty((4, 8), torch.int32),
            "shared_down": self._empty((4, HIDDEN_SIZE), torch.float32),
        }

    def _allocate_attention(self):
        self._attention_epoch = self._empty(GRID_SIZE, torch.int32, zero=True)
        self._attention_q1_state = self._empty(Q1_STATE_BYTES, torch.uint8, zero=True)
        a = self._attention
        a["ids"] = self._empty((4, 2048), torch.int32)
        if self.rank == 0:
            a["ids"].fill_(-1)
        else:
            a["ids"].copy_(
                torch.arange(2048, dtype=torch.int32, device=self.device).expand(4, -1)
            )
        a["lengths"] = self._empty(4, torch.int32, zero=True)
        a["ll"] = self.record_view("ll", 128 << 10).view(torch.int32)
        if self.rank == 0:
            shapes = {
                "qa": (4, 2048),
                "ki": (4, 128),
                "headw": (4, 32),
                "iq": (4, 32, 128),
                "iqrt": (4, 32, 128),
            }
            for name, shape in shapes.items():
                a[name] = self._empty(shape, torch.bfloat16)
            rows = self.max_seq_len + CACHE_PAD
            a["logits"] = self._empty((4, rows), torch.float32)
            a["hist"] = self._empty(4 * 258, torch.int32, zero=True)
            a["ws"] = self._empty((4, rows), torch.int32)
            a["own_slice"] = self._empty((1, HIDDEN_SIZE), torch.bfloat16, zero=True)
            a["id_peers"] = torch.tensor(
                self.region("ll")[1:], dtype=torch.int64, device=self.device
            )
        else:
            shapes = {
                "post_q": (4, 2048),
                "post_kv": (4, 512),
                "post_qnd": (4, 10, 192),
                "post_qpe": (4, 10, 64),
                "post_qnope": (4, 10, 512),
                "post_o": (4, 10, 512),
                "post_uv": (4, 10, 256),
            }
            for name, shape in shapes.items():
                a[name] = self._empty(shape, torch.bfloat16)
            a["o_acc"] = self._empty((4, 32, 10, 512), torch.float32)
            a["lse_acc"] = self._empty((4, 32, 12), torch.float32)

    def order_after_previous_stream(self):
        with torch.cuda.device(self.device):
            self._order_current_device()

    def _order_current_device(self):
        if self._destroyed:
            raise RuntimeError("workspace is destroyed")
        if torch.cuda.is_current_stream_capturing():
            return
        current = torch.cuda.current_stream(self.device)
        previous = self._last_stream
        if previous is not None and previous != current:
            event = torch.cuda.Event()
            event.record(previous)
            current.wait_event(event)
        self._last_stream = current

    def reset_request(self):
        """Clear the consumed-ID channel; preserve all collective generations."""
        if self._destroyed:
            raise RuntimeError("workspace is destroyed")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("reset_request must be outside graph capture")
            torch.cuda.synchronize(self.device)
        self.comm.barrier()
        if "attention" in self.operators:
            self._attention["ll"].zero_()
            if self.rank == 0:
                self._attention["ids"].fill_(-1)
        torch.cuda.synchronize(self.device)
        self.comm.barrier()
        self._last_stream = None

    def memory_usage(self):
        """Report physical symmetric bytes and live local tensor bytes on this rank."""
        if self._destroyed:
            return {"symmetric_bytes": 0, "local_tensor_bytes": 0}
        tensors = list(self._moe.values()) + [
            t for k, t in self._attention.items() if k != "ll"
        ]
        tensors += list(getattr(self, "_moe_states", {}).values())
        tensors += [
            getattr(self, n)
            for n in ("_attention_epoch", "_attention_q1_state")
            if hasattr(self, n)
        ]
        for plan in self._plans:
            tensors += plan._owned_tensors
        return {
            "symmetric_bytes": int(self._memory.get_allocation_size()),
            "local_tensor_bytes": sum(t.numel() * t.element_size() for t in tensors),
        }

    def _release_local(self):
        for plan in list(self._plans):
            plan._invalidate()
        self._plans.clear()
        self._views.clear()
        self._moe.clear()
        self._attention.clear()
        if hasattr(self, "_moe_states"):
            self._moe_states.clear()
        for name in ("_attention_epoch", "_attention_q1_state"):
            if hasattr(self, name):
                delattr(self, name)
        self._memory = None

    def destroy(self):
        """Collectively finish GPU work and release mappings, before the process group."""
        if self._destroyed:
            return
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("destroy must be outside CUDA graph capture")
            torch.cuda.synchronize(self.device)
            self.comm.barrier()
            self._destroyed = True
            self._release_local()
            self.comm.barrier()

    def __del__(self):
        if getattr(self, "_constructed", False) and not getattr(
            self, "_destroyed", True
        ):
            warnings.warn(
                "GLM-5 TP8 workspace was not explicitly destroyed before its process group",
                ResourceWarning,
                stacklevel=2,
            )
