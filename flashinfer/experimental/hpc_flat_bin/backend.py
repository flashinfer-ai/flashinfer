"""Parallel scheduler port; owner: slhslh; tracking: flashinfer-ai/flashinfer#6468.

Explicit use only. Original HPC-Ops planner, adapted to FlashInfer math/merge.
"""

import math
import torch
from .jit import get_module


class HPCFlatBinDecodeWrapper:
    def __init__(self, device=None, cta_per_sm=None, min_process_len=512):
        self.device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        if self.device.type != "cuda":
            raise ValueError("CUDA device required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        props = torch.cuda.get_device_properties(self.device)
        if (props.major, props.minor) != (10, 0):
            raise ValueError("This parallel port targets SM100 only")
        if cta_per_sm is None:
            cta_per_sm = 1
        if not isinstance(cta_per_sm, int) or cta_per_sm < 1:
            raise ValueError("cta_per_sm must be a positive integer")
        if (
            not isinstance(min_process_len, int)
            or min_process_len < 128
            or min_process_len >= 2**31
            or min_process_len % 128
        ):
            raise ValueError("min_process_len must be a positive multiple of 128")
        self.pool = props.multi_processor_count * cta_per_sm
        self.tile_size = 128
        self.min_tiles = min_process_len // 128
        with torch.cuda.device(self.device):
            self.module = get_module()  # JIT is outside plan/run benchmarks.
        self.max_pool = self.module.max_planner_ctas(self.device.index)
        self.shape = None
        self.raw = None

    def plan(
        self,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        pos_encoding_mode="NONE",
        sm_scale=None,
    ):
        if num_qo_heads != 4 * num_kv_heads or num_kv_heads <= 0:
            raise ValueError("Invalid head count")
        if (head_dim, page_size) != (128, 64):
            raise ValueError("Only head128/page64/GQA4 supported")
        if q_data_type != torch.bfloat16 or kv_data_type != torch.bfloat16:
            raise ValueError("BF16 only")
        if pos_encoding_mode != "NONE":
            raise ValueError("No positional encoding in this experiment")
        if (
            indptr.device.type != "cpu"
            or indptr.dtype != torch.int32
            or indptr.ndim != 1
        ):
            raise ValueError("plan requires a CPU int32 indptr")
        ptr = indptr.tolist()
        B = len(ptr) - 1
        if (
            B < 1
            or ptr[0] != 0
            or any(b <= a for a, b in zip(ptr, ptr[1:], strict=False))
        ):
            raise ValueError("All requests must have positive KV lengths")
        if B > 2048:
            raise ValueError("Original planner supports at most 2048 requests")
        if not 1 <= self.pool <= self.max_pool:
            raise ValueError("CTA pool exceeds concurrent planner residency")
        if ptr[-1] * num_kv_heads > 2**30:
            raise ValueError("Work exceeds int32 prototype limit")
        if any(b - a > (2**31 - 1) // 64 for a, b in zip(ptr, ptr[1:], strict=False)):
            raise ValueError("KV length exceeds int32 addressing")
        if (B * num_kv_heads + self.pool) * 4 * 128 >= 2**31:
            raise ValueError("Partial-state storage exceeds int32 addressing")
        for x in (indices, last_page_len):
            if (
                x.device != self.device
                or x.dtype != torch.int32
                or not x.is_contiguous()
            ):
                raise ValueError(
                    "Paged metadata must be contiguous CUDA int32 on the wrapper device"
                )
        if (
            indices.ndim != 1
            or indices.numel() != ptr[-1]
            or last_page_len.shape != (B,)
        ):
            raise ValueError("Invalid paged metadata shape")
        # last_page_len and page IDs are trusted device metadata, as in FI run().
        # Caller must supply 1<=tail<=64 and page IDs in the K/V pool's bounds.
        shape = (B, num_kv_heads, self.pool)
        if self.shape != shape:
            max_tasks = B * num_kv_heads + self.pool

            def ints(n):
                return torch.empty(n, device=self.device, dtype=torch.int32)

            self.indptr = ints(B + 1)
            self.lengths = ints(B)
            self.state_ptr = ints(B * num_kv_heads + 1)
            self.cta_ptr = ints(self.pool + 1)
            self.tasks = ints((max_tasks, 5))
            self.requests = ints(max_tasks)
            self.dummy_tiles = torch.zeros(
                max_tasks, device=self.device, dtype=torch.int32
            )
            self.meta = ints(2)
            self.tmp_v = torch.empty(
                (max_tasks, 4, 128), device=self.device, dtype=torch.bfloat16
            )
            self.tmp_s = torch.empty(
                (max_tasks, 4), device=self.device, dtype=torch.float32
            )
            self.lse = torch.empty(
                (B, num_qo_heads), device=self.device, dtype=torch.float32
            )
            self.shape = shape
        # ceil(number_of_page64 / 2) equals ceil(length / tile128), including
        # partial last pages. No device-to-host copy of tail lengths is needed.
        tiles = sum((b - a + 1) // 2 for a, b in zip(ptr, ptr[1:], strict=False))
        self.capacity = max(
            (tiles * num_kv_heads + self.pool - 1) // self.pool, self.min_tiles
        )
        pad12 = lambda n: (n + 11) // 12 * 12
        raw_size = (
            12 * (1 + (self.capacity + 1) * self.pool)
            + pad12(B * num_kv_heads)
            + 2 * pad12(self.pool)
        )
        if raw_size >= 2**31:
            raise ValueError("Original task map exceeds int32 addressing")
        if self.raw is None or self.raw.numel() < raw_size:
            self.raw = torch.empty(raw_size, device=self.device, dtype=torch.int32)
        self.indptr.copy_(indptr, non_blocking=True)
        self.indices, self.tail = indices, last_page_len
        self.scale = sm_scale if sm_scale is not None else 1 / math.sqrt(128)
        self.assign_gpu()

    def assign_gpu(self, phase=0):
        """Rebuild from current device metadata. Diagnostic phases: 1 prepare,
        2 original parallel assigner, 3 compact-state adaptation; 0 runs all.
        Phase 2 requires phase 1 first after metadata changes. Not thread-safe.
        """
        if self.shape is None:
            raise RuntimeError("Call plan() first")
        if phase not in (0, 1, 2, 3):
            raise ValueError("phase must be 0, 1, 2, or 3")
        self.module.assign(
            self.indptr,
            self.tail,
            self.lengths,
            self.state_ptr,
            self.cta_ptr,
            self.tasks,
            self.requests,
            self.meta,
            self.raw,
            self.shape[1],
            self.min_tiles,
            self.capacity,
            phase,
        )

    def run(self, q, paged_kv_cache, out=None):
        if self.shape is None:
            raise RuntimeError("Call plan() first")
        B, H, _ = self.shape
        k, v = paged_kv_cache
        for x in (q, k, v):
            if (
                x.device != self.device
                or x.dtype != torch.bfloat16
                or not x.is_contiguous()
            ):
                raise ValueError("Q/K/V must be contiguous BF16 on the wrapper device")
        if (
            q.shape != (B, H * 4, 128)
            or k.shape != v.shape
            or k.shape[1:] != (64, H, 128)
        ):
            raise ValueError("Invalid Q/K/V shapes")
        if out is None:
            out = torch.empty_like(q)
        if (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
            or not out.is_contiguous()
        ):
            raise ValueError("Invalid output tensor")
        self.module.run(
            q,
            k,
            v,
            self.indptr,
            self.indices,
            self.tail,
            self.state_ptr,
            self.cta_ptr,
            self.tasks,
            self.requests,
            self.dummy_tiles,
            self.meta,
            self.tmp_v,
            self.tmp_s,
            out,
            self.lse,
            self.scale,
        )
        return out

    def schedule(self):
        """Read actual GPU-generated task map; synchronizes for diagnostics only."""
        capacity, count = self.meta.cpu().tolist()
        return {
            "tile_size": self.tile_size,
            "planner": "hpc_original_parallel",
            "capacity_tiles": capacity,
            "cta_indptr": self.cta_ptr.cpu().tolist(),
            "state_indptr": self.state_ptr.cpu().tolist(),
            "tasks": self.tasks[:count].cpu().tolist(),
        }

    def original_schedule(self):
        """Original 48-byte task-map records, including header/counts/flags."""
        return self.raw.cpu().tolist()
