# Copyright (c) 2026, the FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kernel-level configuration dataclasses for GVR Top-K kernels.

These are internal tuning knobs that most callers should not need to touch.
The public API (``gvr_topk_decode``, ``gvr_topk_lb_*``) accepts an optional
``config`` argument; when ``None`` (the default) it runs heuristic
auto-selection based on hardware and problem shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GvrTopKConfig:
    """Kernel-level configuration for :func:`flashinfer.gvr_topk_decode`.

    All fields have sensible defaults. Most callers should leave this at its
    defaults or omit the ``config`` argument entirely (which triggers
    hardware-aware auto-selection of every knob).

    Parameters
    ----------
    cluster_size : int
        Number of CTAs per thread-block cluster.  ``1`` → single-CTA path;
        ``2`` or ``4`` enable DSMEM aggregation for very long rows (> 64 K).
        B200 GPC limit caps usable values at ~16.
    num_threads_per_block : int
        CTA block size.  Valid values: ``512`` or ``1024``.
    enable_unroll_4 : bool
        Unroll Phase-2/3 inner scan loop 4× for LSU-pipelining ILP.
    enable_phase3_unroll : bool
        Unroll the Phase-3 candidate-collect loop independently of
        ``enable_unroll_4`` (different trade-off: thread-local wc state +
        shared-memory writes).
    use_constant_hint : bool
        Emit ``LDG.E.*.CONSTANT`` (read-only-cache, equivalent to CUDA
        ``__ldg``) for input loads.  Trades write-back on eviction for
        slightly lower DRAM pressure on truly read-only inputs.
    min_blocks_per_mp : int
        ``__launch_bounds__`` hint for the PTX assembler.  Controls the
        register-file budget per thread: lower values → more registers per
        thread → better latency hiding; higher values → more blocks per SM
        → better occupancy for batched workloads.  ``0`` means no hint.
    use_256bit_load : bool
        Use 256-bit vectorised loads (fp32 only).  Halves the LDG count but
        requires 32-byte-aligned addresses and doubles fragment register
        footprint.  Profitable only for long fp32 rows (≥ 16 K elements).
    enable_warp_parallel_reduce : bool
        Replace the ``tid==0`` serial block-reduce with a warp-parallel
        reduce inside warp-0.  Auto-on when ``num_threads_per_block == 1024``
        (32 warps), where the serial cost is meaningful.
    """

    cluster_size: int = 1
    num_threads_per_block: int = 512
    enable_unroll_4: bool = True
    enable_phase3_unroll: bool = True
    use_constant_hint: bool = False
    min_blocks_per_mp: int = 2
    use_256bit_load: bool = False
    enable_warp_parallel_reduce: bool = False

    def __post_init__(self):
        if self.cluster_size < 1 or self.cluster_size > 16:
            raise ValueError(
                f"cluster_size must be in [1, 16] (B200 GPC limit); got {self.cluster_size}"
            )
        if self.num_threads_per_block not in (512, 1024):
            raise ValueError(
                f"num_threads_per_block must be 512 or 1024; got {self.num_threads_per_block}"
            )

    @classmethod
    def auto(
        cls,
        dtype,
        N: int,
        num_rows: int,
        num_sms: int,
    ) -> "GvrTopKConfig":
        """Build a config using the same heuristics as the production runner.

        Parameters
        ----------
        dtype : torch.dtype
            Element type of the logits tensor.
        N : int
            Number of columns (max_seq_len heuristic for CTA-size selection).
        num_rows : int
            Total number of rows in the logits tensor.
        num_sms : int
            Number of streaming multiprocessors on the target device.
        """
        import torch

        # Thread-count heuristic: 1024 threads for large-batch short-row
        # where the extra serial cost in the block-reduce is amortised.
        if dtype != torch.float32:
            n_thresh_t = 131072
        else:
            n_thresh_t = 65536
        num_threads_per_block = 1024 if (num_rows <= num_sms and N >= n_thresh_t) else 512

        use_256bit_load = dtype == torch.float32 and N >= 16384
        enable_warp_parallel_reduce = num_threads_per_block == 1024

        # min_blocks_per_mp heuristic tuned on B200.
        vec_bits = 256 if use_256bit_load else 128
        vec_w = vec_bits // (32 if dtype == torch.float32 else 16)
        n_vec_iters = max(1, N // (num_threads_per_block * vec_w))
        is_fp32 = dtype == torch.float32
        if is_fp32:
            if n_vec_iters < 4:
                min_blocks_per_mp = 0
            elif num_rows <= num_sms:
                min_blocks_per_mp = 1
            elif num_sms * 2 < num_rows <= num_sms * 3 and N <= 32768:
                min_blocks_per_mp = 3
            else:
                min_blocks_per_mp = 2
        else:
            if num_rows > num_sms:
                min_blocks_per_mp = 3
            elif n_vec_iters < 4:
                min_blocks_per_mp = 0
            else:
                min_blocks_per_mp = 1

        return cls(
            cluster_size=1,
            num_threads_per_block=num_threads_per_block,
            enable_unroll_4=True,
            enable_phase3_unroll=True,
            use_constant_hint=False,
            min_blocks_per_mp=min_blocks_per_mp,
            use_256bit_load=use_256bit_load,
            enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        )


@dataclass
class GvrTopKLBConfig:
    """Kernel-level configuration for :func:`flashinfer.gvr_topk_lb_prepare`
    and :func:`flashinfer.gvr_topk_lb_decode`.

    Parameters
    ----------
    max_batch_size : int
        Block size for the prepare kernel and buffer length of ``order_row``.
        Must be a power of 2 in ``[64, 1024]``.
    long_threshold : int
        Requests whose scan length (``seq_lens / compress_ratio``) exceeds
        this value are dispatched to the multi-CTA cluster path.
    cluster_size : int
        CTA-cluster size used for *long* requests in the LB decode kernel.
    num_threads : int
        CTA block size for *short* (single-CTA) requests.  Must be 512 or 1024.
    """

    max_batch_size: int = 1024
    long_threshold: int = 64 * 1024
    cluster_size: int = 4
    num_threads: int = 512

    def __post_init__(self):
        if not (64 <= self.max_batch_size <= 1024) or (
            self.max_batch_size & (self.max_batch_size - 1)
        ) != 0:
            raise ValueError(
                f"max_batch_size must be a power of 2 in [64, 1024]; got {self.max_batch_size}"
            )
        if self.cluster_size < 1 or self.cluster_size > 16:
            raise ValueError(
                f"cluster_size must be in [1, 16]; got {self.cluster_size}"
            )
        if self.num_threads not in (512, 1024):
            raise ValueError(
                f"num_threads must be 512 or 1024; got {self.num_threads}"
            )
