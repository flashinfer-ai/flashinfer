# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Communication-free MXFP8 x MXFP4 persistent FC1/SwiGLU/FC2 launcher.

The shared device assembly has eight compute warps, no dispatch storage,
no peer mapping, and no communication or grid barrier in this specialization.
Input scales are losslessly permuted by a byte-copy kernel before FC12.
"""

from types import SimpleNamespace
from typing import Any, Optional

import torch

from .lean_abi import LeanFc12Config, LeanFc12Inputs, validate_lean_inputs
from .persistent_kernel import PersistentFc12KernelBase, _to_cute


class LeanFc12Launcher(PersistentFc12KernelBase):
    """One warmed launcher owns its scratch and is not concurrently reentrant.

    Call once eagerly before CUDA graph capture. Counts and offsets stay on
    device; changing their contents does not require recompilation.
    """

    def __init__(self, config: LeanFc12Config, *, gate_up_clamp=None, fast_math=False):
        self.config = config
        self._sf_capacity = (
            (config.scale_row_capacity + 127 * config.local_num_experts + 127)
            // 128
            * 128
        )
        self.gate_up_clamp = gate_up_clamp
        self.fast_math = bool(fast_math)
        self.tactic = dict(
            mma_tiler_mnk=(128, 128, 128),
            cluster_shape_mnk=(1, 1, 1),
            num_stages=2,
            flag_batch=1,
            load_balance_mode="static",
        )
        self.workspace = SimpleNamespace(
            rank=0,
            plan=SimpleNamespace(
                config=SimpleNamespace(
                    pool_token_capacity=config.data_row_capacity,
                    pool_sf_capacity=self._sf_capacity,
                    num_experts_per_rank=config.local_num_experts,
                    token_padding_block=config.data_row_alignment,
                    sf_padding_block=128,
                )
            ),
        )
        self._compiled: Any = None
        self._repack: Any = None
        self._device = None

    def _initialize(self, inputs):
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as cutlass_utils
        import cuda.bindings.driver as cuda

        globals().update(cutlass=cutlass, cute=cute)
        cfg = self.config
        device = inputs.activation.device
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up LeanFc12Launcher eagerly before graph capture")
        self._device = device
        self._sf = torch.zeros(
            (self._sf_capacity, cfg.hidden // 32), dtype=torch.uint8, device=device
        )
        self._fc1 = torch.empty(
            (cfg.data_row_capacity, cfg.intermediate),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self._fc1_sf = torch.empty(
            (self._sf_capacity, cfg.intermediate // 32),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        # One FC1-done slot per token tile (the tactic's N) plus one per
        # expert for its partial last tile.
        tile_tokens = self.tactic["mma_tiler_mnk"][1]
        # wait accounting appends per-CTA stats after the counters.
        from .wait_stats import STATS_WORDS, wait_stats_enabled

        self._stats_words = STATS_WORDS if wait_stats_enabled() else 0
        self._done = torch.zeros(
            (cfg.data_row_capacity + tile_tokens - 1) // tile_tokens
            + cfg.local_num_experts
            + self._stats_words,
            dtype=torch.int32,
            device=device,
        )
        self._sf_offsets = torch.empty(
            cfg.local_num_experts + 1, dtype=torch.int32, device=device
        )
        rows, blocks = cfg.scale_row_capacity, cfg.hidden // 32
        experts = cfg.local_num_experts

        @cute.kernel
        def offsets_kernel(counts, offsets):
            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            expert = bid * 128 + tid
            if expert <= experts:
                total = cutlass.Int32(0)
                for e in cutlass.range(0, expert, 1, unroll=1):
                    total += ((counts[e] + 127) // 128) * 128
                offsets[expert] = total

        @cute.kernel
        def repack_kernel(source, destination, counts, source_offsets, dest_offsets):
            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            index = bid * 256 + tid
            if index < rows * blocks:
                row, block = index // blocks, index % blocks
                # Upper-bound handles repeated offsets for empty experts.
                lo, hi = cutlass.Int32(0), cutlass.Int32(experts)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if source_offsets[mid + 1] <= row:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < experts:
                    local_row = row - source_offsets[lo]
                    if local_row < counts[lo]:
                        dest_row = dest_offsets[lo] + local_row
                        offset = ((dest_row // 128) * (blocks // 4) + block // 4) * 512
                        offset += (
                            (dest_row % 32) * 16
                            + ((dest_row % 128) // 32) * 4
                            + block % 4
                        )
                        destination[offset] = source[index]

        @cute.jit
        def repack(source, destination, counts, source_offsets, dest_offsets, stream):
            offsets_kernel(counts, dest_offsets).launch(
                grid=((experts + 128) // 128, 1, 1),
                block=(128, 1, 1),
                stream=stream,
            )
            repack_kernel(
                source, destination, counts, source_offsets, dest_offsets
            ).launch(
                grid=(max(1, (rows * blocks + 255) // 256), 1, 1),
                block=(256, 1, 1),
                stream=stream,
            )

        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        self._repack = cute.compile(
            repack,
            _to_cute(inputs.activation_scales.view(torch.uint8).reshape(-1), 16),
            _to_cute(self._sf.reshape(-1), 16),
            _to_cute(inputs.expert_token_sizes, 4),
            _to_cute(inputs.expert_scale_row_offsets, 4),
            _to_cute(self._sf_offsets, 4),
            stream,
        )
        compute_config = SimpleNamespace(
            hidden=cfg.hidden,
            intermediate=cfg.intermediate,
            num_experts=cfg.local_num_experts,
        )
        cm, cn, _ = self.tactic["cluster_shape_mnk"]
        active = cutlass_utils.HardwareInfo(
            device_id=device.index
        ).get_max_active_clusters(cm * cn)
        self._compiled = cute.compile(
            self._make_launcher(compute_config, active), **self._runtime(inputs)
        )

    def _runtime(self, inputs):
        import cuda.bindings.driver as cuda
        from .packed_weight_views import PackedWeightViews
        from .weight_scale_views import SwizzledWeightScaleViews

        cfg = self.config
        weights = PackedWeightViews(
            inputs.fc1_weight,
            inputs.fc2_weight,
            hidden_size=cfg.hidden,
            intermediate_size=cfg.intermediate,
        ).to_cute()
        scales = SwizzledWeightScaleViews(
            inputs.fc1_weight_scales,
            inputs.fc2_weight_scales,
            hidden_size=cfg.hidden,
            intermediate_size=cfg.intermediate,
        ).to_cute()
        # Communication arguments are compile-time None and generate neither
        # device parameters nor storage in the lean specialization.
        return dict(
            input_token_buffer=None,
            input_sf_buffer=None,
            input_topk_idx=None,
            input_topk_weights=None,
            fc1_weight=weights.fc1_weight,
            fc2_weight=weights.fc2_weight,
            fc1_weight_scales=scales.fc1_weight_scales,
            fc2_weight_scales=scales.fc2_weight_scales,
            expert_send_count=None,
            expert_recv_count=None,
            expert_recv_count_sum=_to_cute(inputs.expert_token_sizes, 4),
            src_token_topk_idx=None,
            token_src_metadata=None,
            l1_arrival_count=None,
            l1_token_buffer=_to_cute(inputs.activation, 16),
            l1_sf_buffer=_to_cute(self._sf, 16),
            l1_topk_weights_buffer=None,
            nvlink_barrier_signal=None,
            nvlink_barrier_counter=None,
            grid_sync_counter=None,
            fc1_done_counter=_to_cute(self._done, 4),
            fc1_output=_to_cute(self._fc1, 16),
            fc1_output_sf=_to_cute(self._fc1_sf, 16),
            route_terms=_to_cute(inputs.output.unsqueeze(1), 16),
            output=None,  # Mega-only all-warp top-k reduction target.
            local_zero_prefix=None,
            shared_zero_prefix=None,
            load_balance_counter=None,
            peer_rank_ptr_mapper_host=None,
            stream=cuda.CUstream(
                torch.cuda.current_stream(inputs.activation.device).cuda_stream
            ),
            expert_data_row_offsets=_to_cute(inputs.expert_data_row_offsets, 4),
            expert_scale_row_offsets=_to_cute(self._sf_offsets, 4),
        )

    def wait_stats(self) -> Optional[torch.Tensor]:
        """Per-CTA wait-accounting words from the last run, if enabled."""

        if not getattr(self, "_stats_words", 0):
            return None
        return self._done[-self._stats_words :].clone()

    def run(self, inputs: LeanFc12Inputs):
        validate_lean_inputs(self.config, inputs)
        if self.config.data_row_capacity == 0:
            return inputs.output
        if self._compiled is None:
            self._initialize(inputs)
        elif self._device != inputs.activation.device:
            raise ValueError("a LeanFc12Launcher cannot be shared between devices")
        runtime = self._runtime(inputs)
        self._done.zero_()
        self._repack(
            _to_cute(inputs.activation_scales.view(torch.uint8).reshape(-1), 16),
            _to_cute(self._sf.reshape(-1), 16),
            _to_cute(inputs.expert_token_sizes, 4),
            _to_cute(inputs.expert_scale_row_offsets, 4),
            _to_cute(self._sf_offsets, 4),
            runtime["stream"],
        )
        self._compiled(**runtime)
        return inputs.output


__all__ = ["LeanFc12Launcher"]
