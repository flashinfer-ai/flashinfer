# Copyright 2026 Cursor Research
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Modified: standalone CUDA module loading; original host launch boundaries.
"""Prepared launchers for the generated MoK training kernels.

Each class owns one generated module (one per schedule identity: precision,
SwiGLU variant and weight-gradient accumulation) and keeps the native host
contract: recurring allocations, counter resets and launch geometry. Token
counts, widths and expert counts remain runtime values.
"""

import math

# Scheduler layouts (EP size, local experts) with generated count/rows kernels.
SCHEDULER_LAYOUTS = (
    (1, 4),
    (4, 4),
    (4, 64),
    (8, 32),
    (16, 16),
    (32, 8),
    (64, 4),
    (8, 36),
    (32, 9),
)
EPILOGUE_TOPKS = (2, 8)
EPI_COLS = 1024  # Epilogue column vector.
SWIGLU_BWD_TILES = 8  # 128 x 128 tiles per backward SwiGLU task half.


def _role(base, clamped=False, fp32_wgrad=False):
    return base + ("_clamped" if clamped else "") + ("_fp32" if fp32_wgrad else "")


def check_swiglu_limit(swiglu_limit, clamped):
    """Native contract: None is plain SwiGLU; a finite positive limit selects the
    clamped kernel, ``silu(min(gate, L)) * clamp(up, -L, L)``."""
    if swiglu_limit is None:
        if clamped:
            raise ValueError(
                "This kernel was built for clamped SwiGLU; pass swiglu_limit"
            )
        return
    if (
        isinstance(swiglu_limit, bool)
        or not isinstance(swiglu_limit, (int, float))
        or not math.isfinite(swiglu_limit)
        or swiglu_limit <= 0
    ):
        raise ValueError("swiglu_limit must be None or a finite positive number")
    if not clamped:
        raise ValueError(
            "This kernel was built for unclamped SwiGLU; use the clamped kernel"
        )


def mxfp8_tile_cols(device):
    """MXFP8 routed GEMM tile width of the generated target variant.

    256 x 512 tiles need 572 tensor-memory columns and are generated only for
    SM107 (576 columns). SM100/SM103 (512 columns) use 256 x 256 tiles with a
    deeper operand ring. The host task counts must match the device variant.
    """
    from .jit import target_arch

    return 512 if target_arch(device) == "sm_107a" else 256


def wgrad_tiles(hidden, intermediate, tile_cols=512):
    """Weight-gradient tasks per expert and kind for 256 x ``tile_cols`` tiles."""
    gate_up = (intermediate // 256) * ((hidden + tile_cols - 1) // tile_cols)
    down = (hidden // 256) * ((intermediate + tile_cols - 1) // tile_cols)
    return max(gate_up, down)


def swiglu_backward_tasks(tiles):
    """Backward SwiGLU tasks covering ``tiles`` 128 x 128 tiles."""
    return (tiles + 2 * SWIGLU_BWD_TILES - 1) // (2 * SWIGLU_BWD_TILES)


def expert_layout(module, counts, capacity, schedule_rank):
    """Device expert layout ``[counts | row offsets | real rows | block expert]``.

    ``module`` is the prepared ``expert_layout`` kernel. ``capacity`` bounds
    the routed rows (a multiple of 256); ``schedule_rank`` marks padding rows
    with a negative peer. Blocks past the scheduled rows are left unwritten;
    the fused kernels never read them.
    """
    import torch
    import tvm_ffi

    experts = counts.numel()
    layout = torch.empty(
        3 * experts + capacity // 256, dtype=torch.int32, device=counts.device
    )
    with tvm_ffi.use_torch_stream():
        module.launch(
            grid=(1, 1, 1),
            counts=counts,
            schedule_rank=schedule_rank,
            layout=layout,
            experts=experts,
        )
    return layout


def _check_mxfp8_weight(pair, experts, rows, cols, device):
    import torch

    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError("MXFP8 routed weights are (data, scales) pairs")
    data, scales = pair
    if (
        data.dtype != torch.float8_e4m3fn
        or tuple(data.shape) != (experts, rows, cols)
        or not data.is_contiguous()
        or data.device != device
    ):
        raise ValueError(
            "MXFP8 weight data must be contiguous E4M3 [experts, rows, cols]"
        )
    if (
        scales.dtype not in (torch.uint8, torch.float8_e8m0fnu)
        or tuple(scales.shape) != (experts * rows // 128, cols // 128, 32, 16)
        or not scales.is_contiguous()
        or scales.device != device
    ):
        raise ValueError(
            "MXFP8 weight scales must be contiguous [E*rows/128, cols/128, 32, 16] UE8M0"
        )
    return data.view(torch.uint8), scales.view(torch.uint8)


def _peer_table(cache, device, *pointer_lists):
    import torch

    key = (*(tuple(pointers) for pointers in pointer_lists), device)
    if key not in cache:
        cache[key] = tuple(
            torch.tensor(pointers, dtype=torch.uint64, device=device)
            for pointers in pointer_lists
        )
    return cache[key]


class MoKForward:
    """Low-level native-signature BF16 forward; caller owns peer barriers."""

    def __init__(self, clamped=False):
        from .jit import load_kernel

        self.clamped = clamped
        self.module = load_kernel(_role("forward", clamped))
        self.layout_module = load_kernel("expert_layout")
        self._peer_tables = {}

    def __call__(
        self,
        x,
        x_ptrs,
        combine_buffer,
        combine_ptrs,
        shared_gate,
        routed_gate,
        shared_up,
        routed_up,
        shared_down,
        routed_down,
        peer_rank,
        peer_token,
        num_tokens,
        counts,
        topk,
        swiglu_limit,
        comm_sms,
        macro_size,
        mini_size,
    ):
        import torch
        import tvm_ffi

        check_swiglu_limit(swiglu_limit, self.clamped)
        local_tokens, hidden = x.shape
        intermediate, experts = shared_gate.shape[0], counts.numel()
        capacity = peer_rank.numel()
        if (
            x.dtype != torch.bfloat16
            or local_tokens % 256
            or hidden % 256
            or intermediate % 256
            or macro_size % mini_size
            or mini_size % 256
            or comm_sms <= 0
            or comm_sms % 2
        ):
            raise ValueError(
                "MoK native BF16 tile and communication geometry is required"
            )
        x_peers, y_peers = _peer_table(
            self._peer_tables, x.device, x_ptrs, combine_ptrs
        )
        options = dict(device=x.device, dtype=torch.bfloat16)
        x_routed = torch.empty((macro_size, hidden), **options)
        gate_shared = torch.empty((local_tokens, intermediate), **options)
        gate_routed = torch.empty((macro_size, intermediate), **options)
        up_shared, up_routed = (
            torch.empty_like(gate_shared),
            torch.empty_like(gate_routed),
        )
        hidden_shared, hidden_routed = (
            torch.empty_like(gate_shared),
            torch.empty_like(gate_routed),
        )
        y_shared, y_routed = torch.empty_like(x), torch.empty_like(x_routed)
        shared_rows, routed_rows = local_tokens // 256, capacity // 256
        # Per 256-row block: fused gate/up/SwiGLU tiles, then 256 x 512 down tiles.
        shared_tasks = shared_rows * (intermediate // 256) + shared_rows * (
            (hidden + 511) // 512
        )
        mini_tasks = (mini_size // 256) * (intermediate // 256) + (mini_size // 256) * (
            (hidden + 511) // 512
        )
        minis = (capacity + mini_size - 1) // mini_size
        counter_opts = dict(dtype=torch.int32, device=x.device)
        x_ready = torch.zeros(minis, **counter_opts)
        hidden_ready = torch.zeros(shared_rows + routed_rows, **counter_opts)
        y_ready = torch.zeros(minis, **counter_opts)
        y_done = torch.zeros(capacity // 128, **counter_opts)
        layout = expert_layout(self.layout_module, counts, capacity, peer_rank)
        with tvm_ffi.use_torch_stream():
            self.module.launch(
                grid=(2 * (shared_tasks + minis * mini_tasks) + comm_sms, 1, 1),
                x_shared=x,
                x_routed=x_routed,
                wg_shared=shared_gate.unsqueeze(0),
                wu_shared=shared_up.unsqueeze(0),
                wd_shared=shared_down.unsqueeze(0),
                wg_routed=routed_gate,
                wu_routed=routed_up,
                wd_routed=routed_down,
                gate_shared_out=gate_shared,
                up_shared_out=up_shared,
                gate_routed_out=gate_routed,
                up_routed_out=up_routed,
                hidden_shared_out=hidden_shared,
                hidden_routed_out=hidden_routed,
                hidden_shared_in=hidden_shared,
                hidden_routed_in=hidden_routed,
                y_shared=y_shared,
                y_routed=y_routed,
                x_routed_ptr=x_routed,
                y_routed_ptr=y_routed,
                x_peers=x_peers,
                y_peers=y_peers,
                schedule_rank=peer_rank,
                schedule_token=peer_token,
                num_tokens=num_tokens,
                counts=layout,
                hidden_ready=hidden_ready,
                x_ready=x_ready,
                y_ready=y_ready,
                y_done=y_done,
                local_tokens=local_tokens,
                hidden=hidden,
                intermediate=intermediate,
                experts=experts,
                topk=topk,
                comm_sms=comm_sms,
                macro_size=macro_size,
                mini_size=mini_size,
                swiglu_limit=0.0 if swiglu_limit is None else float(swiglu_limit),
            )
        return (
            x_routed,
            gate_shared,
            gate_routed,
            up_shared,
            up_routed,
            hidden_shared,
            hidden_routed,
            y_shared,
            y_routed,
        )


class MoKRecompute:
    """Native-signature BF16 context-only recompute; caller owns peer barriers.

    Rebuilds the backward context (first routed macrobatch plus the shared
    expert): dispatch and gate/up/SwiGLU tiles only, without down
    projections, combine or the output epilogue.
    """

    def __init__(self, clamped=False):
        from .jit import load_kernel

        self.clamped = clamped
        self.module = load_kernel(_role("recompute", clamped))
        self.layout_module = load_kernel("expert_layout")
        self._peer_tables = {}

    def __call__(
        self,
        x,
        x_ptrs,
        shared_gate,
        routed_gate,
        shared_up,
        routed_up,
        peer_rank,
        peer_token,
        num_tokens,
        counts,
        topk,
        swiglu_limit,
        comm_sms,
        macro_size,
        mini_size,
    ):
        import torch
        import tvm_ffi

        check_swiglu_limit(swiglu_limit, self.clamped)
        local_tokens, hidden = x.shape
        intermediate, experts = shared_gate.shape[0], counts.numel()
        capacity = peer_rank.numel()
        if (
            x.dtype != torch.bfloat16
            or local_tokens % 256
            or hidden % 256
            or intermediate % 256
            or macro_size % mini_size
            or mini_size % 256
            or comm_sms <= 0
            or comm_sms % 2
        ):
            raise ValueError(
                "MoK native BF16 tile and communication geometry is required"
            )
        (x_peers,) = _peer_table(self._peer_tables, x.device, x_ptrs)
        options = dict(device=x.device, dtype=torch.bfloat16)
        # Identical context storage to the forward ring.
        x_routed = torch.empty((macro_size, hidden), **options)
        gate_shared = torch.empty((local_tokens, intermediate), **options)
        gate_routed = torch.empty((macro_size, intermediate), **options)
        up_shared, up_routed = (
            torch.empty_like(gate_shared),
            torch.empty_like(gate_routed),
        )
        hidden_shared, hidden_routed = (
            torch.empty_like(gate_shared),
            torch.empty_like(gate_routed),
        )
        routed_capacity = min(capacity, macro_size)
        shared_rows, routed_rows = local_tokens // 256, routed_capacity // 256
        shared_tasks = shared_rows * (intermediate // 256)
        mini_tasks = (mini_size // 256) * (intermediate // 256)
        minis = (routed_capacity + mini_size - 1) // mini_size
        counter_opts = dict(dtype=torch.int32, device=x.device)
        x_ready = torch.zeros(minis, **counter_opts)
        hidden_ready = torch.zeros(shared_rows + routed_rows, **counter_opts)
        layout = expert_layout(self.layout_module, counts, capacity, peer_rank)
        with tvm_ffi.use_torch_stream():
            self.module.launch(
                grid=(2 * (shared_tasks + minis * mini_tasks) + comm_sms, 1, 1),
                x_shared=x,
                x_routed=x_routed,
                wg_shared=shared_gate.unsqueeze(0),
                wu_shared=shared_up.unsqueeze(0),
                wg_routed=routed_gate,
                wu_routed=routed_up,
                gate_shared_out=gate_shared,
                up_shared_out=up_shared,
                gate_routed_out=gate_routed,
                up_routed_out=up_routed,
                hidden_shared_out=hidden_shared,
                hidden_routed_out=hidden_routed,
                x_routed_ptr=x_routed,
                x_peers=x_peers,
                schedule_rank=peer_rank,
                schedule_token=peer_token,
                num_tokens=num_tokens,
                counts=layout,
                hidden_ready=hidden_ready,
                x_ready=x_ready,
                local_tokens=local_tokens,
                hidden=hidden,
                intermediate=intermediate,
                experts=experts,
                topk=topk,
                comm_sms=comm_sms,
                macro_size=macro_size,
                mini_size=mini_size,
                swiglu_limit=0.0 if swiglu_limit is None else float(swiglu_limit),
            )
        return (
            x_routed,
            gate_shared,
            gate_routed,
            up_shared,
            up_routed,
            hidden_shared,
            hidden_routed,
        )


class MoKBackward:
    """Native-signature BF16 backward and its separate empty-expert zeroing.

    The router-score gradient is fused into SwiGLU backward as
    ``dot(d_hidden, hidden) / score``, as in native MoK; a zero score yields a
    zero gradient. ``fp32_wgrad`` kernels add every weight-gradient
    contribution into caller-owned FP32 accumulators.
    """

    def __init__(self, clamped=False, fp32_wgrad=False):
        from .jit import load_kernel

        self.clamped, self.fp32_wgrad = clamped, fp32_wgrad
        self.module = load_kernel(_role("backward", clamped, fp32_wgrad))
        self.zero_module = load_kernel("zero")
        self.layout_module = load_kernel("expert_layout")
        self._peer_tables = {}

    def __call__(
        self,
        dy_buffer,
        dy_ptrs,
        dx_buffer,
        dx_ptrs,
        weight_buffer,
        weight_ptrs,
        dweight_buffer,
        dweight_ptrs,
        shared_gate,
        routed_gate,
        shared_up,
        routed_up,
        shared_down,
        routed_down,
        x_routed,
        gate_shared,
        gate_routed,
        up_shared,
        up_routed,
        hidden_shared,
        hidden_routed,
        x,
        x_ptrs,
        peer_rank,
        peer_token,
        num_tokens,
        counts,
        topk,
        swiglu_limit,
        comm_sms,
        macro_size,
        mini_size,
        weight_grad_accumulators=None,
    ):
        """``weight_grad_accumulators`` (FP32 kernels only): caller-owned FP32
        (shared gate, up, down, routed gate, up, down) tensors with the weight
        shapes; this call adds its weight gradients to them and returns them."""
        import torch
        import tvm_ffi

        check_swiglu_limit(swiglu_limit, self.clamped)
        if (weight_grad_accumulators is not None) != self.fp32_wgrad:
            raise ValueError(
                "FP32 weight-gradient kernels require exactly six FP32 accumulators"
            )
        local_tokens, hidden = x.shape
        intermediate, experts, capacity = (
            shared_gate.shape[0],
            counts.numel(),
            peer_rank.numel(),
        )
        if (
            x.dtype != torch.bfloat16
            or local_tokens % 256
            or hidden % 256
            or intermediate % 256
            or macro_size % mini_size
            or mini_size % 256
            or comm_sms <= 0
            or comm_sms % 2
        ):
            raise ValueError(
                "MoK native BF16 tile and communication geometry is required"
            )
        x_peers, dy_peers, dx_peers, weight_peers, dweight_peers = _peer_table(
            self._peer_tables,
            x.device,
            x_ptrs,
            dy_ptrs,
            dx_ptrs,
            weight_ptrs,
            dweight_ptrs,
        )
        options = dict(device=x.device, dtype=torch.bfloat16)
        router_weights = torch.empty(macro_size, dtype=torch.float32, device=x.device)
        partials = torch.empty(
            macro_size, intermediate // 128, dtype=torch.float32, device=x.device
        )
        dy_routed = torch.empty((macro_size, hidden), **options)
        dh_shared = torch.empty((local_tokens, intermediate), **options)
        dh_routed = torch.empty((macro_size, intermediate), **options)
        dg_shared, dg_routed = torch.empty_like(dh_shared), torch.empty_like(dh_routed)
        du_shared, du_routed = torch.empty_like(dh_shared), torch.empty_like(dh_routed)
        dx_shared, dx_routed = torch.empty_like(x), torch.empty_like(dy_routed)
        if self.fp32_wgrad:
            weights = (
                shared_gate,
                shared_up,
                shared_down,
                routed_gate,
                routed_up,
                routed_down,
            )
            if len(weight_grad_accumulators) != 6 or any(
                a.dtype != torch.float32
                or a.shape != w.shape
                or a.device != x.device
                or not a.is_contiguous()
                for a, w in zip(weight_grad_accumulators, weights, strict=True)
            ):
                raise ValueError(
                    "Accumulators must be contiguous FP32 tensors shaped like the weights"
                )
            dwg_shared, dwu_shared, dwd_shared, dwg_routed, dwu_routed, dwd_routed = (
                weight_grad_accumulators
            )
        else:
            dwg_shared, dwg_routed = (
                torch.empty_like(shared_gate),
                torch.empty_like(routed_gate),
            )
            dwu_shared, dwu_routed = (
                torch.empty_like(shared_up),
                torch.empty_like(routed_up),
            )
            dwd_shared, dwd_routed = (
                torch.empty_like(shared_down),
                torch.empty_like(routed_down),
            )
        minis = (capacity + mini_size - 1) // mini_size
        macros = (capacity + macro_size - 1) // macro_size
        shared_rows, routed_rows = local_tokens // 256, capacity // 256
        ib = intermediate // 256  # 256-column counter blocks
        counter_options = dict(dtype=torch.int32, device=x.device)
        dy_ready = torch.zeros(minis, **counter_options)
        dh_ready = torch.zeros((shared_rows + routed_rows) * ib, **counter_options)
        dg_ready = torch.zeros(shared_rows + routed_rows, **counter_options)
        dx_ready = torch.zeros(minis, **counter_options)
        replay_x = torch.zeros(minis, **counter_options)
        replay_gu = torch.zeros(routed_rows * ib, **counter_options)
        replay_h = torch.zeros(routed_rows, **counter_options)
        buffers_done = torch.zeros(macros, **counter_options)
        weight_ready = torch.zeros(macros, **counter_options)
        # 512-column GEMM tasks.
        ibw, hbw = (intermediate + 511) // 512, (hidden + 511) // 512
        shared_tasks = (
            shared_rows * (ibw + hbw)
            + swiglu_backward_tasks((local_tokens // 128) * (intermediate // 128))
            + 3 * wgrad_tiles(hidden, intermediate)
        )
        mini_bwd = (mini_size // 256) * (ibw + hbw) + swiglu_backward_tasks(
            (mini_size // 128) * (intermediate // 128)
        )
        mini_replay = (
            2 * (mini_size // 256) * ibw
            + ((mini_size // 128) * (intermediate // 128) + 5) // 6
        )
        clusters = (
            shared_tasks
            + minis * mini_bwd
            + max(0, minis - macro_size // mini_size) * mini_replay
            + macros * 3 * experts * wgrad_tiles(hidden, intermediate)
        )
        layout = expert_layout(self.layout_module, counts, capacity, peer_rank)
        with tvm_ffi.use_torch_stream():
            self.module.launch(
                grid=(comm_sms + 2 * clusters, 1, 1),
                dy_s=dy_buffer,
                dy_r=dy_routed,
                dg_s=dg_shared,
                dg_r=dg_routed,
                du_s=du_shared,
                du_r=du_routed,
                x_nt_r=x_routed,
                dy_atb_s=dy_buffer,
                dy_atb_r=dy_routed,
                dg_atb_s=dg_shared,
                dg_atb_r=dg_routed,
                du_atb_s=du_shared,
                du_atb_r=du_routed,
                x_atb_s=x,
                x_atb_r=x_routed,
                h_atb_s=hidden_shared,
                h_atb_r=hidden_routed,
                wg_s=shared_gate.unsqueeze(0),
                wu_s=shared_up.unsqueeze(0),
                wd_s=shared_down.unsqueeze(0),
                wg_r=routed_gate,
                wu_r=routed_up,
                wd_r=routed_down,
                wg_nt_r=routed_gate,
                wu_nt_r=routed_up,
                dh_s=dh_shared,
                dh_r=dh_routed,
                dx_s=dx_shared,
                dx_r=dx_routed,
                gate_out_r=gate_routed,
                up_out_r=up_routed,
                dwg_s=dwg_shared.unsqueeze(0),
                dwu_s=dwu_shared.unsqueeze(0),
                dwd_s=dwd_shared.unsqueeze(0),
                dwg_r=dwg_routed,
                dwu_r=dwu_routed,
                dwd_r=dwd_routed,
                dh_sw_s=dh_shared,
                dh_sw_r=dh_routed,
                gate_sw_s=gate_shared,
                gate_sw_r=gate_routed,
                up_sw_s=up_shared,
                up_sw_r=up_routed,
                dg_sw_s=dg_shared,
                dg_sw_r=dg_routed,
                du_sw_s=du_shared,
                du_sw_r=du_routed,
                h_sw_r=hidden_routed,
                x_routed_ptr=x_routed,
                dy_routed_ptr=dy_routed,
                dx_routed_ptr=dx_routed,
                weights=router_weights,
                partials=partials,
                x_peers=x_peers,
                dy_peers=dy_peers,
                dx_peers=dx_peers,
                weight_peers=weight_peers,
                dweight_peers=dweight_peers,
                schedule_rank=peer_rank,
                schedule_token=peer_token,
                num_tokens=num_tokens,
                counts=layout,
                dh_ready=dh_ready,
                dg_ready=dg_ready,
                dy_ready=dy_ready,
                dx_ready=dx_ready,
                replay_x=replay_x,
                replay_gu=replay_gu,
                replay_h=replay_h,
                buffers_done=buffers_done,
                weight_ready=weight_ready,
                local_tokens=local_tokens,
                hidden=hidden,
                intermediate=intermediate,
                experts=experts,
                topk=topk,
                comm_sms=comm_sms,
                macro_size=macro_size,
                mini_size=mini_size,
                swiglu_limit=0.0 if swiglu_limit is None else float(swiglu_limit),
            )
            if not self.fp32_wgrad:
                # Accumulation adds nothing for an empty expert; BF16 outputs are zeroed.
                self.zero_module.launch(
                    grid=(128, experts, 1),
                    gate=dwg_routed.view(torch.uint16),
                    up=dwu_routed.view(torch.uint16),
                    down=dwd_routed.view(torch.uint16),
                    counts=counts,
                    elements=hidden * intermediate,
                )
        return (
            dx_shared,
            dx_routed,
            dg_shared,
            dg_routed,
            du_shared,
            du_routed,
            dh_shared,
            dh_routed,
            dy_routed,
            dwg_shared,
            dwg_routed,
            dwu_shared,
            dwu_routed,
            dwd_shared,
            dwd_routed,
        )


class MoKForwardMXFP8:
    """Native-signature MXFP8 forward; caller owns peer barriers.

    Routed experts run natively in MXFP8 with caller-prequantized weights
    ``(data, scales)``; activations are quantized inside the kernel. The
    shared expert stays BF16. The context saves transposed MXFP8 x and
    activation (weight gradients) and E4M3 gate/up (SwiGLU backward).
    """

    def __init__(self, clamped=False):
        from .jit import load_kernel

        self.clamped = clamped
        self.module = load_kernel(_role("forward_mxfp8", clamped))
        self.layout_module = load_kernel("expert_layout")
        self._peer_tables = {}

    def __call__(
        self,
        x,
        x_ptrs,
        combine_buffer,
        combine_ptrs,
        shared_gate,
        routed_gate,
        shared_up,
        routed_up,
        shared_down,
        routed_down,
        peer_rank,
        peer_token,
        num_tokens,
        counts,
        topk,
        swiglu_limit,
        comm_sms,
        macro_size,
        mini_size,
    ):
        import torch
        import tvm_ffi

        check_swiglu_limit(swiglu_limit, self.clamped)
        local_tokens, hidden = x.shape
        intermediate, experts = shared_gate.shape[0], counts.numel()
        capacity = peer_rank.numel()
        if (
            x.dtype != torch.bfloat16
            or local_tokens % 256
            or hidden % 512
            or intermediate % 256
            or macro_size % mini_size
            or mini_size % 256
            or comm_sms <= 0
            or comm_sms % 2
        ):
            raise ValueError("MoK MXFP8 tile and communication geometry is required")
        wg, wg_sc = _check_mxfp8_weight(
            routed_gate, experts, intermediate, hidden, x.device
        )
        wu, wu_sc = _check_mxfp8_weight(
            routed_up, experts, intermediate, hidden, x.device
        )
        wd, wd_sc = _check_mxfp8_weight(
            routed_down, experts, hidden, intermediate, x.device
        )
        x_peers, y_peers = _peer_table(
            self._peer_tables, x.device, x_ptrs, combine_ptrs
        )
        options = dict(device=x.device, dtype=torch.bfloat16)
        bytes_ = dict(device=x.device, dtype=torch.uint8)
        m_tiles, h_tiles, i_tiles = (
            macro_size // 128,
            hidden // 128,
            intermediate // 128,
        )
        x_q = torch.empty((macro_size, hidden), **bytes_)
        x_sc = torch.empty((m_tiles, h_tiles, 32, 16), **bytes_)
        x_t = torch.empty((hidden, macro_size), **bytes_)
        x_sc_t = torch.empty((h_tiles, m_tiles, 32, 16), **bytes_)
        gate_shared = torch.empty((local_tokens, intermediate), **options)
        up_shared, hidden_shared = (
            torch.empty_like(gate_shared),
            torch.empty_like(gate_shared),
        )
        gate_routed = torch.empty((macro_size, intermediate), **options)
        up_routed = torch.empty_like(gate_routed)
        gate_q = torch.empty((macro_size, intermediate), **bytes_)
        up_q = torch.empty((macro_size, intermediate), **bytes_)
        gate_sc = torch.empty((m_tiles, i_tiles, 32, 16), **bytes_)
        up_sc = torch.empty_like(gate_sc)
        hidden_q = torch.empty((macro_size, intermediate), **bytes_)
        hidden_sc = torch.empty((m_tiles, i_tiles, 32, 16), **bytes_)
        hidden_t = torch.empty((intermediate, macro_size), **bytes_)
        hidden_sc_t = torch.empty((i_tiles, m_tiles, 32, 16), **bytes_)
        y_shared = torch.empty_like(x)
        y_routed = torch.empty((macro_size, hidden), **options)
        shared_rows, routed_rows = local_tokens // 256, capacity // 256
        shared_tasks = shared_rows * (intermediate // 256) + shared_rows * (
            (hidden + 511) // 512
        )
        mini_swiglu = ((mini_size // 128) * (intermediate // 128) + 5) // 6
        tile = mxfp8_tile_cols(x.device)
        # 256 x 512 targets compute and save gate and up in one tile; 256 x 256
        # targets use separate gate and up tiles per 256-column block.
        gate_passes = 1 if tile == 512 else 2
        mini_tasks = (
            gate_passes * (mini_size // 256) * (intermediate // 256)
            + mini_swiglu
            + (mini_size // 256) * ((hidden + tile - 1) // tile)
        )
        minis = (capacity + mini_size - 1) // mini_size
        counter_opts = dict(dtype=torch.int32, device=x.device)
        x_ready = torch.zeros(minis, **counter_opts)
        gate_ready = torch.zeros(
            (shared_rows + routed_rows) * (intermediate // 256), **counter_opts
        )
        hidden_ready = torch.zeros(shared_rows + routed_rows, **counter_opts)
        y_ready = torch.zeros(minis, **counter_opts)
        y_done = torch.zeros(capacity // 128, **counter_opts)
        layout = expert_layout(self.layout_module, counts, capacity, peer_rank)
        with tvm_ffi.use_torch_stream():
            self.module.launch(
                grid=(2 * (shared_tasks + minis * mini_tasks) + comm_sms, 1, 1),
                x_shared=x,
                wg_shared=shared_gate.unsqueeze(0),
                wu_shared=shared_up.unsqueeze(0),
                wd_shared=shared_down.unsqueeze(0),
                gate_shared_out=gate_shared,
                up_shared_out=up_shared,
                hidden_shared_out=hidden_shared,
                hidden_shared_in=hidden_shared,
                y_shared=y_shared,
                x_q_store=x_q,
                x_sc_store=x_sc,
                x_t_store=x_t,
                x_sc_t_store=x_sc_t,
                x_q=x_q,
                x_sc=x_sc,
                wg_q=wg,
                wg_sc=wg_sc,
                wu_q=wu,
                wu_sc=wu_sc,
                wd_q=wd,
                wd_sc=wd_sc,
                gate_routed_out=gate_routed,
                up_routed_out=up_routed,
                gate_q_store=gate_q,
                gate_sc_store=gate_sc,
                up_q_store=up_q,
                up_sc_store=up_sc,
                gate_routed_in=gate_routed,
                up_routed_in=up_routed,
                hidden_q_store=hidden_q,
                hidden_sc_store=hidden_sc,
                hidden_t_store=hidden_t,
                hidden_sc_t_store=hidden_sc_t,
                hidden_q=hidden_q,
                hidden_sc=hidden_sc,
                y_routed=y_routed,
                y_routed_ptr=y_routed,
                x_peers=x_peers,
                y_peers=y_peers,
                schedule_rank=peer_rank,
                schedule_token=peer_token,
                num_tokens=num_tokens,
                counts=layout,
                gate_ready=gate_ready,
                hidden_ready=hidden_ready,
                x_ready=x_ready,
                y_ready=y_ready,
                y_done=y_done,
                local_tokens=local_tokens,
                hidden=hidden,
                intermediate=intermediate,
                experts=experts,
                topk=topk,
                comm_sms=comm_sms,
                macro_size=macro_size,
                mini_size=mini_size,
                swiglu_limit=0.0 if swiglu_limit is None else float(swiglu_limit),
            )
        # Native MXFP8 context: transposed x and activation for the weight
        # gradients, E4M3 gate/up for SwiGLU backward, BF16 shared tensors.
        return (
            (x_t, x_sc_t),
            gate_shared,
            (gate_q, gate_sc),
            up_shared,
            (up_q, up_sc),
            hidden_shared,
            (hidden_t, hidden_sc_t),
            y_shared,
            y_routed,
        )


class MoKRecomputeMXFP8:
    """Native-signature MXFP8 context-only recompute; caller owns peer barriers."""

    def __init__(self, clamped=False):
        from .jit import load_kernel

        self.clamped = clamped
        self.module = load_kernel(_role("recompute_mxfp8", clamped))
        self.layout_module = load_kernel("expert_layout")
        self._peer_tables = {}

    def __call__(
        self,
        x,
        x_ptrs,
        shared_gate,
        routed_gate,
        shared_up,
        routed_up,
        peer_rank,
        peer_token,
        num_tokens,
        counts,
        topk,
        swiglu_limit,
        comm_sms,
        macro_size,
        mini_size,
    ):
        import torch
        import tvm_ffi

        check_swiglu_limit(swiglu_limit, self.clamped)
        local_tokens, hidden = x.shape
        intermediate, experts = shared_gate.shape[0], counts.numel()
        capacity = peer_rank.numel()
        if (
            x.dtype != torch.bfloat16
            or local_tokens % 256
            or hidden % 512
            or intermediate % 256
            or macro_size % mini_size
            or mini_size % 256
            or comm_sms <= 0
            or comm_sms % 2
        ):
            raise ValueError("MoK MXFP8 tile and communication geometry is required")
        wg, wg_sc = _check_mxfp8_weight(
            tuple(routed_gate[:2]), experts, intermediate, hidden, x.device
        )
        wu, wu_sc = _check_mxfp8_weight(
            tuple(routed_up[:2]), experts, intermediate, hidden, x.device
        )
        (x_peers,) = _peer_table(self._peer_tables, x.device, x_ptrs)
        bf16 = dict(device=x.device, dtype=torch.bfloat16)
        u8 = dict(device=x.device, dtype=torch.uint8)
        m_t, h_t, i_t = macro_size // 128, hidden // 128, intermediate // 128
        x_q, x_sc = (
            torch.empty((macro_size, hidden), **u8),
            torch.empty((m_t, h_t, 32, 16), **u8),
        )
        x_t, x_sc_t = (
            torch.empty((hidden, macro_size), **u8),
            torch.empty((h_t, m_t, 32, 16), **u8),
        )
        gate_shared = torch.empty((local_tokens, intermediate), **bf16)
        up_shared, hidden_shared = (
            torch.empty_like(gate_shared),
            torch.empty_like(gate_shared),
        )
        gate_routed, up_routed = (
            torch.empty((macro_size, intermediate), **bf16),
            torch.empty((macro_size, intermediate), **bf16),
        )
        gate_q, up_q = (
            torch.empty((macro_size, intermediate), **u8),
            torch.empty((macro_size, intermediate), **u8),
        )
        gate_sc, up_sc = (
            torch.empty((m_t, i_t, 32, 16), **u8),
            torch.empty((m_t, i_t, 32, 16), **u8),
        )
        h_q, h_sc = (
            torch.empty((macro_size, intermediate), **u8),
            torch.empty((m_t, i_t, 32, 16), **u8),
        )
        h_tq, h_sc_t = (
            torch.empty((intermediate, macro_size), **u8),
            torch.empty((i_t, m_t, 32, 16), **u8),
        )
        routed_capacity = min(capacity, macro_size)
        shared_rows, routed_rows = local_tokens // 256, routed_capacity // 256
        shared_gate_tasks = shared_rows * ((intermediate + 511) // 512)
        mini_gate_tasks = (mini_size // 256) * (intermediate // 256)
        shared_swiglu = ((local_tokens // 128) * (intermediate // 128) + 5) // 6
        mini_swiglu = ((mini_size // 128) * (intermediate // 128) + 5) // 6
        shared_tasks = 2 * shared_gate_tasks + shared_swiglu
        mini_tasks = 2 * mini_gate_tasks + mini_swiglu
        minis = (routed_capacity + mini_size - 1) // mini_size
        c = dict(dtype=torch.int32, device=x.device)
        x_ready = torch.zeros(minis, **c)
        gate_ready = torch.zeros(
            (shared_rows + routed_rows) * (intermediate // 256), **c
        )
        hidden_ready = torch.zeros(shared_rows + routed_rows, **c)
        layout = expert_layout(self.layout_module, counts, capacity, peer_rank)
        with tvm_ffi.use_torch_stream():
            self.module.launch(
                grid=(2 * (shared_tasks + minis * mini_tasks) + comm_sms, 1, 1),
                x_shared=x,
                wg_shared=shared_gate.unsqueeze(0),
                wu_shared=shared_up.unsqueeze(0),
                gate_shared_out=gate_shared,
                up_shared_out=up_shared,
                gate_shared_in=gate_shared,
                up_shared_in=up_shared,
                hidden_shared_out=hidden_shared,
                x_q_store=x_q,
                x_sc_store=x_sc,
                x_t_store=x_t,
                x_sc_t_store=x_sc_t,
                x_q=x_q,
                x_sc=x_sc,
                wg_q=wg,
                wg_sc=wg_sc,
                wu_q=wu,
                wu_sc=wu_sc,
                gate_routed_out=gate_routed,
                up_routed_out=up_routed,
                gate_q_store=gate_q,
                gate_sc_store=gate_sc,
                up_q_store=up_q,
                up_sc_store=up_sc,
                gate_routed_in=gate_routed,
                up_routed_in=up_routed,
                hidden_q_store=h_q,
                hidden_sc_store=h_sc,
                hidden_t_store=h_tq,
                hidden_sc_t_store=h_sc_t,
                x_peers=x_peers,
                schedule_rank=peer_rank,
                schedule_token=peer_token,
                num_tokens=num_tokens,
                counts=layout,
                gate_ready=gate_ready,
                hidden_ready=hidden_ready,
                x_ready=x_ready,
                local_tokens=local_tokens,
                hidden=hidden,
                intermediate=intermediate,
                experts=experts,
                topk=topk,
                comm_sms=comm_sms,
                macro_size=macro_size,
                mini_size=mini_size,
                swiglu_limit=0.0 if swiglu_limit is None else float(swiglu_limit),
            )
        return (
            (x_t, x_sc_t),
            gate_shared,
            (gate_q, gate_sc),
            up_shared,
            (up_q, up_sc),
            hidden_shared,
            (h_tq, h_sc_t),
        )


class MoKBackwardMXFP8:
    """Native-signature MXFP8 backward and its empty-expert zeroing.

    Routed gate/up weights are ``(fp8, sc, fp8_t, sc_t)`` and routed down
    weights ``(fp8_t, sc_t)``, as MoK's ``mxfp8_quantize(w, True, True)``
    produces them; the forward context is the MXFP8 forward's.
    """

    def __init__(self, clamped=False, fp32_wgrad=False):
        from .jit import load_kernel

        self.clamped, self.fp32_wgrad = clamped, fp32_wgrad
        self.module = load_kernel(_role("backward_mxfp8", clamped, fp32_wgrad))
        self.zero_module = load_kernel("zero")
        self.layout_module = load_kernel("expert_layout")
        self._peer_tables = {}

    def __call__(
        self,
        dy_buffer,
        dy_ptrs,
        dx_buffer,
        dx_ptrs,
        weight_buffer,
        weight_ptrs,
        dweight_buffer,
        dweight_ptrs,
        shared_gate,
        routed_gate,
        shared_up,
        routed_up,
        shared_down,
        routed_down,
        x_routed,
        gate_shared,
        gate_routed,
        up_shared,
        up_routed,
        hidden_shared,
        hidden_routed,
        x,
        x_ptrs,
        peer_rank,
        peer_token,
        num_tokens,
        counts,
        topk,
        swiglu_limit,
        comm_sms,
        macro_size,
        mini_size,
        weight_grad_accumulators=None,
    ):
        import torch
        import tvm_ffi

        check_swiglu_limit(swiglu_limit, self.clamped)
        if (weight_grad_accumulators is not None) != self.fp32_wgrad:
            raise ValueError(
                "FP32 weight-gradient kernels require exactly six FP32 accumulators"
            )
        local_tokens, hidden = x.shape
        intermediate, experts, capacity = (
            shared_gate.shape[0],
            counts.numel(),
            peer_rank.numel(),
        )
        if (
            x.dtype != torch.bfloat16
            or local_tokens % 256
            or hidden % 512
            or intermediate % 256
            or macro_size % mini_size
            or mini_size % 256
            or comm_sms <= 0
            or comm_sms % 2
        ):
            raise ValueError("MoK MXFP8 tile and communication geometry is required")
        if (
            not isinstance(routed_gate, tuple)
            or not isinstance(routed_up, tuple)
            or not isinstance(routed_down, tuple)
            or len(routed_gate) != 4
            or len(routed_up) != 4
            or len(routed_down) != 2
        ):
            raise ValueError(
                "Backward needs gate/up (fp8, sc, fp8_t, sc_t) and down (fp8_t, sc_t)"
            )
        dev = x.device
        wg, wg_sc = _check_mxfp8_weight(
            routed_gate[:2], experts, intermediate, hidden, dev
        )
        wg_t, wg_t_sc = _check_mxfp8_weight(
            routed_gate[2:], experts, hidden, intermediate, dev
        )
        wu, wu_sc = _check_mxfp8_weight(
            routed_up[:2], experts, intermediate, hidden, dev
        )
        wu_t, wu_t_sc = _check_mxfp8_weight(
            routed_up[2:], experts, hidden, intermediate, dev
        )
        wd_t, wd_t_sc = _check_mxfp8_weight(
            routed_down, experts, intermediate, hidden, dev
        )
        (x_t, x_sc_t), (gate_q, gate_sc), (up_q, up_sc), (h_t, h_sc_t) = (
            x_routed,
            gate_routed,
            up_routed,
            hidden_routed,
        )
        x_peers, dy_peers, dx_peers, weight_peers, dweight_peers = _peer_table(
            self._peer_tables, dev, x_ptrs, dy_ptrs, dx_ptrs, weight_ptrs, dweight_ptrs
        )
        bf16 = dict(device=dev, dtype=torch.bfloat16)
        u8 = dict(device=dev, dtype=torch.uint8)
        m_t, h_tiles, i_tiles = macro_size // 128, hidden // 128, intermediate // 128
        router_weights = torch.empty(macro_size, dtype=torch.float32, device=dev)
        partials = torch.empty(
            macro_size, intermediate // 128, dtype=torch.float32, device=dev
        )
        dy_q, dy_sc = (
            torch.empty((macro_size, hidden), **u8),
            torch.empty((m_t, h_tiles, 32, 16), **u8),
        )
        dy_t, dy_sc_t = (
            torch.empty((hidden, macro_size), **u8),
            torch.empty((h_tiles, m_t, 32, 16), **u8),
        )
        x_q, x_sc = (
            torch.empty((macro_size, hidden), **u8),
            torch.empty((m_t, h_tiles, 32, 16), **u8),
        )
        gate_bf16, up_bf16 = (
            torch.empty((macro_size, intermediate), **bf16),
            torch.empty((macro_size, intermediate), **bf16),
        )
        h_q, h_sc = (
            torch.empty((macro_size, intermediate), **u8),
            torch.empty((m_t, i_tiles, 32, 16), **u8),
        )
        dh_routed = torch.empty((macro_size, intermediate), **bf16)
        dg_q, du_q = (
            torch.empty((macro_size, intermediate), **u8),
            torch.empty((macro_size, intermediate), **u8),
        )
        dg_sc, du_sc = (
            torch.empty((m_t, i_tiles, 32, 16), **u8),
            torch.empty((m_t, i_tiles, 32, 16), **u8),
        )
        dg_t, du_t = (
            torch.empty((intermediate, macro_size), **u8),
            torch.empty((intermediate, macro_size), **u8),
        )
        dg_sc_t, du_sc_t = (
            torch.empty((i_tiles, m_t, 32, 16), **u8),
            torch.empty((i_tiles, m_t, 32, 16), **u8),
        )
        dx_routed = torch.empty((macro_size, hidden), **bf16)
        dh_shared = torch.empty((local_tokens, intermediate), **bf16)
        dg_shared, du_shared = torch.empty_like(dh_shared), torch.empty_like(dh_shared)
        dx_shared = torch.empty_like(x)
        if self.fp32_wgrad:
            shapes = (
                (intermediate, hidden),
                (intermediate, hidden),
                (hidden, intermediate),
                (experts, intermediate, hidden),
                (experts, intermediate, hidden),
                (experts, hidden, intermediate),
            )
            if len(weight_grad_accumulators) != 6 or any(
                a.dtype != torch.float32
                or tuple(a.shape) != shape
                or a.device != dev
                or not a.is_contiguous()
                for a, shape in zip(weight_grad_accumulators, shapes, strict=True)
            ):
                raise ValueError(
                    "Accumulators must be contiguous FP32 tensors shaped like the weights"
                )
            dwg_shared, dwu_shared, dwd_shared, dwg_routed, dwu_routed, dwd_routed = (
                weight_grad_accumulators
            )
        else:
            dwg_shared, dwu_shared, dwd_shared = (
                torch.empty_like(shared_gate),
                torch.empty_like(shared_up),
                torch.empty_like(shared_down),
            )
            dwg_routed = torch.empty((experts, intermediate, hidden), **bf16)
            dwu_routed = torch.empty_like(dwg_routed)
            dwd_routed = torch.empty((experts, hidden, intermediate), **bf16)
        minis = (capacity + mini_size - 1) // mini_size
        macros = (capacity + macro_size - 1) // macro_size
        shared_rows, routed_rows = local_tokens // 256, capacity // 256
        ib = intermediate // 256
        c = dict(dtype=torch.int32, device=dev)
        dy_ready, dx_ready, replay_x = (
            torch.zeros(minis, **c),
            torch.zeros(minis, **c),
            torch.zeros(minis, **c),
        )
        dh_ready = torch.zeros((shared_rows + routed_rows) * ib, **c)
        dg_ready = torch.zeros(shared_rows + routed_rows, **c)
        replay_gu, replay_h = (
            torch.zeros(routed_rows * ib, **c),
            torch.zeros(routed_rows, **c),
        )
        buffers_done, weight_ready = torch.zeros(macros, **c), torch.zeros(macros, **c)
        tile = mxfp8_tile_cols(dev)
        # Shared BF16 GEMM tasks use 512 columns; routed MXFP8 tasks use the target tile.
        ibw, hbw = (intermediate + 511) // 512, (hidden + 511) // 512
        ibr, hbr = (intermediate + tile - 1) // tile, (hidden + tile - 1) // tile
        shared_tasks = (
            shared_rows * (ibw + hbw)
            + ((local_tokens // 128) * (intermediate // 128) + 3) // 4
            + 3 * wgrad_tiles(hidden, intermediate)
        )
        mini_bwd = (mini_size // 256) * (ibr + hbr) + (
            (mini_size // 128) * (intermediate // 128) + 3
        ) // 4
        mini_replay = (
            2 * (mini_size // 256) * ib
            + ((mini_size // 128) * (intermediate // 128) + 5) // 6
        )
        clusters = (
            shared_tasks
            + minis * mini_bwd
            + max(0, minis - macro_size // mini_size) * mini_replay
            + macros * 3 * experts * wgrad_tiles(hidden, intermediate, tile_cols=tile)
        )
        layout = expert_layout(self.layout_module, counts, capacity, peer_rank)
        with tvm_ffi.use_torch_stream():
            self.module.launch(
                grid=(comm_sms + 2 * clusters, 1, 1),
                dy_s=dy_buffer,
                dg_s=dg_shared,
                du_s=du_shared,
                dy_atb_s=dy_buffer,
                dg_atb_s=dg_shared,
                du_atb_s=du_shared,
                x_atb_s=x,
                h_atb_s=hidden_shared,
                wg_s=shared_gate.unsqueeze(0),
                wu_s=shared_up.unsqueeze(0),
                wd_s=shared_down.unsqueeze(0),
                dh_s=dh_shared,
                dh_r=dh_routed,
                dx_s=dx_shared,
                dx_r=dx_routed,
                gate_out_r=gate_bf16,
                up_out_r=up_bf16,
                dwg_s=dwg_shared.unsqueeze(0),
                dwu_s=dwu_shared.unsqueeze(0),
                dwd_s=dwd_shared.unsqueeze(0),
                dwg_r=dwg_routed,
                dwu_r=dwu_routed,
                dwd_r=dwd_routed,
                dh_sw_s=dh_shared,
                gate_sw_s=gate_shared,
                up_sw_s=up_shared,
                dg_sw_s=dg_shared,
                du_sw_s=du_shared,
                gate_sw_r=gate_bf16,
                up_sw_r=up_bf16,
                dy_q_store=dy_q,
                dy_sc_store=dy_sc,
                dy_t_store=dy_t,
                dy_sc_t_store=dy_sc_t,
                x_q_store=x_q,
                x_sc_store=x_sc,
                x_t_store=x_t,
                x_sc_t_store=x_sc_t,
                dy_q=dy_q,
                dy_sc=dy_sc,
                dy_t=dy_t,
                dy_sc_t=dy_sc_t,
                x_q=x_q,
                x_sc=x_sc,
                x_t=x_t,
                x_sc_t=x_sc_t,
                wg_q=wg,
                wg_sc=wg_sc,
                wu_q=wu,
                wu_sc=wu_sc,
                wd_t_q=wd_t,
                wd_t_sc=wd_t_sc,
                wg_t_q=wg_t,
                wg_t_sc=wg_t_sc,
                wu_t_q=wu_t,
                wu_t_sc=wu_t_sc,
                gate_q_store=gate_q,
                gate_sc_store=gate_sc,
                up_q_store=up_q,
                up_sc_store=up_sc,
                h_q_store=h_q,
                h_sc_store=h_sc,
                h_t_store=h_t,
                h_sc_t_store=h_sc_t,
                h_t=h_t,
                h_sc_t=h_sc_t,
                dh_tile=dh_routed,
                gate_tile=gate_q,
                up_tile=up_q,
                gate_sc=gate_sc,
                up_sc=up_sc,
                dg_store=dg_q,
                dg_sc_store=dg_sc,
                du_store=du_q,
                du_sc_store=du_sc,
                dg_t_store=dg_t,
                dg_sc_t_store=dg_sc_t,
                du_t_store=du_t,
                du_sc_t_store=du_sc_t,
                dg_q=dg_q,
                dg_sc=dg_sc,
                du_q=du_q,
                du_sc=du_sc,
                dg_t=dg_t,
                dg_sc_t=dg_sc_t,
                du_t=du_t,
                du_sc_t=du_sc_t,
                dx_routed_ptr=dx_routed,
                weights=router_weights,
                partials=partials,
                x_peers=x_peers,
                dy_peers=dy_peers,
                dx_peers=dx_peers,
                weight_peers=weight_peers,
                dweight_peers=dweight_peers,
                schedule_rank=peer_rank,
                schedule_token=peer_token,
                num_tokens=num_tokens,
                counts=layout,
                dh_ready=dh_ready,
                dg_ready=dg_ready,
                dy_ready=dy_ready,
                dx_ready=dx_ready,
                replay_x=replay_x,
                replay_gu=replay_gu,
                replay_h=replay_h,
                buffers_done=buffers_done,
                weight_ready=weight_ready,
                local_tokens=local_tokens,
                hidden=hidden,
                intermediate=intermediate,
                experts=experts,
                topk=topk,
                comm_sms=comm_sms,
                macro_size=macro_size,
                mini_size=mini_size,
                swiglu_limit=0.0 if swiglu_limit is None else float(swiglu_limit),
            )
            if not self.fp32_wgrad:
                self.zero_module.launch(
                    grid=(128, experts, 1),
                    gate=dwg_routed.view(torch.uint16),
                    up=dwu_routed.view(torch.uint16),
                    down=dwd_routed.view(torch.uint16),
                    counts=counts,
                    elements=hidden * intermediate,
                )
        return (
            dx_shared,
            dx_routed,
            dg_shared,
            dh_routed,
            du_shared,
            dh_routed,
            dh_shared,
            dh_routed,
            dy_q,
            dwg_shared,
            dwg_routed,
            dwu_shared,
            dwu_routed,
            dwd_shared,
            dwd_routed,
        )


class MoKCommunication:
    """Prepared launchers over the caller-owned symmetric buffers."""

    def __init__(self, chunk_bytes: int = 2048):
        from .jit import load_kernel

        if chunk_bytes != 2048:
            raise ValueError("The exported metadata chunk size is 2048 bytes")
        self.chunk_bytes = chunk_bytes
        self.gather_module = load_kernel("gather")
        self.barrier_module = load_kernel("barrier")

    def all_gather_top_experts(self, routes, gathered, multicast_address, rank):
        import torch
        import tvm_ffi

        if (
            not routes.is_cuda
            or routes.dtype != torch.int32
            or not routes.is_contiguous()
            or routes.ndim != 2
        ):
            raise ValueError(
                "Routes must be contiguous CUDA int32 [local_tokens, top_k]"
            )
        if (
            gathered.ndim != 3
            or tuple(gathered.shape[1:]) != tuple(routes.shape)
            or gathered.dtype != routes.dtype
            or gathered.device != routes.device
            or not gathered.is_contiguous()
        ):
            raise ValueError(
                "Gathered buffer must match the route shape, dtype and device"
            )
        ep = gathered.shape[0]
        if ep not in (1, 4, 8, 16, 32, 64) or not 0 <= rank < ep:
            raise ValueError("Unsupported EP/rank")
        if type(multicast_address) is not int or multicast_address <= 0:
            raise ValueError("A live multicast address is required")
        if routes.numel() * 4 % self.chunk_bytes:
            raise ValueError(
                "Metadata chunk must divide each rank's route-buffer bytes"
            )
        if ep == 1:
            gathered[0].copy_(routes)
        else:
            with tvm_ffi.use_torch_stream():
                self.gather_module.launch(
                    grid=(routes.numel() * 4 // self.chunk_bytes, 1, 1),
                    local_routes=routes,
                    multicast_address=multicast_address,
                    rank=rank,
                    numel=routes.numel(),
                )

    def barrier_all(self, counter, peer_pointers, multicast_address, target):
        import torch
        import tvm_ffi

        ep = len(peer_pointers)
        if ep not in (1, 4, 8, 16, 32, 64):
            raise ValueError("Unsupported EP size")
        if type(multicast_address) is not int or multicast_address <= 0:
            raise ValueError("A live multicast address is required")
        for tensor in (counter, target):
            if (
                not tensor.is_cuda
                or tensor.device != counter.device
                or tensor.dtype != torch.int32
                or tuple(tensor.shape) != (1,)
                or not tensor.is_contiguous()
            ):
                raise ValueError("Barrier counters must be contiguous CUDA int32 [1]")
        if ep > 1:
            with tvm_ffi.use_torch_stream():
                self.barrier_module.launch(
                    grid=(1, 1, 1),
                    local_counter=counter.view(torch.uint32),
                    target_counter=target.view(torch.uint32),
                    multicast_address=multicast_address,
                    ep_size=ep,
                )


class MoKScheduler:
    """Three prepared kernel modules; allocations/resets match native MoK.

    Compilation is setup and must precede graph capture. The launch method
    retains native recurring allocations, zeroes and sentinel initialization.
    """

    def __init__(self, world_size: int, local_experts: int):
        from .jit import load_kernel

        if (world_size, local_experts) not in SCHEDULER_LAYOUTS:
            raise ValueError("Unsupported EP size or expert count")
        self.world_size = world_size
        self.local_experts = local_experts
        self.count = load_kernel(f"count_{world_size}_{local_experts}")
        self.pad = load_kernel(f"pad_{world_size}")
        self.rows = load_kernel(f"rows_{world_size}_{local_experts}")

    def __call__(self, topk_all, schedule_capacity: int, rank: int):
        import torch
        import tvm_ffi

        if (
            not topk_all.is_cuda
            or topk_all.dtype != torch.int32
            or not topk_all.is_contiguous()
            or topk_all.ndim != 3
        ):
            raise ValueError("Expected contiguous CUDA int32 [EP, local_tokens, top_k]")
        world, local_tokens, top_k = topk_all.shape
        if world != self.world_size or not 0 <= rank < world:
            raise ValueError("Prepared scheduler EP/rank mismatch")
        if local_tokens < 256 or local_tokens % 256 or not 0 < top_k <= 255:
            raise ValueError("Expected aligned source rows and top_k in [1, 255]")
        if schedule_capacity < local_tokens * top_k or schedule_capacity % 256:
            raise ValueError(
                "Schedule capacity must be aligned and hold one source rank"
            )
        kwargs = dict(device=topk_all.device, dtype=torch.int32)
        peer_rank = torch.empty(schedule_capacity, **kwargs)
        peer_token = torch.empty(schedule_capacity, **kwargs)
        num_tokens = torch.zeros(1, **kwargs)
        per_expert = torch.empty(self.local_experts, **kwargs)
        counts = torch.zeros(self.local_experts * world, **kwargs)
        peer_rank.fill_(-1)
        # A single stream context preserves the surrounding native/graph stream.
        with tvm_ffi.use_torch_stream():
            self.count.launch(
                grid=((topk_all.numel() + 1023) // 1024, 1, 1),
                topk_all=topk_all,
                counts=counts,
                local_tokens=local_tokens,
                top_k=top_k,
                rank=rank,
            )
            self.pad.launch(
                grid=(self.local_experts, 1, 1),
                counts=counts,
                tokens_per_expert=per_expert,
                num_tokens=num_tokens,
            )
            self.rows.launch(
                grid=(self.local_experts * world, 1, 1),
                topk_all=topk_all,
                counts=counts,
                tokens_per_expert=per_expert,
                num_tokens=num_tokens,
                schedule_peer_rank=peer_rank,
                schedule_peer_token_idx=peer_token,
                local_tokens=local_tokens,
                top_k=top_k,
                capacity=schedule_capacity,
                rank=rank,
            )
        return peer_rank, peer_token, num_tokens, per_expert


class MoKEpilogues:
    """Prepared standalone epilogues: persistent (token, 1024-column) streams.

    Per vector the arithmetic is native: FP32 accumulation from the shared
    row, then the routed rows in ascending expert-slot order, one BF16
    rounding. Top-k changes the physical shared-memory layout.
    """

    def __init__(self, top_k: int):
        from .jit import load_kernel

        if type(top_k) is not int or top_k not in EPILOGUE_TOPKS:
            raise ValueError("Exported epilogues support top_k=2 or 8")
        self.top_k = top_k
        self.forward_module = load_kernel(f"epilogue_forward_{top_k}")
        self.backward_module = load_kernel(f"epilogue_backward_{top_k}")
        self._sms: dict = {}
        self._no_scores: dict = {}

    def _check(self, shared, routed):
        import torch

        if shared.ndim != 2 or min(shared.shape) <= 0 or shared.shape[1] % 8:
            raise ValueError(
                "Expected positive [tokens, hidden] with a 16-byte row stride"
            )
        if tuple(routed.shape) != (shared.shape[0] * self.top_k, shared.shape[1]):
            raise ValueError("Routed buffer must have [tokens * top_k, hidden] shape")
        for tensor in (shared, routed):
            if (
                not tensor.is_cuda
                or tensor.device != shared.device
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
            ):
                raise ValueError("Expected contiguous BF16 buffers on one CUDA device")

    def forward(self, shared, routed, scores):
        import torch
        import tvm_ffi

        self._check(shared, routed)
        tokens, hidden = shared.shape
        if (
            tuple(scores.shape) != (tokens, self.top_k)
            or scores.dtype != torch.float32
            or scores.device != shared.device
            or not scores.is_contiguous()
        ):
            raise ValueError("Scores must be contiguous CUDA FP32 [tokens, top_k]")
        output = torch.empty_like(shared)
        ctas = self._ctas(tokens, hidden, shared.device)
        with tvm_ffi.use_torch_stream():
            self.forward_module.launch(
                grid=(ctas, 1, 1),
                shared=shared,
                routed=routed,
                scores=scores,
                output=output.view(torch.uint32),
                tokens=tokens,
                hidden=hidden,
                ctas=ctas,
            )
        return output

    def backward(self, shared, routed):
        import torch
        import tvm_ffi

        self._check(shared, routed)
        tokens, hidden = shared.shape
        output = torch.empty_like(shared)
        ctas = self._ctas(tokens, hidden, shared.device)
        with tvm_ffi.use_torch_stream():
            self.backward_module.launch(
                grid=(ctas, 1, 1),
                shared=shared,
                routed=routed,
                scores=self._no_scores[shared.device],
                output=output.view(torch.uint32),
                tokens=tokens,
                hidden=hidden,
                ctas=ctas,
            )
        return output

    def _ctas(self, tokens, hidden, device):
        import torch

        if device not in self._sms:
            self._sms[device] = torch.cuda.get_device_properties(
                device
            ).multi_processor_count
            self._no_scores[device] = torch.zeros(1, dtype=torch.float32, device=device)
        # Two resident CTAs per SM (six 18 KB stages each for top-k 8).
        units = tokens * ((hidden + EPI_COLS - 1) // EPI_COLS)
        return max(1, min(units, 2 * self._sms[device]))
