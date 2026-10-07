# Copyright 2026 Cursor Research
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Modified: standalone CUDA module loading; original host launch boundaries.

def mok_swiglu_limit_args(swiglu_limit):
    """Encode the optional clamp as the kernels' ``(limit, clamped)`` scalars.

    ``None`` selects the exact unclamped path. A finite positive float enables
    ``silu(min(gate, L)) * clamp(up, -L, L)`` in both expert MLPs.
    """
    import math
    if swiglu_limit is None:
        return 0.0, 0
    if isinstance(swiglu_limit, bool) or not isinstance(swiglu_limit, (int, float)):
        raise TypeError("swiglu_limit must be None or a positive finite number")
    limit = float(swiglu_limit)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("swiglu_limit must be None or a positive finite number")
    return limit, 1


class MoKForward:
    """Low-level native-signature BF16 forward; caller owns peer barriers."""

    def __init__(self):
        from .jit import load_kernel
        self.module = load_kernel("forward")
        self._peer_tables = {}

    def __call__(self, x, x_ptrs, combine_buffer, combine_ptrs,
                 shared_gate, routed_gate, shared_up, routed_up, shared_down, routed_down,
                 peer_rank, peer_token, num_tokens, counts,
                 topk, swiglu_limit, comm_sms, macro_size, mini_size, source_rows=None,
                 recompute_only=False):
        """``x`` holds this rank's real source rows (any count, including a
        one-row placeholder when ``source_rows == 0``); shared-expert work and
        retained shared activations cover ``source_rows`` rows only.

        ``recompute_only`` rebuilds the backward context only (dispatch, gate/up
        GEMMs, SwiGLU): no down projections, no combine, no ``y`` outputs; the
        down weights and combine pointers are unused placeholders then, and the
        returned ``y_shared``/``y_routed`` are ``None``."""
        import torch
        import tvm_ffi
        limit, clamped = mok_swiglu_limit_args(swiglu_limit)
        local_tokens, hidden = x.shape
        if source_rows is None:
            source_rows = local_tokens
        if type(source_rows) is not int or not 0 <= source_rows <= local_tokens or local_tokens < 1:
            raise ValueError("source_rows must be an integer within the supplied source rows")
        local_tokens = source_rows
        rows_alloc = max(local_tokens, 1)
        intermediate, experts = shared_gate.shape[0], counts.numel()
        capacity = peer_rank.numel()
        if (x.dtype != torch.bfloat16 or hidden % 256
                or intermediate % 256 or macro_size % mini_size or mini_size % 256
                or comm_sms <= 0 or comm_sms % 2):
            raise ValueError("MoK native BF16 tile and communication geometry is required")
        key = (tuple(x_ptrs), tuple(combine_ptrs), x.device)
        if key not in self._peer_tables:
            self._peer_tables[key] = (torch.tensor(x_ptrs, dtype=torch.uint64, device=x.device),
                                      torch.tensor(combine_ptrs, dtype=torch.uint64, device=x.device))
        x_peers, y_peers = self._peer_tables[key]
        recompute = bool(recompute_only)
        options = dict(device=x.device, dtype=torch.bfloat16)
        x_routed = torch.empty((macro_size, hidden), **options)
        # Retained shared activations are sized by the real source rows.
        gate_shared = torch.empty((rows_alloc, intermediate), **options)
        gate_routed = torch.empty((macro_size, intermediate), **options)
        up_shared, up_routed = torch.empty_like(gate_shared), torch.empty_like(gate_routed)
        hidden_shared, hidden_routed = torch.empty_like(gate_shared), torch.empty_like(gate_routed)
        # Every routed activation, including the output, is a macrobatch ring;
        # the score derivative needs no retained routed output (see
        # mok_swiglu_backward).
        if recompute:
            # Unwritten map/pointer placeholders: the recompute has no down GEMM tasks.
            y_shared = y_routed = torch.empty((1, hidden), **options)
        else:
            y_shared = torch.empty((rows_alloc, hidden), **options)
            y_routed = torch.empty_like(x_routed)
        shared_rows, routed_rows = (local_tokens + 255) // 256, capacity // 256
        shared_gate_tasks = shared_rows * (intermediate // 256)
        mini_gate_tasks = (mini_size // 256) * (intermediate // 256)
        shared_swiglu = (shared_rows * 2 * (intermediate // 128) + 5) // 6   # whole 256-row blocks, as the kernel
        mini_swiglu = ((mini_size // 128) * (intermediate // 128) + 5) // 6
        shared_down_tasks = 0 if recompute else shared_rows * (hidden // 256)
        mini_down_tasks = 0 if recompute else (mini_size // 256) * (hidden // 256)
        shared_tasks = 2 * shared_gate_tasks + shared_swiglu + shared_down_tasks
        mini_tasks = 2 * mini_gate_tasks + mini_swiglu + mini_down_tasks
        minis = (capacity + mini_size - 1) // mini_size
        counter_opts = dict(dtype=torch.int32, device=x.device)
        x_ready = torch.zeros(minis, **counter_opts)
        gate_ready = torch.zeros((shared_rows + routed_rows) * (intermediate // 256), **counter_opts)
        hidden_ready = torch.zeros(shared_rows + routed_rows, **counter_opts)
        y_ready = torch.zeros(minis, **counter_opts)
        y_done = torch.zeros(capacity // 128, **counter_opts)
        with tvm_ffi.use_torch_stream():
            self.module.launch(grid=(2 * (shared_tasks + minis * mini_tasks) + comm_sms, 1, 1),
                x_shared=x[:rows_alloc], x_routed=x_routed,
                wg_shared=shared_gate.unsqueeze(0), wu_shared=shared_up.unsqueeze(0), wd_shared=shared_down.unsqueeze(0),
                wg_routed=routed_gate, wu_routed=routed_up, wd_routed=routed_down,
                gate_shared_out=gate_shared, up_shared_out=up_shared,
                gate_routed_out=gate_routed, up_routed_out=up_routed,
                gate_shared_in=gate_shared, up_shared_in=up_shared,
                gate_routed_in=gate_routed, up_routed_in=up_routed,
                hidden_shared_out=hidden_shared, hidden_routed_out=hidden_routed,
                hidden_shared_in=hidden_shared, hidden_routed_in=hidden_routed,
                y_shared=y_shared, y_routed=y_routed, x_routed_ptr=x_routed, y_routed_ptr=y_routed,
                x_peers=x_peers, y_peers=y_peers, schedule_rank=peer_rank, schedule_token=peer_token,
                num_tokens=num_tokens, counts=counts, gate_ready=gate_ready, hidden_ready=hidden_ready,
                x_ready=x_ready, y_ready=y_ready, y_done=y_done,
                local_tokens=local_tokens, hidden=hidden, intermediate=intermediate, experts=experts,
                topk=topk, comm_sms=comm_sms, macro_size=macro_size, mini_size=mini_size,
                swiglu_limit=limit, swiglu_clamped=clamped, recompute_only=int(recompute))
        if recompute:
            return x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, None, None
        return x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed


class MoKBackward:
    """Native-signature BF16 backward and its separate empty-expert zeroing."""

    def __init__(self):
        from .jit import load_kernel
        self.module = load_kernel("backward")
        self.zero_module = load_kernel("zero")
        self._peer_tables = {}

    def __call__(self, dy_buffer, dy_ptrs, dx_buffer, dx_ptrs, weight_buffer, weight_ptrs,
                 dweight_buffer, dweight_ptrs, shared_gate, routed_gate, shared_up, routed_up,
                 shared_down, routed_down, x_routed, gate_shared, gate_routed,
                 up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                 peer_rank, peer_token, num_tokens, counts,
                 topk, swiglu_limit, comm_sms, macro_size, mini_size, source_rows=None):
        """``dy_buffer`` and ``x`` hold this rank's real source rows (any
        count; one-row placeholders when ``source_rows == 0``). The context
        carries the macrobatch rings only; the score derivative is computed
        from the unscaled routed gradient inside the SwiGLU backward."""
        import torch
        import tvm_ffi

        limit, clamped = mok_swiglu_limit_args(swiglu_limit)
        local_tokens, hidden = x.shape
        if source_rows is None:
            source_rows = local_tokens
        if (type(source_rows) is not int or not 0 <= source_rows <= local_tokens or local_tokens < 1
                or dy_buffer.shape[0] < max(source_rows, 1) or dy_buffer.shape[1] != hidden):
            raise ValueError("source_rows must be an integer within the supplied source rows")
        local_tokens = source_rows
        rows_alloc = max(local_tokens, 1)
        intermediate, experts, capacity = shared_gate.shape[0], counts.numel(), peer_rank.numel()
        if (gate_shared.shape[0] < rows_alloc or hidden_shared.shape[0] < rows_alloc
                or up_shared.shape[0] < rows_alloc):
            raise ValueError("The forward context must cover the source rows")
        if (x.dtype != torch.bfloat16 or hidden % 256 or intermediate % 256
                or macro_size % mini_size or mini_size % 256 or comm_sms <= 0 or comm_sms % 2):
            raise ValueError("MoK native BF16 tile and communication geometry is required")
        key = (tuple(x_ptrs), tuple(dy_ptrs), tuple(dx_ptrs), tuple(weight_ptrs), tuple(dweight_ptrs), x.device)
        if key not in self._peer_tables:
            self._peer_tables[key] = tuple(torch.tensor(pointers, dtype=torch.uint64, device=x.device)
                for pointers in (x_ptrs, dy_ptrs, dx_ptrs, weight_ptrs, dweight_ptrs))
        x_peers, dy_peers, dx_peers, weight_peers, dweight_peers = self._peer_tables[key]
        options = dict(device=x.device, dtype=torch.bfloat16)
        router_weights = torch.empty(macro_size, dtype=torch.float32, device=x.device)
        partials = torch.empty(macro_size, intermediate // 128, dtype=torch.float32, device=x.device)
        dy_routed = torch.empty((macro_size, hidden), **options)
        dy_scaled = torch.empty_like(dy_routed)
        dh_shared = torch.empty((rows_alloc, intermediate), **options)
        dh_routed = torch.empty((macro_size, intermediate), **options)
        dg_shared, dg_routed = torch.empty_like(dh_shared), torch.empty_like(dh_routed)
        du_shared, du_routed = torch.empty_like(dh_shared), torch.empty_like(dh_routed)
        dx_shared, dx_routed = torch.empty((rows_alloc, hidden), **options), torch.empty_like(dy_routed)
        x_rows, dy_rows = x[:rows_alloc], dy_buffer[:rows_alloc]
        gate_rows, up_rows, hidden_rows = gate_shared[:rows_alloc], up_shared[:rows_alloc], hidden_shared[:rows_alloc]
        # A rank without source rows has no shared-expert wgrad tiles: the kernel
        # skips empty K ranges (as it skips empty routed experts, whose gradients
        # the separate zeroing launch clears), so its shared gradients are zero.
        shared_grad = torch.zeros_like if local_tokens == 0 else torch.empty_like
        dwg_shared, dwg_routed = shared_grad(shared_gate), torch.empty_like(routed_gate)
        dwu_shared, dwu_routed = shared_grad(shared_up), torch.empty_like(routed_up)
        dwd_shared, dwd_routed = shared_grad(shared_down), torch.empty_like(routed_down)
        minis = (capacity + mini_size - 1) // mini_size
        macros = (capacity + macro_size - 1) // macro_size
        shared_rows, routed_rows = (local_tokens + 255) // 256, capacity // 256
        ib, hb = intermediate // 256, hidden // 256
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
        shared_tasks = shared_rows * (ib + hb) + (shared_rows * 2 * (intermediate // 128) + 3) // 4 + 3 * ib * hb
        mini_bwd = (mini_size // 256) * (ib + hb) + ((mini_size // 128) * (intermediate // 128) + 3) // 4
        mini_replay = 2 * (mini_size // 256) * ib + ((mini_size // 128) * (intermediate // 128) + 5) // 6
        clusters = shared_tasks + minis * mini_bwd + max(0, minis - macro_size // mini_size) * mini_replay + macros * 3 * experts * ib * hb
        with tvm_ffi.use_torch_stream():
            self.module.launch(grid=(comm_sms + 2 * clusters, 1, 1),
                dy_s=dy_rows, dy_r=dy_routed, dg_s=dg_shared, dg_r=dg_routed,
                du_s=du_shared, du_r=du_routed, x_nt_r=x_routed,
                dy_atb_s=dy_rows, dy_atb_r=dy_scaled, dg_atb_s=dg_shared, dg_atb_r=dg_routed,
                du_atb_s=du_shared, du_atb_r=du_routed, x_atb_s=x_rows, x_atb_r=x_routed,
                h_atb_s=hidden_rows, h_atb_r=hidden_routed,
                wg_s=shared_gate.unsqueeze(0), wu_s=shared_up.unsqueeze(0), wd_s=shared_down.unsqueeze(0),
                wg_r=routed_gate, wu_r=routed_up, wd_r=routed_down, wg_nt_r=routed_gate, wu_nt_r=routed_up,
                dh_s=dh_shared, dh_r=dh_routed, dx_s=dx_shared, dx_r=dx_routed,
                gate_out_r=gate_routed, up_out_r=up_routed,
                dwg_s=dwg_shared.unsqueeze(0), dwu_s=dwu_shared.unsqueeze(0), dwd_s=dwd_shared.unsqueeze(0),
                dwg_r=dwg_routed, dwu_r=dwu_routed, dwd_r=dwd_routed,
                dh_sw_s=dh_shared, dh_sw_r=dh_routed, gate_sw_s=gate_rows, gate_sw_r=gate_routed,
                up_sw_s=up_rows, up_sw_r=up_routed, dg_sw_s=dg_shared, dg_sw_r=dg_routed,
                du_sw_s=du_shared, du_sw_r=du_routed, h_sw_r=hidden_routed,
                x_routed_ptr=x_routed, dy_routed_ptr=dy_routed, dy_scaled_ptr=dy_scaled,
                dx_routed_ptr=dx_routed, weights=router_weights, partials=partials,
                x_peers=x_peers, dy_peers=dy_peers, dx_peers=dx_peers,
                weight_peers=weight_peers, dweight_peers=dweight_peers,
                schedule_rank=peer_rank, schedule_token=peer_token, num_tokens=num_tokens, counts=counts,
                dh_ready=dh_ready, dg_ready=dg_ready, dy_ready=dy_ready, dx_ready=dx_ready,
                replay_x=replay_x, replay_gu=replay_gu, replay_h=replay_h,
                buffers_done=buffers_done, weight_ready=weight_ready,
                local_tokens=local_tokens, hidden=hidden, intermediate=intermediate, experts=experts,
                topk=topk, comm_sms=comm_sms, macro_size=macro_size, mini_size=mini_size,
                swiglu_limit=limit, swiglu_clamped=clamped)
            self.zero_module.launch(grid=(128, experts, 1), gate=dwg_routed.view(torch.uint16),
                up=dwu_routed.view(torch.uint16), down=dwd_routed.view(torch.uint16),
                counts=counts, elements=hidden * intermediate)
        return (dx_shared, dx_routed, dg_shared, dg_routed, du_shared, du_routed,
                dh_shared, dh_routed, dy_routed, dwg_shared, dwg_routed,
                dwu_shared, dwu_routed, dwd_shared, dwd_routed)


class MoKCommunication:
    """Prepared launchers over native MoK's caller-owned symmetric buffers."""

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

        if (not routes.is_cuda or routes.dtype != torch.int32
                or not routes.is_contiguous() or routes.ndim != 2):
            raise ValueError("Routes must be contiguous CUDA int32 [local_tokens, top_k]")
        if (gathered.ndim != 3 or tuple(gathered.shape[1:]) != tuple(routes.shape)
                or gathered.dtype != routes.dtype or gathered.device != routes.device
                or not gathered.is_contiguous()):
            raise ValueError("Gathered buffer must match the route shape, dtype and device")
        ep = gathered.shape[0]
        if ep not in (1, 4, 8, 16, 32, 64) or not 0 <= rank < ep:
            raise ValueError("Unsupported EP/rank")
        if type(multicast_address) is not int or multicast_address <= 0:
            raise ValueError("A live multicast address is required")
        if routes.numel() * 4 % self.chunk_bytes:
            raise ValueError("Metadata chunk must divide each rank's route-buffer bytes")
        if ep == 1:
            gathered[0].copy_(routes)
        else:
            with tvm_ffi.use_torch_stream():
                self.gather_module.launch(
                    grid=(routes.numel() * 4 // self.chunk_bytes, 1, 1),
                    local_routes=routes, multicast_address=multicast_address,
                    rank=rank, numel=routes.numel(),
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
            if (not tensor.is_cuda or tensor.device != counter.device
                    or tensor.dtype != torch.int32 or tuple(tensor.shape) != (1,)
                    or not tensor.is_contiguous()):
                raise ValueError("Barrier counters must be contiguous CUDA int32 [1]")
        if ep > 1:
            with tvm_ffi.use_torch_stream():
                self.barrier_module.launch(
                    grid=(1, 1, 1), local_counter=counter.view(torch.uint32),
                    target_counter=target.view(torch.uint32),
                    multicast_address=multicast_address, ep_size=ep,
                )


class MoKScheduler:
    """Three prepared kernel modules; allocations/resets match native MoK.

    Compilation is setup and must precede graph capture. The launch method
    retains native recurring allocations, zeroes and sentinel initialization.
    """

    def __init__(self, world_size: int, local_experts: int):
        from .jit import load_kernel

        if (world_size, local_experts) not in ((1, 4), (4, 4), (16, 16), (64, 4), (4, 64), (8, 32), (32, 8), (8, 36), (32, 9)):
            raise ValueError("Unsupported EP size or expert count")
        self.world_size = world_size
        self.local_experts = local_experts
        self.count = load_kernel(f"count_{world_size}_{local_experts}")
        self.pad = load_kernel(f"pad_{world_size}_{local_experts}")
        self.rows = load_kernel(f"rows_{world_size}_{local_experts}")

    def __call__(self, topk_all, schedule_capacity: int, rank: int):
        import torch
        import tvm_ffi

        if (not topk_all.is_cuda or topk_all.dtype != torch.int32
                or not topk_all.is_contiguous() or topk_all.ndim != 3):
            raise ValueError("Expected contiguous CUDA int32 [EP, local_tokens, top_k]")
        world, local_tokens, top_k = topk_all.shape
        if world != self.world_size or not 0 <= rank < world:
            raise ValueError("Prepared scheduler EP/rank mismatch")
        if local_tokens < 256 or local_tokens % 256 or not 0 < top_k <= 255:
            raise ValueError("Expected aligned source rows and top_k in [1, 255]")
        if schedule_capacity < local_tokens * top_k or schedule_capacity % 256:
            raise ValueError("Schedule capacity must be aligned and hold one source rank")
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
                topk_all=topk_all, counts=counts,
                local_tokens=local_tokens, top_k=top_k, rank=rank,
            )
            self.pad.launch(
                grid=(self.local_experts, 1, 1), counts=counts,
                tokens_per_expert=per_expert, num_tokens=num_tokens,
            )
            self.rows.launch(
                grid=(self.local_experts * world, 1, 1),
                topk_all=topk_all, counts=counts, tokens_per_expert=per_expert,
                num_tokens=num_tokens, schedule_peer_rank=peer_rank,
                schedule_peer_token_idx=peer_token, local_tokens=local_tokens,
                top_k=top_k, capacity=schedule_capacity, rank=rank,
            )
        return peer_rank, peer_token, num_tokens, per_expert


TILE = 128



def _check_input(x_bf16, return_normal, return_transposed):
    import torch

    if x_bf16.ndim not in (2, 3):
        raise ValueError("x_bf16 must have shape (M, N) or (E, M, N)")
    if any(size <= 0 for size in x_bf16.shape):
        raise ValueError("x_bf16 dimensions must be positive")
    if x_bf16.shape[-2] % TILE or x_bf16.shape[-1] % TILE:
        raise ValueError("x_bf16 M and N dimensions must be divisible by 128")
    if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
        raise ValueError("x_bf16 must be a contiguous bfloat16 tensor")
    if type(return_normal) is not bool or type(return_transposed) is not bool:
        raise TypeError("return_normal and return_transposed must be booleans")
    if not return_normal and not return_transposed:
        raise ValueError("at least one quantized layout must be requested")


def mok_scale_tiles(rows, cols):
    """Element count of MoK's ``[rows / 128 * cols / 128, 32, 16]`` E8M0 scale tensor."""
    return (rows // 128) * (cols // 128)


class MoKMxfp8Quantize:
    """Prepared ``mxfp8_quantize`` with MoK's three template variants."""

    def __init__(self):
        from .jit import load_kernel

        self.modules = {
            (True, True): load_kernel("quantize_normal_transposed"),
            (True, False): load_kernel("quantize_normal"),
            (False, True): load_kernel("quantize_transposed"),
        }

    def __call__(self, x_bf16, return_normal, return_transposed):
        import torch
        import tvm_ffi

        _check_input(x_bf16, return_normal, return_transposed)
        if not x_bf16.is_cuda:
            raise ValueError("x_bf16 must be a CUDA tensor")
        x3 = x_bf16 if x_bf16.ndim == 3 else x_bf16.unsqueeze(0)
        experts, rows, cols = x3.shape
        row_blocks, col_blocks = rows // TILE, cols // TILE
        fp8 = dict(device=x_bf16.device, dtype=torch.float8_e4m3fn)
        byte = dict(device=x_bf16.device, dtype=torch.uint8)
        # Unused outputs of a single-layout variant receive one tile of scratch,
        # as MoK's fake global layouts; the compiled-out pass never touches it.
        x_fp8 = torch.empty(x3.shape, **fp8) if return_normal else torch.empty((1, TILE, TILE), **fp8)
        x_sc = (
            torch.empty((experts * row_blocks, col_blocks, 32, 16), **byte)
            if return_normal
            else torch.empty((1, 1, 32, 16), **byte)
        )
        x_fp8_t = (
            torch.empty((experts, cols, rows), **fp8) if return_transposed else torch.empty((1, TILE, TILE), **fp8)
        )
        x_sc_t = (
            torch.empty((experts * col_blocks, row_blocks, 32, 16), **byte)
            if return_transposed
            else torch.empty((1, 1, 32, 16), **byte)
        )
        with tvm_ffi.use_torch_stream():
            self.modules[(return_normal, return_transposed)].launch(
                grid=(experts * row_blocks * col_blocks, 1, 1),
                x_bf16=x3,
                x_fp8=x_fp8.view(torch.uint8),
                x_fp8_t=x_fp8_t.view(torch.uint8),
                x_sc=x_sc.view(-1, 16),
                x_sc_t=x_sc_t.view(-1, 16),
                col_blocks=col_blocks,
                row_blocks=row_blocks,
            )
        squeeze = x_bf16.ndim == 2
        return (
            (x_fp8.squeeze(0) if squeeze else x_fp8) if return_normal else None,
            x_sc if return_normal else None,
            (x_fp8_t.squeeze(0) if squeeze else x_fp8_t) if return_transposed else None,
            x_sc_t if return_transposed else None,
        )


class MoKForwardMxfp8:
    """Low-level native-signature MXFP8 forward (MoK ``dispatch_mlp_swiglu_combine_fwd_mxfp8``);
    caller owns peer barriers. Routed weights are MoK ``mxfp8_quantize(w, True, True)`` tuples."""

    def __init__(self):
        from .jit import load_kernel

        self.module = load_kernel("forward_mxfp8")
        self._peer_tables = {}
        # Rings that MoK does not return (GEMM/SwiGLU inputs); kept for diagnostics.
        self.last_rings = None

    @staticmethod
    def _weight_tuple(weights, name):
        """MoK's forward takes ``(w_fp8, w_sc)`` pairs; the full ``mxfp8_quantize`` 4-tuple is accepted too."""
        import torch

        if not isinstance(weights, (tuple, list)) or len(weights) not in (2, 4):
            raise TypeError(
                f"{name} must be the (w_fp8, w_sc) pair or (w_fp8, w_sc, w_t_fp8, w_t_sc) tuple of mxfp8_quantize"
            )
        w_fp8, w_sc = weights[0], weights[1]
        if w_fp8.dtype != torch.float8_e4m3fn or w_sc.dtype != torch.uint8 or w_fp8.ndim != 3:
            raise TypeError(f"{name}: expected E4M3 [E, N, K] weights with uint8 scale tiles")
        experts, n, k = w_fp8.shape
        if tuple(w_sc.shape) != (experts * (n // 128), k // 128, 32, 16):
            raise ValueError(f"{name}: scale tiles must be [E * N / 128, K / 128, 32, 16]")
        return w_fp8, w_sc

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
        source_rows=None,
        recompute_only=False,
    ):
        import torch
        import tvm_ffi


        limit, clamped = mok_swiglu_limit_args(swiglu_limit)
        local_tokens, hidden = x.shape
        if source_rows is None:
            source_rows = local_tokens
        if type(source_rows) is not int or not 0 <= source_rows <= local_tokens or local_tokens < 1:
            raise ValueError("source_rows must be an integer within the supplied source rows")
        local_tokens = source_rows
        rows_alloc = max(local_tokens, 1)
        intermediate, experts = shared_gate.shape[0], counts.numel()
        capacity = peer_rank.numel()
        recompute = bool(recompute_only)
        wg_fp8, wg_sc = self._weight_tuple(routed_gate, "routed_gate")
        wu_fp8, wu_sc = self._weight_tuple(routed_up, "routed_up")
        if recompute and routed_down is None:
            # MoK's recompute_forward_context takes no down weights: the down projection never runs,
            # the tensor maps only need valid storage.
            wd_fp8, wd_sc = wg_fp8, wg_sc
            down_shape_ok = True
        else:
            wd_fp8, wd_sc = self._weight_tuple(routed_down, "routed_down")
            down_shape_ok = tuple(wd_fp8.shape) == (experts, hidden, intermediate)
        if (
            x.dtype != torch.bfloat16
            or hidden % 256
            or intermediate % 256
            or macro_size % mini_size
            or mini_size % 256
            or comm_sms <= 0
            or comm_sms % 2
            or tuple(wg_fp8.shape) != (experts, intermediate, hidden)
            or tuple(wu_fp8.shape) != (experts, intermediate, hidden)
            or not down_shape_ok
        ):
            raise ValueError("MoK native MXFP8 tile and communication geometry is required")
        key = (tuple(x_ptrs), tuple(combine_ptrs), x.device)
        if key not in self._peer_tables:
            self._peer_tables[key] = (
                torch.tensor(x_ptrs, dtype=torch.uint64, device=x.device),
                torch.tensor(combine_ptrs, dtype=torch.uint64, device=x.device),
            )
        x_peers, y_peers = self._peer_tables[key]
        bf16 = dict(device=x.device, dtype=torch.bfloat16)
        fp8 = dict(device=x.device, dtype=torch.float8_e4m3fn)
        byte = dict(device=x.device, dtype=torch.uint8)
        # Routed macrobatch rings (MoK globals_fwd), both quantized layouts where MoK keeps them.
        x_fp8_routed = torch.empty((macro_size, hidden), **fp8)
        x_sc_routed = torch.empty((mok_scale_tiles(macro_size, hidden), 32, 16), **byte)
        x_fp8_t_routed = torch.empty((hidden, macro_size), **fp8)
        x_sc_t_routed = torch.empty((mok_scale_tiles(hidden, macro_size), 32, 16), **byte)
        gate_routed = torch.empty((macro_size, intermediate), **bf16)
        up_routed = torch.empty_like(gate_routed)
        gate_fp8_routed = torch.empty((macro_size, intermediate), **fp8)
        up_fp8_routed = torch.empty_like(gate_fp8_routed)
        gate_sc_routed = torch.empty((mok_scale_tiles(macro_size, intermediate), 32, 16), **byte)
        up_sc_routed = torch.empty_like(gate_sc_routed)
        hidden_fp8_routed = torch.empty((macro_size, intermediate), **fp8)
        hidden_sc_routed = torch.empty_like(gate_sc_routed)
        hidden_fp8_t_routed = torch.empty((intermediate, macro_size), **fp8)
        hidden_sc_t_routed = torch.empty((mok_scale_tiles(intermediate, macro_size), 32, 16), **byte)
        # Retained shared activations are sized by the real source rows.
        gate_shared = torch.empty((rows_alloc, intermediate), **bf16)
        up_shared, hidden_shared = torch.empty_like(gate_shared), torch.empty_like(gate_shared)
        if recompute:
            y_shared = y_routed = torch.empty((1, hidden), **bf16)
        else:
            y_shared = torch.empty((rows_alloc, hidden), **bf16)
            y_routed = torch.empty((macro_size, hidden), **bf16)
        shared_rows, routed_rows = (local_tokens + 255) // 256, capacity // 256
        shared_gate_tasks = shared_rows * (intermediate // 256)
        mini_gate_tasks = (mini_size // 256) * (intermediate // 256)
        shared_swiglu = (shared_rows * 2 * (intermediate // 128) + 5) // 6
        mini_swiglu = ((mini_size // 128) * (intermediate // 128) + 5) // 6
        shared_down_tasks = 0 if recompute else shared_rows * (hidden // 256)
        mini_down_tasks = 0 if recompute else (mini_size // 256) * (hidden // 256)
        shared_tasks = 2 * shared_gate_tasks + shared_swiglu + shared_down_tasks
        mini_tasks = 2 * mini_gate_tasks + mini_swiglu + mini_down_tasks
        minis = (capacity + mini_size - 1) // mini_size
        counter_opts = dict(dtype=torch.int32, device=x.device)
        x_ready = torch.zeros(minis, **counter_opts)
        gate_ready = torch.zeros((shared_rows + routed_rows) * (intermediate // 256), **counter_opts)
        hidden_ready = torch.zeros(shared_rows + routed_rows, **counter_opts)
        y_ready = torch.zeros(minis, **counter_opts)
        y_done = torch.zeros(capacity // 128, **counter_opts)
        u8 = torch.uint8
        with tvm_ffi.use_torch_stream():
            self.module.launch(
                grid=(2 * (shared_tasks + minis * mini_tasks) + comm_sms, 1, 1),
                x_shared=x[:rows_alloc],
                x_routed=x_fp8_routed.view(u8),
                x_routed_sc=x_sc_routed.view(-1, 16),
                wg_shared=shared_gate.unsqueeze(0),
                wu_shared=shared_up.unsqueeze(0),
                wd_shared=shared_down.unsqueeze(0),
                wg_routed=wg_fp8.view(u8),
                wu_routed=wu_fp8.view(u8),
                wd_routed=wd_fp8.view(u8),
                wg_routed_sc=wg_sc.view(-1, 16),
                wu_routed_sc=wu_sc.view(-1, 16),
                wd_routed_sc=wd_sc.view(-1, 16),
                gate_shared_out=gate_shared,
                up_shared_out=up_shared,
                gate_routed_out=gate_routed,
                up_routed_out=up_routed,
                gate_routed_fp8=gate_fp8_routed.view(u8),
                up_routed_fp8=up_fp8_routed.view(u8),
                gate_routed_sc=gate_sc_routed.view(-1, 16),
                up_routed_sc=up_sc_routed.view(-1, 16),
                gate_shared_in=gate_shared,
                up_shared_in=up_shared,
                gate_routed_in=gate_routed,
                up_routed_in=up_routed,
                hidden_shared_out=hidden_shared,
                hidden_routed_fp8=hidden_fp8_routed.view(u8),
                hidden_routed_sc=hidden_sc_routed.view(-1, 16),
                hidden_routed_fp8_t=hidden_fp8_t_routed.view(u8),
                hidden_routed_sc_t=hidden_sc_t_routed.view(-1, 16),
                hidden_shared_in=hidden_shared,
                hidden_routed_in=hidden_fp8_routed.view(u8),
                hidden_routed_in_sc=hidden_sc_routed.view(-1, 16),
                y_shared=y_shared,
                y_routed=y_routed,
                x_dispatch_fp8=x_fp8_routed.view(u8),
                x_dispatch_sc=x_sc_routed.view(-1, 16),
                x_dispatch_fp8_t=x_fp8_t_routed.view(u8),
                x_dispatch_sc_t=x_sc_t_routed.view(-1, 16),
                y_routed_ptr=y_routed,
                x_peers=x_peers,
                y_peers=y_peers,
                schedule_rank=peer_rank,
                schedule_token=peer_token,
                num_tokens=num_tokens,
                counts=counts,
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
                swiglu_limit=limit,
                swiglu_clamped=clamped,
                recompute_only=int(recompute),
            )
        self.last_rings = dict(
            x_fp8_routed=x_fp8_routed,
            x_sc_routed=x_sc_routed,
            gate_routed=gate_routed,
            up_routed=up_routed,
            hidden_fp8_routed=hidden_fp8_routed,
            hidden_sc_routed=hidden_sc_routed,
        )
        outputs = (None, None) if recompute else (y_shared, y_routed)
        return (
            x_fp8_t_routed,
            x_sc_t_routed,
            gate_shared,
            gate_fp8_routed,
            gate_sc_routed,
            up_shared,
            up_fp8_routed,
            up_sc_routed,
            hidden_shared,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            *outputs,
        )


class MoKBackwardMxfp8:
    """Native-signature MXFP8 backward (MoK ``dispatch_mlp_swiglu_combine_bwd_mxfp8``) and its
    separate empty-expert zeroing; the caller owns the peer barriers.

    Routed weights are MoK ``mxfp8_quantize(w, True, True)`` tuples ``(w_fp8, w_sc, w_t_fp8, w_t_sc)``:
    the normal tuple feeds the replayed gate/up GEMMs, the transposed tuple the input gradients.
    ``wgrad_f32`` returns the routed weight gradients in FP32 with exact FP32 accumulation across
    macrobatches (BF16 otherwise, as the native kernel).
    """

    def __init__(self, wgrad_f32=False):
        from .jit import load_kernel

        self.wgrad_f32 = bool(wgrad_f32)
        self.module = load_kernel("backward_mxfp8_f32" if self.wgrad_f32 else "backward_mxfp8")
        self.zero_module = load_kernel("zero")
        self._peer_tables = {}

    @staticmethod
    def _weight_tuple(weights, name, experts, n, k, transposed_pair=False):
        """Validate one routed weight: the ``mxfp8_quantize`` 4-tuple, or (``transposed_pair``, MoK's down
        projection in the backward) the ``(w_t_fp8, w_t_sc)`` pair; absent members are returned as None."""
        import torch

        if transposed_pair and isinstance(weights, (tuple, list)) and len(weights) == 2:
            weights = (None, None, weights[0], weights[1])
        if not isinstance(weights, (tuple, list)) or len(weights) != 4:
            expected = "(w_t_fp8, w_t_sc) pair or the " if transposed_pair else ""
            raise TypeError(f"{name} must be the {expected}(w_fp8, w_sc, w_t_fp8, w_t_sc) tuple of mxfp8_quantize")
        w_fp8, w_sc, w_t_fp8, w_t_sc = weights
        for tensor, shape, label in ((w_fp8, (experts, n, k), "normal"), (w_t_fp8, (experts, k, n), "transposed")):
            if tensor is not None and (tensor.dtype != torch.float8_e4m3fn or tuple(tensor.shape) != shape):
                raise ValueError(f"{name}: expected E4M3 {label} weights of shape {shape}")
        for tensor, (rows, cols) in ((w_sc, (experts * n, k)), (w_t_sc, (experts * k, n))):
            if tensor is not None and (
                tensor.dtype != torch.uint8 or tuple(tensor.shape) != (rows // 128, cols // 128, 32, 16)
            ):
                raise ValueError(f"{name}: scale tiles must be [E * N / 128, K / 128, 32, 16]")
        return w_fp8, w_sc, w_t_fp8, w_t_sc

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
        x_fp8_t_routed,
        x_sc_t_routed,
        gate_shared,
        gate_fp8_routed,
        gate_sc_routed,
        up_shared,
        up_fp8_routed,
        up_sc_routed,
        hidden_shared,
        hidden_fp8_t_routed,
        hidden_sc_t_routed,
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
        source_rows=None,
    ):
        """``dy_buffer`` and ``x`` hold this rank's real source rows (any count; one-row
        placeholders when ``source_rows == 0``). The context carries MoK's saved MXFP8 tensors
        (``x`` transposed, ``gate``/``up`` normal, ``hidden`` transposed) plus the BF16 shared-expert
        activations; later macrobatches are replayed in the kernel. Returns MoK's 18 outputs."""
        import torch
        import tvm_ffi


        limit, clamped = mok_swiglu_limit_args(swiglu_limit)
        local_tokens, hidden = x.shape
        if source_rows is None:
            source_rows = local_tokens
        if (
            type(source_rows) is not int
            or not 0 <= source_rows <= local_tokens
            or local_tokens < 1
            or dy_buffer.shape[0] < max(source_rows, 1)
            or dy_buffer.shape[1] != hidden
        ):
            raise ValueError("source_rows must be an integer within the supplied source rows")
        local_tokens = source_rows
        rows_alloc = max(local_tokens, 1)
        intermediate, experts, capacity = shared_gate.shape[0], counts.numel(), peer_rank.numel()
        if (
            x.dtype != torch.bfloat16
            or hidden % 256
            or intermediate % 256
            or macro_size % mini_size
            or mini_size % 256
            or comm_sms <= 0
            or comm_sms % 2
        ):
            raise ValueError("MoK native MXFP8 tile and communication geometry is required")
        wg_fp8, wg_sc, wg_t_fp8, wg_t_sc = self._weight_tuple(routed_gate, "routed_gate", experts, intermediate, hidden)
        wu_fp8, wu_sc, wu_t_fp8, wu_t_sc = self._weight_tuple(routed_up, "routed_up", experts, intermediate, hidden)
        _, _, wd_t_fp8, wd_t_sc = self._weight_tuple(
            routed_down, "routed_down", experts, hidden, intermediate, transposed_pair=True
        )
        if gate_shared.shape[0] < rows_alloc or hidden_shared.shape[0] < rows_alloc or up_shared.shape[0] < rows_alloc:
            raise ValueError("The forward context must cover the source rows")
        byte = dict(device=x.device, dtype=torch.uint8)
        fp8 = dict(device=x.device, dtype=torch.float8_e4m3fn)
        bf16 = dict(device=x.device, dtype=torch.bfloat16)
        for tensor, shape, dtype, name in (
            (x_fp8_t_routed, (hidden, macro_size), torch.float8_e4m3fn, "x_fp8_t_routed"),
            (x_sc_t_routed, (mok_scale_tiles(hidden, macro_size), 32, 16), torch.uint8, "x_sc_t_routed"),
            (gate_fp8_routed, (macro_size, intermediate), torch.float8_e4m3fn, "gate_fp8_routed"),
            (gate_sc_routed, (mok_scale_tiles(macro_size, intermediate), 32, 16), torch.uint8, "gate_sc_routed"),
            (up_fp8_routed, (macro_size, intermediate), torch.float8_e4m3fn, "up_fp8_routed"),
            (up_sc_routed, (mok_scale_tiles(macro_size, intermediate), 32, 16), torch.uint8, "up_sc_routed"),
            (hidden_fp8_t_routed, (intermediate, macro_size), torch.float8_e4m3fn, "hidden_fp8_t_routed"),
            (
                hidden_sc_t_routed,
                (mok_scale_tiles(intermediate, macro_size), 32, 16),
                torch.uint8,
                "hidden_sc_t_routed",
            ),
        ):
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(f"{name}: expected {dtype} of shape {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
        key = (tuple(x_ptrs), tuple(dy_ptrs), tuple(dx_ptrs), tuple(weight_ptrs), tuple(dweight_ptrs), x.device)
        if key not in self._peer_tables:
            self._peer_tables[key] = tuple(
                torch.tensor(pointers, dtype=torch.uint64, device=x.device)
                for pointers in (x_ptrs, dy_ptrs, dx_ptrs, weight_ptrs, dweight_ptrs)
            )
        x_peers, dy_peers, dx_peers, weight_peers, dweight_peers = self._peer_tables[key]
        # Replayed forward activations (MoK globals_bwd).
        x_fp8_routed = torch.empty((macro_size, hidden), **fp8)
        x_sc_routed = torch.empty((mok_scale_tiles(macro_size, hidden), 32, 16), **byte)
        gate_routed = torch.empty((macro_size, intermediate), **bf16)
        up_routed = torch.empty_like(gate_routed)
        hidden_fp8_routed = torch.empty((macro_size, intermediate), **fp8)
        hidden_sc_routed = torch.empty((mok_scale_tiles(macro_size, intermediate), 32, 16), **byte)
        router_weights = torch.empty(macro_size, dtype=torch.float32, device=x.device)
        partials = torch.empty(macro_size, intermediate // 128, dtype=torch.float32, device=x.device)
        # Gradients.
        dy_fp8_routed = torch.empty((macro_size, hidden), **fp8)
        dy_sc_routed = torch.empty((mok_scale_tiles(macro_size, hidden), 32, 16), **byte)
        dy_fp8_t_routed = torch.empty((hidden, macro_size), **fp8)
        dy_sc_t_routed = torch.empty((mok_scale_tiles(hidden, macro_size), 32, 16), **byte)
        dh_shared = torch.empty((rows_alloc, intermediate), **bf16)
        dh_routed = torch.empty((macro_size, intermediate), **bf16)
        dg_shared, du_shared = torch.empty_like(dh_shared), torch.empty_like(dh_shared)
        dg_fp8_routed = torch.empty((macro_size, intermediate), **fp8)
        dg_sc_routed = torch.empty((mok_scale_tiles(macro_size, intermediate), 32, 16), **byte)
        dg_fp8_t_routed = torch.empty((intermediate, macro_size), **fp8)
        dg_sc_t_routed = torch.empty((mok_scale_tiles(intermediate, macro_size), 32, 16), **byte)
        du_fp8_routed, du_sc_routed = torch.empty_like(dg_fp8_routed), torch.empty_like(dg_sc_routed)
        du_fp8_t_routed, du_sc_t_routed = torch.empty_like(dg_fp8_t_routed), torch.empty_like(dg_sc_t_routed)
        dx_shared = torch.empty((rows_alloc, hidden), **bf16)
        dx_routed = torch.empty((macro_size, hidden), **bf16)
        # A rank without source rows has no shared-expert wgrad tiles (empty K ranges store nothing).
        shared_grad = torch.zeros_like if local_tokens == 0 else torch.empty_like
        dwg_shared, dwu_shared, dwd_shared = shared_grad(shared_gate), shared_grad(shared_up), shared_grad(shared_down)
        wgrad = dict(device=x.device, dtype=torch.float32 if self.wgrad_f32 else torch.bfloat16)
        dwg_routed = torch.empty((experts, intermediate, hidden), **wgrad)
        dwu_routed = torch.empty((experts, intermediate, hidden), **wgrad)
        dwd_routed = torch.empty((experts, hidden, intermediate), **wgrad)
        x_rows, dy_rows = x[:rows_alloc], dy_buffer[:rows_alloc]
        gate_rows, up_rows, hidden_rows = gate_shared[:rows_alloc], up_shared[:rows_alloc], hidden_shared[:rows_alloc]
        minis = (capacity + mini_size - 1) // mini_size
        macros = (capacity + macro_size - 1) // macro_size
        shared_rows, routed_rows = (local_tokens + 255) // 256, capacity // 256
        ib, hb = intermediate // 256, hidden // 256
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
        shared_tasks = shared_rows * (ib + hb) + (shared_rows * 2 * (intermediate // 128) + 3) // 4 + 3 * ib * hb
        mini_bwd = (mini_size // 256) * (ib + hb) + ((mini_size // 128) * (intermediate // 128) + 3) // 4
        mini_replay = 2 * (mini_size // 256) * ib + ((mini_size // 128) * (intermediate // 128) + 5) // 6
        clusters = (
            shared_tasks
            + minis * mini_bwd
            + max(0, minis - macro_size // mini_size) * mini_replay
            + macros * 3 * experts * ib * hb
        )
        u8 = torch.uint8
        with tvm_ffi.use_torch_stream():
            self.module.launch(
                grid=(comm_sms + 2 * clusters, 1, 1),
                dy_s=dy_rows,
                dg_s=dg_shared,
                du_s=du_shared,
                dy_atb_s=dy_rows,
                dg_atb_s=dg_shared,
                du_atb_s=du_shared,
                x_atb_s=x_rows,
                h_atb_s=hidden_rows,
                wg_s=shared_gate.unsqueeze(0),
                wu_s=shared_up.unsqueeze(0),
                wd_s=shared_down.unsqueeze(0),
                dh_s=dh_shared,
                dx_s=dx_shared,
                dwg_s=dwg_shared.unsqueeze(0),
                dwu_s=dwu_shared.unsqueeze(0),
                dwd_s=dwd_shared.unsqueeze(0),
                dh_sw_s=dh_shared,
                gate_sw_s=gate_rows,
                up_sw_s=up_rows,
                dg_sw_s=dg_shared,
                du_sw_s=du_shared,
                dy_r=dy_fp8_routed.view(u8),
                dy_sc_r=dy_sc_routed.view(-1, 16),
                wd_t_r=wd_t_fp8.view(u8),
                wd_t_sc_r=wd_t_sc.view(-1, 16),
                dh_r=dh_routed,
                dg_r=dg_fp8_routed.view(u8),
                dg_sc_r=dg_sc_routed.view(-1, 16),
                du_r=du_fp8_routed.view(u8),
                du_sc_r=du_sc_routed.view(-1, 16),
                wg_t_r=wg_t_fp8.view(u8),
                wg_t_sc_r=wg_t_sc.view(-1, 16),
                wu_t_r=wu_t_fp8.view(u8),
                wu_t_sc_r=wu_t_sc.view(-1, 16),
                dx_r=dx_routed,
                dy_t_r=dy_fp8_t_routed.view(u8),
                dy_sc_t_r=dy_sc_t_routed.view(-1, 16),
                h_t_r=hidden_fp8_t_routed.view(u8),
                h_sc_t_r=hidden_sc_t_routed.view(-1, 16),
                dwd_r=dwd_routed,
                dg_t_r=dg_fp8_t_routed.view(u8),
                dg_sc_t_r=dg_sc_t_routed.view(-1, 16),
                du_t_r=du_fp8_t_routed.view(u8),
                du_sc_t_r=du_sc_t_routed.view(-1, 16),
                x_t_r=x_fp8_t_routed.view(u8),
                x_sc_t_r=x_sc_t_routed.view(-1, 16),
                dwg_r=dwg_routed,
                dwu_r=dwu_routed,
                x_r=x_fp8_routed.view(u8),
                x_sc_r=x_sc_routed.view(-1, 16),
                wg_r=wg_fp8.view(u8),
                wg_sc_r=wg_sc.view(-1, 16),
                wu_r=wu_fp8.view(u8),
                wu_sc_r=wu_sc.view(-1, 16),
                gate_out_r=gate_routed,
                up_out_r=up_routed,
                gate_fp8_out_r=gate_fp8_routed.view(u8),
                up_fp8_out_r=up_fp8_routed.view(u8),
                gate_sc_r=gate_sc_routed.view(-1, 16),
                up_sc_r=up_sc_routed.view(-1, 16),
                dh_rows_r=dh_routed,
                gate_fp8_r=gate_fp8_routed.view(u8),
                up_fp8_r=up_fp8_routed.view(u8),
                dg_fp8_r=dg_fp8_routed.view(u8),
                du_fp8_r=du_fp8_routed.view(u8),
                dg_fp8_t_r=dg_fp8_t_routed.view(u8),
                du_fp8_t_r=du_fp8_t_routed.view(u8),
                gate_rows_r=gate_routed,
                up_rows_r=up_routed,
                h_fp8_r=hidden_fp8_routed.view(u8),
                h_sc_r=hidden_sc_routed.view(-1, 16),
                h_fp8_t_r=hidden_fp8_t_routed.view(u8),
                dy_disp_fp8=dy_fp8_routed.view(u8),
                dy_disp_fp8_t=dy_fp8_t_routed.view(u8),
                x_disp_fp8=x_fp8_routed.view(u8),
                x_disp_fp8_t=x_fp8_t_routed.view(u8),
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
                counts=counts,
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
                swiglu_limit=limit,
                swiglu_clamped=clamped,
            )
            elements = hidden * intermediate * (2 if self.wgrad_f32 else 1)
            self.zero_module.launch(
                grid=(128, experts, 1),
                gate=dwg_routed.view(torch.uint16),
                up=dwu_routed.view(torch.uint16),
                down=dwd_routed.view(torch.uint16),
                counts=counts,
                elements=elements,
            )
        self.last_rings = dict(
            x_fp8_routed=x_fp8_routed,
            x_sc_routed=x_sc_routed,
            gate_routed=gate_routed,
            up_routed=up_routed,
            hidden_fp8_routed=hidden_fp8_routed,
            hidden_sc_routed=hidden_sc_routed,
            dy_fp8_t_routed=dy_fp8_t_routed,
            dy_sc_t_routed=dy_sc_t_routed,
            dg_fp8_t_routed=dg_fp8_t_routed,
            dg_sc_t_routed=dg_sc_t_routed,
            du_fp8_t_routed=du_fp8_t_routed,
            du_sc_t_routed=du_sc_t_routed,
            partials=partials,
            router_weights=router_weights,
        )
        return (
            dx_shared,
            dx_routed,
            dg_shared,
            dg_fp8_routed,
            dg_sc_routed,
            du_shared,
            du_fp8_routed,
            du_sc_routed,
            dh_shared,
            dh_routed,
            dy_fp8_routed,
            dy_sc_routed,
            dwg_shared,
            dwg_routed,
            dwu_shared,
            dwu_routed,
            dwd_shared,
            dwd_routed,
        )


class MoKEpilogues:
    """Prepared standalone epilogues with the native one-launch boundaries."""

    def __init__(self, top_k: int):
        from .jit import load_kernel

        if type(top_k) is not int or top_k not in (2, 8):
            raise ValueError("Exported epilogues support top_k in (2, 8)")
        self.top_k = top_k
        self.forward_module = load_kernel(f"epilogue_forward_{top_k}")
        self.backward_module = load_kernel(f"epilogue_backward_{top_k}")

    def _check(self, shared, routed):
        import torch

        if shared.ndim != 2 or min(shared.shape) <= 0 or shared.shape[1] % 8:
            raise ValueError("Expected positive [tokens, hidden] with a 16-byte row stride")
        if tuple(routed.shape) != (shared.shape[0] * self.top_k, shared.shape[1]):
            raise ValueError("Routed buffer must have [tokens * top_k, hidden] shape")
        for tensor in (shared, routed):
            if (not tensor.is_cuda or tensor.device != shared.device
                    or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous()):
                raise ValueError("Expected contiguous BF16 buffers on one CUDA device")

    def forward(self, shared, routed, scores):
        import torch
        import tvm_ffi

        self._check(shared, routed)
        tokens, hidden = shared.shape
        if (tuple(scores.shape) != (tokens, self.top_k) or scores.dtype != torch.float32
                or scores.device != shared.device or not scores.is_contiguous()):
            raise ValueError("Scores must be contiguous CUDA FP32 [tokens, top_k]")
        output = torch.empty_like(shared)
        with tvm_ffi.use_torch_stream():
            self.forward_module.launch(grid=((hidden + 1023) // 1024 * ((tokens + 1) // 2), 1, 1),
                                       shared=shared, routed=routed, scores=scores,
                                       output=output, hidden=hidden, tokens=tokens)
        return output

    def backward(self, shared, routed):
        import torch
        import tvm_ffi

        self._check(shared, routed)
        tokens, hidden = shared.shape
        output = torch.empty_like(shared)
        with tvm_ffi.use_torch_stream():
            self.backward_module.launch(grid=((hidden + 1023) // 1024 * tokens, 1, 1),
                                        shared=shared, routed=routed, output=output, hidden=hidden)
        return output
