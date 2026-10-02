# Copyright 2026 Cursor Research
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Modified: standalone CUDA module loading; original host launch boundaries.


class MoKForward:
    """Low-level native-signature BF16 forward; caller owns peer barriers."""

    def __init__(self):
        from .jit import load_kernel

        self.module = load_kernel("forward")
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

        if swiglu_limit is not None:
            raise ValueError(
                "The current BF16 port implements the required unclamped SwiGLU"
            )
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
        key = (tuple(x_ptrs), tuple(combine_ptrs), x.device)
        if key not in self._peer_tables:
            self._peer_tables[key] = (
                torch.tensor(x_ptrs, dtype=torch.uint64, device=x.device),
                torch.tensor(combine_ptrs, dtype=torch.uint64, device=x.device),
            )
        x_peers, y_peers = self._peer_tables[key]
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
        shared_gate_tasks = shared_rows * (intermediate // 256)
        mini_gate_tasks = (mini_size // 256) * (intermediate // 256)
        shared_swiglu = ((local_tokens // 128) * (intermediate // 128) + 5) // 6
        mini_swiglu = ((mini_size // 128) * (intermediate // 128) + 5) // 6
        shared_tasks = (
            2 * shared_gate_tasks + shared_swiglu + shared_rows * (hidden // 256)
        )
        mini_tasks = (
            2 * mini_gate_tasks + mini_swiglu + (mini_size // 256) * (hidden // 256)
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
                gate_shared_in=gate_shared,
                up_shared_in=up_shared,
                gate_routed_in=gate_routed,
                up_routed_in=up_routed,
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


class MoKBackward:
    """Native-signature BF16 backward and its separate empty-expert zeroing."""

    def __init__(self):
        from .jit import load_kernel

        self.module = load_kernel("backward")
        self.zero_module = load_kernel("zero")
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
    ):
        import torch
        import tvm_ffi

        if swiglu_limit is not None:
            raise ValueError(
                "The current BF16 port implements the required unclamped SwiGLU"
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
        key = (
            tuple(x_ptrs),
            tuple(dy_ptrs),
            tuple(dx_ptrs),
            tuple(weight_ptrs),
            tuple(dweight_ptrs),
            x.device,
        )
        if key not in self._peer_tables:
            self._peer_tables[key] = tuple(
                torch.tensor(pointers, dtype=torch.uint64, device=x.device)
                for pointers in (x_ptrs, dy_ptrs, dx_ptrs, weight_ptrs, dweight_ptrs)
            )
        x_peers, dy_peers, dx_peers, weight_peers, dweight_peers = self._peer_tables[
            key
        ]
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
        shared_tasks = (
            shared_rows * (ib + hb)
            + ((local_tokens // 128) * (intermediate // 128) + 3) // 4
            + 3 * ib * hb
        )
        mini_bwd = (mini_size // 256) * (ib + hb) + (
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
            + macros * 3 * experts * ib * hb
        )
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
            )
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

        if (world_size, local_experts) not in ((1, 4), (4, 4), (16, 16)):
            raise ValueError("Unsupported EP size or expert count")
        self.world_size = world_size
        self.local_experts = local_experts
        self.count = load_kernel(f"count_{world_size}_{local_experts}")
        self.pad = load_kernel(f"pad_{world_size}_{local_experts}")
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
    """Prepared standalone epilogues with the native one-launch boundaries."""

    def __init__(self, top_k: int):
        from .jit import load_kernel

        if type(top_k) is not int or top_k not in (2, 8):
            raise ValueError("Exported epilogues support top_k=2 or 8")
        self.top_k = top_k
        self.forward_module = load_kernel(f"epilogue_forward_{top_k}")
        self.backward_module = load_kernel(f"epilogue_backward_{top_k}")

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
        if tokens % 2:
            raise ValueError("The native forward epilogue requires an even token count")
        if (
            tuple(scores.shape) != (tokens, self.top_k)
            or scores.dtype != torch.float32
            or scores.device != shared.device
            or not scores.is_contiguous()
        ):
            raise ValueError("Scores must be contiguous CUDA FP32 [tokens, top_k]")
        output = torch.empty_like(shared)
        with tvm_ffi.use_torch_stream():
            self.forward_module.launch(
                grid=((hidden + 1023) // 1024 * (tokens // 2), 1, 1),
                shared=shared,
                routed=routed,
                scores=scores,
                output=output,
                hidden=hidden,
            )
        return output

    def backward(self, shared, routed):
        import torch
        import tvm_ffi

        self._check(shared, routed)
        tokens, hidden = shared.shape
        output = torch.empty_like(shared)
        with tvm_ffi.use_torch_stream():
            self.backward_module.launch(
                grid=((hidden + 1023) // 1024 * tokens, 1, 1),
                shared=shared,
                routed=routed,
                output=output,
                hidden=hidden,
            )
        return output
