# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Own compilation, scratch storage and launches for NT4 local experts."""

from dataclasses import dataclass

import torch


def _tensor(tensor, shape, dtype, device, name, alignment=16):
    if (
        tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % alignment
    ):
        raise ValueError(
            f"{name} must be contiguous {shape}, {dtype}, on {device}, aligned to {alignment} bytes"
        )


def _check_device(device):
    if device.type != "cuda" or torch.cuda.current_device() != device.index:
        raise ValueError("select the weights' CUDA device before setup")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("pack weights and prepare plans before CUDA graph capture")


def _cute(tensor, alignment=16, static=False):
    import cutlass.torch as ct

    view = ct.from_dlpack(tensor, assumed_align=alignment)
    return (
        view
        if static
        else view.mark_layout_dynamic(leading_dim=ct.get_leading_dim(tensor))
    )


@dataclass(frozen=True)
class PackedWeights:
    """Immutable layout descriptor; tensors must remain unchanged while plans live."""

    fc1: torch.Tensor
    fc1_scale: torch.Tensor
    fc2: torch.Tensor
    fc2_scale: torch.Tensor


def pack_weights(fc1, fc1_scale, fc2, fc2_scale):
    """Pack canonical E4M3 weights and raw E8M0 scales without requantization."""
    device = fc1.device
    _check_device(device)
    for tensor, shape, name in (
        (fc1, (128, 4096, 768), "fc1"),
        (fc2, (128, 768, 4096), "fc2"),
    ):
        _tensor(tensor, shape, torch.float8_e4m3fn, device, name)
    for tensor, shape, name in (
        (fc1_scale, (128, 4096, 24), "fc1_scale"),
        (fc2_scale, (128, 768, 128), "fc2_scale"),
    ):
        if tensor.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError(f"{name} must contain E8M0 bytes")
        _tensor(tensor, shape, tensor.dtype, device, name)
    # Reuse the shared kernel-team layout primitive, not the research frontend.
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import to_blocked

    def scales(raw):
        return torch.stack([to_blocked(s.view(torch.uint8)) for s in raw]).view(
            torch.float8_e8m0fnu
        )

    return PackedWeights(
        fc1.transpose(1, 2).contiguous(),
        scales(fc1_scale),
        fc2.transpose(1, 2).contiguous(),
        scales(fc2_scale),
    )


class LocalExpertPlan:
    """One fixed-capacity local workspace, with no communication or stream switches.

    Keep the plan and packed weights alive until captured graphs are destroyed and
    all queued work completes. A plan cannot be used concurrently by two streams.
    Separate graph buckets require separate plans; packed weights may be shared.
    """

    def __init__(
        self,
        weights,
        max_tokens,
        local_expert_offset,
        workers,
        group_hint,
        hardware_tanh,
    ):
        if type(max_tokens) is not int or not 0 < max_tokens <= (2**31 - 1) // (
            10 * 768
        ):
            raise ValueError(
                "max_tokens must be positive and fit int32 contribution addressing"
            )
        if type(local_expert_offset) is not int or local_expert_offset not in (
            0,
            128,
            256,
            384,
        ):
            raise ValueError("NT4 EP4 requires expert offset 0, 128, 256, or 384")
        if (
            type(workers) is not int
            or workers <= 0
            or type(group_hint) is not int
            or group_hint <= 0
        ):
            raise ValueError("workers and group_hint must be positive integers")
        if type(hardware_tanh) is not bool:
            raise ValueError("hardware_tanh must be a bool")
        if not isinstance(weights, PackedWeights):
            raise TypeError("weights must come from pack_nt4_mxfp8_weights")
        device = weights.fc1.device
        _check_device(device)
        if torch.cuda.get_device_capability(device) != (10, 3):
            raise ValueError("this NT4 candidate targets GB300 / SM103")
        if workers > torch.cuda.get_device_properties(device).multi_processor_count:
            raise ValueError("worker count exceeds the device SM count")
        for tensor, shape, dtype, name in (
            (weights.fc1, (128, 768, 4096), torch.float8_e4m3fn, "packed fc1"),
            (weights.fc2, (128, 4096, 768), torch.float8_e4m3fn, "packed fc2"),
            (
                weights.fc1_scale,
                (128, 4096 * 24),
                torch.float8_e8m0fnu,
                "packed fc1 scale",
            ),
            (
                weights.fc2_scale,
                (128, 768 * 128),
                torch.float8_e8m0fnu,
                "packed fc2 scale",
            ),
        ):
            _tensor(tensor, shape, dtype, device, name)
        self.weights = weights
        self.max_tokens = max_tokens
        self.device = device
        self.local_expert_offset = local_expert_offset
        self.workers, self.group_hint, self.hardware_tanh = (
            workers,
            group_hint,
            hardware_tanh,
        )
        self._closed = False
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import bootstrap_paths

        bootstrap_paths()
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from src.inputs_process import DataPreprocess
        from src.sym_buffer import SymBufferHost
        from .megamoe_kernel_mxfp8 import Sm100Nt4MegaMoEMxfp8Kernel

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        self._local_ids = empty((max_tokens, 10), torch.int64)
        self._local_probs = empty((max_tokens, 10), torch.float32)
        self._x = empty((max_tokens, 768), torch.float8_e4m3fn)
        self._sf = empty((max_tokens, 24), torch.float8_e8m0fnu)
        self._ids = empty((max_tokens, 10), torch.int64)
        self._probs = empty((max_tokens, 10), torch.float32)
        self._kernel = Sm100Nt4MegaMoEMxfp8Kernel(
            group_hint=group_hint,
            max_tokens_per_rank=max_tokens,
            hardware_tanh=hardware_tanh,
        )
        local_bytes, shared_bytes = self._kernel.get_workspace_sizes()
        self._local = torch.zeros(local_bytes, dtype=torch.uint8, device=device)
        self._shared = torch.zeros(shared_bytes, dtype=torch.uint8, device=device)
        region = self._kernel._shared_region_by_name["combine_quant"]
        self._terms = (
            self._shared.narrow(
                0, self._kernel._shared_offsets["combine_quant"], region.nbytes
            )
            .view(torch.bfloat16)
            .view(max_tokens, 10, 768)
        )
        self._args = dict(
            activation=_cute(self._x),
            activation_sf=_cute(self._sf),
            topk_idx=_cute(self._ids),
            topk_weights=_cute(self._probs),
            fc1_weight=_cute(weights.fc1),
            fc1_weight_sf=_cute(weights.fc1_scale),
            fc2_weight=_cute(weights.fc2),
            fc2_weight_sf=_cute(weights.fc2_scale),
            local_workspace=_cute(self._local, static=True),
            shared_workspace=_cute(self._shared),
            peer_rank_ptr_mapper_host=SymBufferHost(
                base_addr=self._shared.data_ptr(),
                offsets=(0,),
                rank_idx=0,
                num_max_ranks=1,
            ),
            fc1_c=None,
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._compiled = cute.compile(
            self._kernel, **self._args, max_active_clusters=workers, stream=stream
        )
        # Compile input staging with a representative tensor; run() binds the
        # caller's gathered input directly, without adding an input copy.
        dummy = empty((max_tokens, 768), torch.bfloat16)
        self._stage_tail = (
            _cute(self._local_ids, 4),
            _cute(self._local_probs, 4),
            None,
            _cute(self._x),
            _cute(self._sf, 4),
            _cute(self._ids, 4),
            _cute(self._probs, 4),
        )
        self._stage = cute.compile(
            DataPreprocess(10, 768, "mxfp8_e4m3"),
            _cute(dummy),
            *self._stage_tail,
            stream,
        )
        self._input_view = None
        # Warm Triton specializations before capture; no expert work is needed.
        from .megamoe_kernel_mxfp8 import route_mask, weighted_partial

        self._ids.fill_(-1)
        self._probs.zero_()
        route_mask(
            self._ids,
            self._probs,
            self._local_ids,
            self._local_probs,
            local_expert_offset // 128,
        )
        weighted_partial(
            self._terms,
            self._local_ids,
            self._local_probs,
            empty((max_tokens, 768), torch.float32),
        )

    def run(self, hidden_states, expert_ids, routing_weights, *, out):
        """Enqueue the complete local path on the current stream, returning ``out``.

        Inputs have exactly the prepared bucket capacity. Use ID -1 for padding;
        other global IDs must be in [0, 512). Output is contiguous FP32 [T, 768].
        Values are not scanned on the host. Input/output storage must not overlap.
        """
        if self._closed:
            raise RuntimeError("NT4 plan is closed")
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("select the plan's CUDA device before running")
        for tensor, shape, dtype, name in (
            (hidden_states, (self.max_tokens, 768), torch.bfloat16, "hidden_states"),
            (expert_ids, (self.max_tokens, 10), torch.int64, "expert_ids"),
            (routing_weights, (self.max_tokens, 10), torch.float32, "routing_weights"),
            (out, (self.max_tokens, 768), torch.float32, "out"),
        ):
            _tensor(tensor, shape, dtype, self.device, name)
        import cuda.bindings.driver as cuda
        from .megamoe_kernel_mxfp8 import route_mask, weighted_partial

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        key = hidden_states.data_ptr()
        entry = self._input_view
        if entry is not None and entry[0].data_ptr() != key:
            entry = None
        if entry is None:
            entry = (hidden_states, _cute(hidden_states))
            self._input_view = entry
        route_mask(
            expert_ids,
            routing_weights,
            self._local_ids,
            self._local_probs,
            self.local_expert_offset // 128,
        )
        self._stage(entry[1], *self._stage_tail, stream)
        self._compiled(**self._args, stream=stream)
        weighted_partial(self._terms, self._local_ids, self._local_probs, out)
        return out

    def contribution_view(self):
        """Borrow BF16 [T, 10, 768] terms for diagnostics; invalid slots are undefined.

        Order reads before the next run. This is not a snapshot and performs no
        synchronization. Consumers must mask loads using the current route IDs.
        """
        if self._closed:
            raise RuntimeError("NT4 plan is closed")
        return self._terms

    def close(self):
        """Release storage after destroying graphs and completing queued work."""
        if self._closed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot close a plan during capture")
        self._closed = True
        self._args.clear()
        self._input_view = None
        self._stage_tail = ()
        self._compiled = self._stage = self._kernel = None
        for name in (
            "_local",
            "_shared",
            "_terms",
            "_x",
            "_sf",
            "_ids",
            "_probs",
            "_local_ids",
            "_local_probs",
            "weights",
        ):
            setattr(self, name, None)
