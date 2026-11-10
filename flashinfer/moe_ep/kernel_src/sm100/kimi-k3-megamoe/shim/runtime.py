"""Per-workspace Kimi launch state and symmetric storage."""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils
import torch
import torch.distributed as dist
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cute.typing import AddressSpace

from ..src.api import ImplDesc, ProblemDesc
from ..src.communication.nvlink_domain.symmetric_buffer import SymmetricBufferHost
from ..src.helpers import prod_config
from ..src.helpers.codegen_options import option_scope
from ..src.helpers.fastquant import _fastquant_launch
from ..src.helpers.fastquant_mx import _fastquant_mx_launch
from ..src.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_fc12_epilogue import (
    SwapABGatedActEpilogue,
)
from ..src.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel import (
    BlockScaledSwapAbMegaMoeKernel,
)
from ..src.quant_def import CombineFormat
from ..src.tactics import PROD1_DEFAULT_TACTICS, PROD1_TOKEN_BUCKETS
from .reduce import KimiTopkReducer
from .staging import StageRoutes


def _tensor(tensor, align=16):
    return cutlass_torch.from_dlpack(tensor, assumed_align=align).mark_layout_dynamic(
        leading_dim=cutlass_torch.get_leading_dim(tensor)
    )


def _pointer(tensor):
    # Byte workspaces can exceed the int32 memref shape limit.
    return cute.runtime.make_ptr(
        cutlass.Uint8, tensor.data_ptr(), AddressSpace.gmem, assumed_align=16
    )


def _launch_counts(preferred, fallback):
    hardware = cutlass.utils.HardwareInfo()
    size = math.prod(preferred)
    count = int(hardware.get_max_active_clusters(size))
    if fallback is None or tuple(fallback) == tuple(preferred):
        return count, None, None
    smaller = math.prod(fallback)
    if smaller <= 0 or size % smaller:
        raise ValueError("fallback cluster size must divide the preferred size")
    capacity = int(hardware.get_max_active_clusters(smaller)) * smaller
    if capacity < count * size:
        raise ValueError("fallback cluster capacity is too small")
    split = size // smaller
    remainder = (capacity - count * size) // smaller // split * split
    return count + remainder // split, count, remainder


def _kernel(config, fleet, world):
    wire = "bf16" if config.combine_dtype == "bf16" else "32e4m3xe8m0"
    bucket = next(b for b in PROD1_TOKEN_BUCKETS if fleet.max_tokens_per_rank <= b)
    tactic = PROD1_DEFAULT_TACTICS[config.quant_kind, wire, bucket]
    keys = (
        "mma_tiler_mnk",
        "cluster_shape_mn",
        "fallback_cluster_shape_mn",
        "schedule_policy",
        "work_id_mode",
        "token_back_mode",
        "fc2_use_bulk",
        "fc2_tma_stages",
        "token_in_flag_batch",
        "epi_flag_batches",
    )
    impl = dict(zip(keys, tactic, strict=True))
    for key in ("cluster_shape_mn", "fallback_cluster_shape_mn"):
        if impl[key] is not None:
            impl[key] = tuple(impl[key][:2])
    impl.update(config.knobs or {})
    tile = tuple(impl["mma_tiler_mnk"])
    launch, preferred, fallback = _launch_counts(
        impl["cluster_shape_mn"], impl["fallback_cluster_shape_mn"]
    )
    impl.update(
        mma_tiler_mnk=tile,
        use_2cta_instrs=tile[0] == 256,
        token_padding_block={224: 32, 240: 16}.get(tile[1], 64),
        sf_padding_block=128,
        launch_cluster_count=launch,
        preferred_cluster_count=preferred,
        fallback_cluster_count=fallback,
        reduce_topk_in_kernel=False,
    )
    if impl["fc2_tma_stages"] is None:
        del impl["fc2_tma_stages"]
    combine = CombineFormat.parse(wire)
    problem = ProblemDesc(
        dict(
            expert_count=fleet.num_experts,
            intermediate_gateup_size=2 * config.intermediate_size,
            hidden_size=fleet.token_hidden_size,
            quant_kind=config.quant_kind,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
            combine_format=combine,
            gate_up_clamp=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            situ_beta=float(config.situ_beta),
            situ_linear_beta=(
                None
                if config.situ_linear_beta is None
                else float(config.situ_linear_beta)
            ),
            world_size=world,
            topk=config.top_k,
            topk_index_dtype=cutlass.Int32,
            max_tokens_per_rank=fleet.max_tokens_per_rank,
            apply_topk_at_fc1=False,
        )
    )
    kernel = BlockScaledSwapAbMegaMoeKernel(problem, ImplDesc(impl))
    if prod_config.flag("MEGAMOE_EPI_ASYNC_FLAG", kernel.quant_kind, combine):
        # Each publisher warp can retain only one pending completion.
        publishers = SwapABGatedActEpilogue._EpilogueWarpCnt
        if any(batch > publishers for batch in kernel.epi_flag_batches):
            raise ValueError(
                f"Kimi async epilogue flag batches must be at most {publishers}; "
                f"got {kernel.epi_flag_batches}"
            )
    kernel._topk_reduce = KimiTopkReducer(
        fleet.token_hidden_size,
        config.top_k,
        combine,
        config.quant_kind,
        fleet.max_tokens_per_rank,
    )
    return kernel


class KimiWorkspace:
    """Own one fixed kernel, its buffers, and its compiled launchers."""

    def __init__(self, config, fleet_params, *, rank, world_size, process_group):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Allocate Kimi workspaces before CUDA graph capture")
        self.config, self.fleet_params = config, fleet_params
        self.rank, self.world_size, self.process_group = rank, world_size, process_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._input_scale = torch.tensor(
            [config.input_norm_const], dtype=torch.float32, device=self.device
        )
        self.options = {"MEGAMOE_SITU_FAST": "1" if config.fast_situ else "0"}
        self._compiled = self._quantizer = None
        self._routes = {}
        self._inputs = None
        self._destroyed = False
        self._symm_handle = None
        capacity, hidden = (
            fleet_params.max_tokens_per_rank,
            fleet_params.token_hidden_size,
        )
        self._nvfp4 = config.quant_kind == "nvfp4"
        self._block = 16 if self._nvfp4 else 32
        if capacity * config.top_k * hidden >= 2**31:
            raise ValueError(
                "Kimi capacity exceeds the int32 combine-plane index limit"
            )
        signature = (
            capacity,
            hidden,
            fleet_params.num_experts,
            config.quant_kind,
            config.combine_dtype,
            config.situ_beta,
            config.situ_linear_beta,
            config.fast_situ,
            config.input_norm_const,
            config.knobs,
        )
        if world_size > 1:
            peers = [None] * world_size
            dist.all_gather_object(peers, signature, group=process_group)
            if any(peer != signature for peer in peers):
                raise ValueError(
                    "All EP ranks must use the same Kimi workspace settings"
                )
        with option_scope(self.options):
            self.kernel = _kernel(config, fleet_params, world_size)
        local_bytes, shared_bytes = map(int, self.kernel.get_workspace_sizes())
        self.local_workspace = torch.empty(
            local_bytes, dtype=torch.uint8, device=self.device
        )
        regions = (
            (
                "activation",
                (capacity, hidden // 2 if self._nvfp4 else hidden),
                torch.uint8,
            ),
            ("activation_sf", (capacity, hidden // self._block), torch.uint8),
            ("scores", (capacity, config.top_k), torch.float32),
            ("output", (capacity, hidden), torch.bfloat16),
            ("shared_workspace", (shared_bytes,), torch.uint8),
        )
        offsets, total = {}, 0
        for name, shape, dtype in regions:
            size = math.prod(shape) * dtype.itemsize
            offsets[name] = (total, size)
            total += (size + 127) // 128 * 128
        if world_size > 1:
            import torch.distributed._symmetric_memory as symm

            group_name = process_group.group_name
            symm.enable_symm_mem_for_group(group_name)
            self._buffer = symm.empty(total, dtype=torch.uint8, device=self.device)
            self._symm_handle = symm.rendezvous(self._buffer, group_name)
            peer_offsets = tuple(
                int(p) - self._buffer.data_ptr() for p in self._symm_handle.buffer_ptrs
            )
        else:
            self._buffer = torch.empty(total, dtype=torch.uint8, device=self.device)
            peer_offsets = (0,)
        for name, shape, dtype in regions:
            offset, size = offsets[name]
            setattr(
                self, name, self._buffer[offset : offset + size].view(dtype).view(shape)
            )
        self.mapper = SymmetricBufferHost(
            base_address=self._buffer.data_ptr(),
            offsets=peer_offsets,
            rank=rank,
            max_ranks=world_size,
        )
        self.ids = torch.full(
            (capacity, config.top_k), -1, dtype=torch.int32, device=self.device
        )
        self._sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._buffer.zero_()
        local_zero, _ = self.kernel.require_zero_workspace_leading_bytes
        self.local_workspace[: int(local_zero)].zero_()
        # No rank may launch until every peer's initial zeros have landed.
        torch.cuda.current_stream().synchronize()
        if world_size > 1:
            dist.barrier(group=process_group)

    def _check_alive(self):
        if self._destroyed:
            raise RuntimeError("Kimi workspace is destroyed")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError("Kimi workspace requires its allocation device")

    def _compile_quantizer(self):
        if self._quantizer is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up Kimi BF16 input before CUDA graph capture")
        address, count = cutlass.Int64(0), cutlass.Int32(0)
        with option_scope(self.options):
            if self._nvfp4:
                self._quantizer = cute.compile(
                    _fastquant_launch,
                    address,
                    address,
                    address,
                    address,
                    count,
                    cutlass.Int32(1),
                    4,
                    256,
                    True,
                    4,
                    cuda.CUstream(0),
                    False,
                )
            else:
                self._quantizer = cute.compile(
                    _fastquant_mx_launch,
                    address,
                    address,
                    address,
                    count,
                    cutlass.Int32(1),
                    2,
                    256,
                    2,
                    cuda.CUstream(0),
                    False,
                )

    def stage_inputs(self, tensors, *, quantize_input):
        self._check_alive()
        self._inputs = tensors
        self._live = tensors.hidden_states.shape[0]
        stream = torch.cuda.current_stream()
        raw_stream = cuda.CUstream(stream.cuda_stream)
        if self._live:
            args = tuple(
                _tensor(t)
                for t in (tensors.topk_ids, tensors.topk_weights, self.ids, self.scores)
            ) + (raw_stream,)
            key = tensors.topk_ids.dtype
            if key not in self._routes:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Warm up this Kimi route ID dtype before capture"
                    )
                self._routes[key] = cute.compile(
                    StageRoutes(self.fleet_params.num_experts, self.config.top_k), *args
                )
            self._routes[key](*args)
        else:
            self.ids.fill_(-1)
            self.scores.zero_()
        if quantize_input:
            self._compile_quantizer()
            if self._live:
                blocks = self._live * self.fleet_params.token_hidden_size // self._block
                unroll, ctas = (4, 4) if self._nvfp4 else (2, 2)
                grid = max(
                    1,
                    min(
                        self._sms * ctas, (blocks + 256 * unroll - 1) // (256 * unroll)
                    ),
                )
                pointers = [
                    tensors.hidden_states.data_ptr(),
                    self.activation.data_ptr(),
                    self.activation_sf.data_ptr(),
                ]
                if self._nvfp4:
                    pointers.append(self._input_scale.data_ptr())
                self._quantizer(
                    *(cutlass.Int64(p) for p in pointers),
                    cutlass.Int32(blocks),
                    cutlass.Int32(grid),
                    raw_stream,
                )
        else:
            self.activation[: self._live].copy_(tensors.hidden_states.view(torch.uint8))
            self.activation_sf[: self._live].copy_(tensors.scales.view(torch.uint8))
        for tensor in (
            tensors.hidden_states,
            tensors.topk_ids,
            tensors.topk_weights,
            tensors.scales,
        ):
            if tensor is not None:
                tensor.record_stream(stream)

    def validate_capture_ready(self, transformed_weights):
        self._check_alive()
        if self._compiled is None:
            raise RuntimeError("Warm up the Kimi workspace before CUDA graph capture")

    def compute(self, transformed_weights, *, output):
        self._check_alive()
        if self._inputs is None:
            raise RuntimeError("Stage Kimi inputs before compute")
        live = self._live
        if output is not None and (
            output.ndim != 2
            or output.shape[0] < live
            or output.shape[1] != self.fleet_params.token_hidden_size
            or output.dtype != torch.bfloat16
            or output.device != self.device
            or not output.is_contiguous()
            or (output.numel() and output.data_ptr() % 16)
        ):
            raise ValueError(
                "Kimi output must be a contiguous, 16-byte-aligned BF16 tensor "
                f"on {self.device} with shape [rows >= {live}, "
                f"{self.fleet_params.token_hidden_size}]"
            )
        return self._compute(transformed_weights, output=output)

    def _compute(self, transformed_weights, *, output):
        """Execute with staged inputs and a validated or internally allocated output."""
        live = self._live
        destination = self.output if output is None or live == 0 else output
        weights = transformed_weights
        stream = torch.cuda.current_stream()
        data_type = torch.float4_e2m1fn_x2
        sf_type = torch.float8_e4m3fn if self._nvfp4 else torch.float8_e8m0fnu
        args = dict(
            activation=_tensor(
                self.activation.view(data_type if self._nvfp4 else torch.float8_e4m3fn)
            ),
            activation_sf=_tensor(self.activation_sf.view(sf_type)),
            topk_indices=_tensor(self.ids),
            topk_scores=_tensor(self.scores),
            fc1_weight=_tensor(weights.w13.view(data_type).transpose(1, 2)),
            fc1_weight_sf=_tensor(weights.w13_scale.view(sf_type)),
            fc2_weight=_tensor(weights.w2.view(data_type).transpose(1, 2)),
            fc2_weight_sf=_tensor(weights.w2_scale.view(sf_type)),
            output_activation=_tensor(destination[: max(1, live)]),
            local_workspace=_pointer(self.local_workspace),
            shared_workspace=_pointer(self.shared_workspace),
            peer_rank_ptr_mapper_host=self.mapper,
            stream=cuda.CUstream(stream.cuda_stream),
        )
        scalars = []
        if self._nvfp4:
            for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
                tensor = getattr(self._inputs, name)
                if tensor is None:
                    tensor = getattr(weights, name)
                args[name] = _tensor(tensor, 4)
                scalars.append(tensor)
        with option_scope(self.options):
            if self._compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm up Kimi before CUDA graph capture")
                self._compiled = cute.compile(self.kernel, **args)
            self._compiled(**args)
        for tensor in (
            self._buffer,
            self.local_workspace,
            self.ids,
            self._input_scale,
            weights.w13,
            weights.w2,
            weights.w13_scale,
            weights.w2_scale,
            destination,
            *scalars,
        ):
            tensor.record_stream(stream)
        return (output if output is not None else self.output)[:live]

    def destroy(self):
        if self._destroyed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Destroy Kimi workspaces outside CUDA graph capture")
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            if self.world_size > 1:
                dist.barrier(group=self.process_group)
        self._inputs = self._compiled = self._quantizer = None
        self._routes.clear()
        for name in (
            "activation",
            "activation_sf",
            "scores",
            "output",
            "ids",
            "local_workspace",
            "shared_workspace",
            "_input_scale",
            "_buffer",
            "_symm_handle",
            "mapper",
        ):
            setattr(self, name, None)
        self._destroyed = True
