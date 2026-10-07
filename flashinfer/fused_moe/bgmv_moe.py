"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import threading
import warnings
from typing import List, Literal, Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api


@functools.cache
def _get_bgmv_moe_module():
    """Lazily load the BGMV MoE CUDA extension.

    Tries in order:
    Loads via FlashInfer's JIT compilation system (TVM-FFI).
    """
    try:
        from ..jit.bgmv_moe import load_bgmv_moe_module

        return load_bgmv_moe_module()
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        raise ImportError(
            f"Failed to load BGMV MoE CUDA extension via JIT. "
            f"Ensure CUDA toolkit is available and csrc/bgmv_moe/ sources exist.\n"
            f"Error: {e}"
        ) from e


@functools.cache
def has_bgmv_moe() -> bool:
    """Return True if the BGMV MoE CUDA extension is available."""
    try:
        _get_bgmv_moe_module()
        return True
    except ImportError:
        return False


@flashinfer_api
def bgmv_moe_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    lora_stride: int,
    *,
    per_pair_input: bool = False,
) -> None:
    """
    MoE LoRA shrink operation: project input through LoRA-A matrices.

    For each (token, expert) pair, computes:
        y[slice, pair, rank] += x[token] @ lora_a[expert, lora_id, :, :]

    Args:
        y: Output tensor [num_slices, num_pairs, rank]. Accumulated in-place.
        x: Input activations [num_tokens, hidden_dim].
        w_ptr: Pointer table [num_slices, num_experts] of int64.
            Each entry points to the start of lora_a weights for (slice, expert).
            The kernel uses lora_stride to index different LoRA adapters.
        sorted_token_ids: Token indices for each pair [num_pairs].
        expert_ids: Expert indices for each pair [num_pairs].
        lora_indices: LoRA adapter ID for each token [num_tokens].
            -1 means no LoRA (pair is skipped).
        lora_stride: Stride (in elements) between consecutive LoRA adapters
            in the weight tensor. For layout [max_loras, num_experts, rank, feat],
            this is num_experts * rank * feat.
        per_pair_input: If False (default, FC1), the input row is the token, so a
            token's hidden row is reused across its k pairs (``x`` is ``[num_tokens, feat_in]``).
            If True (FC2), the input row is the pair itself, i.e. ``x`` is a per-pair
            ``[num_pairs, feat_in]`` buffer (e.g. the gathered post-activation). The
            ``lora_indices``/skip lookup still uses ``sorted_token_ids[pair]``.
    """
    mod = _get_bgmv_moe_module()
    mod.bgmv_moe_shrink(
        y,
        x,
        w_ptr,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        lora_stride,
        per_pair_input,
    )


@flashinfer_api
def bgmv_moe_expand(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_indices: torch.Tensor,
    slice_start_loc: torch.Tensor,
    output_slices: List[int],
    lora_stride: int,
    *,
    finalize: bool = True,
) -> None:
    """
    MoE LoRA expand operation: project through LoRA-B matrices.

    With ``finalize=True`` (default), for each (token, expert) pair computes the
    routing-weighted combine into a per-token row:
        y[token, col_offset:col_offset+feat] += topk_weight * (x[slice, pair, :] @ lora_b[expert, lora_id])
    (``y`` is ``[num_tokens, total_feat_out]`` and must be zero-initialized).

    With ``finalize=False`` (FC1 LoRA delta), writes a per-pair, UNWEIGHTED result with a
    plain store — no ``topk_weight``, no cross-expert combine:
        y[pair, col_offset:col_offset+feat] = (x[slice, pair, :] @ lora_b[expert, lora_id])
    (``y`` is ``[num_pairs, total_feat_out]``). Skipped pairs (lora_id < 0) early-return, so
    ``y`` MUST be zero-initialized by the caller (``torch.zeros``) to define those rows.
    ``topk_weights`` is ignored in this mode but must still be a valid ``[num_pairs]`` float32
    tensor.

    Args:
        y: Output buffer (zero-initialized). ``[num_tokens, total_feat_out]`` (finalize) or
            ``[num_pairs, total_feat_out]`` (no-finalize). Float32.
        x: Shrink output [num_slices, num_pairs, rank].
        w_ptr: Pointer table [num_slices, num_experts] of int64.
        sorted_token_ids: Token indices for each pair [num_pairs].
        expert_ids: Expert indices for each pair [num_pairs].
        topk_weights: Routing weights for each pair [num_pairs]. Float32. (Ignored when
            ``finalize=False``.)
        lora_indices: LoRA adapter ID for each token [num_tokens].
        slice_start_loc: Column offset for each slice [num_slices]. Int64.
        output_slices: Output feature dimension for each slice.
        lora_stride: Stride between LoRA adapters in weight tensor.
        finalize: Combine + weight per token (True) vs per-pair unweighted store (False).
    """
    mod = _get_bgmv_moe_module()
    mod.bgmv_moe_expand(
        y,
        x,
        w_ptr,
        sorted_token_ids,
        expert_ids,
        topk_weights,
        lora_indices,
        slice_start_loc,
        output_slices[0],
        lora_stride,
        finalize,
    )


def fill_w_ptr(
    w_ptr: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    slice_id: int,
) -> int:
    """
    Fill the weight pointer table for a given slice.

    Populates w_ptr[slice_id, 0:num_experts] with data pointers for each expert.
    Works with weight layout [max_loras, num_experts, rank, feat].

    Args:
        w_ptr: Pointer table [num_slices, num_experts] of int64.
        weights: LoRA weight tensor [max_loras, num_experts, rank, feat].
        num_experts: Number of experts.
        slice_id: Which slice to populate.

    Returns:
        lora_stride: The stride (in elements) between LoRA adapters.
    """
    # w shape: [max_loras, num_experts, rank, feat]
    base_ptr = weights.data_ptr()
    expert_stride_bytes = weights.stride(1) * weights.element_size()

    arange = torch.arange(num_experts, dtype=torch.int64, device=weights.device)
    w_ptr[slice_id, :num_experts] = arange * expert_stride_bytes + base_ptr

    # lora_stride = stride along dim 0 (in elements)
    return weights.stride(0)


def _cake_dtype_name(dtype: torch.dtype) -> Literal["bfloat16", "float16"]:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"Cake BGMV MoE requires BF16 or FP16, got {dtype}")


BGMVMoEBackendUsed = Literal["cake", "portable"]
CakeBGMVMoEVariant = Literal["specialized", "generic"]

_CAKE_UNSUPPORTED_DEVICE_MESSAGE = (
    "Cake BGMV MoE requires an exact SM90, SM100 or SM103 CUDA device; "
    "got capability={capability}"
)
_fallback_reasons_warned: set = set()


class _BGMVMoEGraphPlan:
    """Shared pointer-stable CUDA-Graph replay logic for prepared BGMV MoE plans.

    The plan owns caller-visible FP32 accumulation and shrink workspaces. Its
    first eager ``run`` captures the exact launch sequence into a CUDA Graph;
    later calls replay that graph on the same stream. If called while an outer
    CUDA Graph is being captured, the constituent kernels are enqueued directly.
    """

    backend_used: BGMVMoEBackendUsed

    def __init__(
        self,
        *,
        y_accum: torch.Tensor,
        shrink_out: torch.Tensor,
        x: torch.Tensor,
        bound_tensors: Tuple[torch.Tensor, ...],
    ) -> None:
        self.y_accum = y_accum
        self.shrink_out = shrink_out
        self.x = x
        # One (tensor, data_ptr, shape, stride) record per bound tensor. A
        # tensor's dtype and device cannot change in place, so the per-call
        # check compares only what ``set_``/``resize_`` can move.
        self._bound = tuple(
            (tensor, tensor.data_ptr(), tensor.shape, tensor.stride())
            for tensor in bound_tensors
        )
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._owner_stream: Optional[torch.cuda.Stream] = None
        self._lock = threading.RLock()

    def _validate_binding(self) -> None:
        for tensor, data_ptr, shape, stride in self._bound:
            if (
                tensor.data_ptr() != data_ptr
                or tensor.shape != shape
                or tensor.stride() != stride
            ):
                raise RuntimeError(
                    f"{type(self).__name__} tensor storage, shape or stride "
                    "changed after preparation"
                )

    def _launch(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def run(self) -> torch.Tensor:
        """Run or replay the prepared zero+shrink+expand pipeline."""

        self._validate_binding()
        if torch.cuda.is_current_stream_capturing():
            self._launch()
            return self.y_accum

        with self._lock:
            replay_stream = torch.cuda.current_stream(self.x.device)
            graph = self._graph
            if graph is None:
                capture_stream = torch.cuda.Stream(device=self.x.device)
                capture_stream.wait_stream(replay_stream)
                capture_stream.synchronize()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(
                    graph,
                    stream=capture_stream,
                    capture_error_mode="thread_local",
                ):
                    self._launch()
                graph.instantiate()
                self._graph = graph
                self._capture_stream = capture_stream
                self._owner_stream = replay_stream
            elif replay_stream != self._owner_stream:
                raise RuntimeError(
                    f"{type(self).__name__} must replay on its original CUDA stream"
                )
            graph.replay()
        return self.y_accum

    def close(self) -> None:
        """Release graph resources after pending replay work completes."""

        with self._lock:
            if self._graph is None:
                return
            self._owner_stream.synchronize()
            reset = getattr(self._graph, "reset", None)
            if callable(reset):
                reset()
            self._graph = None
            self._capture_stream = None
            self._owner_stream = None


class BGMVMoECakePlan(_BGMVMoEGraphPlan):
    """Pointer-stable SM90/SM100/SM103 Cake BGMV MoE shrink+expand execution plan.

    Runs the generated Cake programs (one owner per output token, no output
    atomics, bitwise-reproducible replays). ``backend_used`` is ``"cake"``;
    ``variant`` is ``"specialized"`` for the hidden 2688/3072 x rank 32 bodies (up to 2048 tokens)
    and ``"generic"`` for the runtime-hidden rank 8/16/32/64 bundles.
    """

    backend_used: BGMVMoEBackendUsed = "cake"

    def __init__(
        self,
        module,
        *,
        y_accum: torch.Tensor,
        shrink_out: torch.Tensor,
        x: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        lora_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        schedule_id: int,
        variant: CakeBGMVMoEVariant = "specialized",
        shrink_launch: Optional[Tuple[int, int]] = None,
        grouped: bool = False,
        order_remap: bool = False,
        pdl_mode: int = 0,
    ) -> None:
        self._module = module
        self.variant: CakeBGMVMoEVariant = variant
        # Generic variant only: (shrink form, hidden splits) chosen at prepare
        # time by ``select_cake_bgmv_moe_generic_shrink`` (form 0 two-stage
        # prefill ring, 1 decode, 2 three-stage prefill ring for small grids).
        self.shrink_launch: Optional[Tuple[int, int]] = shrink_launch
        # Generic variant only: pair-grouped pipeline (each unique (LoRA, expert)
        # pair's weights streamed once per tile of routes, deterministic per-token
        # combine of FP32 route partials); ``False`` runs the per-route kernels.
        self.grouped: bool = bool(grouped)
        # Generic per-route pipeline only: bin-ordered dispatch of the shrink
        # (a single-CTA prologue sorts the routes by (LoRA, expert) bin so the
        # CTAs sharing LoRA-A rows run back to back; bitwise identical to the
        # identity dispatch).  Exclusive with ``grouped``.
        self.order_remap: bool = bool(order_remap)
        # Programmatic dependent launch of the expand behind the shrink:
        # 0 off, 1 shrink triggers at entry, 2 shrink triggers after its tile loop.
        self.pdl_mode: int = int(pdl_mode)
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.sorted_token_ids = sorted_token_ids
        self.expert_ids = expert_ids
        self.lora_indices = lora_indices
        self.topk_weights = topk_weights
        self.schedule_id: Optional[int] = schedule_id
        from ..jit.cake_bgmv_moe import cake_bgmv_moe_route_index_numel

        # Token->pair route index: published by the shrink kernels and read by
        # the expand kernels (arbitrary pair order in O(1) per CTA), followed by
        # the generic shrink's hidden-split partials and arrival counters.
        # Pointer-stable and zero-initialized once; counts are monotonic with
        # launch-parity bases and the split counters are reset by their last
        # arrival, so graph replays never need a memset node.
        self.route_index = torch.zeros(
            cake_bgmv_moe_route_index_numel(int(x.shape[0])),
            dtype=torch.int32,
            device=x.device,
        )
        # Grouped pipeline workspace (generic variant with ``grouped=True``):
        # int32 grouping metadata rebuilt by the grouping kernel every launch and
        # FP32 per-route expand partials [num_pairs, hidden]; 1-element dummies
        # otherwise so the binding signature stays uniform.
        if self.grouped:
            from ..jit.cake_bgmv_moe import cake_bgmv_moe_grouped_workspace_words

            num_pairs = int(sorted_token_ids.shape[0])
            bins = int(lora_a.shape[0]) * int(lora_a.shape[1])
            self.group_workspace = torch.zeros(
                cake_bgmv_moe_grouped_workspace_words(num_pairs, int(x.shape[0]), bins),
                dtype=torch.int32,
                device=x.device,
            )
            self.group_partials = torch.empty(
                num_pairs * int(x.shape[1]), dtype=torch.float32, device=x.device
            )
        else:
            self.group_workspace = torch.zeros(1, dtype=torch.int32, device=x.device)
            self.group_partials = torch.zeros(1, dtype=torch.float32, device=x.device)
        # Lever-27 order workspace (generic variant with ``order_remap=True``):
        # the route permutation rebuilt by the order_build prologue every
        # launch; a 1-word dummy otherwise.
        if self.order_remap:
            from ..jit.cake_bgmv_moe import cake_bgmv_moe_order_workspace_words

            self.order_workspace = torch.zeros(
                cake_bgmv_moe_order_workspace_words(int(sorted_token_ids.shape[0])),
                dtype=torch.int32,
                device=x.device,
            )
        else:
            self.order_workspace = torch.zeros(1, dtype=torch.int32, device=x.device)
        super().__init__(
            y_accum=y_accum,
            shrink_out=shrink_out,
            x=x,
            bound_tensors=(
                y_accum,
                shrink_out,
                x,
                lora_a,
                lora_b,
                sorted_token_ids,
                expert_ids,
                lora_indices,
                topk_weights,
                self.route_index,
                self.group_workspace,
                self.group_partials,
                self.order_workspace,
            ),
        )

    def _launch(self) -> None:
        args = [
            self.y_accum,
            self.shrink_out,
            self.x,
            self.lora_a,
            self.lora_b,
            self.sorted_token_ids,
            self.expert_ids,
            self.lora_indices,
            self.topk_weights,
            self.route_index,
            self.schedule_id,
        ]
        if self.variant == "generic":
            assert self.shrink_launch is not None
            args.extend(self.shrink_launch)
            args.extend([int(self.grouped), self.group_workspace, self.group_partials])
            args.extend([int(self.order_remap), self.order_workspace])
        args.append(int(self.pdl_mode))
        args.append(int(torch.cuda.current_stream(self.x.device).cuda_stream))
        self._module.run(*args)


class BGMVMoEPortablePlan(_BGMVMoEGraphPlan):
    """Prepared plan that runs the portable ``bgmv_moe_shrink`` + ``bgmv_moe_expand`` path.

    Returned by :func:`prepare_bgmv_moe` when the inputs are outside the
    generated Cake support set and ``fallback=True``. Same interface as
    :class:`BGMVMoECakePlan` (``run`` zeroes the workspaces, runs shrink and
    expand, and returns the FP32 accumulator; first eager call captures a CUDA
    Graph). The portable expand accumulates with atomics, so replays are not
    guaranteed bitwise identical. ``backend_used`` is ``"portable"`` and
    ``schedule_id`` is ``None``.
    """

    backend_used: BGMVMoEBackendUsed = "portable"

    def __init__(
        self,
        *,
        y_accum: torch.Tensor,
        shrink_out: torch.Tensor,
        x: torch.Tensor,
        lora_a_weights: List[torch.Tensor],
        lora_b_weights: List[torch.Tensor],
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        lora_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        fallback_reason: str,
    ) -> None:
        self.lora_a_weights = list(lora_a_weights)
        self.lora_b_weights = list(lora_b_weights)
        self.sorted_token_ids = sorted_token_ids
        self.expert_ids = expert_ids
        self.lora_indices = lora_indices
        self.topk_weights = topk_weights
        self.num_experts = int(num_experts)
        self.schedule_id: Optional[int] = None
        self.fallback_reason = fallback_reason
        num_slices = len(self.lora_a_weights)
        device = x.device
        self._w_ptr_a = torch.zeros(
            num_slices, self.num_experts, dtype=torch.int64, device=device
        )
        self._w_ptr_b = torch.zeros(
            num_slices, self.num_experts, dtype=torch.int64, device=device
        )
        self._lora_stride_a = 0
        self._lora_stride_b = 0
        for slice_id in range(num_slices):
            self._lora_stride_a = fill_w_ptr(
                self._w_ptr_a, self.lora_a_weights[slice_id], self.num_experts, slice_id
            )
            self._lora_stride_b = fill_w_ptr(
                self._w_ptr_b, self.lora_b_weights[slice_id], self.num_experts, slice_id
            )
        self._output_slices = [int(weight.shape[2]) for weight in self.lora_b_weights]
        starts = [0]
        for feat_out in self._output_slices[:-1]:
            starts.append(starts[-1] + feat_out)
        self._slice_start_loc = torch.tensor(starts, dtype=torch.int64, device=device)
        super().__init__(
            y_accum=y_accum,
            shrink_out=shrink_out,
            x=x,
            bound_tensors=(
                y_accum,
                shrink_out,
                x,
                *self.lora_a_weights,
                *self.lora_b_weights,
                sorted_token_ids,
                expert_ids,
                lora_indices,
                topk_weights,
            ),
        )

    def _launch(self) -> None:
        self.shrink_out.zero_()
        self.y_accum.zero_()
        bgmv_moe_shrink(
            self.shrink_out,
            self.x,
            self._w_ptr_a,
            self.sorted_token_ids,
            self.expert_ids,
            self.lora_indices,
            self._lora_stride_a,
        )
        bgmv_moe_expand(
            self.y_accum,
            self.shrink_out,
            self._w_ptr_b,
            self.sorted_token_ids,
            self.expert_ids,
            self.topk_weights,
            self.lora_indices,
            self._slice_start_loc,
            self._output_slices,
            self._lora_stride_b,
        )


# Compatible alias for the name exported by the first release of this backend.
BGMVMoEBlackwellPlan = BGMVMoECakePlan

BGMVMoEPlan = Union[BGMVMoECakePlan, BGMVMoEPortablePlan]


def _cake_unsupported_reason(
    *,
    capability: Optional[Tuple[int, int]],
    arch: Optional[str],
    num_slices: int,
    hidden_size: int,
    rank: int,
    feat_outs: List[int],
) -> Optional[str]:
    """Return why the generated Cake programs cannot serve these inputs, or ``None``."""

    from ..jit.cake_bgmv_moe import (
        CAKE_BGMV_MOE_GENERIC_HIDDEN_MULTIPLE,
        CAKE_BGMV_MOE_GENERIC_RANKS,
        cake_bgmv_moe_variant,
    )

    if arch is None:
        return _CAKE_UNSUPPORTED_DEVICE_MESSAGE.format(capability=capability)
    if num_slices != 1:
        return (
            f"Cake BGMV MoE currently requires exactly one LoRA slice, got {num_slices}"
        )
    if rank not in CAKE_BGMV_MOE_GENERIC_RANKS:
        return (
            "Cake BGMV MoE requires LoRA rank in "
            f"{CAKE_BGMV_MOE_GENERIC_RANKS}, got {rank}"
        )
    if cake_bgmv_moe_variant(hidden_size, rank) is None:
        return (
            "Cake BGMV MoE hidden_size must be a positive multiple of "
            f"{CAKE_BGMV_MOE_GENERIC_HIDDEN_MULTIPLE}, got {hidden_size}"
        )
    if feat_outs != [hidden_size]:
        return (
            "Cake BGMV MoE requires LoRA-B feat_out == hidden_size, "
            f"got feat_out={feat_outs[0]} for hidden_size={hidden_size}"
        )
    return None


def _warn_fallback_once(reason: str) -> None:
    if reason in _fallback_reasons_warned:
        return
    _fallback_reasons_warned.add(reason)
    warnings.warn(
        "prepare_bgmv_moe: falling back to the portable bgmv_moe_shrink/"
        f"bgmv_moe_expand path ({reason}). Pass fallback=False to raise instead.",
        RuntimeWarning,
        stacklevel=3,
    )


@flashinfer_api
def prepare_bgmv_moe(
    x: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    lora_b_weights: List[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    *,
    backend: Literal["cake"] = "cake",
    fallback: bool = True,
    shrink_out: Optional[torch.Tensor] = None,
    y_accum: Optional[torch.Tensor] = None,
    grouped: Optional[bool] = None,
    order_remap: Optional[bool] = None,
) -> BGMVMoEPlan:
    """Prepare a graph-replayable BGMV MoE shrink+expand pipeline.

    The generated Cake path supports one LoRA slice, LoRA rank 8, 16, 32 or
    64, any hidden size that is a positive multiple of 8 (LoRA-B feat_out equal
    to it), BF16/FP16 inputs, and exact SM90 (H100/H200), SM100 (B200/GB200) or
    SM103 (B300/GB300) devices; each target runs its own cubin. Hidden sizes
    2688 and 3072 at rank 32 use the specialized measured bodies at up to 2048
    tokens (``plan.variant == "specialized"``); everything else, including those
    shapes at larger token counts, uses the runtime-hidden generic bundles
    (``plan.variant == "generic"``). Routing may be arbitrary; each output has one owner that accumulates
    routes in fixed input order, so identical prepared replays are bitwise
    reproducible. The contiguous top-k=2 layout takes the optimized fast path;
    any other pair order (for example expert-sorted dispatch) is served through
    a token->pair route index that the shrink kernels publish and the expand
    kernels read in O(1) per CTA (``plan.route_index``, 16 slots per token;
    tokens routed to more pairs take an exact serial scan). Pairs already in
    their contiguous position are implicit, so a contiguous launch publishes
    nothing. At small pair counts the generic shrink also splits the hidden
    dimension over extra CTAs (``plan.shrink_launch``) and the last CTA of each
    output tile reduces the FP32 partials in split order, so the result stays
    deterministic and the workspace is never reset.

    Inputs outside that support set (other device capabilities, ranks, hidden
    sizes that are not multiples of 8, multiple slices) are served by a
    :class:`BGMVMoEPortablePlan` running the portable ``bgmv_moe_shrink`` +
    ``bgmv_moe_expand`` kernels when ``fallback=True`` (default); one
    ``RuntimeWarning`` is emitted per distinct reason per process. With
    ``fallback=False`` such inputs raise ``ValueError`` as before. Invalid
    inputs (shape mismatches, wrong index dtypes, out-of-range routing indices,
    non-contiguous or CPU tensors) always raise.

    Args:
        x: Input activations with shape ``[num_tokens, hidden_size]``.
        lora_a_weights: LoRA-A tensors, one per slice, each with shape
            ``[num_loras, num_experts, rank, hidden_size]``.
        lora_b_weights: LoRA-B tensors, one per slice, each with shape
            ``[num_loras, num_experts, feat_out, rank]``.
        sorted_token_ids: Routed token indices with shape ``[num_pairs]``.
        expert_ids: Expert index for each routed pair.
        lora_indices: LoRA index for each input token.
        topk_weights: FP32 routing weight for each routed pair.
        num_experts: Number of experts in the LoRA tensors.
        backend: Backend selector; only ``"cake"`` (the generated Cake
            programs) is supported.
        fallback: Serve unsupported inputs with the portable path instead of
            raising.
        grouped: Generic Cake variant only. ``None`` (default) lets the plan
            choose the pair-grouped pipeline when the routes clearly outnumber
            the ``num_loras * num_experts`` (LoRA, expert) pairs (each pair's
            weights are then streamed once per tile of routes and a
            deterministic per-token combine sums the FP32 route partials in
            ascending pair order; the plan owns an int32 grouping workspace
            and ``num_pairs * hidden`` FP32 partials). ``True`` / ``False``
            force or disable it.
        order_remap: Generic per-route Cake pipeline only. ``None`` (default)
            lets the plan dispatch the shrink in (LoRA, expert)-bin order on
            SM90 where the saved LoRA-A traffic outweighs the single-CTA
            ordering prologue (``select_cake_bgmv_moe_order_remap``); every
            route is still computed by the same code on the same operands, so
            the result is bitwise identical to the identity dispatch. ``True``
            / ``False`` force or disable it; ``True`` is rejected together with
            the grouped pipeline or the specialized variant.
        shrink_out: Optional pointer-stable shrink workspace with shape
            ``[num_slices, num_pairs, rank]`` and the weight dtype.
        y_accum: Optional pointer-stable FP32 output accumulator with shape
            ``[num_tokens, sum(feat_out)]``. The Cake path writes it through
            its row stride, so it may be a column slice of a wider row-major
            buffer; the portable fallback needs a contiguous accumulator.

    Returns:
        A reusable graph-backed execution plan whose ``run`` method returns
        the FP32 accumulated output; ``plan.backend_used`` is ``"cake"`` or
        ``"portable"``.
    """

    if backend != "cake":
        raise ValueError(
            f"prepare_bgmv_moe only supports backend='cake', got {backend}"
        )
    from ..jit.cake_bgmv_moe import cake_bgmv_moe_arch_for_capability

    capability: Optional[Tuple[int, int]] = None
    if torch.cuda.is_available() and x.is_cuda:
        major, minor = torch.cuda.get_device_capability(x.device)
        capability = (int(major), int(minor))
    arch = (
        cake_bgmv_moe_arch_for_capability(capability)
        if capability is not None
        else None
    )
    if capability is None:
        # Neither the generated programs nor the portable kernels run off-GPU.
        raise ValueError(
            _CAKE_UNSUPPORTED_DEVICE_MESSAGE.format(capability=capability)
            + "; the portable fallback also requires CUDA tensors"
        )
    num_slices = len(lora_a_weights)
    if num_slices == 0 or len(lora_b_weights) != num_slices:
        raise ValueError(
            "lora_a_weights and lora_b_weights must be non-empty lists of equal length"
        )
    if x.ndim != 2:
        raise ValueError(f"x must have shape [tokens, hidden], got {tuple(x.shape)}")
    num_tokens, hidden_size = (int(dim) for dim in x.shape)
    weight_dtype = lora_a_weights[0].dtype
    for lora_a, lora_b in zip(lora_a_weights, lora_b_weights, strict=True):
        if lora_a.ndim != 4 or lora_b.ndim != 4:
            raise ValueError("LoRA weights must have rank 4")
        if lora_a.dtype != weight_dtype or lora_b.dtype != weight_dtype:
            raise ValueError("all LoRA weight tensors must have the same dtype")
    if weight_dtype not in (torch.bfloat16, torch.float16) or x.dtype != weight_dtype:
        # Neither the generated programs nor the portable kernels accept other
        # activation/weight dtype combinations.
        raise ValueError(
            "x and the LoRA weights must share one dtype, BF16 or FP16; "
            f"got x={x.dtype}, weights={weight_dtype}"
        )
    num_loras = int(lora_a_weights[0].shape[0])
    rank = int(lora_a_weights[0].shape[2])
    feat_outs: List[int] = []
    for lora_a, lora_b in zip(lora_a_weights, lora_b_weights, strict=True):
        if int(lora_a.shape[0]) != num_loras or int(lora_b.shape[0]) != num_loras:
            raise ValueError(
                "all LoRA weight tensors must have the same num_loras dimension"
            )
        if int(lora_a.shape[1]) != num_experts or int(lora_b.shape[1]) != num_experts:
            raise ValueError("num_experts must match all LoRA weight tensors")
        if tuple(lora_a.shape[2:]) != (rank, hidden_size):
            raise ValueError(
                "LoRA-A must have shape [num_loras, num_experts, rank, hidden_size] "
                f"with one rank for all slices, got {tuple(lora_a.shape)}"
            )
        if int(lora_b.shape[3]) != rank:
            raise ValueError(
                "LoRA-B must have shape [num_loras, num_experts, feat_out, rank], "
                f"got {tuple(lora_b.shape)}"
            )
        feat_outs.append(int(lora_b.shape[2]))
    if len(set(feat_outs)) != 1:
        raise ValueError(
            f"all LoRA-B slices must have the same feat_out, got {feat_outs}"
        )
    num_pairs = int(sorted_token_ids.numel())
    if sorted_token_ids.ndim != 1 or num_pairs <= 0:
        raise ValueError("sorted_token_ids must be a non-empty rank-1 tensor")
    if expert_ids.shape != sorted_token_ids.shape:
        raise ValueError("expert_ids must have the same shape as sorted_token_ids")
    if topk_weights.shape != sorted_token_ids.shape:
        raise ValueError("topk_weights must have the same shape as sorted_token_ids")
    if tuple(lora_indices.shape) != (num_tokens,):
        raise ValueError("lora_indices must have shape [num_tokens]")
    if topk_weights.dtype != torch.float32:
        raise ValueError("topk_weights must have dtype torch.float32")
    named_tensors = [
        ("x", x),
        ("sorted_token_ids", sorted_token_ids),
        ("expert_ids", expert_ids),
        ("lora_indices", lora_indices),
        ("topk_weights", topk_weights),
    ]
    for slice_id in range(num_slices):
        named_tensors.append((f"lora_a_weights[{slice_id}]", lora_a_weights[slice_id]))
        named_tensors.append((f"lora_b_weights[{slice_id}]", lora_b_weights[slice_id]))
    for name, tensor in named_tensors:
        if not tensor.is_cuda or tensor.device != x.device:
            raise ValueError(f"{name} must be on {x.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for name, tensor in (
        ("sorted_token_ids", sorted_token_ids),
        ("expert_ids", expert_ids),
        ("lora_indices", lora_indices),
    ):
        if tensor.dtype != torch.int64:
            raise ValueError(f"{name} must have dtype torch.int64")
    if bool(((expert_ids < 0) | (expert_ids >= num_experts)).any()):
        raise ValueError("expert_ids values must be in [0, num_experts)")
    if bool(((lora_indices < -1) | (lora_indices >= num_loras)).any()):
        raise ValueError("lora_indices values must be -1 or in [0, num_loras)")

    reason = _cake_unsupported_reason(
        capability=capability,
        arch=arch,
        num_slices=num_slices,
        hidden_size=hidden_size,
        rank=rank,
        feat_outs=feat_outs,
    )
    if reason is not None and not fallback:
        raise ValueError(reason)

    expected_shrink = (num_slices, num_pairs, rank)
    if shrink_out is None:
        shrink_out = torch.empty(expected_shrink, dtype=weight_dtype, device=x.device)
    if tuple(shrink_out.shape) != expected_shrink or shrink_out.dtype != weight_dtype:
        raise ValueError(
            f"shrink_out must have shape {expected_shrink} and dtype {weight_dtype}"
        )
    expected_output = (num_tokens, sum(feat_outs))
    if y_accum is None:
        y_accum = torch.empty(expected_output, dtype=torch.float32, device=x.device)
    if tuple(y_accum.shape) != expected_output or y_accum.dtype != torch.float32:
        raise ValueError(
            f"y_accum must have shape {expected_output} and dtype torch.float32"
        )
    for name, tensor in (("shrink_out", shrink_out), ("y_accum", y_accum)):
        if not tensor.is_cuda or tensor.device != x.device:
            raise ValueError(f"{name} must be a tensor on {x.device}")
    if not shrink_out.is_contiguous():
        raise ValueError("shrink_out must be contiguous")
    if y_accum.stride(1) != 1 or y_accum.stride(0) < y_accum.shape[1]:
        raise ValueError(
            "y_accum must be contiguous along its last dimension with a row "
            "stride of at least its width"
        )
    if reason is not None and not y_accum.is_contiguous():
        raise ValueError("y_accum must be contiguous for the portable fallback path")

    if reason is not None:
        _warn_fallback_once(reason)
        return BGMVMoEPortablePlan(
            y_accum=y_accum,
            shrink_out=shrink_out,
            x=x,
            lora_a_weights=lora_a_weights,
            lora_b_weights=lora_b_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            lora_indices=lora_indices,
            topk_weights=topk_weights,
            num_experts=num_experts,
            fallback_reason=reason,
        )

    from ..jit.cake_bgmv_moe import (
        CAKE_BGMV_MOE_GENERIC_SCHEDULE_IDS,
        CAKE_BGMV_MOE_SCHEDULE_IDS,
        cake_bgmv_moe_pdl_mode,
        cake_bgmv_moe_variant,
        get_cake_bgmv_moe_generic_module,
        get_cake_bgmv_moe_module,
        select_cake_bgmv_moe_generic_grouped,
        select_cake_bgmv_moe_generic_schedule,
        select_cake_bgmv_moe_generic_shrink,
        select_cake_bgmv_moe_order_remap,
        select_cake_bgmv_moe_schedule,
    )

    assert arch is not None
    dtype_name = _cake_dtype_name(x.dtype)
    variant = cake_bgmv_moe_variant(hidden_size, rank, num_tokens, arch)
    assert variant is not None
    pdl_mode = cake_bgmv_moe_pdl_mode(
        arch,
        num_tokens,
        hidden_size,
        torch.cuda.get_device_properties(x.device).multi_processor_count,
        variant,
    )
    schedule_id: int
    shrink_launch: Optional[Tuple[int, int]] = None
    use_grouped = False
    if variant == "generic":
        if grouped is None:
            use_grouped = select_cake_bgmv_moe_generic_grouped(
                int(sorted_token_ids.shape[0]),
                num_tokens,
                int(lora_a_weights[0].shape[0]),
                int(lora_a_weights[0].shape[1]),
                hidden_size,
                rank,
            )
        else:
            use_grouped = bool(grouped)
    use_order_remap = False
    if variant == "generic" and not use_grouped:
        if order_remap is None:
            use_order_remap = select_cake_bgmv_moe_order_remap(
                int(sorted_token_ids.shape[0]),
                num_tokens,
                int(lora_a_weights[0].shape[0]),
                int(lora_a_weights[0].shape[1]),
                hidden_size,
                rank,
                arch,
            )
        else:
            use_order_remap = bool(order_remap)
    elif order_remap:
        raise ValueError(
            "order_remap=True requires the per-route generic Cake pipeline "
            f"(variant={variant!r}, grouped={use_grouped})"
        )
    if variant == "specialized":
        schedule = select_cake_bgmv_moe_schedule(hidden_size, num_tokens, arch)
        schedule_id = CAKE_BGMV_MOE_SCHEDULE_IDS[schedule]
        module = get_cake_bgmv_moe_module(hidden_size, dtype_name, arch)
    else:
        generic_schedule = select_cake_bgmv_moe_generic_schedule(
            hidden_size, num_tokens, arch
        )
        schedule_id = CAKE_BGMV_MOE_GENERIC_SCHEDULE_IDS[generic_schedule]
        shrink_launch = select_cake_bgmv_moe_generic_shrink(
            int(sorted_token_ids.shape[0]), rank, hidden_size, arch
        )
        if use_order_remap:
            # The bin-ordered dispatch forms are the two-stage prefill kernel; the
            # selector never picks the remap on a deep-ring grid, a forced remap
            # takes the two-stage form.
            shrink_launch = (0, shrink_launch[1])
        module = get_cake_bgmv_moe_generic_module(rank, dtype_name, arch)
    return BGMVMoECakePlan(
        module,
        y_accum=y_accum,
        shrink_out=shrink_out,
        x=x,
        lora_a=lora_a_weights[0],
        lora_b=lora_b_weights[0],
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        lora_indices=lora_indices,
        topk_weights=topk_weights,
        schedule_id=schedule_id,
        variant=variant,
        shrink_launch=shrink_launch,
        grouped=use_grouped,
        order_remap=use_order_remap,
        pdl_mode=pdl_mode,
    )


@flashinfer_api
def bgmv_moe(
    x: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    lora_b_weights: List[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    output_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    High-level multi-LoRA MoE BGMV: shrink + expand in one call.

    Computes the LoRA delta for MoE:
        delta[token] = Σ_expert (topk_weight * x[token] @ lora_a[expert, lora_id] @ lora_b[expert, lora_id])

    Args:
        x: Input activations [num_tokens, hidden_dim].
        lora_a_weights: List of LoRA-A weight tensors, one per slice.
            Each has shape [max_loras, num_experts, rank, hidden_dim].
        lora_b_weights: List of LoRA-B weight tensors, one per slice.
            Each has shape [max_loras, num_experts, feat_out, rank].
        sorted_token_ids: Token indices for each pair [num_pairs].
        expert_ids: Expert indices for each pair [num_pairs].
        lora_indices: LoRA adapter ID for each token [num_tokens].
        topk_weights: Routing weights for each pair [num_pairs].
        num_experts: Number of experts.
        output_dim: Total output dimension. If None, inferred from lora_b_weights.
    Returns:
        Output tensor [num_tokens, total_feat_out] with LoRA deltas.
    """
    num_slices = len(lora_a_weights)
    num_tokens = x.size(0)
    num_pairs = sorted_token_ids.size(0)
    rank = lora_a_weights[0].size(2)
    device = x.device
    dtype = x.dtype

    # Infer output dimension
    feat_out_per_slice = [lora_b_weights[s].size(2) for s in range(num_slices)]
    total_feat_out = output_dim if output_dim is not None else sum(feat_out_per_slice)

    # Build w_ptr for shrink (lora_a)
    w_ptr_a = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride_a = 0
    for s in range(num_slices):
        lora_stride_a = fill_w_ptr(w_ptr_a, lora_a_weights[s], num_experts, s)

    # Shrink: x @ lora_a -> [num_slices, num_pairs, rank]
    shrink_out = torch.zeros(num_slices, num_pairs, rank, dtype=dtype, device=device)
    bgmv_moe_shrink(
        shrink_out,
        x,
        w_ptr_a,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        lora_stride_a,
    )

    # Build w_ptr for expand (lora_b)
    w_ptr_b = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride_b = 0
    for s in range(num_slices):
        lora_stride_b = fill_w_ptr(w_ptr_b, lora_b_weights[s], num_experts, s)

    # Slice start locations (build on CPU, transfer once to avoid per-element sync)
    slice_start_loc_cpu = torch.zeros(num_slices, dtype=torch.int64)
    loc = 0
    for s in range(num_slices):
        slice_start_loc_cpu[s] = loc
        loc += feat_out_per_slice[s]
    slice_start_loc = slice_start_loc_cpu.to(device=device)

    # Expand: shrink_out @ lora_b -> [num_tokens, total_feat_out]
    y = torch.zeros(num_tokens, total_feat_out, dtype=torch.float32, device=device)
    bgmv_moe_expand(
        y,
        shrink_out,
        w_ptr_b,
        sorted_token_ids,
        expert_ids,
        topk_weights,
        lora_indices,
        slice_start_loc,
        feat_out_per_slice,
        lora_stride_b,
    )

    return y.to(dtype)
