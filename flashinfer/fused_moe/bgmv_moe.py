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
from typing import List, Literal, Optional

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


def _blackwell_tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        int(tensor.data_ptr()),
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _blackwell_dtype_name(dtype: torch.dtype) -> Literal["bfloat16", "float16"]:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"Blackwell BGMV MoE requires BF16 or FP16, got {dtype}")


class BGMVMoEBlackwellPlan:
    """Pointer-stable SM100 BGMV MoE shrink+expand execution plan.

    The plan owns caller-visible FP32 accumulation and shrink workspaces. Its
    first eager ``run`` captures the exact launch sequence into a CUDA Graph;
    later calls replay that graph on the same stream. If called while an outer
    CUDA Graph is being captured, the constituent kernels are enqueued directly.
    """

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
    ) -> None:
        self._module = module
        self.y_accum = y_accum
        self.shrink_out = shrink_out
        self.x = x
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.sorted_token_ids = sorted_token_ids
        self.expert_ids = expert_ids
        self.lora_indices = lora_indices
        self.topk_weights = topk_weights
        self.schedule_id = schedule_id
        self._bound_signatures = tuple(
            _blackwell_tensor_signature(tensor)
            for tensor in (
                y_accum,
                shrink_out,
                x,
                lora_a,
                lora_b,
                sorted_token_ids,
                expert_ids,
                lora_indices,
                topk_weights,
            )
        )
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._owner_stream: Optional[torch.cuda.Stream] = None
        self._lock = threading.RLock()

    def _validate_binding(self) -> None:
        current = tuple(
            _blackwell_tensor_signature(tensor)
            for tensor in (
                self.y_accum,
                self.shrink_out,
                self.x,
                self.lora_a,
                self.lora_b,
                self.sorted_token_ids,
                self.expert_ids,
                self.lora_indices,
                self.topk_weights,
            )
        )
        if current != self._bound_signatures:
            raise RuntimeError(
                "BGMVMoEBlackwellPlan tensor storage, shape, stride, dtype, or "
                "device changed after preparation"
            )

    def _launch(self) -> None:
        self._module.run(
            self.y_accum,
            self.shrink_out,
            self.x,
            self.lora_a,
            self.lora_b,
            self.sorted_token_ids,
            self.expert_ids,
            self.lora_indices,
            self.topk_weights,
            self.schedule_id,
            int(torch.cuda.current_stream(self.x.device).cuda_stream),
        )

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
                    "BGMVMoEBlackwellPlan must replay on its original CUDA stream"
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
    backend: Literal["blackwell"] = "blackwell",
    shrink_out: Optional[torch.Tensor] = None,
    y_accum: Optional[torch.Tensor] = None,
) -> BGMVMoEBlackwellPlan:
    """Prepare the generated SM100 BGMV MoE pipeline for graph replay.

    This optimized path currently supports one LoRA slice, rank 32, hidden
    sizes 2688 or 3072, BF16/FP16 inputs, and exact SM100 devices. Routing may
    be arbitrary; the contiguous top-k=2 layout takes the optimized fast path.

    Args:
        x: Input activations with shape ``[num_tokens, hidden_size]``.
        lora_a_weights: One LoRA-A tensor with shape
            ``[num_loras, num_experts, 32, hidden_size]``.
        lora_b_weights: One LoRA-B tensor with shape
            ``[num_loras, num_experts, hidden_size, 32]``.
        sorted_token_ids: Routed token indices with shape ``[num_pairs]``.
        expert_ids: Expert index for each routed pair.
        lora_indices: LoRA index for each input token.
        topk_weights: FP32 routing weight for each routed pair.
        num_experts: Number of experts in both LoRA tensors.
        backend: Backend selector. Only ``"blackwell"`` is supported.
        shrink_out: Optional pointer-stable FP32 shrink workspace.
        y_accum: Optional pointer-stable FP32 output accumulator.

    Returns:
        A reusable graph-backed execution plan whose ``run`` method returns
        the FP32 accumulated output.
    """

    if backend != "blackwell":
        raise ValueError(
            f"prepare_bgmv_moe only supports backend='blackwell', got {backend}"
        )
    if (
        not torch.cuda.is_available()
        or not x.is_cuda
        or torch.cuda.get_device_capability(x.device) != (10, 0)
    ):
        capability = (
            torch.cuda.get_device_capability(x.device)
            if torch.cuda.is_available() and x.is_cuda
            else None
        )
        raise ValueError(
            "Blackwell BGMV MoE requires an exact SM100 CUDA device; "
            f"got capability={capability}"
        )
    if len(lora_a_weights) != 1 or len(lora_b_weights) != 1:
        raise ValueError("Blackwell BGMV MoE currently requires exactly one LoRA slice")
    if x.ndim != 2:
        raise ValueError(f"x must have shape [tokens, hidden], got {tuple(x.shape)}")
    num_tokens, hidden_size = (int(dim) for dim in x.shape)
    dtype_name = _blackwell_dtype_name(x.dtype)
    lora_a = lora_a_weights[0]
    lora_b = lora_b_weights[0]
    if lora_a.ndim != 4 or lora_b.ndim != 4:
        raise ValueError("LoRA weights must have rank 4")
    if int(lora_a.shape[0]) != int(lora_b.shape[0]):
        raise ValueError("LoRA-A and LoRA-B must have the same num_loras dimension")
    if int(lora_a.shape[1]) != num_experts or int(lora_b.shape[1]) != num_experts:
        raise ValueError("num_experts must match both LoRA weight tensors")
    if tuple(lora_a.shape[2:]) != (32, hidden_size):
        raise ValueError(
            "LoRA-A must have shape [num_loras, num_experts, 32, hidden_size]"
        )
    if tuple(lora_b.shape[2:]) != (hidden_size, 32):
        raise ValueError(
            "LoRA-B must have shape [num_loras, num_experts, hidden_size, 32]"
        )
    if lora_a.dtype != x.dtype or lora_b.dtype != x.dtype:
        raise ValueError("x and both LoRA weight tensors must have the same dtype")
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
    for name, tensor in (
        ("x", x),
        ("lora_a", lora_a),
        ("lora_b", lora_b),
        ("sorted_token_ids", sorted_token_ids),
        ("expert_ids", expert_ids),
        ("lora_indices", lora_indices),
        ("topk_weights", topk_weights),
    ):
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
    num_loras = int(lora_a.shape[0])
    if bool(((lora_indices < -1) | (lora_indices >= num_loras)).any()):
        raise ValueError("lora_indices values must be -1 or in [0, num_loras)")

    expected_shrink = (1, num_pairs, 32)
    if shrink_out is None:
        shrink_out = torch.empty(expected_shrink, dtype=x.dtype, device=x.device)
    if tuple(shrink_out.shape) != expected_shrink or shrink_out.dtype != x.dtype:
        raise ValueError(
            f"shrink_out must have shape {expected_shrink} and dtype {x.dtype}"
        )
    expected_output = (num_tokens, hidden_size)
    if y_accum is None:
        y_accum = torch.empty(expected_output, dtype=torch.float32, device=x.device)
    if tuple(y_accum.shape) != expected_output or y_accum.dtype != torch.float32:
        raise ValueError(
            f"y_accum must have shape {expected_output} and dtype torch.float32"
        )
    for name, tensor in (("shrink_out", shrink_out), ("y_accum", y_accum)):
        if (
            not tensor.is_cuda
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous tensor on {x.device}")

    from ..jit.blackwell_bgmv_moe import (
        BLACKWELL_BGMV_MOE_SCHEDULE_IDS,
        get_blackwell_bgmv_moe_module,
        select_blackwell_bgmv_moe_schedule,
    )

    schedule = select_blackwell_bgmv_moe_schedule(hidden_size, num_tokens)
    module = get_blackwell_bgmv_moe_module(hidden_size, dtype_name)
    return BGMVMoEBlackwellPlan(
        module,
        y_accum=y_accum,
        shrink_out=shrink_out,
        x=x,
        lora_a=lora_a,
        lora_b=lora_b,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        lora_indices=lora_indices,
        topk_weights=topk_weights,
        schedule_id=BLACKWELL_BGMV_MOE_SCHEDULE_IDS[schedule],
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
