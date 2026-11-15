# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public entry point for continuous MQA TopK."""

from __future__ import annotations

from typing import Literal

import torch

from ..api_logging import flashinfer_experimental_api, flashinfer_api


class FP8MQATopKWrapper:
    """Experimental selective TopK on continuous K for prefill.

    Every wrapper shares a process-wide retirement queue. ``plan``, ``prepare_pool`` and
    ``clear_pools`` drain it unless their own stream is capturing. Do not call
    these methods while any thread is performing global capture on another
    stream. Retired modules remain loaded until the next drain or process exit.

    FP8 uses K[N,D] and FP32 scales[N];
    MXFP4 uses packed uint8 K[N,D/2] and UE8M0 uint8 scales[N,4];
    MXFP8 uses FP8 K[N,D] with the same block-scale layout.
    ``num_keys`` describes storage, while plan's prefixes describe live tokens
    before compression. It defaults to floor(max_context_length / ratio).
    MXFP4 and MXFP8 storage must be padded to a multiple of 256 rows.

    Selective scoring infers its model policy from ``num_heads``, ``head_dim``,
    ``top_k`` and ``compression_ratio``. An optional ``model_config`` name
    must agree with those fields. Explicit Qwen configuration requires weights ``1/sqrt(128)``
    rounded to the selected weights dtype and, for FP8, unit K scales;
    these conditions are checked on replay.
    Length policies use physical K rows, including the queries' compressed keys.

    ``weights_dtype`` defaults to ``torch.float32``. Selective scoring also accepts
    ``torch.bfloat16`` for every supported QK, logits and arithmetic mode.
    Weights must match the planned dtype; BF16 weights may have padded rows
    with contiguous heads, including H4 row stride of eight.
    Input storage dtype does not select head accumulation precision.

    Selective ``math_mode=None`` resolves to ``relu_fast`` for FP8/MXFP4 and
    ``relu_strict`` for MXFP8. Fast mode
    checks bounds on device every replay: weights must be finite and zero or
    have magnitude at least 2**-125, and MXFP4 scale bytes must be <= 159.
    Violations set an error bit and return -1 rows.

    ``math_mode="native_bf16"`` and ``math_mode="native_fp16"`` are opt-in
    modes for H4/H32/H64 FP8, MXFP8 or MXFP4 inputs, requiring the
    matching logits dtype.
    QK dots accumulate in FP32. ReLU results and weights are rounded to the
    native dtype, followed by four native FMA chains and the rounded tree
    ``(c0+c2)+(c1+c3)``. FP8 native_fp16 first divides QK by 512 and multiplies
    weights by 512. This power-of-two rebalance follows from the E4M3FN/D128
    maximum dot product; it avoids overflowing the FP16 conversion of raw QK.
    It is independent of input values. Other native modes use no rebalance.
    FP8 applies the original FP32 key scale after the sum, then rounds the
    output. MXFP4 block scales are applied within the dot product.
    Native modes have their own rounding, underflow and overflow behavior;
    they differ from FP32 head accumulation in ``relu_strict``/``relu_fast``.

    Construct outside capture, plan once per step, and run once per layer.
    Prepare every K/scale binding before capture. Keep this wrapper and all
    capture inputs alive until the caller discards its graphs. Calls on a
    wrapper must be serialized. Output is shared int32[Q, top_k] storage with
    request-local indices and -1 padding; ties need not have stable indices.
    """

    is_experimental = True

    @flashinfer_experimental_api(feature="FP8MQATopKWrapper")
    def __init__(
        self,
        *,
        max_context_length: int,
        max_chunk_size: int,
        max_batch_size: int,
        device,
        strategy: Literal["selective"] = "selective",
        backend: Literal["cutedsl"] = "cutedsl",
        num_heads: int,
        head_dim=128,
        top_k=512,
        math_mode=None,
        compression_ratio=1,
        num_keys=None,
        qkv_dtype="fp8",
        logits_dtype=torch.float32,
        weights_dtype=torch.float32,
        model_config=None,
    ):
        if strategy != "selective":
            raise ValueError("continuous TopK supports selective only")
        if backend != "cutedsl":
            raise ValueError("backend must be 'cutedsl'")
        from ..experimental.attn_scores import _continuous_plan_type

        options = dict(
            max_context_length=max_context_length,
            max_chunk_size=max_chunk_size,
            max_batch_size=max_batch_size,
            device=device,
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k,
            compression_ratio=compression_ratio,
            math_mode=math_mode,
            num_keys=num_keys,
        )
        options.update(
            qkv_dtype=qkv_dtype,
            logits_dtype=logits_dtype,
            weights_dtype=weights_dtype,
            model_config=model_config,
        )
        self._capacity_plan = _continuous_plan_type("cutedsl")(**options)
        self._compression_ratio = compression_ratio
        self._backend = backend
        self._strategy = strategy

    @flashinfer_api
    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        *,
        num_query_rows: int | None = None,
        next_n: int | None = None,
        host_length_hints: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Stage original-token prefixes int32[2].

        For query i, visible keys end at
        floor((KV - Q + i + 1) / compression_ratio).
        K stores floor(KV / compression_ratio) keys.
        ``next_n`` optionally asserts exactly that many query rows.
        ``host_length_hints`` optionally supplies host upper bounds; violations
        set error bit 2. Invalid prefixes fail closed (bits 1/2/4). Read
        ``error`` and ``repair_error`` outside capture; the latter sets bit 16
        on fast-mode or explicit Qwen numerical contract violations. Planning
        is capturable after eager preparation.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            CUDA int32 prefix array of shape ``[2]`` containing the query
            offsets in original tokens for the single request.
        kv_indptr : torch.Tensor
            CUDA int32 prefix array of shape ``[2]`` containing the KV offsets
            in original tokens for the single request.
        num_query_rows : int, optional
            Extent of the supplied Q storage. Defaults to ``max_chunk_size``.
        next_n : int, optional
            Assert that the request has exactly this many query rows.
        host_length_hints : tuple[torch.Tensor, torch.Tensor], optional
            ``(query_prefix, kv_bounds)`` as contiguous CPU int32 or int64
            tensors of shapes ``[2]`` and ``[1]``. The query prefix must fit
            the supplied Q extent, and each KV bound must lie in
            ``[0, max_context_length]``. Device violations set error bit 2.
            Omit when capturing ``plan`` in a CUDA graph.

        Returns
        -------
        None
        """
        return self._capacity_plan.plan(
            qo_indptr,
            kv_indptr,
            num_query_rows=num_query_rows,
            next_n=next_n,
            host_length_hints=host_length_hints,
        )

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        *,
        k_scale: torch.Tensor,
        out: torch.Tensor | None = None,
        q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score continuous K directly; no KV packing, page lookup or gather.

        Parameters
        ----------
        q : torch.Tensor
            Contiguous CUDA queries. FP8 and MXFP8 use E4M3FN ``[Q, H, D]``;
            MXFP4 uses packed uint8 ``[Q, H, D/2]``.
        weights : torch.Tensor
            CUDA head weights ``[Q, H]`` with the configured FP32 or BF16 dtype.
            FP32 weights are contiguous; BF16 rows may be padded.
        k : torch.Tensor
            Contiguous CUDA keys: E4M3FN ``[N, D]`` for FP8 and MXFP8, or
            packed uint8 ``[N, D/2]`` for MXFP4. Block-scaled storage has
            ``N`` divisible by 256; the plan specifies the live extent.
        k_scale : torch.Tensor
            Contiguous CUDA FP32 ``[N]`` scales for FP8, or UE8M0 uint8
            ``[N, 4]`` block scales for MXFP4 and MXFP8. K and its scales
            must be 16-byte aligned and prepared before capture.
        out : torch.Tensor, optional
            Contiguous CUDA int32 output buffer ``[Q, top_k]``.
        q_scale : torch.Tensor, optional
            Required for MXFP4 and MXFP8: contiguous CUDA int32 ``[Q, H]``
            or UE8M0 uint8 ``[Q, H, 4]`` block scales. Must be None for FP8.

        Returns
        -------
        torch.Tensor
            CUDA int32 ``[Q, top_k]`` request-local indices, padded with -1.
            Storage is shared across calls unless ``out`` is supplied.
        """
        return self._capacity_plan.run(
            q, weights, k, k_scale=k_scale, out=out, q_scale=q_scale
        )

    def prepare_pool(self, k, *, k_scale):
        """Compile/capture a K/scale binding outside caller graph capture."""
        return self._capacity_plan.prepare_pool(k, k_scale=k_scale)

    def clear_pools(self):
        return self._capacity_plan.clear_pools()

    def input_buffers(self):
        """Stable Q/weight/prefix buffers (plus q_scale for MXFP4/MXFP8)."""
        return self._capacity_plan.input_buffers()

    @property
    def strategy(self):
        return self._strategy

    @property
    def math_mode(self):
        return self._capacity_plan.math_mode

    @property
    def compression_ratio(self):
        return self._compression_ratio

    @property
    def error(self):
        return self._capacity_plan.error

    @property
    def repair_error(self):
        return self._capacity_plan.repair_error

    @property
    def active_shape(self):
        return self._capacity_plan.active_shape


__all__ = ["FP8MQATopKWrapper"]
