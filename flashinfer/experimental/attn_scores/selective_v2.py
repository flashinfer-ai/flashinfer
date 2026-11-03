# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Capacity-planned continuous K adapter for the indexer v2 kernels."""

import torch

from .selective_contiguous import _ContinuousSelectiveTopKPlan
from .selective_capacity import (
    _CapacitySelectiveTopKPlan,
    _CapacityStepInputs,
    _SelectiveCapacity,
    _bind_metadata_module_lifetime,
)


class _V2StepInputs(_CapacityStepInputs):
    """Stage MXFP4 scales with the same cross-alias checks as Q and weights."""

    def _payload_copies(self, q, weights):
        # v2 owns its BF16 contract; v1 retains its FP32-only validation.
        if not all(isinstance(t, torch.Tensor) for t in (q, weights)):
            raise ValueError("step inputs must be tensors")
        if (
            q.device != self.device or q.dtype != self.q.dtype or q.ndim != 3
            or q.shape[1:] != self.q.shape[1:]
            or q.shape[0] > self.capacity.max_chunk_size or not q.is_contiguous()
            or weights.device != self.device or weights.dtype != self.weights.dtype
            or weights.shape != (q.shape[0], self.capacity.num_heads)
            or weights.stride(1) != 1 or weights.stride(0) < self.capacity.num_heads
            or (weights.dtype == torch.float32 and not weights.is_contiguous())
        ):
            raise ValueError("Q and weights must match the planned shapes and dtypes")
        return ((self.q[:q.shape[0]], q), (self.weights[:q.shape[0]], weights))

    def input_buffers(self):
        result = super().input_buffers()
        if getattr(self, "q_scale", None) is not None:
            result["q_scale"] = self.q_scale
        return result

    def stage_payload(self, q, weights):
        copies = self._payload_copies(q, weights)
        if getattr(self, "q_scale", None) is not None:
            copies += ((self.q_scale[: q.shape[0]], self.q_scale_input),)
        self._copy_inputs(copies)


class _CapacityV2TopKPlan(_ContinuousSelectiveTopKPlan):
    """Reuse step validation and staging while owning an independent v2 chain.

    Each layer binds caller-owned K and scales directly. Aligned row storage
    avoids constructor-created padded copies, so live payload updates reach
    all four scoring stages on graph replay.
    """

    def __init__(
        self,
        *,
        max_context_length,
        max_chunk_size,
        max_batch_size,
        num_keys=None,
        device=None,
        num_heads=8,
        head_dim=128,
        top_k=512,
        compression_ratio=1,
        math_mode="relu_strict",
        qkv_dtype="fp8",
        logits_dtype=torch.float32,
        weights_dtype=torch.float32,
        model_config=None,
    ):
        if num_heads not in (4, 32, 64) or head_dim != 128:
            raise ValueError("backend='v2' supports H4/H32/H64 and head_dim=128")
        if max_batch_size != 1:
            raise ValueError(
                "backend='v2' supports single-request prefill only; decode is unsupported"
            )
        if qkv_dtype not in ("fp8", "mxfp4", "mxfp8"):
            raise ValueError("v2 qkv_dtype must be 'fp8', 'mxfp4' or 'mxfp8'")
        if logits_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("v2 logits_dtype must be float16, bfloat16 or float32")
        if weights_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("v2 weights_dtype must be float32 or bfloat16")
        if top_k not in (512, 1024, 2048):
            raise ValueError("backend='v2' supports top_k=512/1024/2048")
        from .indexer_config import resolve_model_config

        model_config = resolve_model_config(
            num_heads, head_dim, top_k, compression_ratio, model_config
        )
        row_alignment = max(8 if num_heads == 32 else 1, 256 // num_heads)
        if (
            type(max_chunk_size) is not int
            or max_chunk_size < row_alignment
            or max_chunk_size % row_alignment
        ):
            raise ValueError(
                f"v2 max_chunk_size must be a positive multiple of {row_alignment}"
            )
        capacity = _SelectiveCapacity(
            num_heads,
            head_dim,
            128,
            compression_ratio,
            max_context_length,
            max_chunk_size,
            max_batch_size,
        )
        physical = capacity.max_context_length // compression_ratio
        if physical > 1048576:
            raise ValueError("backend='v2' supports at most 1048576 keys after pooling")
        self._qkv_dtype = qkv_dtype
        self._logits_dtype = logits_dtype
        self._model_config = model_config
        self._top_k = top_k
        from ._math_policy_v2 import validate_math_mode

        self._math_mode = validate_math_mode(
            math_mode,
            qkv_dtype=qkv_dtype,
            num_heads=num_heads,
            logits_dtype=logits_dtype,
        )
        from .selective_logits import _validate_selective_device

        _validate_selective_device(torch.device(device))
        if num_keys is None:
            num_keys = physical
        if type(num_keys) is not int or not 0 < num_keys <= 1048576:
            raise ValueError("v2 num_keys must be in 1..1048576")
        if qkv_dtype in ("mxfp4", "mxfp8") and num_keys % 256:
            raise ValueError(
                "Block-scaled K storage must have a multiple of 256 rows; plan describes the live extent"
            )
        self._num_keys = num_keys
        self._capacity = capacity
        self._pool_bindings = {}
        self._eager_skip_sampling = False
        self._engine = None
        self._step_prepared = False
        self._state = _V2StepInputs(
            capacity, device=device, num_pages=num_keys, continuous=True
        )
        self._engines = {}
        self._scale_owners = {}
        self._k_scale = None
        if qkv_dtype == "mxfp4":
            self._state.q = torch.zeros(
                (max_chunk_size, num_heads, 64), device=device, dtype=torch.uint8
            )
        else:
            self._state.q.zero_()
        if weights_dtype == torch.bfloat16:
            self._state.weights = torch.empty(
                (max_chunk_size, num_heads), device=device, dtype=weights_dtype
            )
        self._state.weights.fill_(128**-0.5)
        self._q_scale = (
            torch.full(
                (max_chunk_size, num_heads),
                0x7F7F7F7F,
                device=device,
                dtype=torch.int32,
            )
            if qkv_dtype in ("mxfp4", "mxfp8")
            else None
        )
        self._state.q_scale = self._q_scale
        self._state.q_scale_input = self._q_scale
        self._selected = torch.full(
            (max_chunk_size, top_k), -1, device=device, dtype=torch.int32
        )
        self._repair_error = torch.zeros(1, device=device, dtype=torch.int32)
        self._starts = torch.zeros(max_chunk_size, device=device, dtype=torch.int32)
        self._ends = torch.zeros_like(self._starts)
        self._score_ends = torch.zeros_like(self._starts)
        self._keys = None
        self._scales = None
        self._qwen = model_config == "Qwen3.8-Flash-Next"
        self._fast = math_mode == "relu_fast"
        self._metadata_flags = torch.empty(256, device=device, dtype=torch.int32)
        self._metadata_compiled = None
        self._metadata_args = {}
        self._cache_protected_ranges()

    @property
    def _needs_sampling(self):
        # v2 owns sampling; the v1 sample schedule is never built or executed.
        return False

    def _refresh_schedules(self):
        pass

    def _prepared_launches(self):
        return ()

    def input_buffers(self):
        result = super().input_buffers()
        if self._q_scale is not None:
            result["q_scale"] = self._q_scale
        return result

    def _cache_protected_ranges(self):
        from .selective_logits import _byte_range

        super()._cache_protected_ranges()
        # The v2 metadata and bounds buffers also belong to the plan.
        ranges = list(self._protected_ranges)
        ranges.extend(
            _byte_range(value)
            for value in self.__dict__.values()
            if isinstance(value, torch.Tensor) and value.numel()
        )
        merged = []
        for begin, end in sorted(ranges):
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((begin, end))
        self._protected_ranges = tuple(merged)

    def _pool_key(self, keys):
        if self._qkv_dtype != "mxfp8":
            return super()._pool_key(keys)
        scale = self._k_scale
        if (
            not isinstance(keys, torch.Tensor)
            or keys.shape != (self._num_keys, 128)
            or keys.dtype != torch.float8_e4m3fn
            or keys.device != self._state.device
            or not keys.is_contiguous()
            or keys.data_ptr() % 16
        ):
            raise ValueError("MXFP8 K requires aligned contiguous E4M3FN CUDA [N,128]")
        if (
            not isinstance(scale, torch.Tensor)
            or scale.device != keys.device
            or scale.dtype != torch.uint8
            or scale.shape != (self._num_keys, 4)
            or not scale.is_contiguous()
            or scale.data_ptr() % 16
        ):
            raise ValueError("MXFP8 k_scale requires aligned contiguous UE8M0 uint8[N,4]")
        return keys.data_ptr(), scale.data_ptr(), self._layout(keys), self._layout(scale)

    def prepare_pool(self, keys, *, k_scale=None):
        """Warm each K/scale binding; CUDA graphs belong to the caller."""
        if k_scale is not None:
            self._k_scale = k_scale
        key = self._pool_key(keys)
        with torch.cuda.device(self._state.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare_pool must run outside CUDA graph capture")
            self._state.prepare()
            self._state.refresh()
            if key in self._pool_bindings:
                self._engine = self._engines[key]
                return
            previous_engine = self._engine
            previous_keys, previous_scales = self._keys, self._scales
            previous_template = tuple(
                getattr(self, name, None)
                for name in ("_template_engine", "_template_keys", "_template_scales")
            )
            self._keys, self._scales = keys, self._k_scale
            try:
                self._prepare_metadata(keys)
                if previous_template[0] is None:
                    from .kernels.indexer_topk_v2.api import ThrowawayTopK

                    engine = ThrowawayTopK(
                        self._state.q,
                        keys,
                        self._state.weights,
                        self._starts,
                        self._ends,
                        k_scale=self._k_scale,
                        q_scale=self._q_scale,
                        top_k=self.top_k,
                        logits_dtype=self._logits_dtype,
                        model_config=self._model_config,
                        compression_ratio=self._capacity.compression_ratio,
                        math_mode=self.math_mode,
                        out=self._selected,
                        _score_ends=self._score_ends,
                        # Validate live rows/scales on replay; capacity padding is inactive.
                        _validate_numerics=False,
                    )
                    self._template_engine = engine
                    self._template_keys, self._template_scales = keys, self._k_scale
                else:
                    engine = self._bind_engine(keys, self._k_scale)
                self._engines[key] = self._engine = engine
                self._execute_layer(keys)
            except BaseException:
                self._engine = previous_engine
                self._keys, self._scales = previous_keys, previous_scales
                self._template_engine, self._template_keys, self._template_scales = (
                    previous_template
                )
                self._engines.pop(key, None)
                self._metadata_args.pop(key, None)
                raise
            self._scale_owners[key] = self._k_scale
            self._pool_bindings[key] = (keys, ())
            self._cache_protected_ranges()

    def _bind_engine(self, keys, scales):
        """Rebind K/scale views while sharing the four-stage scratch workspace."""
        import copy
        from dataclasses import replace
        from cutlass import testing

        def rebind(arg, source, target):
            if arg.data_ptr() != source.data_ptr():
                # K2 may consume scale storage prepared by K1 in the shared
                # workspace. That argument is deliberately not rebound.
                return arg
            view = target.view(arg.dtype)
            return view.as_strided(arg.shape, arg.stride(), view.storage_offset())

        engine = copy.copy(self._template_engine)
        engine.stages = []
        for name, stage in self._template_engine.stages:
            args = list(stage.arguments.args)
            if name != "K3":
                args[0] = rebind(args[0], self._template_keys, keys)
                args[1] = rebind(args[1], self._template_scales, scales)
            engine.stages.append(
                (
                    name,
                    replace(
                        stage,
                        arguments=testing.JitArguments(*args, **stage.arguments.kwargs),
                    ),
                )
            )
        return engine

    def clear_pools(self):
        super().clear_pools()
        self._engines.clear()
        self._metadata_args.clear()
        self._engine = None
        self._keys = self._scales = None
        self._template_engine = self._template_keys = self._template_scales = None
        self._cache_protected_ranges()

    def _bind_scales(self):
        pass

    def _prepare_layer_inputs(self, keys):
        from .selective_logits import _current_stream

        key = self._pool_key(keys)
        self._engine = self._engines[key]
        self._metadata_compiled(
            *self._metadata_args[key], _current_stream(self._state.device)
        )

    def _prepare_metadata(self, keys):
        from cutlass.cute.runtime import from_dlpack
        from .kernels.selective_v2_metadata import SelectiveV2Metadata
        from .selective_logits import _compile_for_device, _current_stream

        state = self._state
        scales = (
            self._scales.view(torch.int32)
            if self._qkv_dtype in ("mxfp4", "mxfp8")
            else self._scales
        )
        tensors = (
            state.weights.flatten(),
            scales.flatten(),
            self._q_scale.flatten() if self._q_scale is not None else self._starts,
            state.active_shape,
            state.physical_kv_indptr,
            state.row_ends,
            self._starts,
            self._ends,
            self._score_ends,
            self._metadata_flags,
            self._repair_error,
        )
        args = tuple(from_dlpack(tensor, assumed_align=4) for tensor in tensors)
        kernel = SelectiveV2Metadata(
            self._capacity.num_heads,
            self._qwen,
            self._fast,
            self._qkv_dtype in ("mxfp4", "mxfp8"),
            self._metadata_flags.numel(),
            top_k=self.top_k,
        )
        self._metadata_args[self._pool_key(keys)] = args
        if self._metadata_compiled is None:
            self._metadata_compiled = _bind_metadata_module_lifetime(
                _compile_for_device(
                    state.device,
                    kernel,
                    *args,
                    _current_stream(state.device),
                )
            )

    def _execute_layer(self, kv_fused):
        if self._engine is None:
            self._selected.fill_(-1)
            return
        self._prepare_layer_inputs(kv_fused)
        self._engine.run()

    def diagnostics(self, kv_fused):
        """Replay once outside timing and snapshot populations before arena reuse.

        Counts subtract 2**24 per overflowing bank; there are at most 1M
        accepted keys. Modulo therefore recovers the actual population. The
        recovery queue names query blocks, whose entire block is rescored.
        v2 has one exact recovery stage, so no separate terminal repair launch.
        """
        self._pool_key(kv_fused)
        if self._engine is None:
            return dict(
                underfilled_rows=0,
                overflow_rows=0,
                first_repair_rows=0,
                terminal_repair_rows=0,
                repair_trigger_rows=0,
            )
        self._prepare_layer_inputs(kv_fused)
        engine = self._engine
        for _, stage in engine.stages[:3]:
            stage.run()
        counts = engine.counts.cpu().to(torch.int64)
        visible = (self._ends - self._starts).cpu()
        required = visible.clamp(max=self.top_k)
        scoring = visible > self.top_k
        queued = engine.recovery_queue.cpu().tolist()
        tasks = queued[1 : 1 + queued[0]]
        qtile = engine.recovery_qtile
        stats = dict(
            underfilled_rows=int(
                (((counts % (1 << 24)) < required) & scoring).count_nonzero()
            ),
            overflow_rows=int(((counts < 0) & scoring).count_nonzero()),
            repair_trigger_rows=int(((counts < required) & scoring).count_nonzero()),
            first_repair_rows=sum(
                min(qtile, counts.numel() - task * qtile) for task in tasks
            ),
            terminal_repair_rows=0,
            recovery_tasks=tasks,
            encoded_candidate_counts=counts.tolist(),
            counter_scope="untimed replay snapshot after candidate selection, before arena reuse; first repair counts complete query blocks; no separate terminal repair stage",
        )
        engine.stages[3][1].run()
        return stats

    def run(self, q, weights, kv_fused, *, k_scale, out=None, q_scale=None):
        self._state._payload_copies(q, weights)
        self._k_scale = k_scale
        key = self._pool_key(kv_fused)
        if out is not None and isinstance(out, torch.Tensor):
            from .selective_logits import _byte_range, _overlap

            if _overlap(_byte_range(out), _byte_range(k_scale)):
                raise ValueError("out must not overlap k_scale")
        if key not in self._engines and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare each K/k_scale binding before capture")
        if key in self._engines:
            self._engine = self._engines[key]
        if not self._step_prepared:
            raise RuntimeError("plan must be called before run")
        if q.shape[0] != self._supplied_rows:
            raise ValueError("Q rows must match plan num_query_rows")
        if self._qkv_dtype == "fp8":
            if q_scale is not None:
                raise TypeError("q_scale requires MXFP4 or MXFP8")
        else:
            if (
                not isinstance(q_scale, torch.Tensor)
                or q_scale.device != self._state.device
                or not q_scale.is_contiguous()
            ):
                raise ValueError("Block-scaled inputs require contiguous CUDA q_scale")
            if q_scale.dtype == torch.uint8 and q_scale.shape == (*q.shape[:2], 4):
                q_scale = q_scale.view(torch.int32).reshape(q.shape[:2])
            if q_scale.dtype != torch.int32 or q_scale.shape != q.shape[:2]:
                raise ValueError("q_scale must be int32 [R,H] or UE8M0 bytes [R,H,4]")
            if out is not None:
                from .selective_logits import _byte_range, _overlap

                if isinstance(out, torch.Tensor) and _overlap(
                    _byte_range(out), _byte_range(q_scale)
                ):
                    raise ValueError("out must not overlap q_scale")
            self._state.q_scale_input = q_scale
        return _CapacitySelectiveTopKPlan.run(self, q, weights, kv_fused, out=out)
