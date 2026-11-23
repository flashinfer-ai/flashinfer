# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Capacity-planned continuous K adapter for the selective kernels."""

import torch

_MAX_POOLED_KEYS = 1024 * 1024
from .kernels.continuous_topk_metadata import SelectiveContinuousCapacityRefreshKernel

from ._topk_capacity import (
    _CapacityStepInputs,
    _TopKCapacity,
    _bind_metadata_module_lifetime,
    _drain_deferred_metadata_modules,
)


class _SelectiveStepInputs(_CapacityStepInputs):
    """Stage MXFP4 scales with the same cross-alias checks as Q and weights."""

    def input_buffers(self):
        result = super().input_buffers()
        if getattr(self, "q_scale", None) is not None:
            result["q_scale"] = self.q_scale
        return result

    def stage_payload(self, q, weights, copies=None):
        if copies is None:
            copies = self._payload_copies(q, weights)
        if getattr(self, "q_scale", None) is not None:
            copies += ((self.q_scale[: q.shape[0]], self.q_scale_input),)
        self._copy_inputs(copies)


class _CapacitySelectiveTopKPlan:
    """Reuse step validation and staging while owning an independent selective chain.

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
        publish_candidate_blocks=False,
        candidate_capacity=0,
    ):
        if type(publish_candidate_blocks) is not bool:
            raise TypeError("publish_candidate_blocks must be bool")
        if publish_candidate_blocks and max_chunk_size > 65535:
            raise ValueError("candidate publication supports at most 65535 query rows")
        if num_heads not in (4, 32, 64) or head_dim != 128:
            raise ValueError("backend='cutedsl' supports H4/H32/H64 and head_dim=128")
        if max_batch_size != 1:
            raise ValueError(
                "backend='cutedsl' supports single-request prefill only; decode is unsupported"
            )
        if qkv_dtype not in ("fp8", "mxfp4", "mxfp8"):
            raise ValueError("cutedsl qkv_dtype must be 'fp8', 'mxfp4' or 'mxfp8'")
        if logits_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "cutedsl logits_dtype must be float16, bfloat16 or float32"
            )
        if weights_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("cutedsl weights_dtype must be float32 or bfloat16")
        if top_k not in (512, 1024, 2048):
            raise ValueError("backend='cutedsl' supports top_k=512/1024/2048")
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
                f"cutedsl max_chunk_size must be a positive multiple of {row_alignment}"
            )
        capacity = _TopKCapacity(
            num_heads,
            head_dim,
            compression_ratio,
            max_context_length,
            max_chunk_size,
            max_batch_size,
        )
        if type(candidate_capacity) is not int or candidate_capacity not in (0, 2048):
            raise ValueError("candidate_capacity must be 0 or 2048")
        self._candidate_capacity = candidate_capacity
        self._candidate_supplied = False
        # Consumer pools include both history and the current query chunk.
        key_limit = _MAX_POOLED_KEYS + (max_chunk_size if candidate_capacity else 0)
        physical = capacity.max_context_length // compression_ratio
        if physical > key_limit:
            raise ValueError(
                f"backend='cutedsl' supports at most {key_limit} keys after pooling"
            )
        self._qkv_dtype = qkv_dtype
        self._logits_dtype = logits_dtype
        self._model_config = model_config
        self._top_k = top_k
        from ._selective_math_policy import validate_math_mode

        if math_mode is None:
            math_mode = (
                "relu_strict"
                if qkv_dtype == "mxfp8"
                or publish_candidate_blocks
                or candidate_capacity
                else "relu_fast"
            )
        if (
            publish_candidate_blocks or candidate_capacity
        ) and math_mode == "relu_fast":
            raise ValueError(
                "candidate publication and consumption require relu_strict, native_bf16 or native_fp16"
            )
        self._math_mode = validate_math_mode(
            math_mode,
            qkv_dtype=qkv_dtype,
            num_heads=num_heads,
            logits_dtype=logits_dtype,
        )
        if publish_candidate_blocks and not (
            model_config == "DS-V4.1-Flash"
            and compression_ratio == 1
            and qkv_dtype == "mxfp4"
        ):
            raise ValueError(
                "candidate publication requires DS-V4.1-Flash, ratio 1, MXFP4"
            )
        if candidate_capacity and not (
            model_config == "DS-V4.1-Flash"
            and compression_ratio == 1
            and qkv_dtype == "mxfp4"
            and not publish_candidate_blocks
        ):
            raise ValueError(
                "candidate consumption requires DS-V4.1-Flash MXFP4, ratio 1, without publication"
            )
        self._publish_candidate_blocks = publish_candidate_blocks
        from ._topk_runtime import _validate_topk_device

        _validate_topk_device(torch.device(device))
        if num_keys is None:
            num_keys = physical
        if type(num_keys) is not int or not 0 < num_keys <= key_limit:
            raise ValueError(f"cutedsl num_keys must be in 1..{key_limit}")
        if qkv_dtype in ("mxfp4", "mxfp8") and num_keys % 256:
            raise ValueError(
                "Block-scaled K storage must have a multiple of 256 rows; plan describes the live extent"
            )
        self._num_keys = num_keys
        self._capacity = capacity
        self._pool_bindings = {}
        self._engine = None
        # Rows K1-K3 are launched for in this step, and whether the engines of this wrapper follow the step
        # (see _bind_supplied_rows).
        self._step_rows = max_chunk_size
        self._rows_bindable = None
        self._step_prepared = False
        self._state = _SelectiveStepInputs(
            capacity,
            device=device,
            num_keys=num_keys,
            qkv_dtype=qkv_dtype,
            weights_dtype=weights_dtype,
            refresh_type=SelectiveContinuousCapacityRefreshKernel,
        )
        self._engines = {}
        self._scale_owners = {}
        self._k_scale = None
        self._state.q.zero_()
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
        if candidate_capacity:
            self._candidate_input = torch.full(
                (max_chunk_size, candidate_capacity),
                2147483647,
                device=device,
                dtype=torch.int32,
            )
            self._candidate_ends = torch.zeros_like(self._starts)
            self._candidate_lens = torch.zeros_like(self._starts)
            self._candidate_starts = torch.zeros_like(self._starts)
        self._keys = None
        self._scales = None
        self._qwen = model_config == "Qwen3.8-Flash-Next"
        self._fast = math_mode == "relu_fast"
        self._metadata_flags = torch.empty(256, device=device, dtype=torch.int32)
        self._metadata_compiled = None
        self._metadata_args = {}
        self._cache_protected_ranges()

    def _refresh_schedules(self):
        pass

    def input_buffers(self):
        result = self._state.input_buffers()
        if self._candidate_capacity:
            result["candidate_blocks"] = self._candidate_input
        return result

    def _cache_protected_ranges(self):
        from ._topk_runtime import _byte_range

        # The metadata and bounds buffers also belong to the plan.
        owners = [*self.__dict__.values(), *self._state.__dict__.values()]
        if self._engine is not None:
            owners.extend(self._engine.__dict__.values())
        owners.extend(binding[0] for binding in self._pool_bindings.values())
        owners.extend(self._scale_owners.values())
        ranges = sorted(
            _byte_range(t) for t in owners if isinstance(t, torch.Tensor) and t.numel()
        )
        merged = []
        for begin, end in ranges:
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((begin, end))
        self._protected_ranges = tuple(merged)

    @staticmethod
    def _layout(tensor):
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    def _pool_key(self, keys, scale=None):
        if scale is None:
            scale = self._k_scale
        fp4 = self._qkv_dtype == "mxfp4"
        mxfp8 = self._qkv_dtype == "mxfp8"
        width = self._capacity.head_dim // (2 if fp4 else 1)
        if (
            not isinstance(keys, torch.Tensor)
            or keys.shape != (self._num_keys, width)
            or keys.dtype != (torch.uint8 if fp4 else torch.float8_e4m3fn)
            or keys.device != self._state.device
            or not keys.is_contiguous()
            or keys.data_ptr() % 16
        ):
            message = (
                "MXFP8 K requires aligned contiguous E4M3FN CUDA [N,128]"
                if mxfp8
                else "K must be aligned contiguous CUDA storage [num_keys, D] (D/2 for MXFP4)"
            )
            raise ValueError(message)
        block_scaled = fp4 or mxfp8
        if (
            not isinstance(scale, torch.Tensor)
            or scale.device != keys.device
            or not scale.is_contiguous()
            or scale.data_ptr() % 16
            or scale.shape
            != ((self._num_keys, 4) if block_scaled else (self._num_keys,))
            or scale.dtype != (torch.uint8 if block_scaled else torch.float32)
        ):
            message = (
                "MXFP8 k_scale requires aligned contiguous UE8M0 uint8[N,4]"
                if mxfp8
                else "k_scale must be contiguous FP32[N] or MXFP4 UE8M0 uint8[N,4]"
            )
            raise ValueError(message)
        return (
            keys.data_ptr(),
            scale.data_ptr(),
            self._layout(keys),
            self._layout(scale),
        )

    def prepare_pool(self, keys, *, k_scale=None):
        """Warm each K/scale binding; CUDA graphs belong to the caller."""
        _drain_deferred_metadata_modules()
        scale = self._k_scale if k_scale is None else k_scale
        key = self._pool_key(keys, scale)
        with torch.cuda.device(self._state.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare_pool must run outside CUDA graph capture")
            self._k_scale = scale
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
                    from .kernels.selective_topk.api import ThrowawayTopK

                    engine = ThrowawayTopK(
                        self._state.q,
                        keys,
                        self._state.weights,
                        self._candidate_starts
                        if self._candidate_capacity
                        else self._starts,
                        self._candidate_ends
                        if self._candidate_capacity
                        else self._ends,
                        k_scale=self._k_scale,
                        q_scale=self._q_scale,
                        top_k=self.top_k,
                        logits_dtype=self._logits_dtype,
                        model_config=self._model_config,
                        candidate_k_block_ids=self._candidate_input
                        if self._candidate_capacity
                        else None,
                        candidate_valid_lens=self._candidate_lens
                        if self._candidate_capacity
                        else None,
                        _candidate_error=self._repair_error
                        if self._candidate_capacity
                        else None,
                        compression_ratio=self._capacity.compression_ratio,
                        math_mode=self.math_mode,
                        out=self._selected,
                        # Publishers score short rows too, to select candidate blocks.
                        _score_ends=(
                            self._ends
                            if self._publish_candidate_blocks
                            else self._score_ends
                        ),
                        # Validate live rows/scales on replay; capacity padding is inactive.
                        _validate_numerics=False,
                        _filter_config=(
                            dict(
                                publish_blocks=True,
                                prepare_kv_scales=True,
                                shared_kv_queries=False,
                            )
                            if self._publish_candidate_blocks
                            else None
                        ),
                    )
                    if self._candidate_capacity:
                        engine.prepare_live_bounds(
                            self._state.row_ends,
                            self._state.active_shape,
                            self._starts,
                            self._ends,
                            self._score_ends,
                        )
                    self._template_engine = engine
                    self._template_keys, self._template_scales = keys, self._k_scale
                else:
                    engine = self._bind_engine(keys, self._k_scale)
                self._engines[key] = self._engine = engine
                self._prepare_layer_inputs(keys, warmup=True)
                self._engine.run()
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

        def bind_stage(stage):
            args = []
            for arg in stage.arguments.args:
                if isinstance(arg, torch.Tensor):
                    arg = rebind(arg, self._template_keys, keys)
                    arg = rebind(arg, self._template_scales, scales)
                args.append(arg)
            return replace(
                stage,
                arguments=testing.JitArguments(*args, **stage.arguments.kwargs),
            )

        engine = copy.copy(self._template_engine)
        # The copy starts from the stages of max_chunk_size rows, whatever rows the template is bound to now.
        capacity_stages = engine.__dict__.pop("_row_stages", {}).get(
            self._capacity.max_chunk_size, engine.stages
        )
        engine.__dict__.pop("_bound_rows", None)
        engine.stages = [(name, bind_stage(stage)) for name, stage in capacity_stages]
        if self._candidate_capacity:
            if engine.sample_bounds is not None:
                engine.sample_bounds = tuple(
                    bind_stage(stage) for stage in engine.sample_bounds
                )
            engine._candidate_inputs = dict(
                engine._candidate_inputs, k=keys, k_scale=scales
            )

        return engine

    def clear_pools(self):
        """Synchronize and release pool bindings and their scratch workspace.

        Discard caller-owned graphs using these pools before this call and
        recapture them before reuse. Pool bindings retain their tensors until
        cleared or wrapper destruction.
        """
        _drain_deferred_metadata_modules()
        with torch.cuda.device(self._state.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("clear_pools must run outside CUDA graph capture")
            torch.cuda.synchronize(self._state.device)
            self._pool_bindings.clear()
            self._scale_owners.clear()
            self._k_scale = None
            self._engines.clear()
            self._metadata_args.clear()
            self._engine = None
            self._keys = self._scales = None
            self._template_engine = self._template_keys = self._template_scales = None
            self._cache_protected_ranges()

    def _bind_supplied_rows(self, engine):
        """Launch K1-K3 for the Q rows this step supplies, not for max_chunk_size.

        The stages are prepared for max_chunk_size rows. A step that supplies fewer
        (one TP rank's share of a chunk, the last chunk of a prompt) would still
        sample, load and select for every row slot. K1 and K2 schedule the query
        blocks of the rows their output argument has and K3 launches one CTA per
        output row, so binding a prefix view of the output there is the whole
        change: storage, layouts and kernels stay those of max_chunk_size, and
        rows past the view are neither written nor read (K4 takes its tasks from
        the queue K3 fills). Engines that publish or consume candidate blocks keep
        the full launch: their block stage reads every row.
        """
        capacity, rows = self._capacity.max_chunk_size, self._step_rows
        bound = engine.__dict__.get("_bound_rows", capacity)
        engine._bound_rows = rows
        if self._rows_bindable is None:
            self._rows_bindable = not (
                self._candidate_capacity
                or getattr(engine, "block_scores", None) is not None
                or [name for name, _ in engine.stages] != ["K1", "K2", "K3", "K4"]
            )
        if not self._rows_bindable or bound == rows:
            return
        from dataclasses import replace
        from cutlass import testing

        cache = engine.__dict__.setdefault("_row_stages", {})
        if not cache:
            cache[capacity] = engine.stages
        stages = cache.get(rows)
        if stages is None:
            output = engine.output
            view = output[:rows]

            def bind_stage(stage):
                args = tuple(
                    view if arg is output else arg for arg in stage.arguments.args
                )
                return replace(
                    stage,
                    arguments=testing.JitArguments(*args, **stage.arguments.kwargs),
                )

            stages = [
                (name, bind_stage(stage) if name != "K4" else stage)
                for name, stage in cache[capacity]
            ]
            if len(cache) > 8:
                # A step shape is planned once and reused by every layer; keep a few and rebuild the rest.
                cache.pop(next(key for key in cache if key != capacity))
            cache[rows] = stages
        engine.stages = stages

    def _reset_rows_past(self, rows):
        """Leave the rows no stage is launched for as a full launch leaves them.

        A full launch writes -1 selections and zero candidate counts for the rows
        past the live prefix. A step that supplies fewer rows than max_chunk_size
        does not visit the rows past its own, so they are reset here when the step
        is planned. Every such plan resets them: a CUDA graph that holds the plan
        of a longer step can be replayed at any time, so the host cannot know
        which of these rows were written since the last reset.
        """
        if rows >= self._capacity.max_chunk_size or self._rows_bindable is False:
            return
        self._selected[rows:].fill_(-1)
        seen = set()
        for engine in (
            *self._engines.values(),
            getattr(self, "_template_engine", None),
        ):
            counts = getattr(engine, "counts", None)
            if counts is not None and counts.data_ptr() not in seen:
                seen.add(counts.data_ptr())
                counts[rows:].zero_()

    def _prepare_layer_inputs(self, keys, key=None, *, warmup=False):
        from ._topk_runtime import _current_stream

        if key is None:
            key = self._pool_key(keys)
        self._engine = engine = self._engines[key]
        if getattr(engine, "_bound_rows", None) != self._step_rows:
            self._bind_supplied_rows(engine)
        supplied = self._candidate_supplied and not warmup
        if self._candidate_capacity and self._engine.sample_bounds is not None:
            self._engine.stages[0] = (
                "K1",
                self._engine.sample_bounds[int(supplied)],
            )
            return
        args = self._metadata_args[key]
        if self._candidate_capacity:
            import cutlass

            args = (*args, cutlass.Int32(supplied))
        self._metadata_compiled(*args, _current_stream(self._state.device))

    def _prepare_metadata(self, keys):
        from cutlass.cute.runtime import from_dlpack
        from .kernels.selective_prefill_metadata import SelectivePrefillMetadata
        from ._topk_runtime import _compile_for_device, _current_stream

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
        metadata_class = SelectivePrefillMetadata
        if self._candidate_capacity:
            from .kernels.selective_topk.prefill.sparse.bounds import (
                SparseSelectivePrefillMetadata,
            )

            metadata_class = SparseSelectivePrefillMetadata
            tensors = (*tensors, self._candidate_ends, self._candidate_lens)
        args = tuple(from_dlpack(tensor, assumed_align=4) for tensor in tensors)
        kernel = metadata_class(
            self._capacity.num_heads,
            self._qwen,
            self._fast,
            self._qkv_dtype in ("mxfp4", "mxfp8"),
            self._metadata_flags.numel(),
            top_k=self.top_k,
        )
        self._metadata_args[self._pool_key(keys)] = args
        if self._candidate_capacity:
            import cutlass

            args = (*args, cutlass.Int32(1))
        if self._metadata_compiled is None:
            self._metadata_compiled = _bind_metadata_module_lifetime(
                _compile_for_device(
                    state.device,
                    kernel,
                    *args,
                    _current_stream(state.device),
                )
            )

    def run(
        self,
        q,
        weights,
        kv_fused,
        *,
        k_scale,
        out=None,
        q_scale=None,
        candidate_blocks=None,
    ):
        copies = self._state._payload_copies(q, weights)
        direct_candidates = None
        key = self._pool_key(kv_fused, k_scale)
        if out is not None and isinstance(out, torch.Tensor):
            from ._topk_runtime import _byte_range, _overlap

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
                from ._topk_runtime import _byte_range, _overlap

                if isinstance(out, torch.Tensor) and _overlap(
                    _byte_range(out), _byte_range(q_scale)
                ):
                    raise ValueError("out must not overlap q_scale")
            self._state.q_scale_input = q_scale
        if self._candidate_capacity:
            if (
                not isinstance(candidate_blocks, torch.Tensor)
                or candidate_blocks.device != self._state.device
                or candidate_blocks.dtype != torch.int32
                or candidate_blocks.shape != (q.shape[0], self._candidate_capacity)
                or not candidate_blocks.is_contiguous()
            ):
                raise ValueError(
                    "candidate_blocks must be contiguous CUDA int32 [Q, 2048]"
                )
            if out is not None:
                from ._topk_runtime import _byte_range, _overlap

                if _overlap(_byte_range(out), _byte_range(candidate_blocks)):
                    raise ValueError("out must not overlap candidate_blocks")
            if (
                key in self._engines
                and getattr(self._engines[key], "direct_candidates", False)
                and candidate_blocks.shape == self._candidate_input.shape
                and candidate_blocks.data_ptr() % 16 == 0
            ):
                direct_candidates = candidate_blocks
            elif candidate_blocks.data_ptr() != self._candidate_input.data_ptr():
                self._candidate_input[: q.shape[0]].copy_(candidate_blocks)
            self._candidate_supplied = True
        elif candidate_blocks is not None:
            raise ValueError("candidate_blocks requires candidate_capacity=2048")
        return self._run_layer(
            q,
            weights,
            kv_fused,
            copies,
            key,
            k_scale=k_scale,
            out=out,
            candidate_blocks=direct_candidates,
        )

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def candidate_blocks(self):
        if (
            not self._publish_candidate_blocks
            or self._engine is None
            or not self._step_prepared
        ):
            raise RuntimeError("plan and prepare a candidate publisher pool first")
        return self._engine.candidate_blocks[: self._supplied_rows]

    @property
    def candidate_valid_lens(self):
        if (
            not self._publish_candidate_blocks
            or self._engine is None
            or not self._step_prepared
        ):
            raise RuntimeError("plan and prepare a candidate publisher pool first")
        return self._engine.candidate_valid_lens[: self._supplied_rows]

    @property
    def math_mode(self) -> str:
        return self._math_mode

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        *,
        num_query_rows=None,
        next_n=None,
        host_length_hints=None,
    ):
        """Refresh prefixes, map and schedules once for all serialized layers.

        num_query_rows is the supplied Q storage extent, not a device value.
        It defaults to max_chunk_size. Each layer must supply that many rows,
        and K1-K3 are launched for that many (a captured run belongs to it);
        the final device query prefix may be smaller, leaving inactive -1 rows.
        Status is cleared here and survives every layer until the next step.
        """
        _drain_deferred_metadata_modules()
        self._step_prepared = False
        if num_query_rows is None:
            num_query_rows = self._capacity.max_chunk_size
        from ._topk_host_hints import _validate_host_length_hints

        _validate_host_length_hints(
            host_length_hints, qo_indptr.numel() - 1, self._capacity, num_query_rows
        )
        with torch.cuda.device(self._state.device):
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and (
                self._layout_needs_engine() or not self._state.is_prepared
            ):
                raise RuntimeError(
                    "prepare_pool or one eager run must precede capturing plan"
                )
            self._state._host_hint_validation.stage(host_length_hints)
            self._state.stage_step(
                qo_indptr,
                kv_indptr,
                num_query_rows=num_query_rows,
                next_n=next_n,
            )
            self._check_caller_capture(capturing, "plan input copies")
            self._repair_error.zero_()
            self._check_caller_capture(capturing, "plan repair-status reset")
            # Before the first pool is prepared, planning only stages inputs.
            # Pool preparation validates this staged step before any scoring.
            if self._state.is_prepared:
                self._state.refresh()
                self._check_caller_capture(capturing, "plan metadata refresh")
                self._refresh_schedules()
                self._check_caller_capture(capturing, "plan schedule refresh")
            # A step without rows keeps the launch of max_chunk_size.
            self._step_rows = num_query_rows or self._capacity.max_chunk_size
            self._reset_rows_past(self._step_rows)
            self._check_caller_capture(capturing, "plan row reset")
        self._supplied_rows = num_query_rows
        self._step_prepared = True

    def _check_caller_capture(self, capturing, step):
        """Name the first invalid host boundary without synchronizing the GPU."""
        if capturing:
            from ._capture_status import _check_capture_status

            _check_capture_status(torch.cuda.current_stream(self._state.device), step)

    def _layout_needs_engine(self) -> bool:
        """Whether scoring this layout still needs its first engine build."""
        return (
            self._engine is None
            and self._state.num_keys > 0
            and (
                self._capacity.max_context_length // self._capacity.compression_ratio
                > 0
            )
        )

    @property
    def error(self) -> torch.Tensor:
        return self._state.error

    @property
    def repair_error(self) -> torch.Tensor:
        return self._repair_error

    @property
    def active_shape(self) -> torch.Tensor:
        return self._state.active_shape

    def _run_layer(
        self,
        q,
        weights,
        kv_fused,
        copies,
        key,
        *,
        k_scale,
        out=None,
        candidate_blocks=None,
    ):
        """Score a layer; consume/copy the shared output before the next layer."""
        from bisect import bisect_left
        from ._topk_runtime import _byte_range, _overlap

        # Payload metadata checks precede allocation or staging. Output alias
        # checks below include the storage of a lazily prepared first pool.
        selected = self._selected[: q.shape[0]]
        if out is not None:
            if (
                not isinstance(out, torch.Tensor)
                or out.device != self._state.device
                or out.dtype != torch.int32
                or out.shape != (q.shape[0], self.top_k)
                or not out.is_contiguous()
            ):
                raise ValueError("out must be contiguous CUDA int32 [Q, top_k]")
            if out.data_ptr() != selected.data_ptr():
                bounds = _byte_range(out)
                # Static intervals are sorted and disjoint; only the nearest
                # predecessor can overlap. Per-call inputs add three checks.
                index = bisect_left(self._protected_ranges, (bounds[1],))
                static_overlap = index > 0 and _overlap(
                    bounds, self._protected_ranges[index - 1]
                )
                inputs = tuple(
                    _byte_range(tensor)
                    for tensor in (q, weights, kv_fused)
                    if isinstance(tensor, torch.Tensor)
                )
                if static_overlap or any(_overlap(bounds, other) for other in inputs):
                    raise ValueError(
                        "out must not overlap inputs or plan-owned storage"
                    )
        self._k_scale = k_scale
        with torch.cuda.device(self._state.device):
            capturing = torch.cuda.is_current_stream_capturing()
            if key not in self._pool_bindings:
                if capturing:
                    raise RuntimeError("prepare each K/k_scale binding before capture")
                self.prepare_pool(kv_fused)
            self._engine = self._engines[key]
            self._state.stage_payload(q, weights, copies)
            self._check_caller_capture(capturing, "layer payload copies")
            self._prepare_layer_inputs(kv_fused, key)
            if candidate_blocks is None:
                self._engine.run()
            else:
                self._engine.run(candidate_blocks)
            self._check_caller_capture(capturing, "layer execution")
            if out is not None and out.data_ptr() != selected.data_ptr():
                out.copy_(selected)
                self._check_caller_capture(capturing, "layer output copy")
        return selected if out is None else out
