# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Capacity-planned prefill TopK on continuous, request-concatenated K."""

import torch

from .selective_capacity import (
    _CapacitySelectiveTopKPlan,
    _CapacityStepInputs,
    _SelectiveCapacity,
)


class _ContinuousSelectiveTopKPlan(_CapacitySelectiveTopKPlan):
    def __init__(
        self,
        *,
        max_context_length,
        max_chunk_size,
        max_batch_size,
        device,
        num_heads=8,
        head_dim=128,
        top_k=512,
        compression_ratio=1,
        math_mode="relu_strict",
        num_keys=None,
    ):
        from ...attn_scores._math_policy import _validate_math_mode
        from .selective_logits import (
            _validate_selective_device,
            _validate_selective_top_k,
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
        _validate_selective_top_k(top_k)
        from .selective_logits import _SAMPLE_CAPACITY

        if capacity.max_batch_size * _SAMPLE_CAPACITY >= 2**31:
            raise ValueError("sample virtual prefixes must fit int32")
        _validate_selective_device(torch.device(device))
        if num_keys is None:
            num_keys = max_batch_size * (max_context_length // compression_ratio)
        if type(num_keys) is not int or not 0 < num_keys < 2**31:
            raise ValueError("num_keys must be a positive int32 storage capacity")
        self._capacity = capacity
        self._top_k = top_k
        self._math_mode = _validate_math_mode(math_mode)
        self._num_keys = num_keys
        self._state = _CapacityStepInputs(
            capacity, device=device, num_pages=num_keys, continuous=True
        )
        self._state.prepare_sampling(top_k)
        self._selected = torch.full(
            (max_chunk_size, top_k), -1, device=self._state.device, dtype=torch.int32
        )
        self._repair_error = torch.zeros(
            1, device=self._state.device, dtype=torch.int32
        )
        self._pool_bindings = {}
        self._eager_skip_sampling = False
        self._engine = None
        self._step_prepared = False
        self._k_scale = None
        self._protected_ranges = ()
        self._cache_protected_ranges()

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        *,
        num_query_rows=None,
        next_n=None,
        host_length_hints=None,
    ):
        return super().plan(
            qo_indptr,
            kv_indptr,
            None,
            num_query_rows=num_query_rows,
            next_n=next_n,
            host_length_hints=host_length_hints,
        )

    def _pool_key(self, keys):
        scale = self._k_scale
        fp4 = getattr(self, "_qkv_dtype", "fp8") == "mxfp4"
        width = self._capacity.head_dim // (2 if fp4 else 1)
        if (
            not isinstance(keys, torch.Tensor)
            or keys.shape != (self._num_keys, width)
            or keys.dtype != (torch.uint8 if fp4 else torch.float8_e4m3fn)
            or keys.device != self._state.device
            or not keys.is_contiguous()
            or keys.data_ptr() % 16
        ):
            raise ValueError(
                "K must be aligned contiguous CUDA storage [num_keys, D] (D/2 for MXFP4)"
            )
        if (
            not isinstance(scale, torch.Tensor)
            or scale.device != keys.device
            or not scale.is_contiguous()
            or scale.data_ptr() % 16
            or (
                not fp4
                and (scale.shape != (self._num_keys,) or scale.dtype != torch.float32)
            )
            or (
                fp4
                and (scale.shape != (self._num_keys, 4) or scale.dtype != torch.uint8)
            )
        ):
            raise ValueError(
                "k_scale must be contiguous FP32[N] or MXFP4 UE8M0 uint8[N,4]"
            )
        return (
            keys.data_ptr(),
            scale.data_ptr(),
            self._layout(keys),
            self._layout(scale),
        )

    def _bind_scales(self):
        if self._engine is not None:
            from dataclasses import replace

            for name in (
                "_sample_prepared",
                "_candidate_prepared",
                "_repair_prepared",
                "_prepared",
            ):
                prepared = getattr(self._engine, name)
                if prepared.k_scale is not self._k_scale:
                    rebound = replace(prepared, k_scale=self._k_scale)
                    rebound._metadata_modes.update(prepared._metadata_modes)
                    setattr(self._engine, name, rebound)

    def _execute_layer(self, keys):
        self._bind_scales()
        return super()._execute_layer(keys)

    def prepare_pool(self, keys, *, k_scale=None):
        if k_scale is not None:
            self._k_scale = k_scale
        self._bind_scales()
        super().prepare_pool(keys)
        # Keep scales alive for every cached graph, independently of its DLPack
        # bindings. Scale pointer participates in the binding identity.
        self._scale_owners = getattr(self, "_scale_owners", {})
        self._scale_owners[self._pool_key(keys)] = self._k_scale
        self._cache_protected_ranges()

    def _cache_protected_ranges(self):
        from .selective_logits import _byte_range

        super()._cache_protected_ranges()
        ranges = [
            *self._protected_ranges,
            *(_byte_range(t) for t in getattr(self, "_scale_owners", {}).values()),
        ]
        merged = []
        for begin, end in sorted(ranges):
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((begin, end))
        self._protected_ranges = tuple(merged)

    def clear_pools(self):
        super().clear_pools()
        self._scale_owners = {}
        self._k_scale = None
        self._bind_scales()
        self._cache_protected_ranges()

    def run(self, q, weights, keys, *, k_scale, out=None, q_scale=None):
        from .selective_logits import _byte_range, _overlap

        if q_scale is not None:
            raise ValueError("q_scale requires backend='v2' with MXFP4")
        self._k_scale = k_scale
        self._pool_key(keys)
        if isinstance(out, torch.Tensor) and _overlap(
            _byte_range(out), _byte_range(k_scale)
        ):
            raise ValueError("out must not overlap k_scale")
        self._bind_scales()
        return super().run(q, weights, keys, out=out)
