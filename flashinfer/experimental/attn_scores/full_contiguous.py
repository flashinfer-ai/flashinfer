# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Full continuous FP8 scores using the existing dense scorer and GVRv2."""

import torch

from .selective_contiguous import _ContinuousSelectiveTopKPlan


class _DenseFullEngine:
    def __init__(self, plan, keys):
        from ...topk_varlen.kernels.gvr2_topk_host import workspace_bytes
        from .selective_logits import _prepare_fp8_mqa_selective_logits

        state, capacity = plan._state, plan._capacity
        columns, rows = plan._columns, plan._stripe_rows
        self._scores = torch.empty(
            (rows, columns), dtype=torch.float32, device=state.device
        )
        self._counts = torch.zeros((rows, 1), dtype=torch.int32, device=state.device)
        self._starts = torch.zeros(rows, dtype=torch.int32, device=state.device)
        self._empty_lengths = torch.zeros_like(self._starts)
        count = (capacity.max_chunk_size + rows - 1) // rows
        prefix_pitch = (capacity.max_batch_size + 4) // 4 * 4
        self._prefixes = torch.zeros(
            (count, prefix_pitch), dtype=torch.int32, device=state.device
        )
        self._ends = torch.zeros(
            (count, (rows + 3) // 4 * 4), dtype=torch.int32, device=state.device
        )
        self._gvr_workspace = torch.zeros(
            workspace_bytes() // 4, dtype=torch.int32, device=state.device
        )
        self.workspace = {"gvr2_workspace": self._gvr_workspace}
        self.columns, self.top_k = columns, plan.top_k
        self.stripes = []
        self._sample_fused = keys
        for index, begin in enumerate(range(0, capacity.max_chunk_size, rows)):
            end = min(begin + rows, capacity.max_chunk_size)
            q, weights = state.q[begin:end], state.weights[begin:end]
            starts, ends = self._starts[: end - begin], self._ends[index, : end - begin]
            prefix = self._prefixes[index, : capacity.max_batch_size + 1]
            prepared = _prepare_fp8_mqa_selective_logits(
                q,
                keys,
                weights,
                prefix,
                state.physical_kv_indptr,
                state.block_table,
                row_starts=starts,
                row_ends=ends,
                query_tile=2,
                num_heads=capacity.num_heads,
                head_dim=capacity.head_dim,
                math_mode=plan.math_mode,
                _capacity_bounds=True,
                k_scale=plan._k_scale,
            )
            self.stripes.append(
                (
                    begin,
                    end,
                    q,
                    weights,
                    prefix,
                    starts,
                    ends,
                    prepared,
                    self._scores[: end - begin],
                    plan._selected[begin:end],
                    self._empty_lengths[: end - begin],
                )
            )

    def refresh(self, state):
        from .selective_logits import _refresh_fp8_mqa_selective_logits

        for begin, end, _, _, prefix, starts, ends, prepared, *_ in self.stripes:
            torch.clamp(state.safe_qo_indptr, min=begin, max=end, out=prefix)
            prefix.sub_(begin)
            ends.copy_(state.row_ends[begin:end])
            _refresh_fp8_mqa_selective_logits(
                prepared,
                prefix,
                state.physical_kv_indptr,
                row_starts=starts,
                row_ends=ends,
            )

    def _precompile_score_launches(self, q, keys, weights, block_table):
        self.execute(keys, block_table, compile_only=True)

    def execute(self, keys, block_table, *, compile_only=False):
        from ...topk_varlen.topk_varlen import top_k_varlen
        from .selective_logits import _fp8_mqa_selective_logits, _SelectiveLogitsMode

        for (
            _,
            _,
            q,
            weights,
            _,
            _,
            ends,
            prepared,
            scores,
            indices,
            empty,
        ) in self.stripes:
            _fp8_mqa_selective_logits(
                q,
                keys,
                weights,
                block_table,
                prepared,
                mode=_SelectiveLogitsMode.SAMPLE,
                capacity=self.columns,
                values=self._scores,
                index_counts=self._counts,
                _compile_only=compile_only,
            )
            top_k_varlen(
                scores,
                empty if compile_only else ends,
                self.top_k,
                backend="gvr_2",
                out_indices=indices,
                workspace=self.workspace,
            )


class _ContinuousFullTopKPlan(_ContinuousSelectiveTopKPlan):
    """Reuse continuous staging and pool ownership with a bounded dense arena."""

    def __init__(self, *, max_workspace_size=None, **kwargs):
        from ...attn_scores.attn_scores import padded_seq_len

        if max_workspace_size is not None and (
            type(max_workspace_size) is not int or max_workspace_size <= 0
        ):
            raise ValueError("max_workspace_size must be a positive integer")
        super().__init__(**kwargs)
        self._columns = max(
            8192,
            self.top_k,
            padded_seq_len(
                self._capacity.max_context_length // self._capacity.compression_ratio
            ),
        )
        self._score_budget_bytes = (
            512 * 1024 * 1024 if max_workspace_size is None else max_workspace_size
        )
        minimum = 8 * self._columns
        if self._score_budget_bytes < minimum:
            raise ValueError(
                f"max_workspace_size must be at least {minimum} bytes "
                "for one full-score query pair at capacity"
            )
        self._stripe_rows = min(
            (self._capacity.max_chunk_size + 1) // 2 * 2,
            self._score_budget_bytes // (8 * self._columns) * 2,
        )
        self._full_scale_bindings = {}

    @property
    def _needs_sampling(self):
        return False

    def _build_engine(self, keys):
        self._q, self._weights = self._state.q, self._state.weights
        self._engine = _DenseFullEngine(self, keys)
        self._full_scale_bindings[self._k_scale.data_ptr()] = [
            s[7] for s in self._engine.stripes
        ]

    def _refresh_schedules(self):
        if self._engine is not None:
            self._engine.refresh(self._state)

    def _bind_scales(self):
        if self._engine is not None:
            from dataclasses import replace

            key = None if self._k_scale is None else self._k_scale.data_ptr()
            if key not in self._full_scale_bindings:
                prepared = []
                for stripe in self._engine.stripes:
                    rebound = replace(stripe[7], k_scale=self._k_scale)
                    rebound._metadata_modes.update(stripe[7]._metadata_modes)
                    prepared.append(rebound)
                self._full_scale_bindings[key] = prepared
            self._engine.stripes = [
                (*stripe[:7], prepared, *stripe[8:])
                for stripe, prepared in zip(
                    self._engine.stripes, self._full_scale_bindings[key], strict=True
                )
            ]

    def _prepared_launches(self):
        if self._engine is None:
            return ()
        return tuple((s[7], s[7]._launches.copy()) for s in self._engine.stripes)

    def _cache_protected_ranges(self):
        from .selective_logits import _byte_range

        super()._cache_protected_ranges()
        ranges = list(self._protected_ranges)
        if self._engine is not None:
            for stripe in self._engine.stripes:
                ranges.extend(
                    _byte_range(tensor)
                    for tensor in vars(stripe[7]).values()
                    if isinstance(tensor, torch.Tensor) and tensor.numel()
                )
        merged = []
        for begin, end in sorted(ranges):
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((begin, end))
        self._protected_ranges = tuple(merged)

    def _execute_layer(self, keys):
        if self._engine is None:
            self._selected.fill_(-1)
        else:
            self._engine.execute(keys, self._state.block_table)

    def clear_pools(self):
        super().clear_pools()
        self._full_scale_bindings = (
            {None: [s[7] for s in self._engine.stripes]}
            if self._engine is not None
            else {}
        )

    def run(self, q, weights, keys, *, k_scale, out=None, q_scale=None):
        self._k_scale = k_scale
        key = self._pool_key(keys)
        if torch.cuda.is_current_stream_capturing() and key not in self._pool_bindings:
            raise RuntimeError(
                "prepare_pool or one eager run must precede caller capture"
            )
        return super().run(q, weights, keys, k_scale=k_scale, out=out, q_scale=q_scale)
