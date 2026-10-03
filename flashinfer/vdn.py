"""
Copyright (c) 2026 by FlashInfer team.

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

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from numbers import Real

import torch

from .api_logging import flashinfer_api
from .prefill import BatchPrefillWithPagedKVCacheWrapper
from .trace.templates.vdn import vdn_window_attention_run_trace
from .utils import _check_workspace_buffer_alignment, get_compute_capability

_MIN_WORKSPACE_BYTES = 128 * 1024 * 1024
_MAX_INDEX = 2**31 - 1


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum or result > _MAX_INDEX:
        raise ValueError(f"{name} must be in [{minimum}, {_MAX_INDEX}]")
    return result


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _query_groups(
    seq_len: int,
    video_start: int,
    num_frames: int,
    tokens_per_frame: int,
    bounds: Sequence[tuple[int, int]],
    anchor_frames: str,
) -> list[tuple[int, int, tuple[tuple[int, int], ...]]]:
    """Disjoint contiguous query intervals and their deduplicated KV unions."""
    video_end = video_start + num_frames * tokens_per_frame
    full: tuple[tuple[int, int], ...] = ((0, seq_len),)
    globals_ = ((0, video_start), (video_end, seq_len))
    anchors = {0, num_frames - 1} if num_frames else set()
    groups: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []

    def append(start, end, kv_ranges):
        if start == end:
            return
        if not kv_ranges:
            raise ValueError("Every VDN query must have at least one visible key")
        if groups and groups[-1][1] == start and groups[-1][2] == kv_ranges:
            groups[-1] = (groups[-1][0], end, kv_ranges)
        else:
            groups.append((start, end, kv_ranges))

    def frame_range(frame):
        start = video_start + frame * tokens_per_frame
        return start, start + tokens_per_frame

    append(0, video_start, full)
    for frame, (lo, hi) in enumerate(bounds):
        start, end = frame_range(frame)
        if anchor_frames in ("rows", "both") and frame in anchors:
            ranges = full
        else:
            visible_ranges = list(globals_)
            lo, hi = max(lo, 0), min(hi + 1, num_frames)
            if lo < hi:
                visible_ranges.append(
                    (
                        video_start + lo * tokens_per_frame,
                        video_start + hi * tokens_per_frame,
                    )
                )
            if anchor_frames in ("columns", "both"):
                visible_ranges.extend(frame_range(f) for f in anchors)
            ranges = _merge_ranges(visible_ranges)
        append(start, end, ranges)
    append(video_end, seq_len, full)
    return groups


class VDNWindowAttentionWrapper:
    r"""Exact BF16 Video DeltaNet window attention on SM120.

    The video occupies consecutive, frame-major tokens. All remaining tokens are
    global: global queries attend to every key, and all queries attend to global
    keys. Video queries attend to the inclusive frame window specified in
    :meth:`plan`. Optional first/last-frame anchors add full rows or columns to
    this mask. Each query uses one joint softmax over its deduplicated KV union.

    Plans share a single-token paged KV pool, group queries with identical KV
    sets, and schedule longer KV sets first. A fused input kernel permutes Q and
    copies strided V, and a scatter restores output order.

    This wrapper supports eager inference only, with BF16 ``[T, H, 128]`` Q/K/V
    and equal numbers of query and KV heads. Create, plan, and run it on the same
    CUDA stream. Calls must be serialized; concurrent calls or concurrent reuse
    of its workspace by another wrapper are unsupported. Inputs and outputs
    must not share storage with the workspace.

    Parameters
    ----------
    float_workspace_buffer : torch.Tensor
        Caller-owned, contiguous, one-dimensional CUDA uint8 workspace of at
        least 128 MiB, aligned to 16 bytes. Its device must have capability 12.0.
        The wrapper retains this buffer and its own planning metadata until it
        is destroyed. No global plan cache is used.
    """

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        workspace = float_workspace_buffer
        if (
            not isinstance(workspace, torch.Tensor)
            or workspace.layout != torch.strided
            or workspace.device.type != "cuda"
            or workspace.dtype != torch.uint8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.numel() < _MIN_WORKSPACE_BYTES
        ):
            raise ValueError(
                "workspace must be a contiguous CUDA uint8 vector of at least 128 MiB"
            )
        _check_workspace_buffer_alignment(workspace, "float_workspace_buffer")
        if get_compute_capability(workspace.device) != (12, 0):
            raise RuntimeError(
                "VDNWindowAttentionWrapper requires an SM120 CUDA device"
            )
        self._workspace = workspace
        self._device = workspace.device
        self._stream = torch.cuda.current_stream(self._device)
        self._wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None
        self._seq_len = self._num_heads = 0
        self._video_start = self._num_frames = self._tokens_per_frame = 0
        self._window_bounds: tuple[tuple[int, int], ...] = ()
        self._sm_scale = 128**-0.5
        self._anchor_frames = "both"

    def _check_execution(self) -> None:
        if torch.cuda.current_stream(self._device) != self._stream:
            raise RuntimeError(
                "VDN plan/run must use the CUDA stream that created the wrapper"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "VDNWindowAttentionWrapper supports eager execution only"
            )

    def plan(
        self,
        seq_len: int,
        num_heads: int,
        video_start: int,
        num_frames: int,
        tokens_per_frame: int,
        window_bounds: Sequence[tuple[int, int]],
        *,
        sm_scale: float | None = None,
        anchor_frames: str = "both",
    ) -> None:
        r"""Plan the exact window mask without reading device tensor values.

        ``window_bounds[f]`` is an inclusive ``(lo, hi)`` pair for frame ``f``.
        Bounds may extend beyond the clip and are clipped to its frame range.
        ``anchor_frames`` is ``"none"``, ``"rows"``, ``"columns"``, or ``"both"``.
        Anchor columns are included only once when already in a window.
        Zero video frames are supported when the sequence contains global tokens.

        All sequence indices and the total paged-index count must fit int32.
        ``sm_scale`` defaults to ``1 / sqrt(128)`` and must be positive, with
        its base-2 exponential scale representable as a normal finite FP32
        value. Reuse a plan for new Q/K/V
        values with the same geometry; call this method again to change geometry.
        A device-side planning failure invalidates the previous plan because
        the caller-owned workspace may already have been modified.
        """
        seq_len = _integer(seq_len, "seq_len", 1)
        num_heads = _integer(num_heads, "num_heads", 1)
        video_start = _integer(video_start, "video_start")
        num_frames = _integer(num_frames, "num_frames")
        tokens_per_frame = _integer(tokens_per_frame, "tokens_per_frame", 1)
        if num_heads * 128 > _MAX_INDEX:
            raise ValueError("num_heads * 128 must fit int32")
        if video_start + num_frames * tokens_per_frame > seq_len:
            raise ValueError("video tokens must fit inside seq_len")
        if anchor_frames not in ("none", "rows", "columns", "both"):
            raise ValueError("anchor_frames must be none, rows, columns, or both")
        if not isinstance(window_bounds, Sequence) or len(window_bounds) != num_frames:
            raise ValueError("window_bounds must contain one (lo, hi) pair per frame")
        bounds = []
        for bound in window_bounds:
            if not isinstance(bound, Sequence) or len(bound) != 2:
                raise ValueError("each window bound must be an inclusive (lo, hi) pair")
            if any(isinstance(value, bool) for value in bound):
                raise ValueError("window bounds must contain integers")
            try:
                lo, hi = (operator.index(value) for value in bound)
            except TypeError as exc:
                raise ValueError("window bounds must contain integers") from exc
            if lo > hi:
                raise ValueError("window bounds must satisfy lo <= hi")
            bounds.append((lo, hi))
        if sm_scale is None:
            sm_scale = 128**-0.5
        if (
            not isinstance(sm_scale, Real)
            or isinstance(sm_scale, bool)
            or not math.isfinite(sm_scale)
            or not torch.finfo(torch.float32).tiny
            <= sm_scale
            <= torch.finfo(torch.float32).max / math.log2(math.e)
        ):
            raise ValueError(
                "sm_scale must be positive and finite within the supported FP32 range"
            )

        by_keys: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
        for start, end, ranges in _query_groups(
            seq_len, video_start, num_frames, tokens_per_frame, bounds, anchor_frames
        ):
            by_keys.setdefault(ranges, []).append((start, end))
        groups = sorted(
            by_keys.items(), key=lambda item: -sum(hi - lo for lo, hi in item[0])
        )
        q_indptr, kv_indptr = [0], [0]
        for ranges, query_ranges in groups:
            q_indptr.append(q_indptr[-1] + sum(hi - lo for lo, hi in query_ranges))
            kv_indptr.append(kv_indptr[-1] + sum(hi - lo for lo, hi in ranges))
        if kv_indptr[-1] > _MAX_INDEX:
            raise ValueError("total paged KV index count must fit int32")

        with torch.cuda.device(self._device):
            self._check_execution()
            indices: list[torch.Tensor] = []
            query_indices: list[torch.Tensor] = []
            for ranges, query_ranges in groups:
                indices.extend(
                    torch.arange(lo, hi, dtype=torch.int32, device=self._device)
                    for lo, hi in ranges
                )
                query_indices.extend(
                    torch.arange(lo, hi, dtype=torch.int32, device=self._device)
                    for lo, hi in query_ranges
                )
            query_order = torch.cat(query_indices)
            qo_indptr = torch.tensor(q_indptr, dtype=torch.int32, device=self._device)
            paged_indptr = torch.tensor(
                kv_indptr, dtype=torch.int32, device=self._device
            )
            paged_indices = torch.cat(indices)
            last_page_len = torch.ones(
                len(groups), dtype=torch.int32, device=self._device
            )
            self._wrapper = None
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspace, kv_layout="NHD", backend="fa2"
            )
            wrapper.plan(
                qo_indptr,
                paged_indptr,
                paged_indices,
                last_page_len,
                num_heads,
                num_heads,
                128,
                page_size=1,
                causal=False,
                sm_scale=float(sm_scale),
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
                o_data_type=torch.bfloat16,
            )
        self._wrapper = wrapper
        self._query_order = query_order
        self._seq_len, self._num_heads = seq_len, num_heads
        self._video_start, self._num_frames = video_start, num_frames
        self._tokens_per_frame = tokens_per_frame
        self._window_bounds = tuple(bounds)
        self._sm_scale, self._anchor_frames = float(sm_scale), anchor_frames

    def _check_tensor(self, tensor: torch.Tensor, name: str) -> None:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.shape != (self._seq_len, self._num_heads, 128)
            or tensor.dtype != torch.bfloat16
            or tensor.device != self._device
        ):
            raise ValueError(
                f"{name} must be BF16 [seq_len, num_heads, 128] on {self._device}"
            )
        if tensor.requires_grad:
            raise ValueError("VDNWindowAttentionWrapper supports inference only")
        if (
            tensor.untyped_storage().data_ptr()
            == self._workspace.untyped_storage().data_ptr()
        ):
            raise ValueError(f"{name} must not share storage with the workspace")

    @flashinfer_api(trace=vdn_window_attention_run_trace)
    def run(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Run attention with the current plan and new BF16 Q/K/V values.

        Inputs have shape ``[seq_len, num_heads, 128]`` and may be strided.
        ``out``, when supplied, must have the same shape, dtype and device and
        be contiguous. It may alias an input: all input reads precede the final
        output scatter. Otherwise a contiguous output is allocated.
        """
        if self._wrapper is None:
            raise RuntimeError("plan() must be called before run()")
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            self._check_tensor(tensor, name)
        if out is not None:
            self._check_tensor(out, "out")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")

        from .triton.vdn import _pack_query_value, _scatter_output

        with torch.cuda.device(self._device):
            self._check_execution()
            key = key.contiguous()
            if key.data_ptr() % 16:
                key = key.clone()
            query = query if query.stride(2) == 1 else query.contiguous()
            value = value if value.stride(2) == 1 else value.contiguous()
            packed_query = torch.empty(
                query.shape, dtype=query.dtype, device=self._device
            )
            copy_value = not value.is_contiguous() or value.data_ptr() % 16 != 0
            packed_value = torch.empty_like(packed_query) if copy_value else value
            grid = ((query.numel() + 1023) // 1024,)
            _pack_query_value[grid](
                query,
                value,
                self._query_order,
                packed_query,
                packed_value,
                query.numel(),
                self._num_heads * 128,
                query.stride(0),
                query.stride(1),
                value.stride(0),
                value.stride(1),
                copy_value,
                1024,
            )
            attended = self._wrapper.run(
                packed_query, (key.unsqueeze(1), packed_value.unsqueeze(1))
            )
            if out is None:
                out = torch.empty_like(packed_query)
            _scatter_output[grid](
                attended,
                self._query_order,
                out,
                query.numel(),
                self._num_heads * 128,
                1024,
            )
        return out
