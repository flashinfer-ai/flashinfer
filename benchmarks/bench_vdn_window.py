# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Baseline decomposition adapted from OpenVDN/vdn-minimax-h3,
# revision 2f740c9291431d89d4f2330743b093fac4390d09 (Apache-2.0).
# Copyright 2026 the VDN authors.
"""SM120 BF16 VDN window softmax: whole-operator timing, no model required.

PyTorch >= 2.13 supplies the SM120 baseline's varlen_attn. Both arms include
copies, gathers, attention and output scatter; cold planning is reported
separately. This does not benchmark VDN's linear branch or video generation.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.varlen import varlen_attn

import flashinfer
from flashinfer.vdn import VDNWindowAttentionWrapper


@dataclass(frozen=True)
class Layout:
    seq_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int

    @property
    def video_end(self):
        return self.video_start + self.num_frames * self.tokens_per_frame


class TorchWindowBaseline:
    __slots__ = (
        "dense_q",
        "win_q",
        "kv_gather",
        "cu_q",
        "cu_k",
        "max_q",
        "max_k",
        "has_windows",
    )

    def __init__(self, layout, bounds, anchor_frames, device):
        S = layout.seq_len
        F, TPF = layout.num_frames, layout.tokens_per_frame
        vs, ve = layout.video_start, layout.video_end
        anchor_set = (
            {0, F - 1} if anchor_frames in ("columns", "rows", "both") else set()
        )
        dense_row_frames = anchor_set if anchor_frames in ("rows", "both") else set()
        dense_col_frames = anchor_set if anchor_frames in ("columns", "both") else set()

        def frame_rows(f):
            return (vs + f * TPF, vs + (f + 1) * TPF)

        global_ranges = [r for r in ((0, vs), (ve, S)) if r[0] < r[1]]

        def merge(ranges):
            out = []
            for a, b in sorted(ranges):
                if out and out[-1][1] >= a:
                    out[-1] = (out[-1][0], max(out[-1][1], b))
                else:
                    out.append((a, b))
            return out

        def cat_ranges(ranges):
            return torch.cat([torch.arange(a, b, device=device) for a, b in ranges])

        # dense-q rows: globals + anchor-row frames
        dense_ranges = merge(
            global_ranges + [frame_rows(f) for f in sorted(dense_row_frames)]
        )
        self.dense_q = (
            cat_ranges(dense_ranges)
            if dense_ranges
            else torch.empty(0, dtype=torch.long, device=device)
        )

        # window groups: consecutive frames sharing identical bounds (== chunks)
        groups = []
        for f in range(F):
            if f in dense_row_frames:
                continue
            if (
                groups
                and bounds[groups[-1][-1]] == bounds[f]
                and groups[-1][-1] == f - 1
            ):
                groups[-1].append(f)
            else:
                groups.append([f])

        q_idx, kv_idx, q_lens, k_lens = [], [], [], []
        for frames in groups:
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(
                set(range(max(lo, 0), min(hi + 1, F))) | dense_col_frames
            )
            q_r = merge([frame_rows(f) for f in frames])
            kv_r = merge(global_ranges + [frame_rows(f) for f in kv_frames])
            qi, ki = cat_ranges(q_r), cat_ranges(kv_r)
            q_idx.append(qi)
            kv_idx.append(ki)
            q_lens.append(len(qi))
            k_lens.append(len(ki))

        self.has_windows = bool(groups)
        if self.has_windows:
            self.win_q = torch.cat(q_idx)
            self.kv_gather = torch.cat(kv_idx)
            zero = torch.zeros(1, dtype=torch.long)
            self.cu_q = torch.cat([zero, torch.tensor(q_lens).cumsum(0)]).to(
                device, torch.int32
            )
            self.cu_k = torch.cat([zero, torch.tensor(k_lens).cumsum(0)]).to(
                device, torch.int32
            )
            self.max_q, self.max_k = max(q_lens), max(k_lens)
        else:
            self.win_q = torch.empty(0, dtype=torch.long, device=device)

        order = torch.cat([self.dense_q, self.win_q])
        if len(order) != S:
            raise ValueError(f"decomposition covers {len(order)} of {S} rows")

    def run(self, query, key, value, scale):
        key, value = key.contiguous(), value.contiguous()
        output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        if len(self.dense_q):
            with sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                ]
            ):
                output[self.dense_q] = torch.nn.functional.scaled_dot_product_attention(
                    query[self.dense_q].transpose(0, 1).unsqueeze(0),
                    key.transpose(0, 1).unsqueeze(0),
                    value.transpose(0, 1).unsqueeze(0),
                    scale=scale,
                )[0].transpose(0, 1)
        if self.has_windows:
            output[self.win_q] = varlen_attn(
                query[self.win_q],
                key[self.kv_gather],
                value[self.kv_gather],
                self.cu_q,
                self.cu_k,
                self.max_q,
                self.max_k,
                scale=scale,
            )
        return output


def error_stats(actual, expected):
    difference = actual.float() - expected.float()
    return {
        "relative_l2": (
            difference.norm() / expected.float().norm().clamp_min(1e-12)
        ).item(),
        "max_absolute": difference.abs().max().item(),
        "finite": bool(actual.isfinite().all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=107)
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--tokens-per-frame", type=int, default=510)
    parser.add_argument("--prefix", type=int, default=3623)
    parser.add_argument(
        "--projection-width", type=int, choices=(128, 386, 514), default=514
    )
    parser.add_argument("--strided-qk", action="store_true")
    parser.add_argument(
        "--anchor-frames", choices=("none", "rows", "columns", "both"), default="both"
    )
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--window-radius", type=int, default=1)
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="Flush at least 256 MiB before each timed call",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.frames <= 0
        or args.heads <= 0
        or args.tokens_per_frame <= 0
        or args.prefix < 0
    ):
        parser.error(
            "frames, heads and tokens-per-frame must be positive; prefix must be nonnegative"
        )
    if args.repeats < 3 or args.warmups < 1:
        parser.error("at least three repeats and one warmup are required")
    if args.chunk_size < 0 or args.window_radius < 0:
        parser.error("chunk-size and window-radius must be nonnegative")
    if args.strided_qk and args.projection_width == 128:
        parser.error("strided Q/K needs a projection width of 386 or 514")
    if torch.cuda.get_device_capability() != (12, 0):
        parser.error("this benchmark targets SM120")
    torch.manual_seed(20260927)
    layout = Layout(
        args.prefix + args.frames * args.tokens_per_frame,
        args.prefix,
        args.frames,
        args.tokens_per_frame,
    )
    radius, chunk = args.window_radius, args.chunk_size
    bounds = tuple(
        ((f // chunk - radius) * chunk, (f // chunk + radius + 1) * chunk - 1)
        if chunk
        else (f - radius, f + radius)
        for f in range(args.frames)
    )
    shape = (layout.seq_len, args.heads, 128)
    query = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    key = torch.randn_like(query)
    payload = torch.randn(
        layout.seq_len,
        args.heads,
        args.projection_width,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = payload if args.projection_width == 128 else payload[:, :, 256:384]
    if args.strided_qk:
        query, key = payload[:, :, :128], payload[:, :, 128:256]
    scale = 128**-0.5
    torch.cuda.synchronize()
    started = time.perf_counter()
    baseline = TorchWindowBaseline(layout, bounds, args.anchor_frames, query.device)
    torch.cuda.synchronize()
    baseline_plan_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(
        layout.seq_len,
        args.heads,
        args.prefix,
        args.frames,
        args.tokens_per_frame,
        bounds,
        sm_scale=scale,
        anchor_frames=args.anchor_frames,
    )
    torch.cuda.synchronize()
    fi_plan_ms = (time.perf_counter() - started) * 1000
    calls = {
        "torch_varlen": lambda: baseline.run(query, key, value, scale),
        "flashinfer": lambda: wrapper.run(query, key, value),
    }
    expected = calls["torch_varlen"]()
    stats = error_stats(calls["flashinfer"](), expected)
    assert (
        stats["finite"]
        and stats["relative_l2"] < 0.005
        and stats["max_absolute"] < 0.02
    ), stats
    for _ in range(args.warmups):
        for function in calls.values():
            function()
    torch.cuda.synchronize()
    samples = {name: {"host_ms": [], "cuda_event_ms": []} for name in calls}
    flush_bytes = max(
        256 * 1024**2,
        2 * getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0),
    )
    flush = (
        torch.empty(flush_bytes, dtype=torch.uint8, device=query.device)
        if args.cold_l2
        else None
    )
    generator = random.Random(71)
    for _ in range(args.repeats):
        order = list(calls)
        generator.shuffle(order)
        for name in order:
            start_event, end_event = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if flush is not None:
                flush.zero_()
            torch.cuda.synchronize()
            started = time.perf_counter()
            start_event.record()
            output = calls[name]()
            end_event.record()
            torch.cuda.synchronize()
            samples[name]["host_ms"].append((time.perf_counter() - started) * 1000)
            samples[name]["cuda_event_ms"].append(start_event.elapsed_time(end_event))
            del output
    for name, function in calls.items():
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        result = function()
        torch.cuda.synchronize()
        samples[name]["peak_run_extra_bytes"] = (
            torch.cuda.max_memory_allocated() - allocated
        )
        del result
        for timer in ("host_ms", "cuda_event_ms"):
            values = samples[name][timer]
            samples[name][timer + "_summary"] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "stdev": statistics.stdev(values),
            }
    source = Path(flashinfer.__file__).parent
    report = {
        "scope": "BF16 VDN window softmax only; not full hybrid attention or video generation",
        "parameters": vars(args) | {"output": str(args.output)},
        "shape": shape,
        "strides": {
            name: list(t.stride())
            for name, t in (("query", query), ("key", key), ("value", value))
        },
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source / "vdn.py", source / "triton/vdn.py")
            if p.exists()
        },
        "seed": 20260927,
        "random_order_seed": 71,
        "error": stats,
        "plan_ms": {"torch_varlen": baseline_plan_ms, "flashinfer": fi_plan_ms},
        "workspace_bytes": workspace.numel(),
        "l2_flush_bytes": flush_bytes if args.cold_l2 else 0,
        "timing": "warmed eager whole operator; randomized alternating arms; includes copies, allocation and scatter; excludes plan and optional L2 flush; CUDA events plus host through synchronization",
        "methods": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
