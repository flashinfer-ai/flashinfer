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
"""Benchmark MiniMax-H3 BF16 pre-attention on SM100a / SM103a.

The default suite measures the 5-second P8 production center. ``--suite final``
measures all 24 duration/parallelism centers. P2/P4/P8 are the performance
promotion range; P1 is included for correctness and regression visibility but
should retain the segmented fallback. JIT compilation, tensor creation, and
output allocation are outside the timing boundary.

``--operand-layout contract`` (default) uses contiguous ``[9, 5376]`` AdaLN
tables, int64 indices and a per-token RoPE cache with identity positions.
``--operand-layout engine`` uses the engine's own operands: AdaLN tables as
column chunks of a ``[rows, 6 * 5376]`` modulation projection (row stride
``6 * 5376``; ``--adaln-rows``, default 6), int64 ``combined_indices`` and a
shared ``(cos_sin_cache [S, 96], positions [M])`` pair.  In that mode the
``copies ms`` column times the host-side copies a route needed before this
contract (``.contiguous()`` of both tables, the int64 -> int32 cast and the
``index_select`` RoPE gather) so the saving is visible without a second kernel.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from flashinfer.diffusion_ops import minimax_h3_bf16_pre_attention
from flashinfer.testing.utils import bench_gpu_time


HIDDEN = 5376
NUM_HEADS = 56
HEAD_DIM = 128
QKV_KINDS = 3
QKV_WIDTH = NUM_HEADS * QKV_KINDS * HEAD_DIM
ROPE_DIM = 96
ADALN_ROWS = 9
ENGINE_TABLE_CHUNKS = 6
EPS = 1.0e-5

CENTER_SHAPES = [
    (4, 33472, 1),
    (4, 16736, 2),
    (4, 8368, 4),
    (4, 4184, 8),
    (5, 38592, 1),
    (5, 19296, 2),
    (5, 9648, 4),
    (5, 4824, 8),
    (6, 48768, 1),
    (6, 24384, 2),
    (6, 12192, 4),
    (6, 6096, 8),
    (8, 58944, 1),
    (8, 29472, 2),
    (8, 14736, 4),
    (8, 7368, 8),
    (10, 74240, 1),
    (10, 37120, 2),
    (10, 18560, 4),
    (10, 9280, 8),
    (15, 109952, 1),
    (15, 54976, 2),
    (15, 27488, 4),
    (15, 13744, 8),
]
ACTIVE_SHAPES = [(5, 4824, 8)]


def _make_rope_cache(rows: int, device: torch.device):
    positions = torch.arange(rows, dtype=torch.float32, device=device)
    axes = (
        torch.div(positions, 4096, rounding_mode="floor"),
        torch.div(positions, 64, rounding_mode="floor").remainder(64),
        positions.remainder(64),
    )
    inv_freq = torch.pow(
        torch.tensor(10000.0, dtype=torch.float32, device=device),
        -torch.arange(16, dtype=torch.float32, device=device) / 16.0,
    )
    phase = torch.cat([axis[:, None] * inv_freq[None, :] for axis in axes], dim=-1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(torch.bfloat16).contiguous()


def _make_model(device: torch.device, *, layout: str, adaln_rows: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(4532)

    def normal(shape, std):
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return out.normal_(0.0, std, generator=generator)

    def uniform(shape, low, high):
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return out.uniform_(low, high, generator=generator)

    if layout == "contract":
        adaln_scale = uniform((adaln_rows, HIDDEN), -0.05, 0.05)
        adaln_shift = uniform((adaln_rows, HIDDEN), -0.05, 0.05)
    else:
        # Column chunks 0 (shift) and 1 (scale) of the modulation projection.
        proj = uniform((adaln_rows, ENGINE_TABLE_CHUNKS * HIDDEN), -0.05, 0.05)
        adaln_shift = proj[:, :HIDDEN]
        adaln_scale = proj[:, HIDDEN : 2 * HIDDEN]
        assert adaln_scale.stride() == (ENGINE_TABLE_CHUNKS * HIDDEN, 1)

    return {
        "x_norm_weight": uniform((HIDDEN,), 0.9, 1.1),
        "adaln_scale": adaln_scale,
        "adaln_shift": adaln_shift,
        "qkv_weight": normal((QKV_WIDTH, HIDDEN), 0.01),
        "q_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
    }


def _make_case(m: int, p: int, model, device: torch.device, *, layout: str):
    generator = torch.Generator(device=device)
    generator.manual_seed(4532 + m + p)
    x = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
    x.normal_(0.0, 0.5, generator=generator)
    adaln_rows = model["adaln_scale"].shape[0]
    rows = torch.arange(m, dtype=torch.int64, device=device)
    adaln_index = torch.div(rows * adaln_rows, m, rounding_mode="floor").clamp_max(
        adaln_rows - 1
    )
    if layout == "contract":
        rope_cos_sin = _make_rope_cache(m, device)
        rope_positions = None
    else:
        # A shared cache longer than the local token count, addressed by positions.
        cache_rows = m + 4096
        rope_cos_sin = _make_rope_cache(cache_rows, device)
        rope_positions = (rows + 2048).remainder(cache_rows)
    output_shape = (p, m, NUM_HEADS // p, QKV_KINDS, HEAD_DIM)
    return {
        "x": x,
        **model,
        "adaln_index": adaln_index,
        "rope_cos_sin": rope_cos_sin,
        "rope_positions": rope_positions,
        "out": torch.empty(output_shape, dtype=torch.bfloat16, device=device),
        "baseline_out": torch.empty(output_shape, dtype=torch.bfloat16, device=device),
        "ulysses_degree": p,
        "eps": EPS,
    }


def _apply_rope(x, rope_rows):
    rotary = x[..., :ROPE_DIM].float()
    tail = x[..., ROPE_DIM:]
    cos_half = rope_rows[:, :48].float()
    sin_half = rope_rows[:, 48:].float()
    cos = torch.cat((cos_half, cos_half), dim=-1)[:, None, :]
    sin = torch.cat((sin_half, sin_half), dim=-1)[:, None, :]
    rotated_half = torch.cat((-rotary[..., 48:], rotary[..., :48]), dim=-1)
    rotated = (rotary * cos + rotated_half * sin).to(torch.bfloat16)
    return torch.cat((rotated, tail), dim=-1)


def _rope_rows(case):
    positions = case["rope_positions"]
    if positions is None:
        return case["rope_cos_sin"][: case["x"].shape[0]]
    return case["rope_cos_sin"].index_select(0, positions)


def _legacy_operand_copies(case):
    """The host-side copies a route had to make before the engine-operand contract:
    contiguous tables, an int32 index and a per-token RoPE table."""
    scale = case["adaln_scale"].contiguous()
    shift = case["adaln_shift"].contiguous()
    index = case["adaln_index"].to(torch.int32)
    rope = _rope_rows(case).contiguous()
    return scale, shift, index, rope


def _segmented_baseline(case):
    m = case["x"].shape[0]
    p = case["ulysses_degree"]
    norm = F.rms_norm(case["x"], (HIDDEN,), case["x_norm_weight"], eps=EPS).to(
        torch.bfloat16
    )
    index = case["adaln_index"].long()
    scale = case["adaln_scale"].index_select(0, index)
    shift = case["adaln_shift"].index_select(0, index)
    adaln = torch.addcmul(shift, norm, (scale + 1.0).to(torch.bfloat16)).to(
        torch.bfloat16
    )
    qkv = F.linear(adaln, case["qkv_weight"]).to(torch.bfloat16)
    grouped = qkv.view(m, NUM_HEADS, QKV_KINDS, HEAD_DIM)
    q = F.rms_norm(grouped[:, :, 0, :], (HEAD_DIM,), case["q_norm_weight"], eps=EPS).to(
        torch.bfloat16
    )
    k = F.rms_norm(grouped[:, :, 1, :], (HEAD_DIM,), case["k_norm_weight"], eps=EPS).to(
        torch.bfloat16
    )
    rope_rows = _rope_rows(case)
    q = _apply_rope(q, rope_rows)
    k = _apply_rope(k, rope_rows)
    fused = torch.stack((q, k, grouped[:, :, 2, :]), dim=2)
    packed_view = fused.view(m, p, NUM_HEADS // p, QKV_KINDS, HEAD_DIM).permute(
        1, 0, 2, 3, 4
    )
    case["baseline_out"].copy_(packed_view)
    return case["baseline_out"]


def _run_candidate(case):
    return minimax_h3_bf16_pre_attention(
        case["x"],
        case["x_norm_weight"],
        case["adaln_scale"],
        case["adaln_shift"],
        case["adaln_index"],
        case["qkv_weight"],
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["rope_cos_sin"],
        ulysses_degree=case["ulysses_degree"],
        out=case["out"],
        eps=case["eps"],
        rope_positions=case["rope_positions"],
    )


def _median_ms(fn):
    times = bench_gpu_time(fn, enable_cupti=True, dry_run_iters=10, repeat_iters=100)
    return float(np.median(times))


def _bench_shape(duration: int, m: int, p: int, model, device, *, layout: str):
    case = _make_case(m, p, model, device, layout=layout)
    expected = _segmented_baseline(case)
    actual = _run_candidate(case)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)

    baseline_ms = _median_ms(lambda: _segmented_baseline(case))
    candidate_ms = _median_ms(lambda: _run_candidate(case))
    copies_ms = (
        _median_ms(lambda: _legacy_operand_copies(case)) if layout == "engine" else None
    )
    return {
        "duration": duration,
        "M": m,
        "P": p,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "copies_ms": copies_ms,
        "speedup": baseline_ms / candidate_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SM100a / SM103a BF16 pre-attention"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--suite", choices=("active", "final"), default="active")
    parser.add_argument(
        "--operand-layout",
        choices=("contract", "engine"),
        default="contract",
        help="contract: contiguous tables, identity RoPE; engine: strided column-chunk "
        "tables, int64 indices, (cos_sin_cache, positions)",
    )
    parser.add_argument(
        "--adaln-rows",
        type=int,
        default=None,
        help="AdaLN table rows (default 9 for contract, 6 for engine)",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        raise RuntimeError("This benchmark requires compute capability 10.0 or 10.3")

    layout = args.operand_layout
    adaln_rows = args.adaln_rows or (ADALN_ROWS if layout == "contract" else 6)
    shapes = ACTIVE_SHAPES if args.suite == "active" else CENTER_SHAPES
    model = _make_model(device, layout=layout, adaln_rows=adaln_rows)
    print(
        f"GPU: {torch.cuda.get_device_name(device)}, operand layout: {layout}, "
        f"AdaLN rows: {adaln_rows}"
    )
    header = f"{'duration':>8} {'M':>8} {'P':>3} {'baseline ms':>13} {'fused ms':>10} {'speedup':>9}"
    if layout == "engine":
        header += f" {'copies ms':>10}"
    print(header)
    for duration, m, p in shapes:
        result = _bench_shape(duration, m, p, model, device, layout=layout)
        line = (
            f"{duration:>7}s {m:>8} {p:>3} "
            f"{result['baseline_ms']:>13.6f} {result['candidate_ms']:>10.6f} "
            f"{result['speedup']:>8.4f}x"
        )
        if result["copies_ms"] is not None:
            line += f" {result['copies_ms']:>10.6f}"
        print(line)


if __name__ == "__main__":
    main()
