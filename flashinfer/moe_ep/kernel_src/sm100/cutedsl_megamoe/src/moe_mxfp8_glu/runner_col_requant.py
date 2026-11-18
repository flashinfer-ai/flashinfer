# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Correctness and lightweight performance runner for MXFP8 column requant.

Example:

    .venv/bin/python -m moe_mxfp8_glu.runner_col_requant \
        --kind mxfp8_e4m3

The source is ordinary row-scaled MXFP8.  Each expert's data rows are padded
to ``--token_padding_block`` in the shared data pool, while its row-SF plane
is independently padded
to 128 and converted to the 32x4x4 blocked byte layout.  The reference first
dequantizes that row-scaled representation explicitly, then requantizes along
the token axis in blocks of 32.
"""

from __future__ import annotations

import argparse
import math
import traceback
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass.cute as cute
from common.host_utils import (
    mxfp8_quantize_per_block_32_col,
    mxfp8_quantize_per_block_32_row,
)
from moe_nvfp4_swapab.runner_common import round_up, to_blocked


_DATA_ROW_PAD = 128
_SF_ROW_PAD = 128
_MXFP8_BLOCK = 32
_DEFAULT_COUNTS = "1,31,32,33,64,65,127,129"


@dataclass
class _Inputs:
    kind: str
    hidden: int
    counts: List[int]
    data_dtype: torch.dtype
    src_data: torch.Tensor
    src_sf_u8_flat: torch.Tensor
    valid_counts_i32: torch.Tensor
    data_offsets_i32: torch.Tensor
    src_sf_offsets_i32: torch.Tensor
    dequant_per_expert: List[torch.Tensor]
    data_rows: int
    src_sf_rows: int
    token_pad: int = _DATA_ROW_PAD


def _zero_fp8(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate a zero-filled CUDA FP8 tensor through its byte storage."""
    return (
        torch.zeros(math.prod(shape), dtype=torch.uint8, device="cuda")
        .view(dtype)
        .reshape(shape)
    )


def _parse_valid_counts(text: str) -> List[int]:
    try:
        counts = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"valid_counts must be comma-separated integers, got {text!r}"
        ) from exc
    if not counts:
        raise argparse.ArgumentTypeError("valid_counts must contain at least one value")
    if any(count < 0 for count in counts):
        raise argparse.ArgumentTypeError(
            f"valid_counts must be non-negative, got {counts}"
        )
    if max(counts) == 0:
        raise argparse.ArgumentTypeError(
            "valid_counts must contain at least one positive value"
        )
    return counts


def _to_col_blocked(scale_hidden_token: torch.Tensor) -> torch.Tensor:
    """Pack plain ``[hidden, ceil(tokens/32)]`` SF in MN-major atom order."""
    hidden, token_blocks = scale_hidden_token.shape
    hidden_atoms = math.ceil(hidden / 128)
    token_atoms = math.ceil(token_blocks / 4)
    padded = torch.zeros(
        (hidden_atoms * 128, token_atoms * 4),
        dtype=scale_hidden_token.dtype,
        device=scale_hidden_token.device,
    )
    padded[:hidden, :token_blocks] = scale_hidden_token
    return (
        padded.view(hidden_atoms, 128, token_atoms, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
    )


def _make_inputs(
    kind: str,
    hidden: int,
    counts: List[int],
    seed: int,
    bench_only: bool = False,
    dirty_padding: str = "off",
    token_pad: int = _DATA_ROW_PAD,
    src_scale_exp: int = 0,
) -> _Inputs:
    """Build the kernel inputs."""
    data_dtype = {
        "mxfp8_e4m3": torch.float8_e4m3fn,
        "mxfp8_e5m2": torch.float8_e5m2,
    }[kind]
    sf_cols = hidden // _MXFP8_BLOCK

    data_offsets: List[int] = []
    src_sf_offsets: List[int] = []
    data_rows = 0
    src_sf_rows = 0
    for count in counts:
        data_offsets.append(data_rows)
        src_sf_offsets.append(src_sf_rows)
        data_rows += round_up(count, token_pad)
        src_sf_rows += round_up(count, _SF_ROW_PAD)
        # Raw col-SF is (hidden, ceil(count / 32)); to_blocked pads its
        # trailing block count to four, i.e. token rows to 128.

    src_data = _zero_fp8((data_rows, hidden), data_dtype)
    src_sf_u8_flat = torch.zeros(
        src_sf_rows * sf_cols, dtype=torch.uint8, device="cuda"
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dequant_per_expert: List[torch.Tensor] = []
    _FP8_MAX_BYTE = {"mxfp8_e4m3": 0x7E, "mxfp8_e5m2": 0x7B}[kind]
    pad_gen = torch.Generator(device="cuda")
    pad_gen.manual_seed(seed ^ 0x5A5A5A5A)
    for expert, count in enumerate(counts):
        original = torch.randn(
            (count, hidden), dtype=torch.float32, device="cuda"
        )
        if src_scale_exp:
            original = original * float(2.0 ** src_scale_exp)
        q_row, row_sf_plain = mxfp8_quantize_per_block_32_row(
            original, data_dtype
        )

        data_begin = data_offsets[expert]
        src_data[data_begin : data_begin + count].copy_(q_row)

        data_padded = round_up(count, token_pad)
        sf_padded = round_up(count, _SF_ROW_PAD)
        padded = data_padded
        npad = data_padded - count
        npad_sf = sf_padded - count
        sf_plain_full = torch.zeros(
            (sf_padded, sf_cols), dtype=row_sf_plain.dtype, device="cuda"
        )
        sf_plain_full[:count] = row_sf_plain

        if dirty_padding != "off" and npad:
            if row_sf_plain.element_size() != 1:
                raise AssertionError(
                    "dirty padding assumes a 1-byte E8M0 scale, got "
                    f"{row_sf_plain.dtype}"
                )
            pad_orig = torch.randn(
                (npad, hidden), dtype=torch.float32, device="cuda",
                generator=pad_gen,
            )
            pad_q, pad_sf = mxfp8_quantize_per_block_32_row(
                pad_orig, data_dtype
            )
            src_data[data_begin + count : data_begin + padded].copy_(pad_q)
            sf_plain_full[count:sf_padded] = pad_sf[:npad_sf]
            if dirty_padding == "extreme":
                src_data[data_begin + count : data_begin + padded].view(
                    torch.uint8
                ).fill_(_FP8_MAX_BYTE)
                sf_plain_full[count:sf_padded].view(torch.uint8).fill_(0xFE)

        # Keep the quantizer's SF plain for explicit dequantization, but stage
        # the kernel input as one concatenation of independently blocked,
        # 128-row-padded expert segments.
        row_sf_blocked_u8 = (
            to_blocked(sf_plain_full).contiguous().view(torch.uint8)
        )
        expected_sf_bytes = round_up(count, _SF_ROW_PAD) * sf_cols
        if row_sf_blocked_u8.numel() != expected_sf_bytes:
            raise AssertionError(
                f"expert {expert} source SF bytes: got "
                f"{row_sf_blocked_u8.numel()}, expected {expected_sf_bytes}"
            )
        sf_byte_begin = src_sf_offsets[expert] * sf_cols
        src_sf_u8_flat[
            sf_byte_begin : sf_byte_begin + expected_sf_bytes
        ].copy_(row_sf_blocked_u8)

        # Deliberately spell out the row-scaled MXFP8 reconstruction used by
        # both references; do not hide it behind a dequant helper.
        if not bench_only:
            dequant_per_expert.append(
                q_row.float()
                * row_sf_plain.float().repeat_interleave(_MXFP8_BLOCK, dim=1)
            )

    return _Inputs(
        kind=kind,
        hidden=hidden,
        counts=counts,
        data_dtype=data_dtype,
        src_data=src_data,
        src_sf_u8_flat=src_sf_u8_flat,
        valid_counts_i32=torch.tensor(
            counts, dtype=torch.int32, device="cuda"
        ),
        data_offsets_i32=torch.tensor(
            data_offsets, dtype=torch.int32, device="cuda"
        ),
        src_sf_offsets_i32=torch.tensor(
            src_sf_offsets, dtype=torch.int32, device="cuda"
        ),
        dequant_per_expert=dequant_per_expert,
        data_rows=data_rows,
        src_sf_rows=src_sf_rows,
        token_pad=token_pad,
    )


def _reference(
    inputs: _Inputs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Golden for the padded-space tiling."""
    q_ref = _zero_fp8((inputs.data_rows, inputs.hidden), inputs.data_dtype)
    # data_rows is a multiple of 128 and hence of _MXFP8_BLOCK: no tail pad.
    dequant_padded = torch.zeros(
        (inputs.data_rows, inputs.hidden),
        dtype=torch.float32,
        device="cuda",
    )
    for expert, count in enumerate(inputs.counts):
        data_begin = int(inputs.data_offsets_i32[expert].item())
        dequant_padded[data_begin : data_begin + count].copy_(
            inputs.dequant_per_expert[expert]
        )

    q_padded, sf_token_hidden = mxfp8_quantize_per_block_32_col(
        dequant_padded, inputs.data_dtype
    )
    for expert, count in enumerate(inputs.counts):
        data_begin = int(inputs.data_offsets_i32[expert].item())
        q_ref[data_begin : data_begin + count].copy_(
            q_padded[data_begin : data_begin + count]
        )
    sf_token_padded_u8 = _to_col_blocked(sf_token_hidden.t()).contiguous().view(torch.uint8)
    sf_ref_u8 = _select_sf_atom_rows(
        sf_token_padded_u8,
        inputs.hidden,
        inputs.counts,
        [int(v) for v in inputs.data_offsets_i32.cpu().tolist()],
        inputs.token_pad,
    )
    return q_ref, sf_ref_u8, sf_token_hidden


def _select_sf_atom_rows(
    sf_token_padded_u8: torch.Tensor,
    hidden: int,
    counts: Sequence[int],
    data_offsets: Sequence[int],
    token_pad: int,
) -> torch.Tensor:
    """Pack expert SF segments with token atoms contiguous inside hidden atoms."""
    hidden_atoms = math.ceil(hidden / 128)
    atom_bytes = 512
    token_atoms = sf_token_padded_u8.numel() // (hidden_atoms * atom_bytes)
    atoms = sf_token_padded_u8.reshape(hidden_atoms, token_atoms, atom_bytes)
    parts: List[torch.Tensor] = []
    for count, begin in zip(counts, data_offsets):
        first_atom = begin // _SF_ROW_PAD
        num_atoms = round_up(count, _SF_ROW_PAD) // _SF_ROW_PAD
        if num_atoms:
            parts.append(atoms[:, first_atom : first_atom + num_atoms].reshape(-1))
    if not parts:
        return sf_token_padded_u8[:0]
    return torch.cat(parts)



def _flat_index_to_coord(index: int, shape: Sequence[int]) -> Tuple[int, ...]:
    coord: List[int] = []
    for extent in reversed(shape):
        coord.append(index % extent)
        index //= extent
    return tuple(reversed(coord))


def _check_raw_bytes(
    name: str,
    actual_u8: torch.Tensor,
    expected_u8: torch.Tensor,
) -> bool:
    if tuple(actual_u8.shape) != tuple(expected_u8.shape):
        print(
            f"  [FAIL] {name} shape mismatch: actual={tuple(actual_u8.shape)} "
            f"expected={tuple(expected_u8.shape)}"
        )
        return False

    actual_flat = actual_u8.contiguous().reshape(-1)
    expected_flat = expected_u8.contiguous().reshape(-1)
    mismatch = actual_flat != expected_flat
    if not bool(mismatch.any().item()):
        return True

    first = int(torch.nonzero(mismatch, as_tuple=False)[0, 0].item())
    coord = _flat_index_to_coord(first, expected_u8.shape)
    actual_byte = int(actual_flat[first].item())
    expected_byte = int(expected_flat[first].item())
    print(
        f"  [FAIL] first {name} mismatch at {coord} (flat {first}): "
        f"actual=0x{actual_byte:02x}, expected=0x{expected_byte:02x}"
    )
    return False


def _to_cute(tensor: torch.Tensor, assumed_align: int) -> cute.Tensor:
    from cutlass.torch import from_dlpack

    return from_dlpack(
        tensor, assumed_align=assumed_align
    ).mark_layout_dynamic()


def _kernel_kwargs(args) -> dict:
    """CLI flags -> Mxfp8ColRequant kwargs; absent flags are simply not passed."""
    kw = {}
    if args.scaled_cvt != "auto":
        kw["scaled_cvt"] = args.scaled_cvt == "on"
    return kw


def _run_case(
    inputs: _Inputs,
    warmup: int,
    iters: int,
    num_ctas: int,
    check: bool = True,
    poison_dst: bool = False,
    kernel_kwargs: "dict | None" = None,
) -> bool:
    from moe_mxfp8_glu.mxfp8_col_requant import Mxfp8ColRequant

    label = "col_quant"
    ref_data = None
    ref_sf_u8 = None
    ref_sf_token_hidden = None
    dst_sf_bytes = inputs.hidden * (inputs.src_sf_rows // _MXFP8_BLOCK)
    if check:
        ref_data, ref_sf_u8, ref_sf_token_hidden = _reference(inputs)
        if ref_sf_u8.numel() != dst_sf_bytes:
            raise AssertionError(
                f"{label} reference SF bytes: got {ref_sf_u8.numel()}, "
                f"expected {dst_sf_bytes}"
            )

    dst_data = _zero_fp8(
        (inputs.data_rows, inputs.hidden), inputs.data_dtype
    )
    dst_sf_u8_flat = torch.zeros(
        dst_sf_bytes, dtype=torch.uint8, device="cuda"
    )
    if poison_dst:
        dst_data.view(torch.uint8).fill_(0xA5)
        dst_sf_u8_flat.fill_(0xA5)

    op = Mxfp8ColRequant(
        hidden=inputs.hidden,
        num_experts=len(inputs.counts),
        max_total_tokens=sum(inputs.counts),
        quant_type=inputs.kind,
        num_persistent_ctas=num_ctas,
        token_padding_block=inputs.token_pad,
        sf_padding_block=_SF_ROW_PAD,
        **(kernel_kwargs or {}),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    runtime_args = (
        _to_cute(inputs.src_data, 16),
        _to_cute(inputs.src_sf_u8_flat, 16),
        _to_cute(inputs.valid_counts_i32, 4),
        _to_cute(dst_data, 16),
        _to_cute(dst_sf_u8_flat, 16),
    )

    compiled = cute.compile(op, *runtime_args, stream)
    compiled(*runtime_args, stream)
    torch.cuda.synchronize()

    if check:
        pad_bad = []
        for e, cnt in enumerate(inputs.counts):
            begin = int(inputs.data_offsets_i32[e].item())
            padded = round_up(cnt, inputs.token_pad)
            if padded == cnt:
                continue
            region = dst_data[begin + cnt : begin + padded].view(torch.uint8)
            nz = int((region != 0).sum().item())
            if nz:
                pad_bad.append((e, cnt, padded - cnt, nz, region.numel()))
        # This is a verdict, not a diagnostic -- but only when the harness
        # pre-zeroed dst_data.  Under --poison_dst the padding is prefilled and
        # the 1D store epilogue legitimately never writes padding rows, so a
        # nonzero region there says nothing about the kernel.
        pad_ok = True
        if pad_bad:
            print(
                f"[{label}] PADDING dst_data NOT ZERO in "
                f"{len(pad_bad)} expert(s):"
            )
            for e, cnt, rows, nz, tot in pad_bad:
                print(
                    f"    expert {e}: valid={cnt} padding_rows={rows} "
                    f"nonzero_bytes={nz}/{tot}"
                )
            if poison_dst:
                print(
                    f"[{label}] PADDING not a verdict under --poison_dst "
                    f"(prefilled, and the 1D store epilogue does not write "
                    f"padding rows)"
                )
            else:
                pad_ok = False
        else:
            npad = sum(
                round_up(c, inputs.token_pad) - c for c in inputs.counts
            )
            if npad:
                print(f"[{label}] PADDING dst_data all zero ({npad} rows)  OK")

        from tester.col_quant_check import check_col_quant_data_with_tolerance

        data_ok = check_col_quant_data_with_tolerance(
            "FP8 data",
            dst_data,
            ref_data,
            ref_sf_token_hidden,
            counts=inputs.counts,
            token_pad=inputs.token_pad,
        )
        sf_ok = _check_raw_bytes("E8M0 SF", dst_sf_u8_flat, ref_sf_u8)
        if not (data_ok and sf_ok and pad_ok):
            print(
                f"[{label}] FAIL kind={inputs.kind} hidden={inputs.hidden} "
                f"experts={len(inputs.counts)}"
            )
            return False
        del ref_data, ref_sf_u8

    for _ in range(warmup):
        compiled(*runtime_args, stream)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        compiled(*runtime_args, stream)
    stop.record()
    stop.synchronize()
    average_us = start.elapsed_time(stop) * 1.0e3 / iters

    tag = "PASS" if check else "BENCH"
    extra = (
        f" C={op.ColsPerLane} tile_hid={op.TILE_HID} "
        f"warps={op.WarpsPerCta}(1p+{op.ConsumerWarps}c) thr={op.ThreadsPerCta} "
        f"stages={op.NumStages}"
    )
    print(
        f"[{label}] {tag} kind={inputs.kind} hidden={inputs.hidden} "
        f"packed_tokens={sum(inputs.counts)} ctas={op.num_persistent_ctas} "
        f"hpw={op.hidden_tiles_per_work * op.HiddenPerCta} "
        f"groups={op.hidden_groups}{extra} avg={average_us:.3f} us ({iters} iterations)"
    )
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MXFP8 row-to-column requant correctness/perf-lite runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kind",
        choices=["mxfp8_e4m3", "mxfp8_e5m2"],
        default="mxfp8_e4m3",
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--valid_counts",
        "--valid-counts",
        dest="valid_counts",
        type=_parse_valid_counts,
        default=_parse_valid_counts(_DEFAULT_COUNTS),
        help="comma-separated valid token counts, one per expert",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--num_ctas", type=int, default=-1)
    # 'auto' asks the compilation target; 'off' forces the portable FP32
    # consumer, which is what sm_100a takes and what the fallback tests want.
    parser.add_argument(
        "--scaled_cvt",
        "--scaled-cvt",
        dest="scaled_cvt",
        choices=["auto", "on", "off"],
        default="auto",
    )
    parser.add_argument(
        "--bench_only",
        action="store_true",
        help="skip the FP32 reference build and the byte compare; the reference "
             "needs ~3x the FP8 footprint in FP32 and dominates the wall time "
             "at large token counts",
    )
    parser.add_argument(
        "--dirty_padding",
        "--dirty-padding",
        dest="dirty_padding",
        choices=["off", "random", "extreme"],
        default="off",
        help="what to stage in the rows between an expert's valid count and "
             "its 128-padded row count.  'off' is the shipped zero fill, which "
             "cannot test the consumer's dead-token mask at all (a zero never "
             "wins an abs-max).  'extreme' stages ~448*2^127 there.",
    )
    parser.add_argument(
        "--src_scale_exp",
        "--src-scale-exp",
        dest="src_scale_exp",
        type=int,
        default=0,
        help="multiply the pre-quantization source values by 2**N.  Every source "
             "FP8 data byte stays identical and every source E8M0 scale byte "
             "shifts by N, so this reaches scale bytes >= 0x80 that unit-normal "
             "data never produces.  A signed byte load of the scale corrupts the "
             "packed E8M0 pair there; 16 puts E4M3 at 0x87..0x89.",
    )
    parser.add_argument(
        "--poison_dst",
        "--poison-dst",
        dest="poison_dst",
        action="store_true",
        help="pre-fill dst_data/dst_sf with 0xA5 instead of zero, so that "
             "'padding output is zero' means the kernel wrote a zero rather "
             "than the harness having pre-zeroed the buffer",
    )
    parser.add_argument(
        "--token_padding_block",
        "--token-padding-block",
        "--token_padding",
        dest="token_padding_block",
        type=int,
        default=_DATA_ROW_PAD,
        help="expert row padding granularity for the DATA pool and the "
             "destination SF pool.  The kernel accepts 128 and 256 only; the "
             "SOURCE SF pool stays 128-padded either way, so 256 is the "
             "configuration in which the data and SF prefixes drift apart and "
             "an expert can own a tile that is 100%% padding.  Anything else is "
             "expected to raise out of Mxfp8ColRequant.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Gate on the 128-element SF atom, which is the smallest legal TILE_HID and
    # therefore what the kernel actually requires; the kernel raises its own
    # `no legal (ColsPerLane, TILE_HID) pair` error for anything finer.
    if args.hidden <= 0 or args.hidden % 128 != 0:
        parser.error(
            f"hidden must be a positive multiple of the 128-element SF atom "
            f"(the smallest legal TILE_HID), got {args.hidden}"
        )
    if args.warmup < 0:
        parser.error(f"warmup must be non-negative, got {args.warmup}")
    if args.iters <= 0:
        parser.error(f"iters must be positive, got {args.iters}")
    if args.num_ctas <= 0 and args.num_ctas != -1:
        parser.error(f"num_ctas must be positive or -1 (auto), got {args.num_ctas}")

    inputs = _make_inputs(
        kind=args.kind,
        hidden=args.hidden,
        counts=args.valid_counts,
        seed=args.seed,
        bench_only=args.bench_only,
        dirty_padding=args.dirty_padding,
        token_pad=args.token_padding_block,
        src_scale_exp=args.src_scale_exp,
    )
    print("[case] col_quant")
    try:
        all_ok = _run_case(
            inputs,
            args.warmup,
            args.iters,
            args.num_ctas,
            check=not args.bench_only,
            poison_dst=args.poison_dst,
            kernel_kwargs=_kernel_kwargs(args),
        )
    except Exception as exc:  # noqa: BLE001 - standalone diagnostic runner
        print(f"[col_quant] ERROR {type(exc).__name__}: {exc}")
        traceback.print_exc()
        all_ok = False

    if not all_ok:
        print("[overall] FAIL")
        return 1
    # --bench_only skips the reference build and every byte compare, so there is
    # no verdict to report.  Exit non-zero rather than print a green line that
    # nothing backs up.
    if args.bench_only:
        print("[overall] SKIPPED (no reference: --bench_only compares nothing)")
        return 2
    print("[overall] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
