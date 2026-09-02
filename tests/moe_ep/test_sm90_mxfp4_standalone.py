# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone Hopper probes for the SM90 Humming MXFP4 x FP8 path.

The device primitives are imported exclusively through FlashInfer's public
SM90 kernel-package boundary.  Scalar conversion, hybrid-scale, and FC1
handoff expectations are test-owned and do not depend on the kernel donor.
"""

from __future__ import annotations

import importlib
import struct

import pytest


torch = pytest.importorskip("torch")

_PUBLIC_MODULE_NAME = "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel"
_PUBLIC_API = None

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import cutlass.utils.hopper_helpers as sm90_utils
    from cutlass.cute.nvgpu import warpgroup
    from cutlass.cute.runtime import from_dlpack
except ImportError:
    cutlass = None
    cute = None
    utils = None
    sm90_utils = None
    warpgroup = None
    from_dlpack = None


_PRIMITIVE_IMPORT_ERROR: Exception | None = None
_PRIMITIVES_LOADED = False
convert_mxfp4_pair_preprocessed_signs = None
convert_packed_a_kblock = None
convert_packed_a_kblock_from_offset = None
make_expanded_offset_view = None
make_offset_smem_layout = None
make_packed_a_ldsm_views = None


def _public_api():
    """Import the mutually exclusive SM90 tree only inside Hopper tests."""

    global _PUBLIC_API
    if _PUBLIC_API is None:
        try:
            _PUBLIC_API = importlib.import_module(_PUBLIC_MODULE_NAME)
        except RuntimeError as error:
            # The SM100 tree owns the shared top-level vendor module names in
            # this pytest process.  Dedicated SM90 runs exercise this file.
            pytest.skip(f"SM90 kernel tree unavailable in this process: {error}")
    return _PUBLIC_API


def _load_public_primitives() -> None:
    global _PRIMITIVE_IMPORT_ERROR
    global _PRIMITIVES_LOADED
    global convert_mxfp4_pair_preprocessed_signs
    global convert_packed_a_kblock
    global convert_packed_a_kblock_from_offset
    global make_expanded_offset_view
    global make_offset_smem_layout
    global make_packed_a_ldsm_views

    if _PRIMITIVES_LOADED:
        return
    public_api = _public_api()
    try:
        # These attributes are lazy shim exports.  Tests must not import the
        # raw ``moe_hopper_fp8`` package from the vendored source tree.
        convert_mxfp4_pair_preprocessed_signs = (
            public_api.convert_mxfp4_pair_preprocessed_signs
        )
        convert_packed_a_kblock = public_api.convert_packed_a_kblock
        convert_packed_a_kblock_from_offset = (
            public_api.convert_packed_a_kblock_from_offset
        )
        make_expanded_offset_view = public_api.make_expanded_offset_view
        make_offset_smem_layout = public_api.make_offset_smem_layout
        make_packed_a_ldsm_views = public_api.make_packed_a_ldsm_views
    except (AttributeError, ImportError) as error:
        _PRIMITIVE_IMPORT_ERROR = error
    _PRIMITIVES_LOADED = True


def _require_hopper(*, require_primitives: bool = True) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip("SM90 Hopper GPU is required")
    if cutlass is None:
        pytest.skip("nvidia-cutlass-dsl is required")
    if require_primitives:
        _load_public_primitives()
        if _PRIMITIVE_IMPORT_ERROR is not None:
            pytest.fail(
                "SM90 MXFP4 primitives are missing from the FlashInfer public "
                f"boundary {_PUBLIC_MODULE_NAME}: {_PRIMITIVE_IMPORT_ERROR}"
            )


def _as_i32(value: int) -> int:
    return struct.unpack("i", struct.pack("I", value & 0xFFFFFFFF))[0]


def _as_u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _preprocess_signs(word: int) -> int:
    exponent_mantissa = word & 0x77777777
    signs = (
        ((word & 0x00000008) << 4)
        | ((word & 0x00000080) << 8)
        | ((word & 0x00000800) << 12)
        | ((word & 0x00008000) << 16)
        | ((word & 0x00080000) >> 16)
        | ((word & 0x00800000) >> 12)
        | ((word & 0x08000000) >> 8)
        | ((word & 0x80000000) >> 4)
    )
    return (exponent_mantissa | signs) & 0xFFFFFFFF


def _convert_word_reference(
    preprocessed: int, lo_offset: int, hi_offset: int
) -> tuple[int, int]:
    """Independent scalar transcription of the paired PRMT converter."""

    def magnitude(code: int, offset: int) -> int:
        if code == 0:
            return 0
        if code == 1:
            return offset * 8
        if code == 2:
            return offset * 8 + 0x08
        if code == 3:
            return offset * 8 + 0x0C
        return offset * 8 + 0x10 + (code - 4) * 4

    low_bytes: list[int] = []
    high_bytes: list[int] = []
    for index in range(4):
        low_code = (preprocessed >> (index * 4)) & 0x7
        high_code = (preprocessed >> ((index + 4) * 4)) & 0x7
        low_sign = (preprocessed >> (index * 8 + 7)) & 0x1
        high_sign = (preprocessed >> (index * 8 + 3)) & 0x1
        low_bytes.append(magnitude(low_code, lo_offset) | (low_sign << 7))
        high_bytes.append(magnitude(high_code, hi_offset) | (high_sign << 7))

    def pack(values: list[int]) -> int:
        return sum(byte << (8 * index) for index, byte in enumerate(values))

    return pack(low_bytes), pack(high_bytes)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    assert codes.shape[-1] % 2 == 0
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def _reference_e2m1_to_e4m3_bytes(
    codes: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Map logical E2M1 codes and K32 offsets to raw E4M3 bytes."""

    if codes.dtype != torch.uint8 or offsets.dtype != torch.uint8:
        raise ValueError("codes and offsets must have dtype uint8")
    if codes.ndim != 3 or offsets.shape != (
        codes.shape[0],
        codes.shape[1],
        codes.shape[2] // 32,
    ):
        raise ValueError("offsets must contain one byte per logical K32 group")

    magnitude = (codes & 0x7).to(torch.int16)
    expanded_offset = offsets.repeat_interleave(32, dim=-1).to(torch.int16)
    base = expanded_offset * 8
    encoded = torch.where(
        magnitude == 0,
        torch.zeros_like(base),
        torch.where(
            magnitude == 1,
            base,
            torch.where(
                magnitude == 2,
                base + 0x08,
                torch.where(
                    magnitude == 3,
                    base + 0x0C,
                    base + 0x10 + (magnitude - 4) * 4,
                ),
            ),
        ),
    )
    sign = (codes & 0x8).to(torch.int16) << 4
    return (encoded | sign).to(torch.uint8).contiguous()


def _reference_interleave_weight(weight: torch.Tensor) -> torch.Tensor:
    """Test-owned SM90 FP4-for-FP8 physical interleave."""

    experts, rows, packed_k = weight.shape
    logical_k = packed_k * 2
    if rows % 16 or logical_k % 64 or packed_k % 2:
        raise ValueError("weight must satisfy the 16-row/K64 interleave contract")

    source = weight.detach().cpu().contiguous()
    source_u16 = source[..., 0::2].to(torch.int64) | (
        source[..., 1::2].to(torch.int64) << 8
    )
    output_u16 = torch.empty_like(source_u16)
    for expert in range(experts):
        for block_id in range(rows // 2):
            row = (block_id // 8) * 16 + block_id % 8
            for partition in range(logical_k // 64):
                for lane in range(16):
                    destination_row = row + ((lane % 8) // 4) * 8
                    source_column = partition * 16 + lane
                    destination_column = (
                        partition * 16 + (lane // 8) * 8 + (lane % 4) * 2
                    )
                    word = int(source_u16[expert, row, source_column]) | (
                        int(source_u16[expert, row + 8, source_column]) << 16
                    )
                    word = _preprocess_signs(word)
                    output_u16[expert, destination_row, destination_column] = (
                        word & 0xFFFF
                    )
                    output_u16[expert, destination_row, destination_column + 1] = (
                        word >> 16
                    ) & 0xFFFF

    output = torch.empty_like(source)
    output[..., 0::2] = (output_u16 & 0xFF).to(torch.uint8)
    output[..., 1::2] = ((output_u16 >> 8) & 0xFF).to(torch.uint8)
    return output.contiguous()


def _reference_fold_offsets(offsets: torch.Tensor) -> torch.Tensor:
    """Fold logical offsets to ``[E, M/64, K/128, 16, 16]``."""

    experts, rows, k32_groups = offsets.shape
    if rows % 64 or k32_groups % 4:
        raise ValueError("offsets must satisfy the M64/K128 fold contract")
    output = torch.empty(
        (experts, rows // 64, k32_groups // 4, 16, 16),
        dtype=torch.uint8,
    )
    for m64 in range(rows // 64):
        for k128 in range(k32_groups // 4):
            for folded_m in range(16):
                for m_slice in range(4):
                    row = m64 * 64 + m_slice * 16 + folded_m
                    for k32 in range(4):
                        output[:, m64, k128, folded_m, m_slice * 4 + k32] = offsets[
                            :, row, k128 * 4 + k32
                        ]
    return output.contiguous()


if cutlass is not None:

    class _Mxfp4RsK128Probe:
        tile_m = 128
        tile_n = 32
        tile_k = 128
        packed_k = tile_k // 2
        threads = 128

        def __init__(self, *, use_offset_variant: bool):
            self.use_offset_variant = use_offset_variant

        @cute.jit
        def __call__(self, packed_a, folded_offsets, b, expanded_a, c):
            tiled_mma = sm90_utils.make_trivial_tiled_mma(
                cutlass.Float8E4M3FN,
                cutlass.Float8E4M3FN,
                warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.K,
                cutlass.Float32,
                (1, 1, 1),
                (64, self.tile_n),
                warpgroup.OperandSource.RMEM,
            )
            a_smem_layout = sm90_utils.make_smem_layout_a(
                utils.LayoutEnum.ROW_MAJOR,
                (self.tile_m, self.tile_n, self.tile_k),
                cutlass.Float4E2M1FN,
                2,
            )
            b_smem_layout = sm90_utils.make_smem_layout_b(
                utils.LayoutEnum.ROW_MAJOR,
                (self.tile_m, self.tile_n, self.tile_k),
                cutlass.Float8E4M3FN,
                2,
            )
            offset_smem_layout = make_offset_smem_layout(self.tile_m, 2)

            @cute.struct
            class SharedStorage:
                packed_a: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Float4E2M1FN, cute.cosize(a_smem_layout)
                    ],
                    128,
                ]
                b: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Float8E4M3FN, cute.cosize(b_smem_layout)
                    ],
                    128,
                ]
                offsets: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Uint8, cute.cosize(offset_smem_layout)
                    ],
                    16,
                ]

            self.shared_storage = SharedStorage
            self.kernel(
                packed_a,
                folded_offsets,
                b,
                expanded_a,
                c,
                tiled_mma,
                a_smem_layout,
                b_smem_layout,
                offset_smem_layout,
            ).launch(grid=[1, 1, 1], block=[self.threads, 1, 1])

        @cute.kernel
        def kernel(
            self,
            packed_a: cute.Tensor,
            folded_offsets: cute.Tensor,
            b: cute.Tensor,
            expanded_a: cute.Tensor,
            c: cute.Tensor,
            tiled_mma: cute.TiledMma,
            a_smem_layout: cute.ComposedLayout,
            b_smem_layout: cute.ComposedLayout,
            offset_smem_layout: cute.Layout,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            s_a = storage.packed_a.get_tensor(
                a_smem_layout.outer, swizzle=a_smem_layout.inner
            )
            s_b = storage.b.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
            s_offsets = storage.offsets.get_tensor(offset_smem_layout)
            s_a_stage = cute.slice_(s_a, (None, None, 0))
            s_a_bytes_plain = cute.recast_tensor(s_a_stage, cutlass.Uint8)
            s_a_bytes = cute.make_tensor(
                cute.recast_ptr(
                    s_a_stage.iterator,
                    a_smem_layout.inner,
                    dtype=cutlass.Uint8,
                ),
                s_a_bytes_plain.layout,
            )
            s_b_stage = cute.slice_(s_b, (None, None, 0))

            for item in cutlass.range(
                self.tile_m * self.packed_k // self.threads,
                unroll_full=True,
            ):
                linear = tidx + cutlass.Int32(item * self.threads)
                row = linear // cutlass.Int32(self.packed_k)
                column = linear % cutlass.Int32(self.packed_k)
                s_a_bytes[row, column] = packed_a[row, column]

            for item in cutlass.range_constexpr(
                self.tile_n * self.tile_k // self.threads
            ):
                linear = tidx + cutlass.Int32(item * self.threads)
                row = linear // cutlass.Int32(self.tile_k)
                column = linear % cutlass.Int32(self.tile_k)
                s_b_stage[row, column] = b[row, column]

            offset_bytes = (self.tile_m // 64) * 256
            for item in cutlass.range_constexpr(offset_bytes // self.threads):
                linear = tidx + cutlass.Int32(item * self.threads)
                physical_column = linear % cutlass.Int32(16)
                folded_m = (linear // cutlass.Int32(16)) % cutlass.Int32(16)
                m_block = linear // cutlass.Int32(256)
                s_offsets[physical_column, folded_m, m_block, 0, 0] = folded_offsets[
                    linear
                ]

            cute.arch.sync_threads()

            s_a_wg = cute.local_tile(
                s_a,
                cute.slice_(
                    (self.tile_m, self.tile_n, self.tile_k),
                    (None, 0, None),
                ),
                (0, 0, None),
            )
            thread_mma = tiled_mma.get_slice(tidx)
            fp8_a = tiled_mma.make_fragment_A(
                tiled_mma.partition_shape_A((self.tile_m, self.tile_k))
            )
            tiled_copy, smem_partition, copy_view, packed_registers = (
                make_packed_a_ldsm_views(tiled_mma, s_a_wg, fp8_a, tidx)
            )
            cute.copy(
                tiled_copy,
                smem_partition[(None, None, None, 0)],
                copy_view,
            )

            expanded_offsets = make_expanded_offset_view(s_offsets, self.tile_m)
            partitioned_offsets = thread_mma.partition_A(expanded_offsets)
            for k_block in cutlass.range_constexpr(self.tile_k // 32):
                if cutlass.const_expr(self.use_offset_variant):
                    convert_packed_a_kblock_from_offset(
                        packed_registers,
                        fp8_a,
                        partitioned_offsets,
                        k_block,
                        k_block,
                        cutlass.Int32(0),
                    )
                else:
                    convert_packed_a_kblock(
                        packed_registers,
                        fp8_a,
                        partitioned_offsets,
                        k_block,
                        cutlass.Int32(0),
                    )

            expanded_a_partition = thread_mma.partition_A(expanded_a)
            a_store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN
            )
            cute.copy(a_store_atom, fp8_a, expanded_a_partition)

            b_fragment = tiled_mma.make_fragment_B(thread_mma.partition_B(s_b))
            c_partition = thread_mma.partition_C(c)
            accumulators = cute.make_rmem_tensor(c_partition.shape[:3], cutlass.Float32)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()
            for k_block in cutlass.range_constexpr(self.tile_k // 32):
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    fp8_a[(None, None, k_block)],
                    b_fragment[(None, None, k_block, 0)],
                    accumulators,
                )
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float32
            )
            cute.copy(copy_atom, accumulators, c_partition)


@pytest.mark.arch_hopper
def test_prmt_converter_covers_every_e2m1_code_and_offset() -> None:
    _require_hopper()

    @cute.kernel
    def converter_kernel(src, offsets, dst):
        tidx = cute.arch.thread_idx()[0]
        if tidx < cute.size(src, mode=[0]):
            out0, out1, out2, out3 = convert_mxfp4_pair_preprocessed_signs(
                src[tidx, 0],
                src[tidx, 1],
                offsets[tidx, 0],
                offsets[tidx, 1],
            )
            dst[tidx, 0] = out0
            dst[tidx, 1] = out1
            dst[tidx, 2] = out2
            dst[tidx, 3] = out3

    @cute.jit
    def launch_converter(src, offsets, dst):
        converter_kernel(src, offsets, dst).launch(grid=[1, 1, 1], block=[256, 1, 1])

    vectors: list[tuple[int, int, int, int]] = []
    # Every signed E2M1 code at every legal Humming offset is placed in all
    # four low and all four high result bytes of both source words.
    for offset in range(1, 13):
        for code in range(16):
            canonical = sum(code << (4 * index) for index in range(8))
            vectors.append((canonical, canonical, offset, offset))
    # Cross the two offsets as a directed guard against swapped low/high LUTs.
    for low_offset in range(1, 13):
        high_offset = 13 - low_offset
        word0 = sum(index << (4 * index) for index in range(8))
        word1 = sum((15 - index) << (4 * index) for index in range(8))
        vectors.append((word0, word1, low_offset, high_offset))

    preprocessed = [
        (_preprocess_signs(word0), _preprocess_signs(word1), low, high)
        for word0, word1, low, high in vectors
    ]
    src_cpu = torch.tensor(
        [[_as_i32(word0), _as_i32(word1)] for word0, word1, _, _ in preprocessed],
        dtype=torch.int32,
    )
    offsets_cpu = torch.tensor(
        [[low, high] for _, _, low, high in preprocessed], dtype=torch.int32
    )
    expected_rows: list[list[int]] = []
    for word0, word1, low_offset, high_offset in preprocessed:
        word0_low, word0_high = _convert_word_reference(word0, low_offset, high_offset)
        word1_low, word1_high = _convert_word_reference(word1, low_offset, high_offset)
        expected_rows.append(
            [
                _as_i32(word0_low),
                _as_i32(word0_high),
                _as_i32(word1_low),
                _as_i32(word1_high),
            ]
        )
    expected = torch.tensor(expected_rows, dtype=torch.int32)

    src = src_cpu.cuda()
    offsets = offsets_cpu.cuda()
    dst = torch.empty((len(vectors), 4), dtype=torch.int32, device="cuda")
    compiled = cute.compile(
        launch_converter,
        from_dlpack(src),
        from_dlpack(offsets),
        from_dlpack(dst),
    )
    compiled(from_dlpack(src), from_dlpack(offsets), from_dlpack(dst))
    torch.cuda.synchronize()

    actual = dst.cpu()
    if not torch.equal(actual, expected):
        first = int((actual != expected).any(dim=1).nonzero()[0])
        pytest.fail(
            f"PRMT row {first} mismatch: "
            f"actual={[_as_u32(value) for value in actual[first].tolist()]}, "
            f"expected={[_as_u32(value) for value in expected[first].tolist()]}"
        )


@pytest.mark.arch_hopper
@pytest.mark.parametrize("use_offset_variant", (False, True))
def test_k128_ldsm_conversion_and_rs_wgmma(
    use_offset_variant: bool,
) -> None:
    _require_hopper()

    tile_m, tile_n, tile_k = 128, 32, 128
    rows = torch.arange(tile_m, dtype=torch.int64)[:, None]
    columns = torch.arange(tile_k, dtype=torch.int64)[None, :]
    codes = ((rows * 3 + columns * 5 + columns // 17) % 16).to(torch.uint8)
    canonical_a = _pack_codes(codes).unsqueeze(0)
    groups = torch.arange(tile_k // 32, dtype=torch.int64)[None, :]
    offsets = ((rows * 5 + groups * 7) % 12 + 1).to(torch.uint8).unsqueeze(0)

    packed_a = _reference_interleave_weight(canonical_a)[0].contiguous().cuda()
    folded_offsets = _reference_fold_offsets(offsets)[0].reshape(-1).contiguous().cuda()
    b_values = (
        (
            torch.arange(tile_n, dtype=torch.int64)[:, None] * 11
            + torch.arange(tile_k, dtype=torch.int64)[None, :] * 3
        )
        % 7
        - 3
    ).to(torch.float32)
    b = b_values.to(torch.float8_e4m3fn).cuda()
    expanded_a = torch.zeros((tile_m, tile_k), dtype=torch.float8_e4m3fn, device="cuda")
    c = torch.zeros((tile_m, tile_n), dtype=torch.float32, device="cuda")

    probe = _Mxfp4RsK128Probe(use_offset_variant=use_offset_variant)
    compiled = cute.compile(
        probe,
        from_dlpack(packed_a),
        from_dlpack(folded_offsets),
        from_dlpack(b),
        from_dlpack(expanded_a),
        from_dlpack(c),
    )
    compiled(
        from_dlpack(packed_a),
        from_dlpack(folded_offsets),
        from_dlpack(b),
        from_dlpack(expanded_a),
        from_dlpack(c),
    )
    torch.cuda.synchronize()

    expected_bytes = _reference_e2m1_to_e4m3_bytes(codes.unsqueeze(0), offsets)[0]
    actual_bytes = expanded_a.cpu().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)

    a_reference = expected_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    expected_c = a_reference @ b.cpu().to(torch.float32).T
    torch.testing.assert_close(c.cpu(), expected_c, rtol=3.0e-4, atol=0.6)


@pytest.mark.arch_hopper
def test_hybrid_common_scale_recovers_original_mxfp4_values() -> None:
    _require_hopper(require_primitives=False)

    rows, logical_k = 4, 128
    row_ids = torch.arange(rows, dtype=torch.int64)[:, None]
    column_ids = torch.arange(logical_k, dtype=torch.int64)[None, :]
    codes = ((row_ids * 5 + column_ids * 3) % 16).to(torch.uint8)
    offsets = torch.tensor([1, 4, 8, 12], dtype=torch.uint8).view(1, 1, 4)
    offsets = offsets.expand(1, rows, 4).contiguous()

    converted_bytes = _reference_e2m1_to_e4m3_bytes(codes.unsqueeze(0), offsets)[0]
    converted = converted_bytes.view(torch.float8_e4m3fn).to(torch.float32)

    base_exponent = 120
    residual_times_64 = (2.0 ** (base_exponent - 128)) * 64.0
    magnitude_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    magnitude = magnitude_lut[(codes & 0x7).long()]
    signed_e2m1 = torch.where((codes & 0x8) != 0, -magnitude, magnitude)
    exponent = (
        base_exponent + offsets[0].repeat_interleave(32, dim=-1).to(torch.int32) - 1
    )
    original_weight = signed_e2m1 * torch.exp2(exponent.to(torch.float32) - 127.0)

    # The Humming offset is consumed inside WGMMA.  One expert residual and
    # one token activation scale remain, so both are applied once after the
    # complete K128 accumulator.
    torch.testing.assert_close(
        converted * residual_times_64,
        original_weight,
        rtol=0.0,
        atol=0.0,
    )

    activation = (
        (
            (
                torch.arange(3, dtype=torch.int64)[:, None] * 7
                + torch.arange(logical_k, dtype=torch.int64)[None, :] * 5
            )
            % 9
            - 4
        )
        .to(torch.float32)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .cuda()
    )
    token_scale = torch.tensor([0.125, 1.0, 3.5], device="cuda")
    converted_cuda = converted.cuda()
    original_cuda = original_weight.cuda()
    raw_accumulator = converted_cuda @ activation.T
    hybrid_output = raw_accumulator * residual_times_64 * token_scale.unsqueeze(0)
    reference_output = original_cuda @ (activation * token_scale[:, None]).T
    torch.testing.assert_close(
        hybrid_output, reference_output, rtol=2.0e-6, atol=2.0e-5
    )


@pytest.mark.arch_hopper
def test_fc1_handoff_reciprocal_then_multiply_rounding_oracle() -> None:
    _require_hopper(require_primitives=False)

    swiglu = torch.zeros((1, 64), dtype=torch.float32, device="cuda")
    swiglu[0, 0] = -976.5386352539062
    scale = torch.tensor([[4.521012306213379]], device="cuda")

    reciprocal = torch.ones_like(scale) / scale
    handoff = (swiglu * reciprocal).to(torch.float8_e4m3fn)
    division_order = (swiglu / scale).to(torch.float8_e4m3fn)

    assert int(handoff.view(torch.uint8)[0, 0]) == 0xF5
    assert int(division_order.view(torch.uint8)[0, 0]) == 0xF6


@pytest.mark.arch_hopper
def test_fc1_handoff_has_independent_k64_scales_and_exact_bytes() -> None:
    _require_hopper(require_primitives=False)

    quantize_fp8_per_token_block = _public_api().quantize_fp8_per_token_block
    source = torch.zeros((2, 128), dtype=torch.float32, device="cuda")
    source[0, :64] = 1.0
    source[0, 0] = -17.0
    source[0, 64:] = 32.0
    source[1, :64] = torch.arange(64, device="cuda", dtype=torch.float32) / 8.0
    source[1, 64:] = -4.0

    quantized, scale = quantize_fp8_per_token_block(
        source,
        torch.float8_e4m3fn,
        block_k=64,
        use_reciprocal_multiply=True,
    )
    blocks = source.reshape(2, 2, 64)
    expected_scale = blocks.abs().amax(dim=-1) / torch.finfo(torch.float8_e4m3fn).max
    expected_quantized = (
        (blocks * torch.reciprocal(expected_scale.unsqueeze(-1)))
        .reshape(2, 128)
        .to(torch.float8_e4m3fn)
    )

    assert scale.shape == (2, 2)
    torch.testing.assert_close(scale, expected_scale, rtol=0.0, atol=0.0)
    assert torch.equal(
        quantized.contiguous().view(torch.uint8),
        expected_quantized.contiguous().view(torch.uint8),
    )

    changed = source.clone()
    changed[:, 64:] *= 0.125
    changed_quantized, changed_scale = quantize_fp8_per_token_block(
        changed,
        torch.float8_e4m3fn,
        block_k=64,
        use_reciprocal_multiply=True,
    )
    assert torch.equal(scale[:, 0], changed_scale[:, 0])
    assert torch.equal(
        quantized[:, :64].contiguous().view(torch.uint8),
        changed_quantized[:, :64].contiguous().view(torch.uint8),
    )
    assert not torch.equal(scale[:, 1], changed_scale[:, 1])
