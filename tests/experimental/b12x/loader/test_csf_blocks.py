"""Exported block indexes preserve checkpoint bytes across TP boundaries."""

from dataclasses import replace

import numpy as np
import pytest

from b12x.loader._csf_blocks import (
    CsfBlockPlane,
    export_csf_blocks,
    group_block_offsets,
)


def fixture(rows, columns, quantization, *, dense=False, empty=False):
    rng = np.random.default_rng(91)
    bits = 4 if quantization == "nvfp4" else 1
    bases = rng.integers(0, 257 - (1 << bits), size=rows, dtype=np.uint8)
    codes = rng.integers(0, 1 << bits, size=(rows, columns), dtype=np.uint8)
    logical = bases[:, None] + codes
    selected = (
        np.ones_like(logical, dtype=bool) if dense else rng.random(logical.shape) < 0.03
    )
    selected[::32, ::2] = True
    selected[31::32, 1::2] = True
    if empty:
        selected[:] = False
    logical[selected] = rng.integers(0, 256, size=int(selected.sum()), dtype=np.uint8)
    positions = np.flatnonzero(selected).astype(np.uint32)
    exceptions = positions | (logical.ravel()[positions].astype(np.uint32) << 24)
    code_bits = ((codes[:, :, None] >> np.arange(bits)) & 1).reshape(rows, -1)
    packed = np.packbits(code_bits, axis=1, bitorder="little")
    fixed = np.concatenate(
        (bases.reshape(-1, 16), packed.reshape(rows // 16, -1)), axis=1
    )
    return fixed, exceptions, logical


def decode(chunk):
    """Decode with scalar tile traversal independent of the export permutation."""
    result = np.empty((chunk.rows, chunk.columns), dtype=np.uint8)
    block = group = code_offset = 0
    bits = 4 if chunk.quantization == "nvfp4" else 1
    for tile_r in range(0, chunk.rows, 128):
        for tile_c in range(0, chunk.columns, 4):
            for c in range(tile_c, min(tile_c + 4, chunk.columns), 2):
                for r in range(tile_r, min(tile_r + 128, chunk.rows), 32):
                    width = min(2, chunk.columns - c)
                    count = 32 * width * bits // 8
                    code = chunk.codes[code_offset : code_offset + count]
                    fields = np.unpackbits(code, bitorder="little").reshape(-1, bits)
                    offsets = (fields * (1 << np.arange(bits))).sum(axis=1)
                    values = (
                        chunk.bases[r : r + 32, None] + offsets.reshape(32, width)
                    ).astype(np.uint8)
                    page = group // 32
                    offset = block + page
                    start, end = map(int, chunk.offsets[offset : offset + 2])
                    origin = int(chunk.page_bases[page]) // 2
                    for record in chunk.exceptions[origin + start : origin + end]:
                        row, column = divmod(int(record) & 255, 2)
                        values[row, column] = int(record) >> 8
                    result[r : r + 32, c : c + width] = values
                    block += 1
                    code_offset += count
            group += 1
    assert code_offset == chunk.codes.size
    return result


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize(
    "hidden,intermediate", [(2560, 640), (4096, 2048), (5120, 2304)]
)
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("down", [False, True])
def test_tp_partitions_reconstruct_exact_scale_bytes(
    tp, down, hidden, intermediate, quantization
):
    group_size = 16 if quantization == "nvfp4" else 32
    if intermediate // tp % 32 or intermediate // tp % group_size:
        pytest.skip("Partition does not contain whole 32-row blocks")
    rows, columns = (
        (hidden, intermediate // group_size)
        if down
        else (intermediate, hidden // group_size)
    )
    fixed, exceptions, logical = fixture(rows, columns, quantization)
    for rank in range(tp):
        rr = (0, rows) if down else (rank * rows // tp, (rank + 1) * rows // tp)
        cr = (
            (rank * columns // tp, (rank + 1) * columns // tp) if down else (0, columns)
        )
        chunk = export_csf_blocks(
            fixed,
            exceptions,
            rows=rows,
            columns=columns,
            quantization=quantization,
            row_range=rr,
            column_range=cr,
        )
        chunk.validate()
        np.testing.assert_array_equal(
            decode(chunk), logical[rr[0] : rr[1], cr[0] : cr[1]]
        )


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("dense", [False, True])
def test_full_index_page_and_short_terminal_page(dense, quantization):
    fixed, exceptions, logical = fixture(
        160, 160, quantization, dense=dense, empty=not dense
    )
    chunk = export_csf_blocks(
        fixed, exceptions, rows=160, columns=160, quantization=quantization
    )
    chunk.validate()
    block_bytes = 32 if quantization == "nvfp4" else 8
    assert chunk.codes.shape == (400 * block_bytes,)
    assert chunk.offsets.shape == (403,)
    if dense:
        assert chunk.offsets[256] == 16384
        assert chunk.page_bases[1] == 32768
    np.testing.assert_array_equal(decode(chunk), logical)


def test_rejects_exception_indexes_outside_their_blocks():
    fixed, exceptions, _ = fixture(32, 2, "nvfp4", dense=True)
    chunk = export_csf_blocks(
        fixed, exceptions, rows=32, columns=2, quantization="nvfp4"
    )
    records = chunk.exceptions.copy()
    records[0] |= 64
    with pytest.raises(ValueError, match="positions"):
        replace(chunk, exceptions=records).validate()


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("columns", [1, 2, 3, 5, 9, 10])
def test_short_groups_keep_complete_exception_spans_in_one_page(quantization, columns):
    rows = 2688
    fixed, exceptions, logical = fixture(rows, columns, quantization, dense=True)
    plane = export_csf_blocks(
        fixed, exceptions, rows=rows, columns=columns, quantization=quantization
    )
    plane.validate()
    groups = group_block_offsets(rows, columns)
    for group, (begin, end) in enumerate(zip(groups[:-1], groups[1:], strict=True)):
        page = group // 32
        start, stop = plane.offsets[[begin + page, end + page]].astype(np.int64)
        assert 0 <= start < stop <= 16384
    np.testing.assert_array_equal(decode(plane), logical)


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
def test_stacked_up_gate_seam_is_inside_a_group(quantization):
    fixed, exceptions, logical = fixture(640, 160, quantization, dense=True)
    plane = export_csf_blocks(
        fixed, exceptions, rows=640, columns=160, quantization=quantization
    )
    plane.validate()
    decoded = decode(plane)
    np.testing.assert_array_equal(decoded[256:384], logical[256:384])
    assert plane.codes.size == 640 * 160 // (2 if quantization == "nvfp4" else 8)


@pytest.mark.parametrize("quantization,base", [("nvfp4", 241), ("mxfp4", 255)])
def test_rejects_overflowing_row_palette(quantization, base):
    fixed, exceptions, _ = fixture(32, 2, quantization)
    fixed[0, 0] = base
    with pytest.raises(ValueError, match="palette"):
        export_csf_blocks(
            fixed, exceptions, rows=32, columns=2, quantization=quantization
        )


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
def test_block_to_block_tp_slice_fusion_and_value_conversion(quantization):
    from b12x.loader._csf_blocks import (
        concatenate_csf_blocks,
        slice_csf_blocks,
        transform_csf_blocks,
    )

    fixed, exceptions, expected = fixture(640, 19, quantization, dense=True)
    plane = export_csf_blocks(
        fixed, exceptions, rows=640, columns=19, quantization=quantization
    )
    first = slice_csf_blocks(plane, row_range=(32, 192), column_range=(9, 19))
    second = slice_csf_blocks(plane, row_range=(320, 480), column_range=(9, 19))
    fused = concatenate_csf_blocks((first, second))
    logical = np.concatenate((expected[32:192, 9:19], expected[320:480, 9:19]))
    np.testing.assert_array_equal(decode(fused), logical)
    table = ((np.arange(256, dtype=np.uint16) * 3) % 251).astype(np.uint8)
    converted = transform_csf_blocks(fused, table)
    converted.validate()
    np.testing.assert_array_equal(decode(converted), table[logical])


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
def test_tp_weight_padding_preserves_logical_scales(quantization):
    from b12x.loader._csf_blocks import pad_csf_blocks

    fixed, exceptions, expected = fixture(160, 3, quantization, dense=True)
    plane = export_csf_blocks(
        fixed, exceptions, rows=160, columns=3, quantization=quantization
    )
    padded = pad_csf_blocks(plane, rows=192, columns=5, row_base=56)
    padded.validate()
    actual = decode(padded)
    np.testing.assert_array_equal(actual[:160, :3], expected)
    np.testing.assert_array_equal(
        actual[:160, 3:], np.repeat(plane.bases[:, None], 2, 1)
    )
    assert np.all(actual[160:] == 56)


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
def test_block_fusion_keeps_positions_above_legacy_24_bit_limit(quantization):
    from b12x.loader._csf_blocks import (
        _from_logical_components,
        concatenate_csf_blocks,
        logical_csf_components,
        slice_csf_blocks,
    )

    rows, columns = 8192, 1025
    bases = np.full(rows, 120, dtype=np.uint8)
    fields = np.zeros((rows, columns), dtype=np.uint8)
    positions = np.array([rows * columns - 1], dtype=np.int64)
    source = _from_logical_components(
        bases, fields, positions, np.array([37], dtype=np.uint8), quantization
    )
    fused = concatenate_csf_blocks((source, source))
    fused.validate()
    _, _, actual, values = logical_csf_components(fused)
    np.testing.assert_array_equal(actual, [rows * columns - 1, 2 * rows * columns - 1])
    np.testing.assert_array_equal(values, [37, 37])
    tail = slice_csf_blocks(fused, row_range=(2 * rows - 32, 2 * rows))
    assert decode(tail)[-1, -1] == 37


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
def test_checkpoint_conversion_preserves_scales_and_uncompressed_tensors(
    tmp_path, quantization
):
    import json
    from pathlib import Path
    import subprocess
    import sys

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    rows, columns = 160, 10
    fixed, exceptions, logical = fixture(rows, columns, quantization, dense=True)
    source, destination = tmp_path / "source", tmp_path / "blocks"
    (source / "tensors").mkdir(parents=True)
    prefix = "model.layers.0.mlp.experts.0.up_proj"
    weight = (
        torch.arange(rows * columns * (8 if quantization == "nvfp4" else 16))
        .to(torch.uint8)
        .reshape(rows, -1)
    )
    calibration = torch.tensor([0.125], dtype=torch.float32)
    norm = torch.arange(32, dtype=torch.bfloat16)
    first = {
        f"{prefix}.weight": weight,
        f"{prefix}.weight_scale_2": calibration,
        f"{prefix}.weight_scale.{quantization}_csf_fixed": torch.from_numpy(fixed),
    }
    second = {
        f"{prefix}.weight_scale.{quantization}_csf_exceptions": torch.from_numpy(
            exceptions
        ),
        "model.norm.weight": norm,
    }
    mapping = {}
    for number, tensors in enumerate((first, second)):
        filename = f"tensors/part-{number}.safetensors"
        save_file(tensors, source / filename)
        mapping.update({k: filename for k in tensors})
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    (source / "config.json").write_text(
        json.dumps({"quantization_config": {"weight_scale_encoding": "csf"}})
    )
    (source / "NOTICE").write_text("Model attribution.\n")
    script = Path(__file__).resolve().parents[4] / "scripts" / "convert_csf_blocks.py"
    result = subprocess.run(
        [sys.executable, str(script), str(source), str(destination), "--workers", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert destination.stat().st_mode == source.stat().st_mode
    updated = json.loads((destination / "model.safetensors.index.json").read_text())
    tensors = {}
    for filename in set(updated["weight_map"].values()):
        with safe_open(destination / filename, framework="pt") as stream:
            tensors.update({name: stream.get_tensor(name) for name in stream.keys()})
    torch.testing.assert_close(tensors[f"{prefix}.weight"], weight, rtol=0, atol=0)
    torch.testing.assert_close(
        tensors[f"{prefix}.weight_scale_2"], calibration, rtol=0, atol=0
    )
    torch.testing.assert_close(tensors["model.norm.weight"], norm, rtol=0, atol=0)
    components = {
        name: tensors[f"{prefix}.weight_scale.csf32_{name}"].numpy()
        for name in ("bases", "codes", "page_bases", "offsets", "exceptions")
    }
    decoded = CsfBlockPlane(rows, columns, quantization, **components)
    decoded.validate()
    np.testing.assert_array_equal(decode(decoded), logical)
    assert updated["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in tensors.values()
    )
    assert (
        json.loads((destination / "config.json").read_text())["quantization_config"][
            "weight_scale_encoding"
        ]
        == "csf_32x2_v1"
    )
    assert (destination / "NOTICE").read_bytes() == (source / "NOTICE").read_bytes()
    assert not any(
        "_csf_fixed" in name or "_csf_exceptions" in name for name in tensors
    )
