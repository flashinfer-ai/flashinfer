"""CPU-only codec and checkpoint tests; run with --confcutdir=scripts."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codec = load_module("encode_csf", ROOT / "scripts/encode_csf.py")
blocks = load_module(
    "csf_blocks_reference",
    ROOT / "flashinfer/experimental/b12x/loader/_csf_blocks.py",
)


@pytest.mark.parametrize("quantization,bits", [("mxfp4", 1), ("nvfp4", 4)])
@pytest.mark.parametrize(
    "shape",
    [(32, 1), (96, 7), (160, 10), (256, 133), (32, 4097), (3072, 112), (3584, 96)],
)
def test_exact_blocks_match_existing_exporter(quantization, bits, shape):
    rng = np.random.default_rng(532)
    source = rng.integers(0, 256, shape, dtype=np.uint8)
    parts = codec.encode_scales(source, quantization)
    plane = blocks.CsfBlockPlane(*shape, quantization, **parts)
    plane.validate()
    bases, fields, positions, values = blocks.logical_csf_components(plane)
    expanded = bases[:, None] + fields
    expanded.ravel()[positions] = values
    np.testing.assert_array_equal(expanded, source)

    # Recreate slab streams from independently chosen row palettes.
    width = 1 << bits
    expected_bases = np.array(
        [
            np.argmax(
                np.convolve(
                    np.bincount(row, minlength=256),
                    np.ones(width, dtype=int),
                    mode="valid",
                )
            )
            for row in source
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(parts["bases"], expected_bases)
    delta = source.astype(np.int16) - expected_bases[:, None]
    outside = (delta < 0) | (delta >= width)
    delta[outside] = 0
    packed = np.packbits(
        ((delta[:, :, None] >> np.arange(bits)) & 1).reshape(shape[0], -1),
        axis=1,
        bitorder="little",
    )
    fixed = np.concatenate(
        (expected_bases.reshape(-1, 16), packed.reshape(shape[0] // 16, -1)), axis=1
    )
    positions = np.flatnonzero(outside).astype(np.uint32)
    exceptions = positions | (source.ravel()[positions].astype(np.uint32) << 24)
    expected = blocks.export_csf_blocks(
        fixed, exceptions, rows=shape[0], columns=shape[1], quantization=quantization
    )
    for component in codec.COMPONENTS:
        np.testing.assert_array_equal(parts[component], getattr(expected, component))


@pytest.mark.parametrize("quantization", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("value", [0, 127, 255])
def test_constant_rows_have_no_exceptions(quantization, value):
    source = np.full((160, 13), value, dtype=np.uint8)
    parts = codec.encode_scales(source, quantization)
    assert not parts["exceptions"].size
    np.testing.assert_array_equal(
        codec.decode_scales(parts, source.shape, quantization), source
    )


def checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rng = np.random.default_rng(91)
    scale_name = (
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale"
    )
    tensors = {
        scale_name: rng.integers(118, 122, (160, 10), dtype=np.uint8),
        scale_name.replace("weight_scale", "weight_packed"): rng.integers(
            0, 256, (160, 160), dtype=np.uint8
        ),
        "retained": np.array([1, -2, 3], dtype=np.float32),
        "empty": np.empty(0, dtype=np.uint16),
    }
    save_file(tensors, source / "model.safetensors", metadata={"format": "pt"})
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "text_config": {
                    "model_type": "kimi_linear",
                    "quantization_config": {"quant_method": "compressed-tensors"},
                },
            }
        )
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
                "weight_map": {name: "model.safetensors" for name in tensors},
            }
        )
    )
    (source / "tokenizer.json").write_text("{}\n")
    return source, scale_name, tensors


def test_checkpoint_writer_preserves_bytes_and_metadata(tmp_path):
    source, scale_name, tensors = checkpoint(tmp_path)
    output = tmp_path / "output"
    codec.convert(source, output, quantization="mxfp4", workers=1)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert scale_name not in index["weight_map"]
    with safe_open(output / "model.safetensors", framework="numpy") as reader:
        assert reader.metadata() == {"format": "pt"}
        for name, expected in tensors.items():
            if name != scale_name:
                np.testing.assert_array_equal(reader.get_tensor(name), expected)
        parts = {
            c: reader.get_tensor(f"{scale_name}.csf32_{c}") for c in codec.COMPONENTS
        }
        plane = blocks.CsfBlockPlane(160, 10, "mxfp4", **parts)
        plane.validate()
        bases, fields, positions, values = blocks.logical_csf_components(plane)
        expanded = bases[:, None] + fields
        expanded.ravel()[positions] = values
        np.testing.assert_array_equal(expanded, tensors[scale_name])
        assert index["metadata"]["total_size"] == sum(
            reader.get_tensor(n).nbytes for n in reader.keys()
        )
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["model_type"] == "kimi_linear"
    assert config["text_config"]["quantization_config"] == {
        "quant_method": "compressed-tensors",
        "weight_scale_encoding": "csf_32x2_v1",
    }
    assert (output / "tokenizer.json").read_bytes() == (
        source / "tokenizer.json"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        codec.convert(source, output, quantization="mxfp4")


def test_mismatched_quantization_rejected_before_output(tmp_path):
    source, _, _ = checkpoint(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="geometry or dtype"):
        codec.convert(source, output, quantization="nvfp4")
    assert not output.exists()
    assert not list(tmp_path.glob(".output.partial-*"))
