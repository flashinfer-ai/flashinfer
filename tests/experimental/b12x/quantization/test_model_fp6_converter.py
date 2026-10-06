from __future__ import annotations

import json
import pathlib

import pytest
import torch

from b12x.quantization.mxfp6.model_fp6 import (
    SafetensorsModel,
    convert_dense_model_to_fp6,
    convert_moe_model_to_fp6,
    discover_dense_linears,
    discover_moe_experts,
)
from b12x.quantization.mxfp6.fp6_safetensors_export import (
    export_dense_model_to_fp6_safetensors,
    export_moe_model_to_fp6_safetensors,
)

safetensors_torch = pytest.importorskip("safetensors.torch")


def _write_ckpt(path: pathlib.Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(
        {k: v.contiguous() for k, v in tensors.items()}, str(path / "model.safetensors")
    )
    (path / "config.json").write_text(json.dumps({"model_type": "test"}))


def _fake_moe(path: pathlib.Path, *, e: int, layers: int, k: int, n: int) -> None:
    t: dict[str, torch.Tensor] = {}
    pre = "model.language_model.layers.{L}.mlp"
    for L in range(layers):
        base = pre.format(L=L)
        for ei in range(e):
            t[f"{base}.experts.{ei}.gate_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
            t[f"{base}.experts.{ei}.up_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
            t[f"{base}.experts.{ei}.down_proj.weight"] = torch.randn(k, n, dtype=torch.bfloat16) * 0.1
        # decoys discovery must ignore:
        t[f"{base}.gate.weight"] = torch.randn(e, k, dtype=torch.bfloat16)
        t[f"model.language_model.layers.{L}.linear_attn.in_proj.weight"] = torch.randn(k, k, dtype=torch.bfloat16)
    t["visual.blocks.0.mlp.fc1.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
    _write_ckpt(path, t)


def _fake_moe_packed(path: pathlib.Path, *, e: int, layers: int, k: int, n: int) -> None:
    """Experts stacked in one 3-D tensor: experts.gate_up_proj / experts.down_proj."""
    t: dict[str, torch.Tensor] = {}
    pre = "model.language_model.layers.{L}.mlp"
    for L in range(layers):
        base = pre.format(L=L)
        t[f"{base}.experts.gate_up_proj"] = torch.randn(e, 2 * n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.experts.down_proj"] = torch.randn(e, k, n, dtype=torch.bfloat16) * 0.1
        t[f"{base}.gate.weight"] = torch.randn(e, k, dtype=torch.bfloat16)
        t[f"model.language_model.layers.{L}.input_layernorm.weight"] = torch.randn(k, dtype=torch.bfloat16)
    t["model.language_model.embed_tokens.weight"] = torch.randn(16, k, dtype=torch.bfloat16)
    _write_ckpt(path, t)


def _fake_dense(path: pathlib.Path, *, layers: int, k: int, n: int, attn_on: set[int]) -> None:
    t: dict[str, torch.Tensor] = {}
    for L in range(layers):
        base = f"model.language_model.layers.{L}"
        t[f"{base}.mlp.gate_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.mlp.up_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.mlp.down_proj.weight"] = torch.randn(k, n, dtype=torch.bfloat16) * 0.1
        if L in attn_on:
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                t[f"{base}.self_attn.{p}.weight"] = torch.randn(k, k, dtype=torch.bfloat16) * 0.1
    _write_ckpt(path, t)


def test_discover_moe_experts(tmp_path) -> None:
    _fake_moe(tmp_path / "m", e=4, layers=2, k=64, n=32)
    model = SafetensorsModel(tmp_path / "m")
    scheme = discover_moe_experts(model)
    assert scheme is not None
    assert scheme.num_experts == 4
    assert scheme.layers == [0, 1]
    assert scheme.gate_name == "gate_proj" and scheme.up_name == "up_proj"
    assert scheme.down_name == "down_proj" and not scheme.is_fused
    assert scheme.prefix_template == "model.language_model.layers.{L}.mlp"
    assert scheme.expert_key(1, 3, "down_proj") == (
        "model.language_model.layers.1.mlp.experts.3.down_proj.weight"
    )


def test_moe_dryrun_writes_nothing(tmp_path) -> None:
    _fake_moe(tmp_path / "m", e=4, layers=2, k=64, n=32)
    report = convert_moe_model_to_fp6(
        tmp_path / "m", tmp_path / "out", dry_run=True, device="cpu", verbose=False
    )
    assert report.arch == "moe"
    assert report.layers == [0, 1]
    assert report.tensors_written == 0
    assert not (tmp_path / "out").exists()


def test_moe_convert_cpu_roundtrip(tmp_path) -> None:
    _fake_moe(tmp_path / "m", e=2, layers=1, k=64, n=32)
    report = convert_moe_model_to_fp6(
        tmp_path / "m",
        tmp_path / "out",
        limit_layers=1,
        device="cpu",
        use_gpu=False,
        verbose=False,
    )
    assert report.tensors_written == 2
    assert (tmp_path / "out" / "manifest.json").is_file()
    art = tmp_path / "out" / "layer_0.moe_fp6.safetensors"
    assert art.is_file()

    from b12x.quantization.mxfp6 import load_fp6_moe_weights

    w = load_fp6_moe_weights(str(art), device="cpu")
    assert w.num_experts == 2 and w.k == 64 and w.n == 32
    assert w.w1_fp6.shape == (2, 64, 48)  # (E, 2N, 3K/4)
    assert w.w2_fp6.shape == (2, 64, 24)  # (E, K, 3N/4)


def test_discover_dense_and_dryrun(tmp_path) -> None:
    _fake_dense(tmp_path / "d", layers=2, k=64, n=128, attn_on={0})
    model = SafetensorsModel(tmp_path / "d")
    scheme = discover_dense_linears(model)
    assert scheme is not None
    assert scheme.layers == [0, 1]
    assert scheme.mlp_gate == "mlp.gate_proj" and scheme.mlp_down == "mlp.down_proj"
    assert scheme.mlp_fused_gate_up is None
    assert scheme.attn_projs == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]

    report = convert_dense_model_to_fp6(
        tmp_path / "d", tmp_path / "out_d", dry_run=True, device="cpu", verbose=False
    )
    assert report.arch == "dense"
    assert report.tensors_written == 0
    assert not (tmp_path / "out_d").exists()


def _read_ckpt(out: pathlib.Path) -> tuple[dict, dict, dict]:
    """Return (state_dict, index, config) for an exported FP6 safetensors dir."""
    index = json.loads((out / "model.safetensors.index.json").read_text())
    config = json.loads((out / "config.json").read_text())
    state: dict[str, torch.Tensor] = {}
    for shard in set(index["weight_map"].values()):
        state.update(safetensors_torch.load_file(str(out / shard)))
    return state, index, config


def test_export_moe_per_expert_safetensors(tmp_path) -> None:
    _fake_moe(tmp_path / "m", e=2, layers=1, k=64, n=32)
    out = tmp_path / "out"
    report = export_moe_model_to_fp6_safetensors(
        tmp_path / "m", out, limit_layers=1, device="cpu", use_gpu=False, verbose=False
    )
    assert report.arch == "moe"
    # 2 experts x 3 projections (gate/up/down) = 6 quantized linears.
    assert report.quantized_tensors == 6
    state, index, config = _read_ckpt(out)

    base = "model.language_model.layers.0.mlp.experts"
    for proj, (o, i) in {"gate_proj": (32, 64), "up_proj": (32, 64), "down_proj": (64, 32)}.items():
        w = state[f"{base}.0.{proj}.weight"]
        sc = state[f"{base}.0.{proj}.weight_scale"]
        assert w.dtype == torch.uint8 and tuple(w.shape) == (o, i * 3 // 4)
        assert sc.dtype == torch.uint8 and tuple(sc.shape) == (o, i // 32)
        assert f"{base}.0.{proj}.weight_scale_2" in state
        assert f"{base}.0.{proj}.input_scale" in state
    # Non-quantized tensors are copied through (router gate, decoy linear-attn).
    assert "model.language_model.layers.0.mlp.gate.weight" in state
    assert "model.language_model.layers.0.linear_attn.in_proj.weight" in state
    # quantization_config present with the W6A6 contract.
    qc = config["quantization_config"]
    assert qc["quant_method"] == "modelopt" and qc["quant_algo"] == "W6A6"
    assert qc["group_size"] == 32 and qc["exclude_modules"]


def test_export_moe_packed_expands_to_per_expert(tmp_path) -> None:
    _fake_moe_packed(tmp_path / "mp", e=2, layers=1, k=64, n=32)
    out = tmp_path / "outp"
    report = export_moe_model_to_fp6_safetensors(
        tmp_path / "mp", out, limit_layers=1, device="cpu", use_gpu=False, verbose=False
    )
    assert report.quantized_tensors == 6  # packed source -> per-expert keys
    state, _, _ = _read_ckpt(out)
    base = "model.language_model.layers.0.mlp.experts"
    for e in (0, 1):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"{base}.{e}.{proj}.weight" in state
    # Original packed keys must be gone (replaced by per-expert).
    assert f"{base}.gate_up_proj" not in state
    assert f"{base}.down_proj" not in state
    # Embedding copied through.
    assert "model.language_model.embed_tokens.weight" in state


def test_load_fp6_moe_checkpoint_cpu(tmp_path) -> None:
    """Export -> load round-trip: loader reconstructs FP6MoEWeights from the keys."""
    _fake_moe(tmp_path / "m", e=2, layers=1, k=64, n=32)
    out = tmp_path / "out"
    export_moe_model_to_fp6_safetensors(
        tmp_path / "m", out, limit_layers=1, device="cpu", use_gpu=False, verbose=False
    )
    from b12x.quantization.mxfp6.fp6_safetensors_load import load_fp6_moe_checkpoint

    layers = load_fp6_moe_checkpoint(str(out), device="cpu")
    assert set(layers) == {0}
    w = layers[0]
    assert w.num_experts == 2 and w.k == 64 and w.n == 32
    assert w.w1_fp6.shape == (2, 64, 48)  # (E, 2N, 3K/4)
    assert w.w2_fp6.shape == (2, 64, 24)  # (E, K, 3N/4)
    # Export default has been mxfp6_w6a8 (E2M3 weights, E4M3 activations)
    # since Jun 13; the loader must report it back verbatim.
    assert w.source_format == "mxfp6_w6a8" and w.weight_fmt == "e2m3"


def test_load_fp6_dense_checkpoint_cpu(tmp_path) -> None:
    """Export -> load round-trip for dense linears (loader reconstructs FP6DenseWeight)."""
    _fake_dense(tmp_path / "d", layers=1, k=128, n=128, attn_on={0})
    out = tmp_path / "outd2"
    export_dense_model_to_fp6_safetensors(
        tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False
    )
    from b12x.quantization.mxfp6.fp6_safetensors_load import load_fp6_dense_checkpoint

    weights = load_fp6_dense_checkpoint(str(out), device="cpu")
    name = "model.language_model.layers.0.mlp.gate_proj"
    assert name in weights
    w = weights[name]
    assert w.out_features == 128 and w.in_features == 128
    assert w.packed.shape == (128, 96)  # (out, 3*in/4)
    assert float(w.global_scale.reshape(-1)[0]) == 1.0
    # Attention projection also quantized + loadable.
    assert "model.language_model.layers.0.self_attn.q_proj" in weights


def test_export_dense_safetensors(tmp_path) -> None:
    _fake_dense(tmp_path / "d", layers=2, k=128, n=128, attn_on={0})
    out = tmp_path / "outd"
    report = export_dense_model_to_fp6_safetensors(
        tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False
    )
    assert report.arch == "dense"
    state, _, config = _read_ckpt(out)
    w = state["model.language_model.layers.0.mlp.gate_proj.weight"]
    assert w.dtype == torch.uint8 and tuple(w.shape) == (128, 128 * 3 // 4)
    assert "model.language_model.layers.0.mlp.gate_proj.weight_scale" in state
    assert config["quantization_config"]["quant_algo"] == "W6A6"
