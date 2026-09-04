"""Host-side contracts for the SM120 NVFP4 x NVFP4 MegaMoE backend."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from flashinfer.moe_ep import BootstrapConfig, FleetParams
from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cutedsl import (
    Sm120Nvfp4Nvfp4CutedslMegaKernelBackend,
    Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
    select_graph_compile_bucket,
)
from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel.shim.weights import (
    interleave_gate_up_8,
    interleave_gate_up_16,
    scale_storage_size,
)


def test_config_uses_post_swiglu_intermediate() -> None:
    config = Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=2048,
        top_k=6,
    )
    assert config.intermediate_size == 2048
    assert config.kernel_name == "sm120_nvfp4_nvfp4_bf16_cutedsl"


def test_gate_up_interleave_is_grouped_in_eight_rows() -> None:
    rows = torch.arange(32, dtype=torch.int64).view(1, 32, 1)
    actual = interleave_gate_up_8(rows, full_width=32).flatten().tolist()
    assert actual == list(range(0, 8)) + list(range(16, 24)) + list(
        range(8, 16)
    ) + list(range(24, 32))


def test_legacy_gate_up_interleave_is_grouped_in_sixteen_rows() -> None:
    rows = torch.arange(64, dtype=torch.int64).view(1, 64, 1)
    actual = interleave_gate_up_16(rows, full_width=64).flatten().tolist()
    assert actual == (
        list(range(0, 16))
        + list(range(32, 48))
        + list(range(16, 32))
        + list(range(48, 64))
    )


def test_scale_storage_uses_block16_and_atom_padding() -> None:
    assert scale_storage_size(4096, 4096) == 4096 * 256
    assert scale_storage_size(33, 65) == 128 * 8


def test_decode_graph_compile_bucket_selection() -> None:
    capacity = 8192
    expected = {
        1: 7,
        7: 7,
        8: 16,
        16: 16,
        17: 32,
        127: 128,
        129: 168,
        168: 168,
        169: 256,
        256: 256,
        257: capacity,
    }
    for requested, bucket in expected.items():
        assert select_graph_compile_bucket(requested, capacity) == bucket


def test_workspace_pool_key_covers_nvfp4_contract(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    fleet = FleetParams(
        num_experts=256,
        max_tokens_per_rank=8192,
        token_hidden_size=4096,
    )

    def key(*, norm_const: float = 1.0, gate_up_clamp: float | None = None):
        backend = Sm120Nvfp4Nvfp4CutedslMegaKernelBackend(
            Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=4096,
                top_k=6,
                input_norm_const=norm_const,
                gate_up_clamp=gate_up_clamp,
            )
        )
        backend.bind_ep_bootstrap(
            BootstrapConfig(world_size=4, rank=1, auto_bootstrap=False)
        )
        return backend._workspace_pool_key(fleet)

    assert key() == key()
    assert key(norm_const=2.0) != key()
    assert key(gate_up_clamp=10.0) != key()


def test_production_modules_do_not_import_mega_runner() -> None:
    package = Path(__file__).parents[2] / "flashinfer" / "moe_ep"
    roots = (
        package / "backends" / "mega" / "kernel" / "sm120" / "nvfp4_nvfp4_bf16_cutedsl",
        package / "kernel_src" / "sm120" / "nvfp4_split_cutedsl_megakernel" / "shim",
    )
    offenders: list[str] = []
    for root in roots:
        for source in root.rglob("*.py"):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any("mega_runner" in name for name in names):
                    offenders.append(str(source.relative_to(package)))
    assert not offenders
