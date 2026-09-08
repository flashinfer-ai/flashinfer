"""Host-side contracts for the SM120 MXFP4 x MXFP8 MegaMoE backend."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from flashinfer.moe_ep import BootstrapConfig, FleetParams
from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cutedsl import (
    Sm120Mxfp4Mxfp8CutedslMegaKernelBackend,
    Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm120.split_cutedsl_megakernel.shim.weights import (
    interleave_gate_up_8,
    to_blocked,
)


def test_config_uses_post_swiglu_intermediate() -> None:
    config = Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=2048,
        top_k=6,
    )
    assert config.intermediate_size == 2048
    assert config.kernel_name == "sm120_mxfp4_mxfp8_bf16_cutedsl"


def test_gate_up_interleave_is_grouped_in_eight_rows() -> None:
    rows = torch.arange(32, dtype=torch.int64).view(1, 32, 1)
    actual = interleave_gate_up_8(rows, full_width=32).flatten().tolist()
    assert actual == list(range(0, 8)) + list(range(16, 24)) + list(
        range(8, 16)
    ) + list(range(24, 32))


def test_scale_swizzle_preserves_all_bytes() -> None:
    scale = torch.arange(128 * 4, dtype=torch.int64).view(128, 4)
    blocked = to_blocked(scale)
    assert blocked.numel() == scale.numel()
    assert torch.equal(torch.sort(blocked).values, torch.sort(scale.flatten()).values)


def test_workspace_pool_key_covers_geometry(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    fleet = FleetParams(
        num_experts=256,
        max_tokens_per_rank=8192,
        token_hidden_size=4096,
    )

    def key(*, intermediate: int = 4096, top_k: int = 6):
        backend = Sm120Mxfp4Mxfp8CutedslMegaKernelBackend(
            Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=top_k,
            )
        )
        backend.bind_ep_bootstrap(
            BootstrapConfig(
                world_size=4,
                rank=1,
                auto_bootstrap=False,
            )
        )
        return backend._workspace_pool_key(fleet)

    assert key() == key()
    assert key(intermediate=2048) != key()
    assert key(top_k=8) != key()


def test_production_modules_do_not_import_mega_runner() -> None:
    package = Path(__file__).parents[2] / "flashinfer" / "moe_ep"
    roots = (
        package
        / "backends"
        / "mega"
        / "kernel"
        / "sm120"
        / "mxfp4_mxfp8_bf16_cutedsl",
        package
        / "kernel_src"
        / "sm120"
        / "split_cutedsl_megakernel"
        / "shim",
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
