"""Adapter imports must resolve without initializing CUDA."""

import subprocess
import sys

import pytest


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "package,child,symbol",
    [
        (
            "shims.sm100.cutedsl_megamoe.pre_unification",
            "nvfp4",
            "MegaMoENvfp4Config",
        ),
        (
            "shims.sm100.cutedsl_megamoe.sources",
            "genphase",
            "Sm100GenphaseMoeConfig",
        ),
        (
            "shims.sm107.cutedsl_megamoe",
            "block_scaled",
            "Sm107BlockScaledMoeConfig",
        ),
        (
            "shims.sm90.cutedsl_megamoe",
            "hopper_fp8",
            "MegaMoEHopperFp8Config",
        ),
        (
            "shims.sm120.split_cutedsl",
            "runtime",
            "MegaMoESm120W4A8Config",
        ),
        (
            "shims.sm120.swapab_cutedsl",
            "mxfp8",
            "MegaMoESm120Mxfp8Config",
        ),
        (
            "shims.sm100.kimi_k3",
            "weights",
            "KimiWeights",
        ),
        (
            "shims.sm90.cake_bf16",
            "cake_weights",
            "Sm90CakeBf16Weights",
        ),
        (
            "shims.sm120.cake_mxfp4",
            "cake_jit",
            "gen_cake_sm120_megamoe_module",
        ),
        (
            "shims.sm90.push_style_megamoe",
            "protocol",
            "Sm90PushConfig",
        ),
        (
            "kernels.flashinfer.bf16_nvfp4",
            "frontend",
            "MegaMoEBf16Nvfp4Config",
        ),
        (
            "kernels.flashinfer.bf16_rank_major",
            "session",
            "BlackwellBf16RankMajorSession",
        ),
    ],
)
def test_canonical_adapter_import(package, child, symbol):
    # Legacy providers claim process-wide raw package names, so isolate them.
    script = """
import importlib
import sys
import torch

package, child, symbol = sys.argv[1:]
package = "flashinfer.moe_ep." + package
module = importlib.import_module(package)
child_module = importlib.import_module(package + "." + child)
assert getattr(module, symbol) is getattr(child_module, symbol)
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, package, child, symbol],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cpu_only
def test_blackwell_adapter_module_cli_accepts_help():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--val-atol" in result.stdout
