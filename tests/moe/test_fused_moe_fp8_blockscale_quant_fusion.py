"""Bitwise equivalence test for the fused FP8 block-scale activation quant in the
CUTLASS MoE backend (port of TensorRT-LLM #16849).

The FLASHINFER_MOE_FUSED_BLOCKSCALE_QUANT toggle is read when the block-scale GEMM
runner is constructed, so each arm runs in a subprocess with the env var set, writes
its outputs to a file, and the parent asserts bit-identity (cf. the ON/OFF equivalence
pattern used for PDL in test coverage of the FC1->FC2 quant launch).

Target shapes: DeepSeek-style FP8 block-scale MoE (group 1x128 activations, 128x128
weights), hidden 4096, intermediate 1536, 128 experts, top-8 — token counts spanning
decode to large prefill.

Requires SM90 (the fusion is SM90-only; on other archs the toggle is inert and the
test is skipped).
"""

import os
import subprocess
import sys
import tempfile

import pytest
import torch

TOKEN_COUNTS = [64, 1452, 4096, 17083]
HIDDEN = 4096
INTERMEDIATE = 1536
NUM_EXPERTS = 128
TOP_K = 8

_WORKER = r"""
import os, sys, torch
from flashinfer.fused_moe import cutlass_fused_moe

out_path, toggle = sys.argv[1], sys.argv[2]
assert os.environ.get("FLASHINFER_MOE_FUSED_BLOCKSCALE_QUANT") == toggle

torch.manual_seed(0)
dev = "cuda"
H, I, E, K = {HIDDEN}, {INTERMEDIATE}, {NUM_EXPERTS}, {TOP_K}

# FP8 block-scale weights: fc1 (2*I x H, gate+up), fc2 (H x I), 128x128 weight scales.
w1 = torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
w2 = torch.randn(E, H, I, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
w1_scale = torch.rand(E, 2 * I // 128, H // 128, device=dev, dtype=torch.float32) * 2
w2_scale = torch.rand(E, H // 128, I // 128, device=dev, dtype=torch.float32) * 2

results = {{}}
for T in {TOKEN_COUNTS}:
    x = (torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 3).contiguous()
    router = torch.randn(T, E, device=dev, dtype=torch.float32)
    weights, ids = torch.topk(torch.sigmoid(router), K, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
    out = cutlass_fused_moe(
        x, ids.to(torch.int), weights, w1, w2, torch.bfloat16,
        quant_scales=[w1_scale, w2_scale],
        use_deepseek_fp8_block_scale=True,
    )
    out = out[0] if isinstance(out, (list, tuple)) else out
    results[T] = out.cpu()
torch.save(results, out_path)
""".format(
    HIDDEN=HIDDEN,
    INTERMEDIATE=INTERMEDIATE,
    NUM_EXPERTS=NUM_EXPERTS,
    TOP_K=TOP_K,
    TOKEN_COUNTS=TOKEN_COUNTS,
)


def _run_arm(toggle: str, out_path: str) -> None:
    env = dict(os.environ, FLASHINFER_MOE_FUSED_BLOCKSCALE_QUANT=toggle)
    subprocess.run(
        [sys.executable, "-c", _WORKER, out_path, toggle], env=env, check=True
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="fusion is SM90-only",
)
def test_fused_blockscale_quant_bitwise_identical():
    with tempfile.TemporaryDirectory() as d:
        base, fused = os.path.join(d, "base.pt"), os.path.join(d, "fused.pt")
        _run_arm("0", base)
        _run_arm("1", fused)
        a, b = torch.load(base), torch.load(fused)
        for t in TOKEN_COUNTS:
            mism = (a[t].view(torch.uint16) != b[t].view(torch.uint16)).sum().item()
            assert mism == 0, f"T={t}: {mism}/{a[t].numel()} values differ bitwise"
