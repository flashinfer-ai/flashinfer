"""Bitwise equivalence test for the fused FP8 block-scale activation quant in the
CUTLASS MoE backend (port of TensorRT-LLM #16849).

The FLASHINFER_MOE_FUSED_BLOCKSCALE_QUANT toggle is read when the block-scale GEMM
runner is constructed, so each arm runs in a subprocess with the env var set, writes
its outputs to a file, and the parent asserts bit-identity.

Requires SM90: the fusion is SM90-only; on other archs the toggle is inert and the
test is skipped.
"""

import os
import subprocess
import sys
import tempfile

import pytest
import torch

TOKEN_COUNTS = [64, 1452, 4096]

_WORKER = """
import sys, torch
from flashinfer.fused_moe import cutlass_fused_moe

out_path = sys.argv[1]
H, I, E, K = 4096, 1536, 32, 8

def build_inputs(T, seed=0):
    \"\"\"DeepSeek-style FP8 block-scale MoE inputs at token count T.\"\"\"
    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    # fp8 block-quantized weights + 128x128 scales ([E, N/128, K/128] float32)
    w1 = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w2 = (torch.randn(E, H, I, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w1s = torch.rand(E, 2 * I // 128, H // 128, device=dev, dtype=torch.float32) * 0.02
    w2s = torch.rand(E, H // 128, I // 128, device=dev, dtype=torch.float32) * 0.02
    logits = torch.randn(T, E, device=dev)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return x, w1, w2, w1s, w2s, topk_ids.to(torch.int), topk_weights.to(torch.float32)

results = {}
for T in TOKEN_COUNTS:
    x, w1, w2, w1s, w2s, ids, wts = build_inputs(T)
    out = cutlass_fused_moe(
        input=x,
        token_selected_experts=ids,
        token_final_scales=wts,
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        output_dtype=torch.bfloat16,
        quant_scales=[w1s, w2s],
        use_deepseek_fp8_block_scale=True,
    )
    out = out[0] if isinstance(out, (list, tuple)) else out
    torch.cuda.synchronize()
    results[T] = out.cpu()
torch.save(results, out_path)
""".replace("TOKEN_COUNTS", repr(TOKEN_COUNTS))


def _run_arm(toggle: str, out_path: str) -> None:
    """Run one toggle arm in a subprocess and save its MoE outputs."""
    env = dict(os.environ, FLASHINFER_MOE_FUSED_BLOCKSCALE_QUANT=toggle)
    subprocess.run([sys.executable, "-c", _WORKER, out_path], env=env, check=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="fused block-scale activation quant is SM90-only",
)
def test_fused_blockscale_quant_bitwise_identical():
    """Fused (toggle=1) and unfused (toggle=0) MoE outputs must be bit-identical."""
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.pt")
        fused = os.path.join(d, "fused.pt")
        _run_arm("0", base)
        _run_arm("1", fused)
        a, b = torch.load(base), torch.load(fused)
        for t in TOKEN_COUNTS:
            mism = (a[t].view(torch.uint16) != b[t].view(torch.uint16)).sum().item()
            assert mism == 0, f"T={t}: {mism}/{a[t].numel()} values differ bitwise"
