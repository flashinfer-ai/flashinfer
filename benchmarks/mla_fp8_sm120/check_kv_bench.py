import json
from pathlib import Path
import torch
from kv_bench_kernels import ExperimentalMLA, quantize_rows, dequantize

torch.manual_seed(47)
torch.set_num_threads(4)
q = torch.randn(2, 20, 576, device="cuda", dtype=torch.bfloat16)
kv = torch.randn(22, 16, 576, device="cuda", dtype=torch.bfloat16)
kv8, scales = quantize_rows(kv)
pages = torch.randperm(22, device="cuda", dtype=torch.int32).reshape(2, 11).contiguous()
lengths = torch.tensor([63, 173], dtype=torch.int32, device="cuda")
decoded = dequantize(kv8, scales, torch.empty_like(kv))


def ref(qq, kk):
    qq, kk, pp = qq.cpu().double(), kk.cpu().double(), pages.cpu().long()
    outs = []
    for b, length in enumerate(lengths.tolist()):
        logical = kk[pp[b]].reshape(-1, 576)[:length]
        outs.append(((qq[b] @ logical.t()) / 16).softmax(-1) @ logical[:, :512])
    return torch.stack(outs)


original_ref = ref(q, kv)
results = []
for mode, bn in [(0, 16), (1, 16), (2, 64)]:
    runner = ExperimentalMLA(q, kv if mode == 0 else kv8, pages, lengths, mode, 4, bn)
    out = runner.run(scales)
    torch.cuda.synchronize()
    actual = out.cpu().double()
    if mode == 0:
        arithmetic_ref = original_ref
    elif mode == 1:
        arithmetic_ref = ref(q, decoded)
    else:
        q_dec = runner.q8.float() * runner.qs[:, :, None]
        kv_dec = kv8.float() * scales[:, :, None]
        arithmetic_ref = ref(q_dec, kv_dec)
    row = {
        "mode": mode,
        "bn": bn,
        "max_abs_error_vs_original": (actual - original_ref).abs().max().item(),
        "relative_l2_vs_original": (
            (actual - original_ref).norm() / original_ref.norm()
        ).item(),
        "relative_l2_vs_quantized_reference": (
            (actual - arithmetic_ref).norm() / arithmetic_ref.norm()
        ).item(),
        "shared_memory": runner.compiled_kernel.metadata.shared,
        "registers": runner.compiled_kernel.n_regs,
    }
    assert torch.isfinite(actual).all()
    assert row["relative_l2_vs_quantized_reference"] < (0.005 if mode < 2 else 0.05), (
        row
    )
    root = Path(__file__).resolve().parent
    (root / f"kv_bench_mode{mode}.ptx").write_text(runner.compiled_kernel.asm["ptx"])
    results.append(row)
    print(json.dumps(row), flush=True)
(root / "kv_bench_correctness.json").write_text(json.dumps(results, indent=2) + "\n")
print("PASS", flush=True)
