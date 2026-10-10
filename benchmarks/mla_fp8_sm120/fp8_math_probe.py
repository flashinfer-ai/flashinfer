"""SM120 FP8xFP8 feasibility probe for GLM's 20-head MLA shapes.

This uses separate PyTorch scaled GEMMs, materializes attention scores, and pads
20 heads to 32. It is NOT a fused paged-attention kernel or a serving benchmark.
Both QK and PV use FP8 E4M3 operands with FP32 outputs/accumulation.
"""

import json
from pathlib import Path

import torch

torch.manual_seed(47)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
device = "cuda"
heads, padded_heads, length = 20, 32, 1024
q = torch.randn(heads, 576, device=device, dtype=torch.bfloat16).float()
kv = torch.randn(length, 576, device=device, dtype=torch.bfloat16).float()
q_padded = torch.nn.functional.pad(q, (0, 0, 0, padded_heads - heads))


def quantize(x):
    scale = x.abs().amax().clamp_min(1e-12) / 448.0
    return (x / scale).to(torch.float8_e4m3fn), scale


q8, sq = quantize(q_padded)
kv8, skv = quantize(kv)
# _scaled_mm requires the B operand in column-major layout.
v8_column_major = kv8[:, :512].t().contiguous().t()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    logits8 = torch._scaled_mm(
        q8,
        kv8.t(),
        scale_a=sq,
        scale_b=skv,
        out_dtype=torch.float32,
        use_fast_accum=False,
    ) * (1 / 16)
    p = logits8.softmax(dim=-1)
    p8, sp = quantize(p)
    out8 = torch._scaled_mm(
        p8,
        v8_column_major,
        scale_a=sp,
        scale_b=skv,
        out_dtype=torch.float32,
        use_fast_accum=False,
    )[:heads]
    torch.cuda.synchronize()

qd, kvd = q.cpu().double(), kv.cpu().double()
logits_ref = (qd @ kvd.t()) * (1 / 16)
out_ref = logits_ref.softmax(-1) @ kvd[:, :512]
actual = out8.cpu().double()
logits_actual = logits8[:heads].cpu().double()
# This branch isolates Q/K quantization effects before P/V quantization.
qk_only = logits_actual.softmax(-1) @ kvd[:, :512]
result = {
    "gpu": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "torch": torch.__version__,
    "shapes": {
        "heads": heads,
        "padded_heads": padded_heads,
        "kv_len": length,
        "qk_dim": 576,
        "v_dim": 512,
    },
    "operand_dtype": "float8_e4m3fn",
    "accum_output_dtype": "float32",
    "scale_policy": "per-tensor absmax/448; FP32 softmax; independent P scale",
    "q_scale": sq.item(),
    "kv_scale": skv.item(),
    "p_scale": sp.item(),
    "qk_max_abs_error": (logits_actual - logits_ref).abs().max().item(),
    "qk_only_output_relative_l2": ((qk_only - out_ref).norm() / out_ref.norm()).item(),
    "both_gemms_fp8_output_max_abs_error": (actual - out_ref).abs().max().item(),
    "both_gemms_fp8_output_relative_l2": (
        (actual - out_ref).norm() / out_ref.norm()
    ).item(),
    "all_finite": bool(torch.isfinite(actual).all()),
    "cuda_gemm_kernel_names": sorted(
        set(
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
            and "gemm" in e.name.lower()
        )
    ),
    "scope": "Feasibility on synthetic BF16-origin inputs; not model accuracy or fused-kernel performance.",
}
root = Path(__file__).resolve().parent
prof.export_chrome_trace(str(root / "trace_fp8_math.json"))
(root / "fp8_math_results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
