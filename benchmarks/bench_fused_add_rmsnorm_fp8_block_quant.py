"""Benchmark: fused Add+RMSNorm+1x128-fp8-block-quant (one kernel) vs the 2-kernel producer
(flashinfer.fused_add_rmsnorm then a per-token-group 1x128 fp8 quant), which is what runs before
fp8 block-scaled GEMMs in deepseek_v2-family / MoE models.

Shows the kernel-level win: the fused kernel does add+norm+quant in a single memory pass, saving
the intermediate bf16 round-trip.
"""

import numpy as np
import torch

import flashinfer
from flashinfer.quantization import per_token_group_quant_8bit
from flashinfer.testing import bench_gpu_time


def _round_up(x, m):
    return (x + m - 1) // m * m


def _run_fused(
    out, block_scale, normed_out, input, residual, weight, eps, enable_pdl=False
):
    """Call the fused op via the public API if present, else the JIT module export directly."""
    if hasattr(flashinfer, "fused_add_rmsnorm_fp8_block_quant"):
        flashinfer.fused_add_rmsnorm_fp8_block_quant(
            out,
            block_scale,
            normed_out,
            input,
            residual,
            weight,
            eps,
            enable_pdl=enable_pdl,
        )
    else:
        flashinfer.norm.get_norm_module().fused_add_rmsnorm_fp8_block_quant(
            out, block_scale, normed_out, input, residual, weight, eps, enable_pdl
        )


def _ref_group_quant_fp8(normed, block=128):
    """Reference 1x128 per-token-group fp8 quant (torch) -> (fp8, scale[M, H/128])."""
    M, H = normed.shape
    x = normed.float().view(M, H // block, block)
    amax = x.abs().amax(dim=-1).clamp_min(1e-4)
    scale = amax / 448.0
    q = (x / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(M, H), scale


def bench_one(M, H, scale_ue8m0=False, dtype=torch.bfloat16):
    """Median time per call (CUDA graph, cold L2) of the 2-kernel producer and of the fused
    kernel with and without normed_out. With scale_ue8m0 the fused kernel writes packed UE8M0
    scales, while the baseline quant writes power-of-two fp32 scales that SM100 DeepGEMM still
    has to pack."""
    dev = "cuda"
    input = torch.randn(M, H, dtype=dtype, device=dev) * 0.1
    residual = torch.randn(M, H, dtype=dtype, device=dev) * 0.1
    weight = torch.randn(H, dtype=dtype, device=dev)
    eps = 1e-6

    out = torch.empty(M, H, dtype=torch.float8_e4m3fn, device=dev)
    num_blocks = H // 128
    block_scale = torch.empty(
        (num_blocks + 3) // 4 if scale_ue8m0 else num_blocks,
        _round_up(M, 4),
        dtype=torch.int32 if scale_ue8m0 else torch.float32,
        device=dev,
    )
    normed_out = torch.empty(M, H, dtype=dtype, device=dev)

    # Timed under CUDA graphs so host gaps between the two baseline kernels do not count; the
    # rotated input/residual copies keep L2 cold. Memory traffic does not depend on the values.
    def baseline(input, residual):
        flashinfer.fused_add_rmsnorm(input, residual, weight, eps)  # input <- normed
        per_token_group_quant_8bit(
            input,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
            backend="cuda",
        )

    def fused(input, residual, normed=None):
        _run_fused(out, block_scale, normed, input, residual, weight, eps)

    def us(fn):
        times = bench_gpu_time(
            fn, input_args=(input, residual), use_cuda_graph=True, cold_l2_cache=True
        )
        return 1e3 * np.median(times)

    t_b = us(baseline)
    t_f = us(lambda i, r: fused(i, r, normed_out))
    t_n = us(fused)
    print(
        f"M={M:>5} H={H:>5} ue8m0={int(scale_ue8m0)} | 2-kernel {t_b:7.2f}us  "
        f"fused {t_f:7.2f}us ({t_b / t_f:4.2f}x)  fused w/o normed_out {t_n:7.2f}us "
        f"({t_b / t_n:4.2f}x)"
    )
    return t_b / t_f, t_b / t_n


def check_correctness(M=99, H=4096, dtype=torch.bfloat16):
    dev = "cuda"
    input = torch.randn(M, H, dtype=dtype, device=dev) * 0.1
    residual = torch.randn(M, H, dtype=dtype, device=dev) * 0.1
    weight = torch.randn(H, dtype=dtype, device=dev)
    eps = 1e-6
    # reference
    x = input.float() + residual.float()
    var = x.pow(2).mean(-1, keepdim=True)
    normed_ref = (x * torch.rsqrt(var + eps) * weight.float()).to(
        dtype
    )  # round to bf16 like kernel
    q_ref, scale_ref = _ref_group_quant_fp8(normed_ref)  # scale_ref [M, H/128]
    # fused
    out = torch.empty(M, H, dtype=torch.float8_e4m3fn, device=dev)
    block_scale = torch.empty(
        H // 128, _round_up(M, 4), dtype=torch.float32, device=dev
    )
    normed_out = torch.empty(M, H, dtype=dtype, device=dev)
    r = residual.clone()
    _run_fused(out, block_scale, normed_out, input, r, weight, eps)
    # checks
    torch.testing.assert_close(r, x.to(dtype), rtol=1e-2, atol=1e-2)  # residual
    torch.testing.assert_close(
        normed_out.float(), normed_ref.float(), rtol=1e-2, atol=1e-2
    )  # normed
    scale_got = block_scale.transpose(0, 1)[:M]  # -> [M, H/128]
    torch.testing.assert_close(
        scale_got, scale_ref, rtol=1e-3, atol=1e-6
    )  # block scale
    # dequant fp8 vs reference normed
    deq = out.float().view(M, H // 128, 128) * scale_got.unsqueeze(-1)
    torch.testing.assert_close(deq.view(M, H), normed_ref.float(), rtol=0.1, atol=0.3)
    print("correctness: OK (residual, normed_out, block_scale, dequant all match)")


if __name__ == "__main__":
    print(f"flashinfer {flashinfer.__version__} | {torch.cuda.get_device_name()}")
    check_correctness()
    for scale_ue8m0 in (False, True):
        speedups = []
        for M in (1, 32, 256, 1024, 4096, 16384):
            for H in (4096, 5120, 7168):
                speedups.append(bench_one(M, H, scale_ue8m0))
        print(
            f"geomean speedup vs 2-kernel (ue8m0={int(scale_ue8m0)}): "
            f"fused {np.exp(np.mean(np.log([s[0] for s in speedups]))):.2f}x, "
            f"w/o normed_out {np.exp(np.mean(np.log([s[1] for s in speedups]))):.2f}x\n"
        )
