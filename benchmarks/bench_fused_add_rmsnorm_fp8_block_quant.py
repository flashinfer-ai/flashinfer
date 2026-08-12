"""Benchmark: fused Add+RMSNorm+1x128-fp8-block-quant (one kernel) vs the 2-kernel producer
(flashinfer.fused_add_rmsnorm then a per-token-group 1x128 fp8 quant), which is what runs before
fp8 block-scaled GEMMs in deepseek_v2-family / MoE models.

Shows the kernel-level win: the fused kernel does add+norm+quant in a single memory pass, saving
the intermediate bf16 round-trip.
"""

import numpy as np
import torch

import flashinfer


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


def bench_one(M, H, dtype=torch.bfloat16, iters=100, warmup=20):
    dev = "cuda"
    input = torch.randn(M, H, dtype=dtype, device=dev) * 0.1
    residual = torch.randn(M, H, dtype=dtype, device=dev) * 0.1
    weight = torch.randn(H, dtype=dtype, device=dev)
    eps = 1e-6

    # fused outputs
    out = torch.empty(M, H, dtype=torch.float8_e4m3fn, device=dev)
    block_scale = torch.empty(
        H // 128, _round_up(M, 4), dtype=torch.float32, device=dev
    )
    normed_out = torch.empty(M, H, dtype=dtype, device=dev)

    # try to use a production per-token-group quant for the baseline's 2nd kernel; else torch ref.
    try:
        from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8

        def baseline_quant(n):
            return per_token_group_quant_fp8(n, 128)
    except Exception:

        def baseline_quant(n):
            return _ref_group_quant_fp8(n)

    # Timing loops run the kernels in place (no setup clones); residual grows ~linearly (stays bounded),
    # and memory traffic per call is identical regardless of values -> fair kernel-time comparison.

    def fused():
        _run_fused(out, block_scale, normed_out, input, residual, weight, eps)

    def baseline():
        flashinfer.fused_add_rmsnorm(
            input, residual, weight, eps
        )  # residual <- add+norm (bf16), in place
        baseline_quant(residual)  # 2nd kernel: 1x128 fp8 quant of the bf16 normed

    def time_it(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(iters):
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return float(np.median(ts))  # ms

    t_f = time_it(fused)
    t_b = time_it(baseline)
    # fused byte model: read input+residual+weight (bf16), write residual+normed(bf16)+fp8+fp32 scale
    bytes_f = (
        (3 * M * H * 2)
        + (2 * M * H * 2)
        + (M * H * 1)
        + (H // 128 * _round_up(M, 4) * 4)
    )
    gbps = bytes_f / (t_f * 1e-3) / 1e9
    print(
        f"M={M:>5} H={H:>5} | fused {t_f * 1e3:7.1f}us  2-kernel {t_b * 1e3:7.1f}us  "
        f"speedup {t_b / t_f:4.2f}x  | fused {gbps:6.0f} GB/s"
    )
    return t_b / t_f


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
    speedups = []
    for M in (256, 1024, 4096):
        for H in (4096, 6144, 8192):
            speedups.append(bench_one(M, H))
    print(
        f"\ngeomean speedup (fused vs 2-kernel): {np.exp(np.mean(np.log(speedups))):.2f}x"
    )
