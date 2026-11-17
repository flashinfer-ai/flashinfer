"""Experimental fused projection, Sinkhorn, residual mixing and RMSNorm."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_mega_mhc(
    *,
    x,
    residual,
    post_mix,
    comb_res_mix,
    shifted_prev_mix,
    fn,
    mix_scales,
    mix_bases,
    rmsnorm_weight,
    hc_mult=4,
    hc_norm_eps=2e-5,
    hc_pre_eps=3e-4,
    hc_post_scale=1.25,
    sinkhorn_eps=2e-6,
    num_sinkhorn_iters=20,
    rmsnorm_eps=7e-6,
    rmsnorm_scale=1.25,
    sf_layout="col",
    shared_sf_block_m=224,
    out=None,
    deterministic=False,
    scratch=None,
    split_barriers=None,
    launch_epochs=None,
    fnp=None,
    fp8_sf_ue8m0=True,
    fp4_global_scale=None,
):
    """Prepare a reusable mHC plan; plan.run() returns the output tensor dict.

    BF16 x[T,H], residual[T,4,H] and rmsnorm_weight[H] combine with FP32
    projection fn[24,4H], mixing coefficients and optional shifted_prev_mix.
    Results include updated residual/mixing tensors and the normalized BF16
    values; with sf_layout 'col' or 'extra' (hidden 5120) also FP8 E4M3 values
    and exact packed UE8M0 scales: 'col' emits GEMM scale storage, 'extra'
    emits routed and padded shared-expert scale storage. sf_layout 'bf16'
    (hidden 4096, shifted_prev_mix=None) emits the BF16 output alone through
    one of three programs chosen by the split count and the task waves the
    token count yields on the device (the tailpipe program up to 22 splits,
    its weight-staging twin up to 16 splits over five or more waves, the BF16
    program at 32 and 64; bitwise the same output); all read fn through the BF16 pieces
    fnp[3,32,4H], built here from fn or taken from the caller's fnp, and
    plan.update_fn() re-splits a changed fn. The quantized layouts of hidden 4096 emit the BF16 output and its
    quantization, bitwise what vLLM computes from y_bf16: 'fp8_group128' adds
    y_fp8 (float8_e4m3fn [T,H]) and y_fp8_sf (float32 [T,H//128]), equal to
    per_token_group_quant_fp8(y_bf16, 128, use_ue8m0=fp8_sf_ue8m0) (True keeps
    every scale a power of two); 'nvfp4' adds y_fp4 (uint8 [T,H//2], element 2i
    in the low nibble of byte i) and y_fp4_sf (float8_e4m3fn [T,H//16], linear
    layout), equal to scaled_fp4_quant(y_bf16, fp4_global_scale,
    is_sf_swizzled_layout=False) with its zero-scale guard; fp4_global_scale is
    a float32 [1] device tensor and required there. A layout not exported for
    hidden on the device's architecture raises NotImplementedError naming the
    exported ones. The eps, scale and iteration keywords are the model's
    constants (the defaults are DeepSeek V4's). Any token count in [1, 2**20]
    is accepted. Plans sharing a launch_epochs (caller-owned or another plan's)
    must launch the same grid (one epoch slot per active CTA); a reuse with
    another grid is refused, and a caller-owned launch_epochs must own its whole
    storage (a sliced or offset view is refused). See MegaMHCPlan for
    caller-owned output/workspace layouts and invocation-state ownership. The
    prepared plan follows the current PyTorch stream and CUDA Graph replay.
    """
    from .experimental.deepgemm_mega_mhc.mega_mhc import MegaMHCPlan

    return MegaMHCPlan(
        x=x,
        residual=residual,
        post_mix=post_mix,
        comb_res_mix=comb_res_mix,
        shifted_prev_mix=shifted_prev_mix,
        fn=fn,
        mix_scales=mix_scales,
        mix_bases=mix_bases,
        rmsnorm_weight=rmsnorm_weight,
        hc_mult=hc_mult,
        hc_norm_eps=hc_norm_eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_scale=hc_post_scale,
        sinkhorn_eps=sinkhorn_eps,
        num_sinkhorn_iters=num_sinkhorn_iters,
        rmsnorm_eps=rmsnorm_eps,
        rmsnorm_scale=rmsnorm_scale,
        sf_layout=sf_layout,
        shared_sf_block_m=shared_sf_block_m,
        out=out,
        deterministic=deterministic,
        scratch=scratch,
        split_barriers=split_barriers,
        launch_epochs=launch_epochs,
        fnp=fnp,
        fp8_sf_ue8m0=fp8_sf_ue8m0,
        fp4_global_scale=fp4_global_scale,
    )
