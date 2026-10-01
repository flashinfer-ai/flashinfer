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
    descriptor_workspace=None,
):
    """Prepare a reusable mHC plan; plan.run() returns the output tensor dict.

    BF16 x[T,H], residual[T,4,H] and rmsnorm_weight[H] combine with FP32
    projection fn[24,4H], mixing coefficients and optional shifted_prev_mix.
    Results include updated residual/mixing tensors, normalized BF16 and FP8
    E4M3 values, and exact packed UE8M0 scales. 'col' emits GEMM scale storage;
    'extra' emits routed and padded shared-expert scale storage. See MegaMHCPlan
    for caller-owned output/workspace layouts and invocation-state ownership.
    The prepared plan follows the current PyTorch stream and CUDA Graph replay.
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
        descriptor_workspace=descriptor_workspace,
    )
