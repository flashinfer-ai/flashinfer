"""Experimental grouped FP4 E2M1 GEMM using prepacked UE8M0 scales."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_fp4_k_grouped_gemm(
    a,
    b,
    a_scales,
    b_scales,
    *,
    m,
    group_ks,
    k_alignment=256,
    use_psum_layout=True,
    output_dtype="bf16",
    accumulate=False,
    num_stages=7,
    out=None,
    grouped_layout=None,
    descriptor_workspace=None,
):
    """Prepare independently reduced A_g @ B_g.T products, optionally adding to out.

    A is [ceil(m/256)*256, sum(padded_K)/2], B is [N,sum(padded_K)/2].
    E2M1 nibbles store even K low; each group pads independently to k_alignment.
    Padding must contain zero. Scales contain four UE8M0 bytes per int32/uint32
    word: [sum(padded_K)/128, physical_M] and [sum(padded_K)/128,N], packed-K
    major, one scale per32 K. Empty groups overwrite zero or preserve initial C.
    Output storage is [groups,physical_M,N]; plan.output retains its logical-M view.
    accumulate=True requires FP32 and initialized caller-owned out. Each run adds
    once in-place; it never resets output. PSUM metadata stores physical group
    starts plus logical K; padded layout stores each group's padded K length.
    Preparation may allocate and read metadata. run() is current-stream and graph
    replay safe. Nonempty shapes must have an exported schedule in the catalog.
    """
    from .experimental.deepgemm_kgroup_gemm.kgroup_gemm import GroupedFP4Plan

    return GroupedFP4Plan(
        a,
        b,
        a_scales,
        b_scales,
        m=m,
        group_ks=group_ks,
        k_alignment=k_alignment,
        use_psum_layout=use_psum_layout,
        output_dtype=output_dtype,
        accumulate=accumulate,
        num_stages=num_stages,
        out=out,
        grouped_layout=grouped_layout,
        descriptor_workspace=descriptor_workspace,
    )
