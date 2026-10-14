"""Explicit opt-in for the local, experimental HPC-Ops scheduling port."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def create_hpc_flat_bin_decode_wrapper(
    device=None, cta_per_sm=None, min_process_len=512
):
    """Create a BF16/NHD/GQA4/head128/page64 persistent flat-bin decoder.

    Original parallel HPC-Ops assigner, SM100 tile128 / default one CTA per SM.
    Uses FlashInfer's CUDA-core math/merge, not HPC-Ops's attention kernel.
    No automatic dispatch, CUDA Graph, PDL, RoPE or sliding-window support.
    """
    from .experimental.hpc_flat_bin.backend import HPCFlatBinDecodeWrapper

    return HPCFlatBinDecodeWrapper(device, cta_per_sm, min_process_len)
