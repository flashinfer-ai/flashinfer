# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
from ..api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def paged_fp8_k_nvfp4_v_fmha(
    q,
    k,
    v,
    v_block_scales,
    block_table,
    seq_lens,
    qo_indptr,
    *,
    max_q_len,
    out,
    qk_scale,
    v_scale,
):
    """Experimental causal paged GQA on SM100a/103a; needs CUDA 13.4+ ptxas.

    Q is ragged E4M3 [tokens, Hq, D]. K, packed V, and per-16-channel V
    scales are HND views, including views into a single padded physical page.
    Scale bytes must use the NVFP4 KV writer's four-token swizzle, not linear
    [token, D/16] order. Page size must be a multiple of four.
    D is 128 or 256. Q rows are the final Q-length positions of each sequence.
    Callers own the device-resident int32 page table, valid sequence lengths
    and cumulative Q offsets, and supply a host upper bound on query length.
    qk_scale includes softmax and Q/K tensor scales; v_scale is the V tensor
    scale. Output is written into the supplied BF16 tensor. No full-page
    dequantization, TRTLLM cubins, or host reads of device metadata.
    The short, low-batch H128/P64 decode schedule uses graph-capturable
    FP32 partial outputs for 32, 128 or 256 sequence splits; it does not expand
    the V cache. Scratch size in bytes is `4 * tokens * Hq * splits * (D + 2)`.

    A single page with all V elements equal to one (E2M1 code 2 in both
    nibbles, unit block and tensor scales) produces ones for every query::

        import torch
        from flashinfer.attention.mixed_kv import paged_fp8_k_nvfp4_v_fmha

        device = "cuda"
        fp8 = torch.float8_e4m3fn
        q = torch.zeros((2, 4, 128), device=device, dtype=fp8)
        k = torch.zeros((1, 1, 64, 128), device=device, dtype=fp8)
        v = torch.full((1, 1, 64, 64), 0x22, device=device, dtype=torch.uint8)
        sf = torch.ones((1, 1, 64, 8), device=device, dtype=fp8)
        table = torch.tensor([[0]], device=device, dtype=torch.int32)
        lens = torch.tensor([2], device=device, dtype=torch.int32)
        ptr = torch.tensor([0, 2], device=device, dtype=torch.int32)
        out = torch.empty_like(q, dtype=torch.bfloat16)
        paged_fp8_k_nvfp4_v_fmha(q, k, v, sf, table, lens, ptr, max_q_len=2,
                                 out=out, qk_scale=128**-0.5, v_scale=1.0)
        torch.testing.assert_close(out, torch.ones_like(out))
    """
    from ..experimental.mixed_kv_fmha import run

    return run(
        q,
        k,
        v,
        v_block_scales,
        block_table,
        seq_lens,
        qo_indptr,
        max_q_len,
        out,
        qk_scale,
        v_scale,
    )
