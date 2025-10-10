import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability

global_workspace_buffer = None  # can.be empty initialized
global_trtllm_gen_fmha_workspace_buffer = None  # must be zero initialized
workspace_size = 256 * 1024 * 1024


def create_workspace_buffers(device):
    # Lazily initialize and reuse global workspace buffers
    global global_workspace_buffer, global_trtllm_gen_fmha_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    if global_trtllm_gen_fmha_workspace_buffer is None:
        global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    return global_trtllm_gen_fmha_workspace_buffer, global_workspace_buffer


def test_trtllm_gen_prefill_deepseek(
    batch_size, s_qo, s_kv, num_kv_heads, head_grp_size, causal
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] in [11, 12]:
        pytest.skip("trtllm-gen does not support SM110/SM120/SM121 GPUs.")
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    num_qo_heads = num_kv_heads * head_grp_size
    head_dim_qk = 192
    head_dim_vo = 128

    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    cumsum_s_kv = torch.sum(actual_seq_lens_kv)

    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )

    k_cache = torch.randn(
        (cumsum_s_kv, num_kv_heads, head_dim_qk),
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        (cumsum_s_kv, num_kv_heads, head_dim_vo),
        device=device,
        dtype=torch.bfloat16,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim_qk**0.5))

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(device)

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # kv_indptr = torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * s_kv

    # Create kv_indptr as cumulative sum of actual_seq_lens_kv
    kv_indptr = torch.cat(
        [
            torch.tensor(
                [0],
                device=device,
            ),
            torch.cumsum(actual_seq_lens_kv.view(-1), dim=0),
        ]
    ).int()

    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer_ref,
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    output_ref, lse_ref = wrapper.run(q, k_cache, v_cache, return_lse=True)
    output = torch.empty_like(output_ref)

    bmm1_scale = scale
    bmm2_scale = 1.0
    output_trtllm, lse_trtllm = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q,
        k_cache,
        v_cache,
        workspace_buffer,
        actual_seq_lens_kv,
        s_qo,
        s_kv,
        bmm1_scale,
        bmm2_scale,
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
        out=output,
    )
    torch.testing.assert_close(
        output_trtllm,
        output_ref,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        lse_trtllm,
        lse_ref,
        atol=1e-3,
        rtol=1e-3,
    )
    # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()


if __name__ == "__main__":
    test_trtllm_gen_prefill_deepseek(
        batch_size=1,
        s_qo=1024,
        s_kv=1024,
        num_kv_heads=128,
        head_grp_size=1,
        causal=True,
    )
