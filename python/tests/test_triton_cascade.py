import torch

import flashinfer
import flashinfer.triton


def test_merge_state():
    seq_len = 2048
    num_heads = 32
    head_dim = 128
    va = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    sa = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    vb = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    sb = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    v_merged, s_merged = flashinfer.triton.cascade.merge_state(va, sa, vb, sb)
    v_merged_std, s_merged_std = flashinfer.merge_state(va, sa, vb, sb)

    assert torch.allclose(v_merged, v_merged_std, atol=1e-2)
    assert torch.allclose(s_merged, s_merged_std, atol=1e-2)


def test_merge_state_in_place():
    seq_len = 2048
    num_heads = 32
    head_dim = 128
    v = torch.randn(seq_len, num_heads, head_dim).half()
    v_std = v.clone()
    v, v_std = v.to("cuda:0"), v_std.to("cuda:0")
    s = torch.randn(seq_len, num_heads, dtype=torch.float32)
    s_std = s.clone()
    s, s_std = s.to("cuda:0"), s_std.to("cuda:0")
    v_other = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    s_other = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    flashinfer.merge_state_in_place(v_std, s_std, v_other, s_other)
    flashinfer.triton.cascade.merge_state_in_place(v, s, v_other, s_other)

    assert torch.allclose(v, v_std, atol=1e-2)
    assert torch.allclose(s, s_std, atol=1e-2)


def test_merge_states():
    seq_len = 2048
    num_heads = 32
    head_dim = 128
    num_states = 100
    v = torch.randn(seq_len, num_states, num_heads, head_dim).half().to("cuda:0")
    s = torch.randn(seq_len, num_states, num_heads, dtype=torch.float32).to("cuda:0")
    v_merged_std, s_merged_std = flashinfer.merge_states(v, s)
    v_merged, s_merged = flashinfer.triton.cascade.merge_states(v, s)

    assert torch.allclose(v_merged, v_merged_std, atol=1e-2)
    assert torch.allclose(s_merged, s_merged_std, atol=1e-2)


if __name__ == "__main__":
    test_merge_state()
    test_merge_state_in_place()
    test_merge_states()
