import pytest
import torch

import flashinfer
import flashinfer.triton


@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_merge_state(seq_len, num_heads, head_dim):
    va = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    sa = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    vb = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    sb = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    v_merged, s_merged = flashinfer.triton.cascade.merge_state(va, sa, vb, sb)
    v_merged_std, s_merged_std = flashinfer.merge_state(va, sa, vb, sb)

    assert torch.allclose(v_merged, v_merged_std, atol=1e-2)
    assert torch.allclose(s_merged, s_merged_std, atol=1e-2)


@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_merge_state_in_place(seq_len, num_heads, head_dim):
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


@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("num_states", [100])
def test_merge_states(seq_len, num_states, num_heads, head_dim):
    v = torch.randn(seq_len, num_states, num_heads, head_dim).half().to("cuda:0")
    s = torch.randn(seq_len, num_states, num_heads, dtype=torch.float32).to("cuda:0")
    v_merged_std, s_merged_std = flashinfer.merge_states(v, s)
    v_merged, s_merged = flashinfer.triton.cascade.merge_states(v, s)

    assert torch.allclose(v_merged, v_merged_std, atol=1e-2)
    assert torch.allclose(s_merged, s_merged_std, atol=1e-2)


@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_variable_length_merge_states(seq_len, num_heads, head_dim):
    max_index_sets = 512
    lengths = torch.randint(low=1, high=max_index_sets, size=(seq_len,))
    indptr = [0]
    for i in range(seq_len):
        indptr.append(indptr[-1] + lengths[i])
    v = torch.randn(indptr[-1], num_heads, head_dim).half().to("cuda:0")
    s = torch.randn(indptr[-1], num_heads, dtype=torch.float32).to("cuda:0")
    indptr = torch.tensor(indptr, dtype=torch.int32).to("cuda:0")
    v_merged, s_merged = flashinfer.triton.cascade.variable_length_merge_states(
        v, s, indptr
    )
    for i in range(seq_len):
        sub_v = v[indptr[i] : indptr[i + 1]]
        sub_s = s[indptr[i] : indptr[i + 1]]
        sub_v = torch.unsqueeze(sub_v, 0)
        sub_s = torch.unsqueeze(sub_s, 0)
        v_merged_std, s_merged_std = flashinfer.merge_states(sub_v, sub_s)
        v_merged_std = torch.squeeze(v_merged_std, 0)
        s_merged_std = torch.squeeze(s_merged_std, 0)
        assert v_merged[i].shape == v_merged_std.shape
        assert torch.allclose(v_merged[i], v_merged_std, atol=1e-2)
        assert torch.allclose(s_merged[i], s_merged_std, atol=1e-2)
