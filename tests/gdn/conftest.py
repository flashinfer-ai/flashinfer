import functools

import pytest
import torch
import torch.distributions as dist


def multidist_randn(num_dists, dim, mean_mean=0.0, mean_std=1.0, scale_lower=0.5, scale_upper=1.5):
    means = torch.distributions.Normal(mean_mean, mean_std).sample((num_dists,))
    scales = torch.distributions.Uniform(scale_lower, scale_upper).sample((num_dists,))
    data = torch.distributions.Normal(means, scales).sample((dim,))
    return data.T.contiguous()


def multidist_randu(num_dists, dim, mean_mean=0.0, mean_std=1.0, lower=-1.0, upper=1.0):
    means = torch.distributions.Normal(mean_mean, mean_std).sample((num_dists,))
    data = torch.distributions.Uniform(means + lower, means + upper).sample((dim,))
    return data.T.contiguous()


def gen_qkv(seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=torch.float16):
    # qkv_rng = functools.partial(multidist_randn, mean_std=0.1)
    qkv_rng = functools.partial(multidist_randu, mean_std=0.05, lower=-0.25, upper=0.25)

    total_seq_lens = sum(seq_lens)
    q = qkv_rng(total_seq_lens * num_q_heads, head_size)
    k = qkv_rng(total_seq_lens * num_k_heads, head_size)
    v = qkv_rng(total_seq_lens * num_v_heads, head_size)

    q = q.reshape(total_seq_lens, num_q_heads, head_size).to(dtype).contiguous()
    k = k.reshape(total_seq_lens, num_k_heads, head_size).to(dtype).contiguous()
    v = v.reshape(total_seq_lens, num_v_heads, head_size).to(dtype).contiguous()

    return q, k, v


@pytest.fixture()
def qkv_factory():
    return gen_qkv
