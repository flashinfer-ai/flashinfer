import random
import cutlass
from flashinfer.cute_dsl.blockscaled_gemm import (
    create_scale_factor_tensor,
    grouped_gemm_nt_masked,  # deepgemm-like python interface for DLFW integration
)
import torch
import cutlass.torch as cutlass_torch
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    is_cute_dsl_available,
)


def bench_one(num_groups, max_m, expected_m_per_group, n, k):
    data = create_data(num_groups=num_groups, max_m=max_m, expected_m_per_group=expected_m_per_group, n=n, k=k)

    def test_func():
        grouped_gemm_nt_masked(
            TODO=TODO,
        )

    valid_m = data['masked_m'].sum().item()
    t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
    print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): '
          f'{t * 1e6:4.0f} us | '
          f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
          f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')


# ref: DeepGEMM
def enumerate_m_grouped_masked():
    max_m = 4096
    for num_groups, expected_m_per_group in (
            # GB200 cases
            (6, 1024), (6, 512),
            # DeepGEMM default cases
            (1, 1024), (2, 512), (4, 256),
            # TODO scan more cases
    ):
        for n, k in ((4096, 7168), (7168, 2048),):
            yield dict(num_groups=num_groups, max_m=max_m, expected_m_per_group=expected_m_per_group, n=n, k=k)


# Copy and modified from test_cute_dsl_blockscaled_gemm.py, may extract common logic later if needed
def create_data(
        num_groups, max_m, expected_m_per_group, n, k
):
    l = num_groups
    m = max_m

    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"
    sf_vec_size = 16

    # DeepGEMM case
    a_major = "k"
    b_major = "k"
    c_major = "n"

    a_ref = cutlass_torch.matrix(l, m, k, a_major == "m", cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n, c_major == "m", cutlass.Float32)

    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref,
        get_cutlass_dtype(c_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )

    # for deepgemm-like python interface
    if ab_dtype == "float4_e2m1fn":
        m, k, l = a_torch.shape
        n, k, l = b_torch.shape
        # slice into half after flatten
        half_len_a = a_torch.numel() // 2
        half_len_b = b_torch.numel() // 2
        a_torch = (
            a_torch.permute(2, 0, 1)
            .flatten()[:half_len_a]
            .reshape(l, m, k // 2)
            .permute(1, 2, 0)
        )
        b_torch = (
            b_torch.permute(2, 0, 1)
            .flatten()[:half_len_b]
            .reshape(l, n, k // 2)
            .permute(1, 2, 0)
        )

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(
        l, m, k, sf_vec_size, get_cutlass_dtype(sf_dtype)
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, get_cutlass_dtype(sf_dtype)
    )

    masked_m_tensor = create_masked_m(num_groups=num_groups, expected_m_per_group=expected_m_per_group, max_m=max_m)

    return dict(
        a=(a_torch, sfa_torch),
        b=(b_torch, sfb_torch),
        c=c_torch,
        masked_m=masked_m_tensor,
    )

def create_masked_m(num_groups, expected_m_per_group, max_m):
    """Align with DeepGEMM :: generate_m_grouped_masked"""
    masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m
    return masked_m


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    for config in enumerate_m_grouped_masked():
        bench_one(**config)
