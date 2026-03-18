"""
This is the test file for MaskedBatchedMatmulCuteDSL kernel with combine fusion.
`test_blockscaled_gemm_python_interface` is the python interface test. For pytorch DLFW, refer to this.

USAGE: torchrun --nproc_per_node=4 test_multi_gpu_cute_dsl_blockscaled_gemm_fusion.py
"""

import os
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

import torch.distributed._symmetric_memory as torch_symmetric_memory

from flashinfer.cute_dsl.blockscaled_gemm import (
    Sm100BlockScaledPersistentDenseGemmKernel,  # not used in python interface
    grouped_gemm_nt_masked,  # deepgemm-like python interface for DLFW integration
    create_scale_factor_tensor,
)
from flashinfer.cute_dsl.utils import (
    cutlass_to_torch_dtype,
    get_cutlass_dtype,
    is_cute_dsl_available,
)

# WAR for https://github.com/pytorch/pytorch/issues/162429
c_torch_handle_list = []
barrier_flag_local_handle_list = []


def test_blockscaled_gemm_python_interface(
    lm: Tuple[int, int],
    kn: Tuple[int, int],
    ab_dtype: cutlass.dtype,
    sf_dtype: cutlass.dtype,
    sf_vec_size: int,
    c_dtype: cutlass.dtype,
    a_major: str,
    b_major: str,
    c_major: str,
    fuse_alpha: bool,
    alpha_dtype: cutlass.dtype,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    iterations: int,
):
    if not is_cute_dsl_available():
        print("Skipping: Please `pip install nvidia-cutlass-dsl`")
        return

    torch.manual_seed(42)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    major, minor = torch.cuda.get_device_capability(device)

    if not (major == 10 and minor == 0):
        print("Skipping: Cute-dsl backend is only supported on SM100.")
        return

    l, m = lm
    k, n = kn

    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        get_cutlass_dtype(ab_dtype),
        get_cutlass_dtype(sf_dtype),
        sf_vec_size,
        get_cutlass_dtype(c_dtype),
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        print(
            f"Skipping: Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )
        return

    if not (a_major == "k" and b_major == "k" and c_major == "n"):
        # not supported since we try to align deepgemm for now
        print(
            f"Skipping: Non deepgemm-like cases {a_major}, {b_major}, {c_major}. Might be added later"
        )
        return

    a_ref = cutlass_torch.matrix(
        l,
        m,
        k,
        a_major == "m",
        cutlass.Float32,
        device=device,  # init_type=cutlass_torch.TensorInitType.SCALAR, init_config=cutlass_torch.ScalarInitConfig(value=1.0)
    )
    b_ref = cutlass_torch.matrix(
        l,
        n,
        k,
        b_major == "n",
        cutlass.Float32,
        device=device,  # init_type=cutlass_torch.TensorInitType.SCALAR, init_config=cutlass_torch.ScalarInitConfig(value=1.0)
    )

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

    alpha_tensor = (
        torch.randn(l, dtype=torch.float32, device=device) if fuse_alpha else None
    )
    # print(f"c_torch: {c_torch.shape}:{c_torch.stride()}, is_contiguous: {c_torch.is_contiguous()}")
    # c_torch2 = c_torch.permute(2, 0, 1)
    # print(f"c_torch2: {c_torch2.shape}:{c_torch2.stride()}, is_contiguous: {c_torch2.is_contiguous()}")
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
        l, m, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )
    masked_m_tensor = torch.randint(0, m + 1, (l,), dtype=torch.int32, device=device)

    my_rank = torch.distributed.get_rank()
    topk_weights_tensor = torch.arange(
        2, 2 + m, dtype=torch.float32, device=device
    ).repeat(l, 1)
    num_ranks = torch.distributed.get_world_size()
    # Balanced communication: 1/num_ranks tokens go to each peer (contiguous chunks)
    tokens_per_rank = m // num_ranks
    token_indices = torch.arange(m, dtype=torch.int32, device=device)
    rank_src_info_tensor = (
        (token_indices // tokens_per_rank).unsqueeze(0).expand(l, -1).contiguous()
    )
    # Each output row accumulates exactly l contributions from different ranks
    expert_indices = torch.arange(l, dtype=torch.int32, device=device)
    idx_src_info_tensor = (
        (token_indices % tokens_per_rank).unsqueeze(0) * num_ranks
        + (expert_indices + my_rank).unsqueeze(1) % num_ranks
    ).to(torch.int32)
    c_torch = torch_symmetric_memory.empty(
        (m, n), dtype=cutlass_to_torch_dtype(get_cutlass_dtype(c_dtype)), device=device
    )
    copied_c_torch = torch.empty_like(c_torch)
    c_torch.fill_(0)
    global c_torch_handle_list
    c_torch_handle = torch_symmetric_memory.rendezvous(
        c_torch, group=torch.distributed.group.WORLD
    )
    c_torch_handle_list.append(c_torch_handle)
    copied_c_tensor = from_dlpack(copied_c_torch, assumed_align=16)
    c_ref = c_torch.to(torch.float32)
    out_ptrs_tensor = torch.tensor(
        c_torch_handle.buffer_ptrs, dtype=torch.int64, device=device
    )

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    barrier_flag_local = torch_symmetric_memory.empty(
        (num_sms,),
        device=device,
        dtype=torch.int32,
    )
    barrier_flag_local.fill_(0)
    global barrier_flag_local_handle_list
    barrier_flag_local_handle = torch_symmetric_memory.rendezvous(
        barrier_flag_local, group=torch.distributed.group.WORLD
    )
    barrier_flag_local_handle_list.append(barrier_flag_local_handle)
    barrier_flag_multicast = cutlass_torch.as_tensor(
        barrier_flag_local_handle.multicast_ptr,
        barrier_flag_local.shape,
        barrier_flag_local.dtype,
    )

    torch.distributed.barrier()

    for _ in range(iterations):
        c_torch.zero_()
        torch.distributed.all_reduce(torch.ones(1, dtype=torch.float, device="cuda"))
        if my_rank == 1:
            # Inject jitter to trigger potential race conditions
            torch.cuda._sleep(1000000000)
        grouped_gemm_nt_masked(
            (a_torch, sfa_torch),
            (b_torch, sfb_torch),
            c_torch,
            masked_m_tensor,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            alpha=alpha_tensor,
            alpha_dtype=alpha_dtype,
            topk_weights=topk_weights_tensor,
            idx_src_info=idx_src_info_tensor,
            rank_src_info=rank_src_info_tensor,
            out_ptrs=out_ptrs_tensor,
            num_ranks=num_ranks,
            barrier_flag_local=barrier_flag_local,
            barrier_flag_multicast=barrier_flag_multicast,
            is_combine_fusion=True,
            is_swap_ab=True,
        )
        # Copy to side tensor to capture race conditions
        copied_c_torch.copy_(c_torch)
        torch.cuda.synchronize()

    torch.distributed.barrier()

    # compute ref output
    if not fuse_alpha:
        alpha_tensor = torch.ones(l, dtype=torch.float32, device=device)
    res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
    res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
    ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)
    ref = torch.einsum("mnl,l->mnl", ref, alpha_tensor)

    # Convert c back to f32 for comparison.
    torch.cuda.synchronize()
    cute.testing.convert(
        copied_c_tensor,
        from_dlpack(c_ref, assumed_align=16).mark_layout_dynamic(
            leading_dim=(1 if c_major == "n" else 0)
        ),
    )

    if c_dtype in ("float32", "float16", "bfloat16"):
        # Each output row r maps to token (start_token + r // num_ranks) and
        # accumulates exactly l contributions (one per expert, each from a different source rank)
        tokens_per_rank = m // num_ranks
        start_token = my_rank * tokens_per_rank
        token_for_row = start_token + torch.arange(m, device=device) // num_ranks
        acc_ref = torch.zeros((m, n), dtype=torch.float32, device=device)
        for i in range(l):
            valid = (token_for_row < masked_m_tensor[i]).unsqueeze(1).float()
            acc_ref += (
                ref[token_for_row, :, i]
                * topk_weights_tensor[i, token_for_row].unsqueeze(1)
                * valid
            )
        ref = acc_ref

        torch.testing.assert_close(
            c_ref,
            ref,
            atol=tolerance,
            rtol=2e-01,
        )


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)

    for BATCH_SIZE in [16, 64, 128, 256]:
        if torch.distributed.get_rank() == 0:
            print(f"\n{'=' * 60}")
            print(f"BATCH_SIZE={BATCH_SIZE}")
            print(f"{'=' * 60}")
        test_blockscaled_gemm_python_interface(
            lm=(8, BATCH_SIZE),
            kn=(2048, 7168),
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            sf_vec_size=16,
            c_dtype="bfloat16",
            a_major="k",
            b_major="k",
            c_major="n",
            fuse_alpha=True,
            alpha_dtype="float32",
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            tolerance=10000,
            iterations=1,
        )
    # WAR for https://github.com/pytorch/pytorch/issues/162429
    os._exit(0)
