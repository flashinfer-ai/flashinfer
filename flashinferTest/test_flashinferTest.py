import pytest

import flashinferTest


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("s_kv", [1024, 2048])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("is_cuda_graph_compatible", [False, True])
def test_BatchDecodeWithPagedKVCacheWrapper_routine(
    batch_size, s_kv, page_size, is_cuda_graph_compatible
):
    args = flashinferTest.parse_args(
        f"--routine BatchDecodeWithPagedKVCacheWrapper --backends fa2 fa2_tc --page_size {page_size} --batch_size {batch_size} --s_qo 1 --s_kv {s_kv} --num_qo_heads 64 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len -vv --refcheck {f'--no_cuda_graph' if not is_cuda_graph_compatible else ''}".split()
    )
    flashinferTest.run_test(args)


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("s_kv", [1024, 2048])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("is_cuda_graph_compatible", [False])
def test_BatchPrefillWithPagedKVCacheWrapper_routine(
    batch_size, s_kv, page_size, is_cuda_graph_compatible
):
    args = flashinferTest.parse_args(
        f"--routine BatchPrefillWithPagedKVCacheWrapper --backends fa2 --page_size {page_size} --batch_size {batch_size} --s_qo {s_kv} --s_kv {s_kv} --num_qo_heads 8 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len -vv --refcheck --causal {f'--no_cuda_graph' if not is_cuda_graph_compatible else ''}".split()
    )
    flashinferTest.run_test(args)


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("s_kv", [1024, 2048])
@pytest.mark.parametrize("is_cuda_graph_compatible", [False])
def test_BatchPrefillWithRaggedKVCacheWrapper_routine(
    batch_size, s_kv, is_cuda_graph_compatible
):
    args = flashinferTest.parse_args(
        f"--routine BatchPrefillWithRaggedKVCacheWrapper --backends fa2 --batch_size {batch_size} --s_qo {s_kv} --s_kv {s_kv} --num_qo_heads 128 --num_kv_heads 128 --head_dim_qk 192 --head_dim_vo 128 -vv --refcheck --causal {f'--no_cuda_graph' if not is_cuda_graph_compatible else ''}".split()
    )
    flashinferTest.run_test(args)


@pytest.mark.parametrize("m", [1024, 4096])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize("mma_sm", [1, 2])
def test_gemm_fp8_nt_groupwise(m, n, k, mma_sm):
    args = flashinferTest.parse_args(
        f"--routine gemm_fp8_nt_groupwise --m {m} --n {n} --k {k} --mma_sm {mma_sm} --no_cuda_graph --refcheck -vv".split()
    )
    flashinferTest.run_test(args)


@pytest.mark.parametrize("m", [1024, 4096])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize("mma_sm", [1, 2])
@pytest.mark.parametrize("group_size", [1, 2])
def test_group_gemm_fp8_nt_groupwise(m, n, k, mma_sm, group_size):
    args = flashinferTest.parse_args(
        f"--routine group_gemm_fp8_nt_groupwise --m {m} --n {n} --k {k} --mma_sm {mma_sm} --group_size {group_size} --no_cuda_graph --refcheck -vv".split()
    )
    flashinferTest.run_test(args)
