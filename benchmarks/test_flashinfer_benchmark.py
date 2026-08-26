import math
from types import SimpleNamespace

import flashinfer_benchmark
import numpy as np
import pytest
import torch
from routines import attention as attention_routine
from routines.flashinfer_benchmark_utils import routine_cc_to_supported_backends


PRIMS_TS_ATTENTION_ROUTINES = (
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
    "BatchMLAPagedAttentionWrapper",
)


class _RecordingPrimsTSWrapper:
    def __init__(self):
        self.constructor_call = None
        self.plan_calls = []
        self.run_calls = []

    def plan(self, *args, **kwargs):
        self.plan_calls.append((args, kwargs))

    def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        kwargs["out"].zero_()
        return kwargs["out"]


@pytest.fixture
def mocked_prims_ts_benchmark(monkeypatch):
    """Run the benchmark's Python adapter path without a GPU kernel compile."""

    benchmark_outputs = []
    real_torch_empty = torch.empty

    monkeypatch.setattr(
        attention_routine, "get_device", lambda _args: torch.device("cpu")
    )
    monkeypatch.setattr(
        attention_routine,
        "filter_backends_by_compute_capability",
        lambda backends, *_args: list(backends),
    )
    monkeypatch.setattr(attention_routine, "print_perf_metrics", lambda *_args: None)

    def small_workspace_empty(*args, **kwargs):
        if args == (512 * 1024 * 1024,):
            return real_torch_empty(1, **kwargs)
        return real_torch_empty(*args, **kwargs)

    monkeypatch.setattr(attention_routine.torch, "empty", small_workspace_empty)

    def fake_bench_gpu_time(*, fn, input_args, **_kwargs):
        benchmark_outputs.append(fn(*input_args))
        return np.array([1.0])

    monkeypatch.setattr(attention_routine, "bench_gpu_time", fake_bench_gpu_time)

    def install_wrapper(wrapper_name):
        wrapper = _RecordingPrimsTSWrapper()

        def constructor(*args, **kwargs):
            wrapper.constructor_call = (args, kwargs)
            return wrapper

        module = SimpleNamespace(**{wrapper_name: constructor})
        monkeypatch.setattr(attention_routine, "_get_prims_ts_module", lambda: module)
        return wrapper

    return install_wrapper, benchmark_outputs


def _use_deterministic_ones(monkeypatch):
    monkeypatch.setattr(
        attention_routine.torch,
        "randn",
        lambda *args, **kwargs: torch.ones(*args, **kwargs),
    )


def _parse_prims_ts_case(routine, extra_args):
    return flashinfer_benchmark.parse_args(
        [
            "--routine",
            routine,
            "--backends",
            "prims-ts",
            "--no_cuda_graph",
            "--num_iters",
            "1",
            "--dry_run_iters",
            "0",
            *extra_args,
        ]
    )


def test_prims_ts_backend_alias_is_canonicalized():
    args = flashinfer_benchmark.parse_args(
        [
            "--routine",
            "BatchDecodeWithPagedKVCacheWrapper",
            "--backends",
            "prims_ts",
            "--page_size",
            "32",
            "--batch_size",
            "1",
            "--s_qo",
            "1",
            "--s_kv",
            "128",
            "--num_qo_heads",
            "8",
            "--num_kv_heads",
            "2",
            "--head_dim_qk",
            "128",
            "--head_dim_vo",
            "128",
        ]
    )
    assert args.backends == ["prims-ts"]


@pytest.mark.parametrize("routine", PRIMS_TS_ATTENTION_ROUTINES)
def test_prims_ts_backend_is_blackwell_only(routine):
    support = routine_cc_to_supported_backends[routine]
    assert "prims-ts" in support["10.0"]
    assert "prims-ts" in support["10.3"]
    assert all(
        "prims-ts" not in backends
        for compute_capability, backends in support.items()
        if compute_capability not in ("10.0", "10.3")
    )


def test_dsv4_sparse_mla_backend_is_sm100_or_sm103_only():
    support = routine_cc_to_supported_backends["trtllm_batch_decode_sparse_mla_dsv4"]
    assert support["10.0"] == ["trtllm-gen"]
    assert support["10.3"] == ["trtllm-gen"]
    assert all(
        not backends
        for compute_capability, backends in support.items()
        if compute_capability not in ("10.0", "10.3")
    )


def test_dsv4_sparse_mla_adapter_contract(monkeypatch):
    calls = []

    monkeypatch.setattr(
        attention_routine, "get_device", lambda _args: torch.device("cpu")
    )
    monkeypatch.setattr(
        attention_routine,
        "filter_backends_by_compute_capability",
        lambda backends, *_args: list(backends),
    )
    monkeypatch.setattr(attention_routine, "print_perf_metrics", lambda *_args: None)
    monkeypatch.setattr(attention_routine, "_DSV4_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(
        attention_routine.torch,
        "randn",
        lambda *shape, **kwargs: torch.ones(*shape, **kwargs),
    )

    def fake_sparse_mla(**kwargs):
        calls.append(kwargs)
        kwargs["out"].fill_(0.05)
        return kwargs["out"]

    monkeypatch.setattr(
        attention_routine.flashinfer.mla,
        "trtllm_batch_decode_sparse_mla_dsv4",
        fake_sparse_mla,
    )

    def fake_bench_gpu_time(*, fn, input_args, **_kwargs):
        fn(*input_args)
        return np.array([1.0, 1.0])

    monkeypatch.setattr(attention_routine, "bench_gpu_time", fake_bench_gpu_time)
    args = flashinfer_benchmark.parse_args(
        [
            "--routine",
            "trtllm_batch_decode_sparse_mla_dsv4",
            "--batch_size",
            "1",
            "--s_qo",
            "2",
            "--s_kv",
            "256",
            "--num_qo_heads",
            "16",
            "--num_kv_heads",
            "1",
            "--head_dim_qk",
            "512",
            "--head_dim_vo",
            "512",
            "--compressed_topk",
            "16",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
            "--refcheck",
        ]
    )

    assert args.backends == ["trtllm-gen"]
    assert args.page_size == 256
    assert args.causal is True
    results = attention_routine.testTrtllmBatchDecodeSparseMlaDsv4(args)

    assert len(calls) == 2
    call = calls[0]
    assert call["query"].shape == (2, 16, 512)
    assert call["swa_kv_cache"].shape == (1, 1, 256, 512)
    assert call["compressed_kv_cache"].shape == (1, 1, 64, 512)
    assert call["sparse_indices"].shape == (2, 144)
    assert call["sparse_topk_lens"].tolist() == [144, 144]
    assert call["seq_lens"].tolist() == [256]
    assert call["cum_seq_lens_q"].tolist() == [0, 2]
    assert call["max_q_len"] == 2
    assert call["backend"] == "trtllm-gen"
    assert call["kv_layout"] == "HND"
    assert call["out"].dtype == torch.bfloat16
    assert results[0]["median_time"] == 1.0
    assert results[0]["compressed_kv_len"] == 64
    assert results[0]["causal"] is True


@pytest.mark.parametrize(
    ("backend", "out_dtype", "message"),
    [
        ("cudnn", "float16", "cuDNN decode requires BF16 output"),
        ("fa2_tc", "fp8_e4m3", "FA2_TC backend does not support FP8 output"),
        (
            "auto",
            "fp8_e4m3",
            "auto backend may select an implementation without FP8 output support",
        ),
    ],
)
def test_decode_out_dtype_option_drops_incompatible_backend(
    mocked_prims_ts_benchmark, capsys, backend, out_dtype, message
):
    args = flashinfer_benchmark.parse_args(
        [
            "--routine",
            "BatchDecodeWithPagedKVCacheWrapper",
            "--backends",
            backend,
            "--page_size",
            "16",
            "--batch_size",
            "1",
            "--s_qo",
            "1",
            "--s_kv",
            "16",
            "--num_qo_heads",
            "2",
            "--num_kv_heads",
            "1",
            "--head_dim_qk",
            "128",
            "--head_dim_vo",
            "128",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
            "--out_dtype",
            out_dtype,
            "--no_cuda_graph",
        ]
    )

    assert attention_routine.testBatchDecodeWithPagedKVCacheWrapper(args) == []
    assert message in capsys.readouterr().out


@pytest.mark.parametrize(
    ("routine", "wrapper_name"),
    (
        ("BatchDecodeWithPagedKVCacheWrapper", "BatchDecodePagedTSWrapper"),
        ("BatchPrefillWithPagedKVCacheWrapper", "BatchPrefillPagedTSWrapper"),
        ("BatchPrefillWithRaggedKVCacheWrapper", "BatchPrefillTSWrapper"),
    ),
)
def test_prims_ts_fmha_adapters_accept_fp16(
    mocked_prims_ts_benchmark, routine, wrapper_name
):
    """The common benchmark filters must not hide PrimTS FP16 support."""

    install_wrapper, benchmark_outputs = mocked_prims_ts_benchmark
    wrapper = install_wrapper(wrapper_name)
    args = _parse_prims_ts_case(
        routine,
        [
            "--page_size",
            "16",
            "--batch_size",
            "2",
            "--s_qo",
            "2",
            "--s_kv",
            "16",
            "--num_qo_heads",
            "2",
            "--num_kv_heads",
            "1",
            "--head_dim_qk",
            "128",
            "--head_dim_vo",
            "128",
            "--q_dtype",
            "float16",
            "--kv_dtype",
            "float16",
            "--out_dtype",
            "float16",
            "--causal",
        ],
    )

    getattr(attention_routine, f"test{routine}")(args)

    assert len(wrapper.plan_calls) == len(wrapper.run_calls) == 1
    runtime_q = wrapper.run_calls[0][0][0]
    runtime_out = wrapper.run_calls[0][1]["out"]
    assert runtime_q.dtype == runtime_out.dtype == torch.float16
    assert benchmark_outputs[0].dtype == torch.float16


def test_prims_ts_decode_rejects_non_fp16_output_for_fp16_input(
    mocked_prims_ts_benchmark, capsys
):
    args = _parse_prims_ts_case(
        "BatchDecodeWithPagedKVCacheWrapper",
        [
            "--page_size",
            "16",
            "--batch_size",
            "1",
            "--s_qo",
            "1",
            "--s_kv",
            "16",
            "--num_qo_heads",
            "2",
            "--num_kv_heads",
            "1",
            "--head_dim_qk",
            "128",
            "--head_dim_vo",
            "128",
            "--q_dtype",
            "float16",
            "--kv_dtype",
            "float16",
            "--out_dtype",
            "bfloat16",
        ],
    )

    assert attention_routine.testBatchDecodeWithPagedKVCacheWrapper(args) == []
    assert "requires FP16 output for FP16 inputs" in capsys.readouterr().out


def test_prims_ts_paged_context_adapter_contract(
    monkeypatch, mocked_prims_ts_benchmark
):
    install_wrapper, benchmark_outputs = mocked_prims_ts_benchmark
    wrapper = install_wrapper("BatchPrefillPagedTSWrapper")
    _use_deterministic_ones(monkeypatch)
    args = _parse_prims_ts_case(
        "BatchPrefillWithPagedKVCacheWrapper",
        [
            "--page_size",
            "16",
            "--batch_size",
            "2",
            "--s_qo",
            "2",
            "--s_kv",
            "16",
            "--num_qo_heads",
            "2",
            "--num_kv_heads",
            "1",
            "--head_dim_qk",
            "128",
            "--head_dim_vo",
            "128",
            "--q_dtype",
            "fp8_e4m3",
            "--kv_dtype",
            "fp8_e4m3",
            "--causal",
        ],
    )

    attention_routine.testBatchPrefillWithPagedKVCacheWrapper(args)

    assert wrapper.constructor_call == (("HND",), {})
    assert len(wrapper.plan_calls) == len(wrapper.run_calls) == 1
    plan_args, plan_kwargs = wrapper.plan_calls[0]
    q, k_cache, v_cache = plan_args[:3]
    assert q.shape == (4, 2, 128)
    assert k_cache.shape == v_cache.shape == (2, 1, 16, 128)
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert k_cache.stride() == v_cache.stride() == (2048, 2048, 128, 1)
    assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
    assert plan_args[3].tolist() == [0, 2, 4]
    assert plan_args[4].tolist() == [0, 1, 2]
    assert plan_args[6].tolist() == [16, 16]
    assert plan_kwargs["page_size"] == 16
    assert plan_kwargs["mask_type"] == "causal"
    assert plan_kwargs["out_dtype"] == torch.float8_e4m3fn
    # With deterministic identical Q/K/V input, all three dequantization
    # scales match. This relation proves both Q and K scales are folded into
    # softmax scale, while V is forwarded as the output scale.
    output_scale = plan_kwargs["output_scale"]
    assert plan_kwargs["sm_scale"] == pytest.approx(
        output_scale**2 / math.sqrt(128), rel=1e-6
    )

    run_args, run_kwargs = wrapper.run_calls[0]
    assert all(
        runtime_tensor is planned_tensor
        for runtime_tensor, planned_tensor in zip(
            run_args, (q, k_cache, v_cache), strict=True
        )
    )
    assert run_kwargs["out"].shape == q.shape
    assert run_kwargs["out"].dtype == torch.float8_e4m3fn
    assert benchmark_outputs[0].shape == q.shape


def test_prims_ts_ragged_context_adapter_contract(
    monkeypatch, mocked_prims_ts_benchmark
):
    install_wrapper, benchmark_outputs = mocked_prims_ts_benchmark
    wrapper = install_wrapper("BatchPrefillTSWrapper")
    _use_deterministic_ones(monkeypatch)
    args = _parse_prims_ts_case(
        "BatchPrefillWithRaggedKVCacheWrapper",
        [
            "--batch_size",
            "2",
            "--s_qo",
            "3",
            "--s_kv",
            "3",
            "--num_qo_heads",
            "2",
            "--num_kv_heads",
            "1",
            "--head_dim_qk",
            "128",
            "--head_dim_vo",
            "128",
            "--q_dtype",
            "fp8_e4m3",
            "--kv_dtype",
            "fp8_e4m3",
            "--causal",
        ],
    )

    attention_routine.testBatchPrefillWithRaggedKVCacheWrapper(args)

    assert wrapper.constructor_call == ((), {})
    assert len(wrapper.plan_calls) == len(wrapper.run_calls) == 1
    plan_args, plan_kwargs = wrapper.plan_calls[0]
    q, k, v = plan_args
    assert q.shape == (6, 2, 128)
    assert k.shape == v.shape == (6, 1, 128)
    assert q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
    assert plan_kwargs["qo_indptr"].tolist() == [0, 3, 6]
    assert plan_kwargs["kv_indptr"].tolist() == [0, 3, 6]
    assert plan_kwargs["mask_type"] == "causal"
    assert plan_kwargs["out_dtype"] == torch.float8_e4m3fn
    assert plan_kwargs["sm_scale"] == pytest.approx((1.0 / 256) ** 2 / math.sqrt(128))
    assert plan_kwargs["output_scale"] == pytest.approx(1.0 / 256)

    run_args, run_kwargs = wrapper.run_calls[0]
    assert all(
        runtime_tensor is planned_tensor
        for runtime_tensor, planned_tensor in zip(run_args, (q, k, v), strict=True)
    )
    assert run_kwargs["out"].shape == q.shape
    assert run_kwargs["out"].dtype == torch.float8_e4m3fn
    assert benchmark_outputs[0].shape == q.shape


@pytest.mark.parametrize(
    ("spec_dec_mask", "expected_plan_mask", "expected_recorded_causal"),
    (("causal", "causal", True), ("full", "dense", False)),
)
def test_prims_ts_fmha_decode_sq_gt_one_adapter_contract(
    monkeypatch,
    mocked_prims_ts_benchmark,
    spec_dec_mask,
    expected_plan_mask,
    expected_recorded_causal,
):
    metric_causal = []

    def record_tflops(*args):
        metric_causal.append(args[5])
        return 0.0

    monkeypatch.setattr(
        attention_routine,
        "attention_tflops_per_sec_with_actual_seq_lens",
        record_tflops,
    )
    install_wrapper, benchmark_outputs = mocked_prims_ts_benchmark
    wrapper = install_wrapper("BatchDecodePagedTSWrapper")
    args = _parse_prims_ts_case(
        "BatchDecodeWithPagedKVCacheWrapper",
        [
            "--page_size",
            "16",
            "--batch_size",
            "2",
            "--s_qo",
            "3",
            "--s_kv",
            "16",
            "--num_qo_heads",
            "2",
            "--num_kv_heads",
            "1",
            "--head_dim_qk",
            "128",
            "--head_dim_vo",
            "128",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
            "--spec_dec_mask",
            spec_dec_mask,
            "--output_path",
            "unused.csv",
        ],
    )

    results = attention_routine.testBatchDecodeWithPagedKVCacheWrapper(args)

    assert wrapper.constructor_call == (("HND",), {})
    assert len(wrapper.plan_calls) == len(wrapper.run_calls) == 1
    plan_args, plan_kwargs = wrapper.plan_calls[0]
    assert plan_args[0].tolist() == [0, 1, 2]
    assert plan_args[2].tolist() == [16, 16]
    assert plan_args[3:] == (2, 1, 128, 16)
    assert plan_kwargs == {
        "seq_len_q": 3,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
        "mask_type": expected_plan_mask,
        "max_kv_len": 16,
    }
    assert len(results) == 1
    assert results[0]["causal"] is expected_recorded_causal
    assert metric_causal == [expected_recorded_causal]

    run_args, run_kwargs = wrapper.run_calls[0]
    runtime_q, runtime_kv_cache = run_args
    assert runtime_q.shape == (2, 3, 2, 128)
    assert runtime_kv_cache.shape == (2, 2, 1, 16, 128)
    assert runtime_kv_cache.is_contiguous()
    assert run_kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(128))
    assert run_kwargs["bmm2_scale"] == 1.0
    assert run_kwargs["out"].shape == runtime_q.shape
    # The public benchmark schema remains packed even though PrimTS receives
    # the rank-4 multi-query form.
    assert benchmark_outputs[0].shape == (6, 2, 128)


def test_prims_ts_fmha_decode_sq_one_keeps_causal_plan(
    mocked_prims_ts_benchmark,
):
    install_wrapper, _ = mocked_prims_ts_benchmark
    wrapper = install_wrapper("BatchDecodePagedTSWrapper")
    args = _parse_prims_ts_case(
        "BatchDecodeWithPagedKVCacheWrapper",
        [
            "--page_size",
            "16",
            "--batch_size",
            "1",
            "--s_qo",
            "1",
            "--s_kv",
            "16",
            "--num_qo_heads",
            "2",
            "--num_kv_heads",
            "1",
            "--head_dim_qk",
            "128",
            "--head_dim_vo",
            "128",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
            # The speculative-mask option has no effect when SQ == 1.
            "--spec_dec_mask",
            "full",
            "--output_path",
            "unused.csv",
        ],
    )

    results = attention_routine.testBatchDecodeWithPagedKVCacheWrapper(args)

    assert wrapper.plan_calls[0][1]["mask_type"] == "causal"
    assert len(results) == 1
    assert results[0]["causal"] is False


def test_prims_ts_mla_decode_sq_gt_one_adapter_contract(
    mocked_prims_ts_benchmark,
):
    install_wrapper, benchmark_outputs = mocked_prims_ts_benchmark
    wrapper = install_wrapper("BatchMLADecodePagedTSWrapper")
    args = _parse_prims_ts_case(
        "BatchMLAPagedAttentionWrapper",
        [
            "--page_size",
            "16",
            "--batch_size",
            "2",
            "--s_qo",
            "3",
            "--s_kv",
            "16",
            "--num_qo_heads",
            "2",
            "--num_kv_heads",
            "1",
            "--head_dim_ckv",
            "512",
            "--head_dim_kpe",
            "64",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
        ],
    )

    attention_routine.testBatchMLAPagedAttentionWrapper(args)

    assert wrapper.constructor_call == ((), {})
    assert len(wrapper.plan_calls) == len(wrapper.run_calls) == 1
    plan_args, plan_kwargs = wrapper.plan_calls[0]
    assert plan_args[0].shape == (2, 1)
    assert plan_args[1].tolist() == [16, 16]
    assert plan_args[2:] == (2, 512, 64, 16)
    assert plan_kwargs == {
        "seq_len_q": 3,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
        "mask_type": "causal",
        "max_kv_len": 16,
    }

    run_args, run_kwargs = wrapper.run_calls[0]
    runtime_q, runtime_kv_cache = run_args
    assert runtime_q.shape == (2, 3, 2, 576)
    assert runtime_kv_cache.shape == (2, 16, 576)
    assert runtime_kv_cache.is_contiguous()
    assert run_kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(192))
    assert run_kwargs["bmm2_scale"] == 1.0
    assert run_kwargs["out"].shape == (2, 3, 2, 512)
    assert run_kwargs["out"].dtype == torch.bfloat16
    assert benchmark_outputs[0].shape == (6, 2, 512)


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("s_kv", [1024, 2048])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("is_cuda_graph_compatible", [False, True])
def test_BatchDecodeWithPagedKVCacheWrapper_routine(
    batch_size, s_kv, page_size, is_cuda_graph_compatible
):
    args = flashinfer_benchmark.parse_args(
        f"--routine BatchDecodeWithPagedKVCacheWrapper --backends fa2 fa2_tc --page_size {page_size} --batch_size {batch_size} --s_qo 1 --s_kv {s_kv} --num_qo_heads 64 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len -vv --refcheck {'--no_cuda_graph' if not is_cuda_graph_compatible else ''}".split()
    )
    flashinfer_benchmark.run_test(args)


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("s_kv", [1024, 2048])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("is_cuda_graph_compatible", [False])
def test_BatchPrefillWithPagedKVCacheWrapper_routine(
    batch_size, s_kv, page_size, is_cuda_graph_compatible
):
    args = flashinfer_benchmark.parse_args(
        f"--routine BatchPrefillWithPagedKVCacheWrapper --backends fa2 --page_size {page_size} --batch_size {batch_size} --s_qo {s_kv} --s_kv {s_kv} --num_qo_heads 8 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len -vv --refcheck --causal {'--no_cuda_graph' if not is_cuda_graph_compatible else ''}".split()
    )
    flashinfer_benchmark.run_test(args)


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("s_kv", [1024, 2048])
@pytest.mark.parametrize("is_cuda_graph_compatible", [False])
def test_BatchPrefillWithRaggedKVCacheWrapper_routine(
    batch_size, s_kv, is_cuda_graph_compatible
):
    args = flashinfer_benchmark.parse_args(
        f"--routine BatchPrefillWithRaggedKVCacheWrapper --backends fa2 --batch_size {batch_size} --s_qo {s_kv} --s_kv {s_kv} --num_qo_heads 128 --num_kv_heads 128 --head_dim_qk 192 --head_dim_vo 128 -vv --refcheck --causal {'--no_cuda_graph' if not is_cuda_graph_compatible else ''}".split()
    )
    flashinfer_benchmark.run_test(args)


@pytest.mark.parametrize("m", [1024, 4096])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize("mma_sm", [1, 2])
def test_gemm_fp8_nt_groupwise(m, n, k, mma_sm):
    args = flashinfer_benchmark.parse_args(
        f"--routine gemm_fp8_nt_groupwise --m {m} --n {n} --k {k} --mma_sm {mma_sm} --no_cuda_graph --refcheck -vv".split()
    )
    flashinfer_benchmark.run_test(args)


@pytest.mark.parametrize("m", [1024, 4096])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize("mma_sm", [1, 2])
@pytest.mark.parametrize("group_size", [1, 2])
def test_group_gemm_fp8_nt_groupwise(m, n, k, mma_sm, group_size):
    args = flashinfer_benchmark.parse_args(
        f"--routine group_gemm_fp8_nt_groupwise --m {m} --n {n} --k {k} --mma_sm {mma_sm} --group_size {group_size} --no_cuda_graph --refcheck -vv".split()
    )
    flashinfer_benchmark.run_test(args)
