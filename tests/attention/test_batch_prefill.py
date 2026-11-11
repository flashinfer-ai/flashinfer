import pytest
import torch

from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_effect(dtype):
    torch.manual_seed(42)

    H_QO, H_KV, N_CTX, HEAD_DIM, PAGE_SIZE = 1, 1, 8, 64, 16
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE

    # Create paged KV cache
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    paged_kv_cache = (k_cache, v_cache)

    # Create query tensor and indptrs
    q = torch.randn(N_CTX, H_QO, HEAD_DIM, dtype=dtype, device="cuda")
    qo_indptr = torch.tensor([0, N_CTX], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor(
        [N_CTX % PAGE_SIZE or PAGE_SIZE], dtype=torch.int32, device="cuda"
    )

    workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)

    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    out1, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=0.1, v_scale=0.1)
    out2, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=2.0, v_scale=2.0)

    assert not torch.allclose(out1, out2, atol=1e-3), (
        "Output should change when k_scale/v_scale values are different. "
        "This may indicate that the arguments are not passed correctly."
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_math_property(dtype: torch.dtype):
    torch.manual_seed(0)

    # ---------------- parameters ----------------
    N_CTX, PAGE_SIZE = 128, 16
    H_QO, H_KV, HEAD_DIM = 1, 1, 64  # Explicitly specify H_QO
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE

    # ---------------- paged KV cache ----------------
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn_like(k_cache)
    paged_kv_cache = (k_cache, v_cache)

    # ---------------- query and indptr ----------------
    q = torch.randn(N_CTX, H_QO, HEAD_DIM, dtype=dtype, device="cuda")
    qo_indptr = torch.tensor([0, N_CTX], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor(
        [N_CTX % PAGE_SIZE or PAGE_SIZE], dtype=torch.int32, device="cuda"
    )

    # ---------------- wrapper ----------------
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)

    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # ---------------- scale factors ----------------
    k_scale = 0.5
    v_scale = 2.0

    # -------- case 1: k_scale only ----------
    out1, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=k_scale)
    out1_ref, _ = wrapper.forward_return_lse(q * k_scale, paged_kv_cache)
    torch.testing.assert_close(out1, out1_ref, rtol=1e-2, atol=1e-3)

    # -------- case 2: v_scale only ----------
    out2, _ = wrapper.forward_return_lse(q, paged_kv_cache, v_scale=v_scale)
    out2_ref, _ = wrapper.forward_return_lse(q, paged_kv_cache)
    torch.testing.assert_close(out2, out2_ref * v_scale, rtol=1e-2, atol=1e-3)

    # -------- case 3: both k_scale and v_scale ----------
    out3, _ = wrapper.forward_return_lse(
        q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale
    )
    out3_ref, _ = wrapper.forward_return_lse(q * k_scale, paged_kv_cache)
    torch.testing.assert_close(out3, out3_ref * v_scale, rtol=1e-2, atol=1e-3)


def _small_plan_inputs(q_lens, metadata_device="cuda", sm=(9, 0)):
    """Two disjoint 256-token KV sequences for backend regressions."""
    if get_compute_capability(torch.device("cuda")) != sm:
        pytest.skip(f"Backend regression requires SM{sm[0]}{sm[1]}")
    metadata = dict(dtype=torch.int32, device=metadata_device)
    return dict(
        qo_indptr=torch.tensor([0, q_lens[0], sum(q_lens)], **metadata),
        paged_kv_indptr=torch.tensor([0, 16, 32], **metadata),
        paged_kv_indices=torch.arange(32, **metadata),
        paged_kv_last_page_len=torch.tensor([16, 16], **metadata),
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim_qk=128,
        page_size=16,
        causal=True,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )


def _small_wrapper(plan_inputs, backend="auto", use_cuda_graph=False):
    metadata_buffers = {
        name + "_buf": torch.empty_like(value, device="cuda")
        for name, value in plan_inputs.items()
        if use_cuda_graph and isinstance(value, torch.Tensor)
    }
    return BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend=backend,
        use_cuda_graph=use_cuda_graph,
        **metadata_buffers,
    )


def _small_data(q_lens):
    torch.manual_seed(42)
    q = torch.randn(sum(q_lens), 32, 128, dtype=torch.float16, device="cuda")
    kv = torch.randn((32, 2, 16, 8, 128), dtype=torch.float16, device="cuda")
    return q, kv


def _explicit_output(plan_inputs, q, kv):
    reference = _small_wrapper(plan_inputs, backend="fa2")
    reference.plan(**plan_inputs)
    return reference.run(q, kv)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_short_query_auto_backend_reselection(dtype):
    wrapper = _small_wrapper(_small_plan_inputs((2, 2)))
    # Include the threshold, unequal request lengths, and a return to short queries.
    for q_lens, expected in [
        ((2, 2), "fa2"),
        ((64, 64), "fa2"),
        ((65, 63), "fa3"),
        ((256, 256), "fa3"),
        ((2, 2), "fa2"),
    ]:
        plan_inputs = _small_plan_inputs(q_lens)
        plan_inputs.update(
            q_data_type=dtype, kv_data_type=dtype, o_data_type=torch.float16
        )
        wrapper.plan(**plan_inputs)
        assert wrapper._backend == (expected if dtype == torch.float16 else "fa3")


@pytest.mark.parametrize("initial_q_lens", [(64, 64), (65, 63)])
def test_cuda_graph_replan_preserves_backend_and_replay(initial_q_lens):
    """Cross the threshold without changing total rows or batch size."""
    other_q_lens = (65, 63) if initial_q_lens == (64, 64) else (64, 64)
    plan_inputs = _small_plan_inputs(initial_q_lens)
    wrapper = _small_wrapper(plan_inputs, use_cuda_graph=True)
    wrapper.plan(**plan_inputs)
    expected_backend = "fa2" if max(initial_q_lens) <= 64 else "fa3"
    assert wrapper._backend == expected_backend
    q, kv = _small_data(initial_q_lens)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(q, kv)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = wrapper.run(q, kv)
    for q_lens in (initial_q_lens, other_q_lens, initial_q_lens):
        plan_inputs = _small_plan_inputs(q_lens)
        # Changed page IDs also expose stale captured metadata.
        plan_inputs["paged_kv_indices"] = plan_inputs["paged_kv_indices"].roll(1)
        if expected_backend == "fa2":
            reference = _small_wrapper(plan_inputs, backend="fa2", use_cuda_graph=True)
            assert wrapper.workspace_size(**plan_inputs) == reference.workspace_size(
                **plan_inputs
            )
        wrapper.plan(**plan_inputs)
        # Fail before replaying an incompatible scheduler layout on an unfixed build.
        assert wrapper._backend == expected_backend
        graph.replay()
        torch.testing.assert_close(
            captured_output,
            _explicit_output(plan_inputs, q, kv),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "metadata_device,use_cuda_graph", [("cpu", False), ("cuda", True)]
)
def test_short_query_workspace_size_matches_plan(metadata_device, use_cuda_graph):
    plan_inputs = _small_plan_inputs((2, 2), metadata_device)
    wrapper = _small_wrapper(plan_inputs, use_cuda_graph=use_cuda_graph)
    float_size, int_size = wrapper.workspace_size(**plan_inputs)
    wrapper.reset_workspace_buffer(
        torch.empty(float_size, dtype=torch.uint8, device="cuda"),
        torch.empty(int_size, dtype=torch.uint8, device="cuda"),
    )
    wrapper.plan(**plan_inputs)
    assert wrapper._backend == "fa2"
    q, kv = _small_data((2, 2))
    torch.testing.assert_close(
        wrapper.run(q, kv),
        _explicit_output(plan_inputs, q, kv),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "initial_backend,incompatible_feature",
    [("fa2", "fp8_query"), ("fa3", "custom_mask"), ("fa3", "token_head_sf")],
)
def test_cuda_graph_rejects_incompatible_backend_replan(
    initial_backend, incompatible_feature
):
    q_lens = (64, 64) if initial_backend == "fa2" else (65, 63)
    plan_inputs = _small_plan_inputs(q_lens)
    wrapper = _small_wrapper(plan_inputs, use_cuda_graph=True)
    wrapper.plan(**plan_inputs)
    assert wrapper._backend == initial_backend
    q, kv = _small_data(q_lens)
    baseline = wrapper.run(q, kv)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(q, kv)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = wrapper.run(q, kv)
    plan_inputs = _small_plan_inputs((2, 2))
    plan_inputs.update(head_dim_qk=64, head_dim_vo=64)
    if incompatible_feature == "fp8_query":
        # FP8 queries require FA3, which cannot replace a captured FA2 plan.
        plan_inputs.update(
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.float16,
        )
    elif incompatible_feature == "custom_mask":
        # Custom masks require FA2, which cannot replace a captured FA3 plan.
        plan_inputs["custom_mask"] = torch.ones(
            4 * 256, dtype=torch.bool, device="cuda"
        )
    else:
        # Per-token/head KV scales force FA2 through a separate selection path.
        plan_inputs.update(
            kv_data_type=torch.float8_e4m3fn,
            use_token_head_sf=True,
        )
    with pytest.raises(ValueError, match="CUDA.graph|backend"):
        wrapper.plan(**plan_inputs)
    assert wrapper._backend == initial_backend
    # Check allocation dimensions before running: stale values can underallocate
    # output storage for the previously accepted native kernel.
    assert (
        wrapper._qo_indptr_last,
        wrapper._head_dim_qk,
        wrapper._head_dim_vo,
        wrapper._max_q_len,
    ) == (
        sum(q_lens),
        128,
        128,
        max(q_lens),
    )
    torch.testing.assert_close(wrapper.run(q, kv), baseline, rtol=1e-3, atol=1e-3)
    graph.replay()
    torch.testing.assert_close(captured_output, baseline, rtol=1e-3, atol=1e-3)


def test_sm120_fp8_cuda_graph_replan_and_replay():
    plan_inputs = _small_plan_inputs((2, 2), sm=(12, 0))
    plan_inputs.update(
        head_dim_qk=256,
        head_dim_vo=256,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.float16,
    )
    wrapper = _small_wrapper(plan_inputs, use_cuda_graph=True)
    wrapper.plan(**plan_inputs)
    assert wrapper._backend == "fa2"
    # Use a supported SM120 FP8 VO-split configuration.
    torch.manual_seed(42)
    q = torch.randn(4, 32, 256, dtype=torch.float16, device="cuda").to(
        torch.float8_e4m3fn
    )
    kv = torch.randn(32, 2, 16, 8, 256, dtype=torch.float16, device="cuda").to(
        torch.float8_e4m3fn
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(q, kv)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = wrapper.run(q, kv)
    for _ in range(2):
        plan_inputs["paged_kv_indices"] = plan_inputs["paged_kv_indices"].roll(1)
        wrapper.plan(**plan_inputs)
        graph.replay()
        torch.testing.assert_close(
            captured_output, _explicit_output(plan_inputs, q, kv), rtol=1e-3, atol=1e-3
        )


def test_sizing_and_failed_plan_do_not_pin_cuda_graph_backend():
    plan_inputs = _small_plan_inputs((64, 64))
    wrapper = _small_wrapper(plan_inputs, use_cuda_graph=True)
    wrapper.workspace_size(**plan_inputs)
    # A failed FA2 plan must not prevent the first successful plan choosing FA3.
    plan_inputs["paged_kv_indices"] = plan_inputs["paged_kv_indices"].repeat(2)
    with pytest.raises(ValueError, match="paged_kv_indices exceeds"):
        wrapper.plan(**plan_inputs)
    wrapper.plan(**_small_plan_inputs((65, 63)))
    assert wrapper._backend == "fa3"
