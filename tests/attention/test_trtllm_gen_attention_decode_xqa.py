"""XQA decode tests extracted from ``test_trtllm_gen_attention_decode.py``.

Split out so the CI parallel runner (``scripts/task_run_unit_tests.sh``)
schedules the ``backend="xqa"`` variant of ``test_trtllm_batch_decode`` on
its own GPU, parallel with the ``backend="trtllm-gen"`` variant (which
stays in ``test_trtllm_gen_attention_decode.py``) and the prefill shard.

The XQA backend has a highly parametrized JIT URI
(``input_dtype, kv_cache_dtype, output_dtype, page_size, head_dim,
head_group_ratio, use_sliding_window, use_spec_dec, spec_q_seq_len``),
so its first-touch compile cluster (~4.6s per unique URI) dominated the
previous single-file wall time.  Isolating it lets that cluster run
concurrently instead of serialized with every other decode case.

The parametrize matrix below is identical to the original
``test_trtllm_batch_decode`` except ``backend`` is restricted to
``["xqa"]``.  ``_test_trtllm_batch_decode`` and all helpers continue to
live in the decode file and are imported here.
"""

import pytest
import torch

import flashinfer

from tests.attention.test_trtllm_gen_attention_decode import (
    _test_trtllm_batch_decode,
)

pytestmark = pytest.mark.long_running


@pytest.mark.parametrize("backend", ["xqa"])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (4, 1, 16, 2, 1),
        (4, 1, 32, 2, 5),
        (4, 2, 64, 2, 5),
        (4, 3, 32, 2, 5),
        (4, 3, 64, 2, 1),
        (4, 4, 64, 4, 1),
        (4, 5, 64, 4, 8),
        (128, 1, 64, 2, 5),
        (128, 2, 32, 4, 1),
        (128, 3, 16, 4, 8),
        (128, 4, 16, 2, 5),
        (128, 5, 16, 2, 5),
        (256, 1, 64, 4, 8),
        (256, 2, 16, 2, 8),
        (256, 3, 64, 4, 5),
        (256, 4, 32, 2, 8),
        (256, 5, 32, 2, 1),
    ],
)
@pytest.mark.parametrize("window_left", [-1, 127])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("bf16", "fp8", "bf16"),
        ("fp16", "fp8", "fp16"),
        ("bf16", "fp8", "fp8"),
        ("fp16", "fp8", "fp8"),
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
        ("fp8", "nvfp4", "fp8"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("max_in_kv_len", [110])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("non_contiguous_query", [False, True])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_decode(
    backend: str,
    kv_layout: str,
    batch_size: int,
    q_len_per_req: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_in_kv_len: int,
    head_dim: int,
    non_contiguous_query: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    # xqa backend does not support non-contiguous query yet
    if backend == "xqa" and non_contiguous_query:
        pytest.skip("xqa backend does not support non-contiguous query")

    # General set of tests for xqa decode
    _test_trtllm_batch_decode(
        backend,
        kv_layout,
        batch_size,
        q_len_per_req,
        page_size,
        num_kv_heads,
        head_grp_size,
        window_left,
        q_dtype,
        o_dtype,
        kv_dtype,
        enable_pdl,
        enable_sink,
        max_in_kv_len,
        head_dim,
        kv_dtype in ("fp8", "nvfp4"),
        non_contiguous_query=non_contiguous_query,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("q_len_per_req", [1, 2])
def test_xqa_query_and_output_contiguity_contract(monkeypatch, q_len_per_req):
    batch_size = 2
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    num_tokens = batch_size * q_len_per_req
    device = torch.device("cuda")

    non_contiguous_query = torch.randn(
        num_tokens,
        head_dim,
        num_qo_heads,
        dtype=torch.float16,
        device=device,
    ).transpose(1, 2)
    assert not non_contiguous_query.is_contiguous()

    kv_cache = tuple(
        torch.randn(
            batch_size,
            1,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
        for _ in range(2)
    )
    workspace_buffer = torch.empty(
        8 * 1024 * 1024 + 1, dtype=torch.uint8, device=device
    )
    block_tables = torch.arange(batch_size, dtype=torch.int32, device=device).view(
        batch_size, 1
    )
    seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)

    captured = {}

    def capture_xqa(query, _k, _v, _block_tables, _seq_lens, out, *_args, **_kwargs):
        captured["query"] = query
        captured["out"] = out

    monkeypatch.setattr(flashinfer.decode, "xqa", capture_xqa)

    with pytest.raises(ValueError, match="query must be contiguous"):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            non_contiguous_query,
            kv_cache,
            workspace_buffer,
            block_tables,
            seq_lens,
            max_seq_len=1,
            backend="xqa",
            kv_layout="NHD",
            enable_pdl=False,
            q_len_per_req=q_len_per_req,
        )
    assert not captured

    query = non_contiguous_query.contiguous()
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len=1,
        backend="xqa",
        kv_layout="NHD",
        enable_pdl=False,
        q_len_per_req=q_len_per_req,
    )

    assert output.is_contiguous()
    assert captured["query"].is_contiguous()
    assert captured["out"].is_contiguous()

    non_contiguous_out = torch.empty(
        num_tokens,
        head_dim,
        num_qo_heads,
        dtype=query.dtype,
        device=device,
    ).transpose(1, 2)
    with pytest.raises(ValueError, match="out must be contiguous"):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query,
            kv_cache,
            workspace_buffer,
            block_tables,
            seq_lens,
            max_seq_len=1,
            out=non_contiguous_out,
            backend="xqa",
            kv_layout="NHD",
            enable_pdl=False,
            q_len_per_req=q_len_per_req,
        )

    invalid_shape_out = torch.empty(
        (*query.shape[:-1], head_dim - 1), dtype=query.dtype, device=device
    )
    with pytest.raises(ValueError, match="Invalid shape of out"):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query,
            kv_cache,
            workspace_buffer,
            block_tables,
            seq_lens,
            max_seq_len=1,
            out=invalid_shape_out,
            backend="xqa",
            kv_layout="NHD",
            enable_pdl=False,
            q_len_per_req=q_len_per_req,
        )

    invalid_device_out = torch.empty(query.shape, dtype=query.dtype, device="cpu")
    with pytest.raises(ValueError, match="Invalid device of out"):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query,
            kv_cache,
            workspace_buffer,
            block_tables,
            seq_lens,
            max_seq_len=1,
            out=invalid_device_out,
            backend="xqa",
            kv_layout="NHD",
            enable_pdl=False,
            q_len_per_req=q_len_per_req,
        )
