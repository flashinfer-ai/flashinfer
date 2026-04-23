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

from tests.attention.test_trtllm_gen_attention_decode import (
    _test_trtllm_batch_decode,
)


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

    # fixme(qsang-nv): failing tests for xqa + head dim 256.
    if backend == "xqa" and head_dim == 256:
        pytest.skip("xqa backend + head dim 256 cases have precision issues")

    # General set of tests for trtllm-gen decode
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
