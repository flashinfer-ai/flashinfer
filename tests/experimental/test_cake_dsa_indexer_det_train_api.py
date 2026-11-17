"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests of the public entry points of the deterministic Cake DSA indexer top-k
(flashinfer-ai/flashinfer#5676, renamed by flashinfer-ai/flashinfer#8928):
``flashinfer.cake_dsa_indexer_det_train.cake_dsa_indexer_det_train_topk`` and
``cake_dsa_indexer_det_train_topk_workspace_size`` against the backend package,
and their stable-API contract (no experimental warning).  The backend contract
itself (registry, dispatch policy, validation, binding, the reference and the
operator tests) lives in ``test_cake_dsa_indexer_det_train.py``, which imports
only the package; the naming rules live in
``test_cake_dsa_indexer_det_train_names.py``.  The operator tests here need a
compute capability 10.0 / 10.3 / 10.7 device with registered generated programs
and skip otherwise.
"""

import warnings

import pytest
import torch
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.cake_dsa_indexer_det_train import (
    cake_dsa_indexer_det_train_topk,
    cake_dsa_indexer_det_train_topk_workspace_size,
)
from flashinfer.experimental.cake.cake_dsa_indexer_det_train import cake_backend
from flashinfer.experimental.cake.cake_dsa_indexer_det_train.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    generated_program_available,
)

from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program():
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated DSA indexer programs not registered for this device")


def _run(inputs: ref.IndexerInputs, **extra):
    """The backend call the public entry point must reproduce bit for bit."""
    out = cake_backend.cake_dsa_indexer_det_train_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        **extra,
    )
    assert isinstance(out, tuple) and len(out) == 2
    return out


@pytest.mark.parametrize(
    "fn",
    [cake_dsa_indexer_det_train_topk, cake_dsa_indexer_det_train_topk_workspace_size],
)
def test_public_api_is_stable(fn):
    """#8928: the facades are stable APIs (no experimental marker, no ExperimentalWarning)."""
    assert not getattr(fn, "is_experimental", False)


def test_public_entry_point_matches_backend_without_warning():
    _require_program()
    inputs = ref.make_random_inputs(
        [200, 300], [200, 2500], top_k=128, seed=70, device="cuda"
    )
    expected = _run(inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        got = cake_dsa_indexer_det_train_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
        )
        again = cake_dsa_indexer_det_train_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            max_seqlen_q=max(inputs.seg_q_len),
            max_seqlen_k=max(inputs.seg_k_len),
        )
        indices_only = cake_dsa_indexer_det_train_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            return_scores=False,
        )
    assert ref.same_bits(got, expected)
    assert ref.same_bits(again, expected)
    assert indices_only[1] is None
    assert torch.equal(indices_only[0], expected[0])
    assert cake_dsa_indexer_det_train_topk_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=128,
        device=inputs.device,
    ) == cake_backend.cake_dsa_indexer_det_train_topk_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=128,
        device=inputs.device,
    )


def _bf16_host_args():
    return (
        torch.zeros((6, 32, 128), dtype=torch.bfloat16),
        torch.zeros((10, 128), dtype=torch.bfloat16),
        torch.zeros((6, 32), dtype=torch.float32),
        torch.tensor((0, 2, 6), dtype=torch.int32),
        torch.tensor((0, 3, 10), dtype=torch.int32),
    )


def test_public_entry_point_host_argument_rules():
    """The facade's own checks (before any device rule): segment boundaries are required, ``scores`` goes with
    ``return_scores=True``, and the keywords of the SM120 serving form (``backend``, ``q_scale``, ``block_table``,
    ``context_lens``, ``enable_pdl``) are not parameters of the Cake entry point."""
    q, k, w, cu_q, cu_k = _bf16_host_args()
    with pytest.raises(ValueError, match="cu_seqlens_q and cu_seqlens_k"):
        cake_dsa_indexer_det_train_topk(q, k, w, None, cu_k, top_k=4)
    with pytest.raises(ValueError, match="return_scores=False"):
        cake_dsa_indexer_det_train_topk(
            q,
            k,
            w,
            cu_q,
            cu_k,
            top_k=4,
            return_scores=False,
            scores=torch.empty((6, 4), dtype=torch.float32),
        )
    for keyword in ("backend", "q_scale", "block_table", "context_lens", "enable_pdl"):
        with pytest.raises(TypeError, match=keyword):
            cake_dsa_indexer_det_train_topk(
                q, k, w, cu_q, cu_k, top_k=4, **{keyword: None}
            )


def _fp8_host_call(**extra):
    """The public entry point on host FP8 tensors (every rule checked here comes before the device rule)."""
    q = torch.zeros((6, 32, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
    k = torch.zeros((10, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
    w = torch.zeros((6, 32), dtype=torch.float32)
    cu_q = torch.tensor((0, 2, 6), dtype=torch.int32)
    cu_k = torch.tensor((0, 3, 10), dtype=torch.int32)
    return cake_dsa_indexer_det_train_topk(q, k, w, cu_q, cu_k, top_k=4, **extra)


def test_public_entry_point_fp8_argument_rules():
    """The FP8 form takes ``k_scale`` for float8_e4m3fn q / k with ``softmax_scale=None``; a BF16 call takes no
    ``k_scale``; ``q_scale`` (the MXFP4 form of the SM120 serving entry) is not a parameter here."""
    k_scale = torch.ones(10, dtype=torch.float32)
    with pytest.raises(ValueError, match="CUDA"):
        _fp8_host_call(k_scale=k_scale)
    with pytest.raises(TypeError, match="q_scale"):
        _fp8_host_call(k_scale=k_scale, q_scale=torch.ones((6, 32), dtype=torch.int32))
    with pytest.raises(ValueError, match="softmax_scale=None"):
        _fp8_host_call(k_scale=k_scale, softmax_scale=0.5)
    with pytest.raises(ValueError, match="k_scale"):
        _fp8_host_call()
    with pytest.raises(ValueError, match="k_scale"):
        cake_dsa_indexer_det_train_topk(*_bf16_host_args(), top_k=4, k_scale=k_scale)


def test_public_entry_point_fp8_matches_backend():
    _require_program()
    if not generated_program_available(torch.device("cuda"), dtype=torch.float8_e4m3fn):
        pytest.skip("generated FP8 DSA indexer programs not registered for this device")
    inputs = ref.make_fp8_inputs(
        [200, 300], [200, 2500], top_k=128, seed=71, device="cuda"
    )
    expected = _run(inputs)
    got = cake_dsa_indexer_det_train_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
    )
    assert ref.same_bits(got, expected)
    sizes = [
        fn(
            inputs.num_queries,
            inputs.num_keys,
            inputs.num_segments,
            top_k=128,
            device=inputs.device,
            dtype=torch.float8_e4m3fn,
        )
        for fn in (
            cake_dsa_indexer_det_train_topk_workspace_size,
            cake_backend.cake_dsa_indexer_det_train_topk_workspace_size,
        )
    ]
    assert sizes[0] == sizes[1]
