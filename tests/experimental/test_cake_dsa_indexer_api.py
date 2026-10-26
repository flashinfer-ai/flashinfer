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

Tests of the flashinfer public entry point of the experimental Cake DSA
indexer top-k backend (flashinfer-ai/flashinfer#5676):
``flashinfer.dsa_indexer.dsa_indexer_topk`` and
``dsa_indexer_topk_workspace_size`` against the backend package, and the
experimental-API warning contract of that entry point.  The backend contract
itself (registry, dispatch policy, validation, binding, the reference and the
operator tests) lives in ``test_cake_dsa_indexer.py``, which imports only the
package.  The operator test here needs a compute capability 10.0 / 10.3 / 10.7
device with registered generated programs and skips otherwise.
"""

import contextlib
import warnings

import pytest
import torch
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.dsa_indexer import dsa_indexer_topk, dsa_indexer_topk_workspace_size
from flashinfer.experimental.cake_dsa_indexer import cake_backend
from flashinfer.experimental.cake_dsa_indexer.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    dsa_indexer_workspace_size,
    generated_program_available,
)

from tests.test_helpers import cake_dsa_indexer_reference as ref


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program():
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated DSA indexer programs not registered for this device")


@contextlib.contextmanager
def _quiet_experimental():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


def _run(inputs: ref.IndexerInputs, **extra):
    """The backend call the public entry point must reproduce bit for bit."""
    out = cake_backend.dsa_indexer_topk(
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


def test_public_api_is_marked_experimental():
    assert dsa_indexer_topk.is_experimental


def test_public_entry_point_matches_backend_and_warns_once():
    _require_program()
    inputs = ref.make_random_inputs(
        [200, 300], [200, 2500], top_k=128, seed=70, device="cuda"
    )
    expected = _run(inputs)
    with pytest.warns(ExperimentalWarning):
        got = dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
        )
    assert ref.same_bits(got, expected)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        again = dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            max_seqlen_q=max(inputs.seg_q_len),
            max_seqlen_k=max(inputs.seg_k_len),
        )
    assert ref.same_bits(again, expected)
    assert dsa_indexer_topk_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=128,
        device=inputs.device,
    ) == dsa_indexer_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=128,
        device=inputs.device,
    )
    with pytest.raises(ValueError, match="backend"), _quiet_experimental():
        dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            backend="other",
        )
