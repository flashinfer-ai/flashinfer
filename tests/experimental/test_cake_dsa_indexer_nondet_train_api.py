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

Tests of the public entry points of the non-deterministic Cake DSA indexer top-k
(flashinfer-ai/flashinfer#5676 under the set-equality semantics of
flashinfer-ai/flashinfer#8946, named by the rules of flashinfer-ai/flashinfer#8928):
``flashinfer.cake_dsa_indexer_nondet_train.cake_dsa_indexer_nondet_train_topk``
and ``cake_dsa_indexer_nondet_train_topk_workspace_size`` against the backend
package, and their stable-API contract (no experimental warning).  A row's slot
order may differ between two calls, so results are compared as sets: per row the
valid count and the sorted (id, score bits) pairs.  The backend contract itself
(registry, dispatch policy, validation, binding, the reference and the operator
tests) lives in ``test_cake_dsa_indexer_nondet_train.py``, which imports only the
package; the naming rules live in ``test_cake_dsa_indexer_nondet_train_names.py``.
The operator tests here need a compute capability 10.0 / 10.3 / 10.7 device with
registered generated programs and skip otherwise.
"""

import contextlib
import inspect
import re
import warnings
from pathlib import Path

import pytest
import torch
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.cake_dsa_indexer_nondet_train import (
    cake_dsa_indexer_nondet_train_topk,
    cake_dsa_indexer_nondet_train_topk_workspace_size,
)
from flashinfer.experimental.cake.cake_dsa_indexer_nondet_train import cake_backend
from flashinfer.experimental.cake.cake_dsa_indexer_nondet_train.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    TOPK_ENTRY,
    generated_program_available,
)

from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref

ALERT = "does not have a deterministic implementation"


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
    """The backend call whose per-row sets the public entry point must reproduce bit for bit."""
    out = cake_backend.cake_dsa_indexer_nondet_train_topk(
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


def _public(inputs: ref.IndexerInputs, **extra):
    return cake_dsa_indexer_nondet_train_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        **extra,
    )


def _row_pairs(result) -> torch.Tensor:
    """int64 ``[T, K]``: every slot as ``id << 32 | score bits``, each row sorted."""
    ids, scores = result
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((ids.to(torch.int64) << 32) | bits).sort(dim=1).values


def _canonical(result):
    """``result`` with every row in the reference's slot order (the valid slots by ascending id with their scores,
    then the padding), for ``ref.judge``; a per-row permutation that keeps the row's pairs."""
    ids, scores = result
    key = torch.where(ids >= 0, ids.to(torch.int64), 1 << 32)
    order = key.sort(dim=1, stable=True).indices
    return ids.gather(1, order), scores.gather(1, order)


def _same_sets(a, b) -> bool:
    """Per row the same valid count and the same (id, score bits) pairs; slot order and padding positions free."""
    if tuple(a[0].shape) != tuple(b[0].shape) or tuple(a[1].shape) != tuple(b[1].shape):
        return False
    return bool(torch.equal((a[0] >= 0).sum(dim=1), (b[0] >= 0).sum(dim=1))) and bool(
        torch.equal(_row_pairs(a), _row_pairs(b))
    )


@contextlib.contextmanager
def _torch_deterministic(*, warn_only: bool):
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)


@pytest.mark.parametrize(
    "fn",
    [
        cake_dsa_indexer_nondet_train_topk,
        cake_dsa_indexer_nondet_train_topk_workspace_size,
    ],
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
        got = _public(inputs)
        again = _public(
            inputs,
            max_seqlen_q=max(inputs.seg_q_len),
            max_seqlen_k=max(inputs.seg_k_len),
        )
        indices_only = _public(inputs, return_scores=False)
    assert _same_sets(got, expected)
    assert _same_sets(again, expected)
    assert indices_only[1] is None
    # the ids of every row as a set (a row's ids are unique; its padding is -1)
    assert torch.equal(
        indices_only[0].sort(dim=1).values, expected[0].sort(dim=1).values
    )
    verdict = ref.judge(_canonical(got), ref.select_reference(inputs))
    assert verdict.passed, str(verdict)
    assert cake_dsa_indexer_nondet_train_topk_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=128,
        device=inputs.device,
    ) == cake_backend.cake_dsa_indexer_nondet_train_topk_workspace_size(
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
    ``return_scores=True``, and neither the keywords of the SM120 serving form (``backend``, ``q_scale``,
    ``block_table``, ``context_lens``, ``enable_pdl``) nor the deterministic package's FP8 ``k_scale`` nor a
    ``deterministic`` switch are parameters of this entry point."""
    q, k, w, cu_q, cu_k = _bf16_host_args()
    with pytest.raises(ValueError, match="cu_seqlens_q and cu_seqlens_k"):
        cake_dsa_indexer_nondet_train_topk(q, k, w, None, cu_k, top_k=4)
    with pytest.raises(ValueError, match="return_scores=False"):
        cake_dsa_indexer_nondet_train_topk(
            q,
            k,
            w,
            cu_q,
            cu_k,
            top_k=4,
            return_scores=False,
            scores=torch.empty((6, 4), dtype=torch.float32),
        )
    for keyword in (
        "backend",
        "q_scale",
        "block_table",
        "context_lens",
        "enable_pdl",
        "k_scale",
        "deterministic",
    ):
        with pytest.raises(TypeError, match=keyword):
            cake_dsa_indexer_nondet_train_topk(
                q, k, w, cu_q, cu_k, top_k=4, **{keyword: None}
            )
    # host tensors with every rule above satisfied reach the backend's device rule
    with pytest.raises(ValueError, match="CUDA"):
        cake_dsa_indexer_nondet_train_topk(q, k, w, cu_q, cu_k, top_k=4)


def test_public_signatures_take_no_k_scale():
    """v1 ships the BF16 form only: no public or backend entry point names ``k_scale``; the workspace bound keeps
    the deterministic package's ``dtype`` keyword (default ``torch.bfloat16``)."""
    for fn in (
        cake_dsa_indexer_nondet_train_topk,
        cake_dsa_indexer_nondet_train_topk_workspace_size,
        cake_backend.cake_dsa_indexer_nondet_train_topk,
        cake_backend.cake_dsa_indexer_nondet_train_topk_workspace_size,
        cake_backend.prepare_dsa_indexer_nondet_topk,
        cake_backend.validate_dsa_indexer_nondet_inputs,
    ):
        parameters = inspect.signature(fn).parameters
        assert "k_scale" not in parameters, fn.__qualname__
        assert "deterministic" not in parameters, fn.__qualname__
    parameters = inspect.signature(
        cake_dsa_indexer_nondet_train_topk_workspace_size
    ).parameters
    assert parameters["dtype"].default is torch.bfloat16


def _fp8_host_args():
    q = torch.zeros((6, 32, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
    k = torch.zeros((10, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
    w = torch.zeros((6, 32), dtype=torch.float32)
    cu_q = torch.tensor((0, 2, 6), dtype=torch.int32)
    cu_k = torch.tensor((0, 3, 10), dtype=torch.int32)
    return q, k, w, cu_q, cu_k


def test_public_entry_point_rejects_the_fp8_form():
    """float8_e4m3fn q / k (one of them or both) raise ``ValueError`` saying this package offers the BF16 form only,
    before the device rule (host tensors here); the workspace bound raises the same for ``dtype=float8_e4m3fn``,
    before it reads a device, and names ``bfloat16`` for any other dtype."""
    fq, fk, w, cu_q, cu_k = _fp8_host_args()
    q, k, *_ = _bf16_host_args()
    for pair in ((fq, fk), (fq, k), (q, fk)):
        with pytest.raises(ValueError, match="BF16 form only"):
            cake_dsa_indexer_nondet_train_topk(*pair, w, cu_q, cu_k, top_k=4)
    with pytest.raises(ValueError, match="BF16 form only"):
        cake_dsa_indexer_nondet_train_topk_workspace_size(
            6, 10, 2, top_k=4, dtype=torch.float8_e4m3fn
        )
    with pytest.raises(ValueError, match="bfloat16"):
        cake_dsa_indexer_nondet_train_topk_workspace_size(
            6, 10, 2, top_k=4, dtype=torch.float16
        )


def test_public_entry_point_under_torch_deterministic_mode():
    """Under ``torch.use_deterministic_algorithms(True)`` the public entry point raises PyTorch's ``RuntimeError``
    naming the backend's eager entry (no deterministic package named as a fallback); with ``warn_only=True`` it warns
    once, pointing at this file through the facade and its logging decorator, and returns the sets of a call outside
    the mode.  The workspace bound never alerts."""
    _require_program()
    inputs = ref.make_random_inputs(
        [200, 300], [200, 2500], top_k=128, seed=72, device="cuda"
    )
    expected = _run(inputs)
    size = cake_dsa_indexer_nondet_train_topk_workspace_size(
        inputs.num_queries, inputs.num_keys, inputs.num_segments, top_k=128
    )
    match = re.escape(f"{TOPK_ENTRY} {ALERT}")
    with _torch_deterministic(warn_only=False):
        with pytest.raises(RuntimeError, match=match) as info:
            _public(inputs)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                cake_dsa_indexer_nondet_train_topk_workspace_size(
                    inputs.num_queries, inputs.num_keys, inputs.num_segments, top_k=128
                )
                == size
            )
    assert "_det_train" not in str(info.value)
    with (
        _torch_deterministic(warn_only=True),
        pytest.warns(UserWarning, match=ALERT) as record,
    ):
        got = _public(inputs)
    alerts = [w for w in record if str(w.message).startswith(f"{TOPK_ENTRY} {ALERT}")]
    assert len(alerts) == 1, [str(w.message) for w in record]
    assert Path(alerts[0].filename).name == Path(__file__).name, alerts[0].filename
    assert _same_sets(got, expected)
