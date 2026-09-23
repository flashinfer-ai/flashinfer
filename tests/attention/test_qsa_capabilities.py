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
"""

import sys

import pytest
import torch

import flashinfer
from flashinfer import qsa_capabilities as qsa_capabilities_module  # noqa: F401
from flashinfer.qsa_capabilities import (
    QSA_CAP_ATTENTION_PAGED,
    QSA_CAP_FP8,
    QSA_CAP_NVFP4,
    QSA_CAP_OUTPUT_GATE,
    QSA_CAP_SELECTION,
    _capabilities,
    qsa_capabilities,
    qsa_capability_names,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def test_this_build_reports_what_it_carries(device):
    """Every capability this branch adds, reported by the query."""
    bits = qsa_capabilities()
    for bit, name in (
        (QSA_CAP_SELECTION, "selection"),
        (QSA_CAP_ATTENTION_PAGED, "attention_paged"),
        (QSA_CAP_NVFP4, "nvfp4"),
        (QSA_CAP_FP8, "fp8"),
        (QSA_CAP_OUTPUT_GATE, "output_gate"),
    ):
        assert bits & bit, f"{name} is missing from a build that has it"
        assert name in qsa_capability_names()


def test_the_gate_capability_comes_from_the_compiled_module(device):
    """The binary answers for itself, so a stale artifact cannot lie.

    A build that predates the query has no symbol to ask, and a capability
    nobody can confirm is one the query does not claim.
    """
    module = sys.modules["flashinfer.qsa_output_gate"]
    loaded = module.get_qsa_output_gate_module()
    query = getattr(loaded, "qsa_output_gate_capabilities", None)
    assert query is not None, "the compiled module does not answer for itself"
    # Two for bfloat16, which is what the QSA route needs.
    assert int(query()) & 2


@pytest.mark.parametrize(
    "module_name,loader,lost",
    [
        ("flashinfer.sparse_scores", "get_sparse_scores_module", QSA_CAP_SELECTION),
        ("flashinfer.sparse_route", "get_sparse_route_module", QSA_CAP_SELECTION),
        ("flashinfer.topk", "get_topk_module", QSA_CAP_SELECTION),
        (
            "flashinfer.qsa_output_gate",
            "get_qsa_output_gate_module",
            QSA_CAP_OUTPUT_GATE | QSA_CAP_ATTENTION_PAGED | QSA_CAP_NVFP4 | QSA_CAP_FP8,
        ),
    ],
)
def test_one_missing_piece_takes_its_capability_down(
    device, monkeypatch, module_name, loader, lost
):
    """Each kernel the answer rests on, removed one at a time.

    A build missing the scorer has no selection, however much Python is
    present. A build missing the gate has no gated attention either, because
    the route this API serves is gated on the way out.
    """
    _capabilities.cache_clear()
    module = sys.modules[module_name]

    def absent():
        raise RuntimeError("this build does not carry it")

    monkeypatch.setattr(module, loader, absent)
    try:
        bits = qsa_capabilities()
        assert not bits & lost, f"{module_name} is gone and the bits say otherwise"
        # What did not depend on it survives.
        if lost == QSA_CAP_SELECTION:
            assert bits & QSA_CAP_ATTENTION_PAGED
    finally:
        _capabilities.cache_clear()


def test_a_module_missing_one_symbol_takes_its_capability_down(device, monkeypatch):
    """Not only "does it load" -- the symbols the path calls have to be there."""
    _capabilities.cache_clear()
    module = sys.modules["flashinfer.topk"]
    loaded = module.get_topk_module()

    class _Older:
        """The same module without the descriptor-based size query."""

        def __getattr__(self, name):
            if name == "cub_topk_ragged_transform_workspace_size_for":
                raise AttributeError(name)
            return getattr(loaded, name)

    monkeypatch.setattr(module, "get_topk_module", _Older)
    try:
        assert not qsa_capabilities() & QSA_CAP_SELECTION
    finally:
        _capabilities.cache_clear()


def test_a_missing_capability_is_reported_as_missing(device, monkeypatch):
    """Zero is an answer, not an error.

    A build whose gate module will not load has no gate; the query says so
    instead of raising in the middle of whatever asked.
    """
    _capabilities.cache_clear()
    module = sys.modules["flashinfer.qsa_output_gate"]

    def absent():
        raise ImportError("no gate in this build")

    monkeypatch.setattr(module, "get_qsa_output_gate_module", absent)
    try:
        bits = qsa_capabilities()
        assert not bits & QSA_CAP_OUTPUT_GATE
        assert "output_gate" not in qsa_capability_names()
        # The rest of the answer survives one missing piece.
        assert bits & QSA_CAP_SELECTION
    finally:
        _capabilities.cache_clear()


def test_a_build_without_the_query_symbol_claims_no_gate(device, monkeypatch):
    """An older artifact that cannot say what it built is not taken on trust."""
    _capabilities.cache_clear()
    module = sys.modules["flashinfer.qsa_output_gate"]
    loaded = module.get_qsa_output_gate_module()

    class _Older:
        """The same module without the capability symbol."""

        def __getattr__(self, name):
            if name == "qsa_output_gate_capabilities":
                raise AttributeError(name)
            return getattr(loaded, name)

    monkeypatch.setattr(module, "get_qsa_output_gate_module", _Older)
    try:
        assert not qsa_capabilities() & QSA_CAP_OUTPUT_GATE
    finally:
        _capabilities.cache_clear()


def test_the_names_and_the_bits_agree(device):
    """One answer, two shapes."""
    bits = qsa_capabilities()
    names = qsa_capability_names()
    assert len(names) == bin(bits).count("1")
    assert flashinfer.qsa_capabilities() == bits


def test_the_answer_is_per_device(device):
    """The kernels are compiled for an architecture, so the query takes one."""
    assert qsa_capabilities(device) == qsa_capabilities(torch.device("cuda"))
    assert qsa_capabilities(device) == qsa_capabilities(
        torch.device("cuda", torch.cuda.current_device())
    )
    # No argument means the current device, not a device-independent answer.
    assert qsa_capabilities() == qsa_capabilities(device)


def test_there_is_no_self_allocating_support_probe(device):
    """Whether a geometry works is answered by building it, not by a probe.

    A probe would allocate a float workspace of its own -- and how much room
    split-k has changes what the planner lays down, so it would answer for a
    workspace nobody runs with. It would also compile and allocate everything
    the caller is about to compile and allocate again. So there is none, and
    a shape the library cannot serve raises when it is built, with its reason.
    """
    assert not hasattr(flashinfer, "supports_qsa_config")

    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    common = {
        "max_rows": 4,
        "num_qo_heads": 4,
        "num_kv_heads": 1,
        "route_width": 35,
        "q_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
    }
    # A geometry error is a ValueError that says which geometry, not a False.
    with pytest.raises(ValueError, match="multiple of sixteen"):
        flashinfer.QSAAttention(
            workspace,
            head_dim=100,
            kv_data_type=torch.uint8,
            kv_cache_format="nvfp4",
            **common,
        )
    with pytest.raises(ValueError, match="whole number of query heads"):
        flashinfer.QSAAttention(
            workspace,
            head_dim=128,
            kv_data_type=torch.bfloat16,
            kv_cache_format="dense",
            **{**common, "num_qo_heads": 3, "num_kv_heads": 2},
        )
