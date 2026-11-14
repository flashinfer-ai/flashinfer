# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Descriptor and lifetime contracts for Hopper ragged batch cache bounds."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from flashinfer.cudnn import prefill

pytestmark = pytest.mark.cpu_only


class _CudaMetadata(torch.Tensor):
    """CPU storage with a device tag for the hardware eligibility guard only."""

    @property
    def device(self):
        return torch.device("cuda", 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("backend", [92400, 92700])
def test_hopper_direct_half_batch_eligibility(monkeypatch, dtype, group, backend):
    monkeypatch.setattr(prefill, "CUDNN_AVAILABLE", True)
    monkeypatch.setattr(
        prefill,
        "cudnn",
        SimpleNamespace(__version__="1.32.3", backend_version=lambda: backend),
    )
    monkeypatch.setattr(prefill, "get_compute_capability", lambda device: (9, 0))
    monkeypatch.setattr(prefill, "_cudnn_supports_bounded_ragged", lambda **kw: False)
    prefill._cudnn_supports_direct_seqlens.cache_clear()
    try:
        ptr = torch.tensor([0, 256, 512], dtype=torch.int32).as_subclass(_CudaMetadata)
        meta = prefill._PrefillMetadata(
            256, 4096, False, False, batch_offsets_q=ptr, batch_offsets_k=ptr
        ).resolve_from_plan(dtype, 32, 32 // group, 256, 256)
        assert meta._bounded_batch
        assert meta._token_offsets
        assert not meta._bounded_ragged
        assert meta.cu_seq_lens_q is ptr
        assert not torch.cuda.is_initialized()
    finally:
        prefill._cudnn_supports_direct_seqlens.cache_clear()


@pytest.mark.parametrize(
    "change",
    [
        "old_backend",
        "blackwell",
        "d128",
        "different_v_dim",
        "single_q",
        "explicit_q",
        "explicit_kv",
        "element_offsets",
        "fp8",
        "nonintegral_gqa",
        "unsupported_gqa",
        "cpu",
        "paged",
        "window",
        "sinks",
    ],
)
def test_other_declarations_keep_existing_batch_cache(monkeypatch, change):
    backend = 92300 if change == "old_backend" else 92400
    monkeypatch.setattr(prefill, "CUDNN_AVAILABLE", True)
    monkeypatch.setattr(
        prefill,
        "cudnn",
        SimpleNamespace(__version__="1.32.3", backend_version=lambda: backend),
    )
    monkeypatch.setattr(
        prefill,
        "get_compute_capability",
        lambda device: (10, 0) if change == "blackwell" else (9, 0),
    )
    monkeypatch.setattr(prefill, "_cudnn_supports_bounded_ragged", lambda **kw: False)
    monkeypatch.setattr(
        prefill, "_cudnn_supports_paged_token_offsets", lambda *a, **kw: True
    )
    prefill._cudnn_supports_direct_seqlens.cache_clear()
    try:
        ptr = torch.tensor([0, 256, 512], dtype=torch.int32)
        if change != "cpu":
            ptr = ptr.as_subclass(_CudaMetadata)
        lengths = torch.tensor([256, 256], dtype=torch.int32)
        meta = prefill._PrefillMetadata(
            1 if change == "single_q" else 256,
            4096,
            False,
            False,
            batch_offsets_q=ptr,
            batch_offsets_k=ptr,
            actual_seq_lens_q=lengths if change == "explicit_q" else None,
            actual_seq_lens_kv=lengths if change in ("explicit_kv", "paged") else None,
            block_tables=(
                torch.empty(2, 256, dtype=torch.int32) if change == "paged" else None
            ),
        )
        if change == "window":
            meta.window_left = 127
        if change == "sinks":
            meta.sinks = torch.empty(32, dtype=torch.float32)
        meta.resolve_from_plan(
            torch.float8_e4m3fn if change == "fp8" else torch.bfloat16,
            30 if change in ("nonintegral_gqa", "unsupported_gqa") else 32,
            10 if change == "unsupported_gqa" else 8,
            128 if change == "d128" else 256,
            128 if change in ("d128", "different_v_dim") else 256,
            batch_offsets_units="elements" if change == "element_offsets" else "tokens",
        )
        assert not meta._bounded_batch
        assert not torch.cuda.is_initialized()
    finally:
        prefill._cudnn_supports_direct_seqlens.cache_clear()


@pytest.fixture
def plan_boundary(monkeypatch):
    monkeypatch.setattr(prefill, "_cudnn_supports_bounded_ragged", lambda **kw: True)
    monkeypatch.setattr(
        prefill, "_cudnn_supports_direct_seqlens", lambda *a, **kw: True
    )
    monkeypatch.setattr(prefill, "_cudnn_supports_shape_override", lambda: True)
    monkeypatch.setattr(
        prefill,
        "_build_prefill_graph",
        lambda **kw: (SimpleNamespace(get_workspace_size=lambda: 0), []),
        raising=False,
    )

    def metadata(batch, *, bounded_batch=True, bounded_ragged=False, capacity=1024):
        ptr = torch.arange(batch + 1, dtype=torch.int32) * 256
        meta = prefill._PrefillMetadata(
            256,
            4096,
            False,
            True,
            batch_offsets_q=ptr,
            batch_offsets_k=ptr,
            batch_offsets_stats=ptr,
            max_total_num_rows=capacity,
        ).resolve_from_plan(torch.bfloat16, 16, 4, 256, 256)
        meta._bounded_batch = bounded_batch
        meta._bounded_ragged = bounded_ragged
        return meta

    return metadata


@pytest.mark.parametrize(
    "next_batch,expected_reuse", [(2, False), (3, True), (4, True), (5, False)]
)
def test_prepared_graph_reuse_obeys_batch_bucket(
    plan_boundary, next_batch, expected_reuse
):
    q = torch.empty(1024, 16, 256, dtype=torch.bfloat16)
    kv = torch.empty(1024, 4, 256, dtype=q.dtype)
    original = plan_boundary(3)
    plan = prefill._CudnnPrefillPlan.prepare(original, q.dtype, q.device)
    prepared = prefill.prepare_cudnn_batch_prefill(
        q, kv, kv, 0.5, torch.empty(0), metadata=original
    )
    next_plan = prefill._CudnnPrefillPlan.prepare(
        plan_boundary(next_batch), q.dtype, q.device, plan
    )
    assert plan.override == (4, 65536, 65536)
    assert next_plan.execution_shape == (next_batch, 256, 4096)
    assert (plan.override_keys == next_plan.override_keys) is expected_reuse
    assert prepared.matches_plan(q, kv, kv, 0.5, next_plan, True) is expected_reuse
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("source_mode,target_mode", [(False, True), (True, False)])
def test_same_inputs_cannot_reuse_a_different_declaration_mode(
    plan_boundary, source_mode, target_mode
):
    meta = plan_boundary(3, bounded_batch=source_mode)
    first = prefill._CudnnPrefillPlan.prepare(meta, torch.bfloat16, torch.device("cpu"))
    changed = replace(meta, _bounded_batch=target_mode)
    second = prefill._CudnnPrefillPlan.prepare(
        changed, torch.bfloat16, torch.device("cpu"), first
    )
    assert second is not first
    assert second.exact_keys == first.exact_keys
    assert second.override_keys != first.override_keys
    assert first.override[0] == (4 if source_mode else 4096)
    assert second.override[0] == (4 if target_mode else 4096)
    # The detached old plan must not acquire the new metadata mode.
    assert first.metadata._bounded_batch is source_mode


def test_batch_bound_preserves_packed_capacity_and_pointer_rebinding(plan_boundary):
    meta = plan_boundary(3)
    first = prefill._CudnnPrefillPlan.prepare(meta, torch.bfloat16, torch.device("cpu"))
    rebound = prefill._CudnnPrefillPlan.prepare(
        plan_boundary(3), torch.bfloat16, torch.device("cpu"), first
    )
    assert rebound is not first
    assert rebound.override_keys is first.override_keys
    assert (
        rebound.metadata.batch_offsets_q.data_ptr()
        != first.metadata.batch_offsets_q.data_ptr()
    )
    assert first.override_keys[True][0][0][3] is None
    assert first.metadata.max_total_num_rows == 1024
    assert prefill._prefill_graph_total_q(None, 1024, bounded_batch=True) == 1024
    # Existing FROST bounded-ragged capacity remains separate from batch bounds.
    frost = prefill._CudnnPrefillPlan.prepare(
        plan_boundary(3, bounded_batch=False, bounded_ragged=True),
        torch.bfloat16,
        torch.device("cpu"),
    )
    assert frost.override_keys[True][0][0][3] == max(
        1024, frost.override[0] * frost.override[1]
    )
    broad = prefill._CudnnPrefillPlan.prepare(
        plan_boundary(3, bounded_batch=False), torch.bfloat16, torch.device("cpu")
    )
    assert broad.override_keys[True][0][0][3] is None
