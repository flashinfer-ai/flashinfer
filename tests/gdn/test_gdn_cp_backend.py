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

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flashinfer
import flashinfer.gdn_prefill as gdn_prefill
from flashinfer.gdn_kernels.blackwell import gdn_cp_backend as gdn_cp
from flashinfer.jit import gdn_cp_backend as gdn_cp_jit
from flashinfer.utils import is_sm100a_supported

from .reference_delta_rule import delta_rule


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "csrc" / "gdn" / "gdn_cp"


def _assert_oracle_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Require exact non-finite masks and the public tolerance elsewhere."""

    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2, equal_nan=True)


def test_generated_source_inventory_and_hashes() -> None:
    root = _source_root()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == (
        "flashinfer-pr4078-sm100-cp-prefill-standalone-export-v3"
    )
    assert manifest["baseline_revision"] == ("6cb2e70995d92edbc443b1bfc317ecacac907640")
    assert manifest["support_contract"]["external_fallbacks_allowed"] == 0
    assert manifest["support_contract"]["focus_contract"] == {
        "row_count": 150,
        "canonical_stream_sha256": "d4f3fad233af91b8afac35271d6848df8f0f090b08f17807b9e2830139dd37ab",
    }
    assert manifest["support_contract"]["full_regression_contract"] == {
        "row_count": 822,
        "canonical_stream_sha256": "0dff83c89b9a17f67e0a2db9bb9c20ed77506fa3b38cc55d7772864021553592",
    }
    assert manifest["support_contract"]["checkpoint"] == {
        "cu_starts_dtypes": ["int32", "int64"],
        "interval": (
            "zero disables checkpoints; otherwise a positive multiple of 64 "
            "that becomes the CP chunk length"
        ),
        "ordering": "sequence-major complete CP chunk boundaries",
        "shape": "[sum(seq_len // interval), H, 128, 128]",
        "state_dtype": "float32",
    }
    assert manifest["frozen_performance_shape_count"] == 120
    assert len(manifest["frozen_performance_shapes"]) == 120
    assert len(tuple(path for path in root.rglob("*") if path.is_file())) == 72
    assert manifest["launch_order"] == [
        "t_precompute",
        "mn_precompute",
        "state_fixup",
        "cp_prefill",
    ]
    assert manifest["launch_policy"]["tensor_map_abi"].startswith("grid_constant")
    assert len(manifest["cuda_headers"]) == 1
    assert manifest["cuda_headers"][0]["path"] == "cuda/gdn_cp_common.cuh"
    assert manifest["cuda_headers"][0]["sha256"] == (
        "f42b51a944b0b0ed1481a8a05daa48bf4e9b6cd354f9eac97cce17be60fa3af3"
    )
    assert [record["name"] for record in manifest["kernels"]] == [
        "qk_norm",
        "qk_norm_bf16",
        "t_precompute",
        "t_precompute_bf16",
        "t_precompute_gb300_hv48_min6",
        "mn_precompute",
        "mn_precompute_bf16",
        "state_fixup_simt_row4",
        "normalized_final_state",
        "normalized_final_state_bf16",
        "state_gather_fp32",
        "state_gather_fp16",
        "state_gather_bf16",
        "state_gather_fp32_int64",
        "state_gather_fp16_int64",
        "state_gather_bf16_int64",
        "state_scatter_fp32",
        "state_scatter_fp16",
        "state_scatter_bf16",
        "state_scatter_fp32_int64",
        "state_scatter_fp16_int64",
        "state_scatter_bf16_int64",
        "state_fixup_utcmma64",
        "state_fixup_utcmma128",
        "cp_prefill",
        "cp_prefill_checkpoint",
        "cp_prefill_equal_head",
        "cp_prefill_equal_head_checkpoint",
        "cp_prefill_equal_head_h32",
        "cp_prefill_bf16",
        "cp_prefill_generic",
        "cp_prefill_generic_checkpoint",
        "cp_prefill_generic_bf16",
    ]
    assert len(manifest["kernels"]) == 33
    for record in manifest["kernels"]:
        host = record["host_binding"]
        host_path = root / host["path"]
        assert hashlib.sha256(host_path.read_bytes()).hexdigest() == host["sha256"]
        assert host["arg_plan"][-3:] == [
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ]
        for output in record["outputs"]:
            source = root / output["path"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == output["sha256"]
    common_header = root / manifest["cuda_headers"][0]["path"]
    assert common_header.stat().st_size == manifest["cuda_headers"][0]["size_bytes"]
    assert (
        hashlib.sha256(common_header.read_bytes()).hexdigest()
        == (manifest["cuda_headers"][0]["sha256"])
    )


def test_jit_loader_accepts_checked_in_manifest() -> None:
    gdn_cp_jit._manifest.cache_clear()
    assert gdn_cp_jit._manifest()["schema"] == (
        "flashinfer-pr4078-sm100-cp-prefill-standalone-export-v3"
    )


@pytest.mark.parametrize(
    ("arch", "hq", "hv", "expected_t", "expected_chunk", "expected_source"),
    [
        ("sm_100a", 16, 48, "t_precompute", None, None),
        ("sm_103a", 16, 48, "t_precompute_gb300_hv48_min6", None, None),
        ("sm_103a", 16, 64, "t_precompute", 4096, 32768),
        ("sm_103a", 32, 32, "t_precompute", None, None),
    ],
)
def test_long_row_dispatch_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    arch: str,
    hq: int,
    hv: int,
    expected_t: str,
    expected_chunk: int | None,
    expected_source: int | None,
) -> None:
    monkeypatch.setattr(gdn_cp, "_arch_for", lambda _device: arch)
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    q = SimpleNamespace(
        shape=(65536, hq, 128), device=SimpleNamespace(), dtype=torch.float16
    )
    k = SimpleNamespace(shape=(65536, hq, 128))
    v = SimpleNamespace(shape=(65536, hv, 128))
    plan = gdn_cp._build_plan(q, k, v, (65536,))
    assert plan.t_kernel == expected_t
    if expected_chunk is not None:
        assert plan.cp_chunk_len == expected_chunk
    if expected_source is not None:
        assert plan.source_cp_chunk_len == expected_source


@pytest.mark.parametrize(
    ("arch", "seq_lens", "dtype", "heads", "expected_prefill"),
    [
        ("sm_100a", (128,), torch.float16, (1, 1, 1), "cp_prefill_equal_head_checkpoint"),
        ("sm_100a", (128,), torch.float16, (32, 32, 32), "cp_prefill_equal_head_checkpoint"),
        ("sm_103a", (128,), torch.float16, (32, 32, 32), "cp_prefill_equal_head_h32"),
        ("sm_100a", (128,), torch.bfloat16, (1, 1, 1), "cp_prefill_bf16"),
        ("sm_100a", (65,), torch.bfloat16, (1, 1, 2), "cp_prefill_generic_bf16"),
        ("sm_100a", (128, 129), torch.float16, (4, 2, 2), "cp_prefill_generic_checkpoint"),
    ],
)
def test_generic_plan_selects_head_dtype_and_tail_routes(
    monkeypatch: pytest.MonkeyPatch,
    arch: str,
    seq_lens: tuple[int, ...],
    dtype: torch.dtype,
    heads: tuple[int, int, int],
    expected_prefill: str,
) -> None:
    monkeypatch.setattr(gdn_cp, "_arch_for", lambda _device: arch)
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    total = sum(seq_lens)
    hq, hk, hv = heads
    device = SimpleNamespace()
    q = SimpleNamespace(shape=(total, hq, 128), device=device, dtype=dtype)
    k = SimpleNamespace(shape=(total, hk, 128))
    v = SimpleNamespace(shape=(total, hv, 128))

    plan = gdn_cp._build_plan(q, k, v, seq_lens)

    suffix = "_bf16" if dtype == torch.bfloat16 else ""
    assert plan.t_kernel == f"t_precompute{suffix}"
    assert plan.mn_kernel == f"mn_precompute{suffix}"
    assert plan.prefill_kernel == expected_prefill
    assert plan.num_sab_heads == max(hq, hv)


def test_checkpoint_interval_becomes_cp_chunk_and_maps_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdn_cp, "_arch_for", lambda _device: "sm_103a")
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    device = torch.device("cpu")
    q = SimpleNamespace(shape=(640, 1, 128), device=device, dtype=torch.float16)
    k = SimpleNamespace(shape=(640, 1, 128))
    v = SimpleNamespace(shape=(640, 1, 128))

    plan = gdn_cp._build_plan(
        q,
        k,
        v,
        (256, 384),
        checkpoint_every_n_tokens=128,
    )

    assert plan.cp_chunk_len == 128
    assert plan.checkpoint_count == 5
    assert gdn_cp._checkpoint_fixed_state_indices(plan, device).tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_bf16_two_block_factor_keeps_source_and_physical_chunks_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdn_cp, "_arch_for", lambda _device: "sm_100a")
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    q = SimpleNamespace(
        shape=(256, 64, 128), device=SimpleNamespace(), dtype=torch.bfloat16
    )
    k = SimpleNamespace(shape=(256, 64, 128))
    v = SimpleNamespace(shape=(256, 64, 128))

    plan = gdn_cp._build_plan(q, k, v, (256,))

    assert plan.source_cp_chunk_len == 128
    assert plan.cp_chunk_len == 64


def test_all_contract_plans_match_frozen_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    manifest = json.loads(
        (_source_root() / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["frozen_performance_shapes"]) == 120
    for shape in manifest["frozen_performance_shapes"]:
        total = sum(shape["seq_lens"])
        q = SimpleNamespace(
            shape=(total, shape["Hq"], shape["D"]),
            device=SimpleNamespace(),
            dtype=torch.float16,
        )
        k = SimpleNamespace(shape=(total, shape["Hk"], shape["D"]))
        v = SimpleNamespace(shape=(total, shape["Hv"], shape["D"]))
        for arch, expected in shape["dispatch"].items():
            monkeypatch.setattr(gdn_cp, "_arch_for", lambda _device, value=arch: value)
            plan = gdn_cp._build_plan(q, k, v, tuple(shape["seq_lens"]))
            assert {
                "cp_chunk_len": plan.cp_chunk_len,
                "t_kernel": plan.t_kernel,
                "fixup_kernel": plan.fixup_kernel,
                "total_t_blocks": plan.total_t_blocks,
                "total_cp_chunks": plan.total_cp_chunks,
                "t_grid": list(plan.t_grid),
                "fixup_grid": list(plan.fixup_grid),
                "cp_grid": list(plan.cp_grid),
            } == expected


def test_gdn_cp_prepared_launcher_is_not_a_new_public_api() -> None:
    assert not hasattr(gdn_prefill, "prepare_gdn_cp_prefill")
    assert not hasattr(gdn_prefill, "GDNCPPrefill")
    assert not hasattr(gdn_prefill, "GDNCPPrefillPlan")
    assert not hasattr(flashinfer, "prepare_gdn_cp_prefill")
    assert not hasattr(flashinfer, "GDNCPPrefill")
    assert not hasattr(flashinfer, "GDNCPPrefillPlan")


def test_indexed_state_rows_must_not_overlap() -> None:
    inner = 128 * 128
    storage = torch.empty(inner * 2, dtype=torch.float32)
    overlapping = storage.as_strided(
        (2, 1, 128, 128),
        (inner - 1, inner, 128, 1),
    )
    plan = SimpleNamespace(num_sab_heads=1, num_seqs=2)

    with pytest.raises(ValueError, match="non-overlapping"):
        gdn_cp._validate_state(
            overlapping,
            name="initial_state",
            plan=plan,
            device=torch.device("cpu"),
            indexed=True,
        )


def test_arbitrary_positive_non_overlapping_state_layout_is_accepted() -> None:
    storage = torch.empty(140000, dtype=torch.float32)
    state = storage.as_strided(
        (2, 2, 128, 128),
        (70000, 33000, 257, 2),
    )
    plan = SimpleNamespace(num_sab_heads=2, num_seqs=2)

    gdn_cp._validate_state(
        state,
        name="initial_state",
        plan=plan,
        device=torch.device("cpu"),
        indexed=False,
    )
    assert gdn_cp._state_carrier(state).numel() == 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(state.shape, state.stride(), strict=True)
    )


def test_zero_length_plans_keep_semantic_fixup_and_skip_empty_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdn_cp, "_arch_for", lambda _device: "sm_100a")
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    device = SimpleNamespace()

    def plan_for(seq_lens: tuple[int, ...]) -> gdn_cp.GDNCPPrefillPlan:
        total = sum(seq_lens)
        q = SimpleNamespace(
            shape=(total, 1, 128), device=device, dtype=torch.float16
        )
        k = SimpleNamespace(shape=(total, 1, 128))
        v = SimpleNamespace(shape=(total, 1, 128))
        return gdn_cp._build_plan(q, k, v, seq_lens)

    mixed = plan_for((0, 64, 0))
    assert mixed.fixup_kernel == "state_fixup_simt_row4"
    assert mixed.total_t_blocks == 3
    assert mixed.total_cp_chunks == 3

    empty = plan_for((0, 0, 0))
    assert empty.fixup_kernel == "state_fixup_simt_row4"
    assert empty.source_cp_chunk_len == 64
    assert empty.total_t_blocks == 0
    assert empty.total_cp_chunks == 0
    assert empty.t_grid[0] == 0
    assert empty.cp_grid[0] == 0


@pytest.mark.parametrize("io_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_qk_l2norm_in_kernel", [False, True])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_alpha_absent_requires_output_and_requested_final_state_recurrences(
    io_dtype: torch.dtype,
    use_qk_l2norm_in_kernel: bool,
    output_final_state: bool,
) -> None:
    needs_final_state, needs_output, normalize_qk = gdn_cp._recurrence_requirements(
        io_dtype=io_dtype,
        alpha_was_none=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output_final_state=output_final_state,
    )

    assert needs_final_state is output_final_state
    assert needs_output is True
    assert normalize_qk == int(use_qk_l2norm_in_kernel)


@pytest.mark.parametrize("io_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_qk_l2norm_in_kernel", [False, True])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_explicit_alpha_recomputes_requested_bf16_final_state(
    io_dtype: torch.dtype,
    use_qk_l2norm_in_kernel: bool,
    output_final_state: bool,
) -> None:
    needs_final_state, needs_output, normalize_qk = gdn_cp._recurrence_requirements(
        io_dtype=io_dtype,
        alpha_was_none=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output_final_state=output_final_state,
    )

    assert needs_final_state is (
        output_final_state
        and (use_qk_l2norm_in_kernel or io_dtype == torch.bfloat16)
    )
    assert needs_output is (
        io_dtype == torch.bfloat16 and not use_qk_l2norm_in_kernel
    )
    assert normalize_qk == int(use_qk_l2norm_in_kernel)


def test_public_gdn_cp_cache_reuses_equal_metadata_and_rebinds_tensor_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparations: list[dict[str, object]] = []
    dynamic_launches: list[dict[str, object]] = []
    replays: list[str] = []

    class FakePrepared:
        def launch_with_bindings(self, **kwargs) -> None:
            dynamic_launches.append(kwargs)

        def replay(self) -> None:
            replays.append("replay")

    def fake_prepare(*_args, **kwargs):
        preparations.append(kwargs)
        return FakePrepared()

    monkeypatch.setattr(gdn_cp, "prepare_gdn_cp_prefill", fake_prepare)
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(gdn_cp.torch.cuda, "is_current_stream_capturing", lambda: False)
    gdn_cp._public_key = None
    gdn_cp._public_metadata_binding = None
    gdn_cp._public_prepared = None

    cu_seqlens = torch.tensor([0, 2], dtype=torch.int64)
    alpha = torch.ones((2, 1), dtype=torch.float32)
    beta = torch.ones((2, 1), dtype=torch.float32)
    initial_state = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
    for _ in range(2):
        q = torch.zeros((2, 1, 128), dtype=torch.float16)
        output = torch.empty_like(q)
        output_state = torch.empty_like(initial_state)
        gdn_cp.chunk_gated_delta_rule_gdn_cp_sm100(
            output,
            output_state,
            q,
            q,
            q,
            alpha,
            beta,
            cu_seqlens,
            0.125,
            initial_state=initial_state,
            state_indices=None,
            output_final_state=True,
        )

    assert len(preparations) == 1
    assert preparations[0]["_capture_graph"] is False
    assert len(dynamic_launches) == 1
    assert dynamic_launches[0]["output"] is output
    assert dynamic_launches[0]["q"] is q

    gdn_cp.chunk_gated_delta_rule_gdn_cp_sm100(
        output,
        output_state,
        q,
        q,
        q,
        alpha,
        beta,
        cu_seqlens.clone(),
        0.125,
        initial_state=initial_state,
        state_indices=None,
        output_final_state=True,
    )
    assert len(preparations) == 1

    gdn_cp._reset_gdn_cp_prefill_cache()
    assert gdn_cp._public_key is None
    assert gdn_cp._public_metadata_binding is None
    assert gdn_cp._public_prepared is None

    gdn_cp.chunk_gated_delta_rule_gdn_cp_sm100(
        output,
        initial_state,
        q,
        q,
        q,
        alpha,
        beta,
        cu_seqlens,
        0.125,
        initial_state=initial_state,
        state_indices=None,
        output_final_state=True,
    )
    assert replays == ["replay"]


def test_direct_binding_refreshes_unnormalized_qk_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = object.__new__(gdn_cp.GDNCPPrefill)
    prepared._graph = None
    prepared._stream = SimpleNamespace(cuda_stream=7)
    prepared._qk_norm = None
    prepared._gather = None
    prepared._scatter = None
    prepared.q_normalized = torch.full((1,), -1.0)
    prepared.k_normalized = torch.full((1,), -2.0)
    prepared.alpha = torch.ones((1,))
    prepared.beta = torch.ones((1,))
    prepared._retained_tensors = ()
    prepared._refresh_retained_tensors = lambda: None
    launched: list[tuple[torch.Tensor, torch.Tensor]] = []
    prepared._launch_direct = lambda: launched.append(
        (prepared.q_normalized, prepared.k_normalized)
    )
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(gdn_cp.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(gdn_cp.tvm_ffi, "use_torch_stream", nullcontext)

    q = torch.zeros((1,))
    k = torch.ones((1,))
    prepared.launch_with_bindings(
        q=q,
        k=k,
        v=torch.full((1,), 2.0),
        alpha=torch.full((1,), 3.0),
        beta=torch.full((1,), 4.0),
        initial_state=None,
        output=torch.empty((1,)),
        output_state=None,
        state_checkpoints=None,
    )

    assert launched == [(q, k)]
    assert prepared.q_normalized is q
    assert prepared.k_normalized is k


def test_public_gdn_cp_cache_accepts_inference_metadata_during_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparations: list[str] = []
    dynamic_launches: list[str] = []
    capturing = False

    class FakePrepared:
        def launch_with_bindings(self, **_kwargs) -> None:
            dynamic_launches.append("launch")

        def replay(self) -> None:
            pass

    def fake_prepare(*_args, **_kwargs):
        preparations.append("prepare")
        return FakePrepared()

    monkeypatch.setattr(gdn_cp, "prepare_gdn_cp_prefill", fake_prepare)
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing,
    )
    gdn_cp._reset_gdn_cp_prefill_cache()

    with torch.inference_mode():
        cu_seqlens = torch.tensor([0, 2], dtype=torch.int64)
        state_indices = torch.tensor([0], dtype=torch.int64)
        checkpoint_cu_starts = torch.tensor([0, 1], dtype=torch.int64)
    assert all(
        torch.is_inference(tensor)
        for tensor in (cu_seqlens, state_indices, checkpoint_cu_starts)
    )

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    output = torch.empty_like(q)
    alpha = torch.ones((2, 1), dtype=torch.float32)
    beta = torch.ones((2, 1), dtype=torch.float32)
    state = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
    output_state = torch.empty_like(state)
    state_checkpoints = torch.empty_like(state)

    def invoke() -> None:
        gdn_cp.chunk_gated_delta_rule_gdn_cp_sm100(
            output,
            output_state,
            q,
            q,
            q,
            alpha,
            beta,
            cu_seqlens,
            0.125,
            initial_state=state,
            state_indices=state_indices,
            output_final_state=True,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=64,
        )

    invoke()
    capturing = True
    invoke()

    assert preparations == ["prepare"]
    assert dynamic_launches == ["launch"]
    assert gdn_cp._public_metadata_binding is not None
    assert gdn_cp._public_metadata_binding[3:6] == (None, None, None)


def test_public_gdn_cp_cache_preserves_alpha_absence_across_address_rebinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparations: list[torch.Tensor | None] = []
    dynamic_launches: list[dict[str, object]] = []

    class FakePrepared:
        def launch_with_bindings(self, **kwargs) -> None:
            dynamic_launches.append(kwargs)

        def replay(self) -> None:
            pass

    def fake_prepare(
        _q,
        _k,
        _v,
        alpha,
        _beta,
        _cu_seqlens,
        _initial_state,
        **_kwargs,
    ):
        preparations.append(alpha)
        return FakePrepared()

    monkeypatch.setattr(gdn_cp, "prepare_gdn_cp_prefill", fake_prepare)
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(gdn_cp.torch.cuda, "is_current_stream_capturing", lambda: False)
    gdn_cp._reset_gdn_cp_prefill_cache()

    cu_seqlens = torch.tensor([0, 2], dtype=torch.int64)
    beta = torch.ones((2, 1), dtype=torch.float32)
    state = torch.zeros((1, 1, 128, 128), dtype=torch.float32)

    def invoke(q: torch.Tensor, alpha: torch.Tensor | None) -> None:
        gdn_cp.chunk_gated_delta_rule_gdn_cp_sm100(
            torch.empty_like(q),
            torch.empty_like(state),
            q,
            q,
            q,
            alpha,
            beta,
            cu_seqlens,
            0.125,
            initial_state=state,
            state_indices=None,
            output_final_state=True,
        )

    first_q = torch.zeros((2, 1, 128), dtype=torch.float16)
    rebound_q = torch.ones_like(first_q)
    invoke(first_q, None)
    invoke(rebound_q, None)
    explicit_alpha = torch.ones((2, 1), dtype=torch.float32)
    invoke(rebound_q, explicit_alpha)

    assert len(preparations) == 2
    assert preparations[0] is None
    assert preparations[1] is explicit_alpha
    assert len(dynamic_launches) == 1
    assert dynamic_launches[0]["q"] is rebound_q
    assert dynamic_launches[0]["alpha"] is None


def test_public_gdn_cp_cache_detects_metadata_writes_without_version_bump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparations: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    class FakePrepared:
        def launch_with_bindings(self, **_kwargs) -> None:
            pass

        def replay(self) -> None:
            pass

    def fake_prepare(
        _q,
        _k,
        _v,
        _alpha,
        _beta,
        cu_seqlens,
        _initial_state,
        *,
        state_indices,
        checkpoint_cu_starts,
        **_kwargs,
    ):
        preparations.append(
            (
                tuple(int(value) for value in cu_seqlens.tolist()),
                tuple(int(value) for value in state_indices.tolist()),
                tuple(int(value) for value in checkpoint_cu_starts.tolist()),
            )
        )
        return FakePrepared()

    monkeypatch.setattr(gdn_cp, "prepare_gdn_cp_prefill", fake_prepare)
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(gdn_cp.torch.cuda, "is_current_stream_capturing", lambda: False)
    gdn_cp._reset_gdn_cp_prefill_cache()

    q = torch.zeros((4, 1, 128), dtype=torch.float16)
    output = torch.empty_like(q)
    alpha = torch.ones((4, 1), dtype=torch.float32)
    beta = torch.ones((4, 1), dtype=torch.float32)
    state = torch.zeros((2, 1, 128, 128), dtype=torch.float32)
    output_state = torch.empty_like(state)
    state_checkpoints = torch.empty((2, 1, 128, 128), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1, 4], dtype=torch.int64)
    state_indices = torch.tensor([0, 1], dtype=torch.int32)
    checkpoint_cu_starts = torch.tensor([0, 1, 2], dtype=torch.int64)

    def invoke() -> None:
        gdn_cp.chunk_gated_delta_rule_gdn_cp_sm100(
            output,
            output_state,
            q,
            q,
            q,
            alpha,
            beta,
            cu_seqlens,
            0.125,
            initial_state=state,
            state_indices=state_indices,
            output_final_state=True,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=64,
        )

    invoke()
    cu_version = int(cu_seqlens._version)
    indices_version = int(state_indices._version)
    checkpoint_version = int(checkpoint_cu_starts._version)
    cu_seqlens.numpy()[1] = 2
    state_indices.numpy()[:] = (1, 0)
    checkpoint_cu_starts.numpy()[1] = 0
    assert int(cu_seqlens._version) == cu_version
    assert int(state_indices._version) == indices_version
    assert int(checkpoint_cu_starts._version) == checkpoint_version
    invoke()

    assert preparations == [
        ((0, 1, 4), (0, 1), (0, 1, 2)),
        ((0, 2, 4), (1, 0), (0, 0, 2)),
    ]


def test_public_gdn_cp_cache_requires_warmed_metadata_during_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparations: list[str] = []
    dynamic_launches: list[str] = []
    capturing = False

    class FakePrepared:
        def launch_with_bindings(self, **_kwargs) -> None:
            dynamic_launches.append("launch")

        def replay(self) -> None:
            pass

    def fake_prepare(*_args, **_kwargs):
        preparations.append("prepare")
        return FakePrepared()

    monkeypatch.setattr(gdn_cp, "prepare_gdn_cp_prefill", fake_prepare)
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(
        gdn_cp.torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing,
    )
    gdn_cp._reset_gdn_cp_prefill_cache()

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    output = torch.empty_like(q)
    alpha = torch.ones((2, 1), dtype=torch.float32)
    beta = torch.ones((2, 1), dtype=torch.float32)
    state = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
    output_state = torch.empty_like(state)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int64)

    def invoke(metadata: torch.Tensor) -> None:
        gdn_cp.chunk_gated_delta_rule_gdn_cp_sm100(
            output,
            output_state,
            q,
            q,
            q,
            alpha,
            beta,
            metadata,
            0.125,
            initial_state=state,
            state_indices=None,
            output_final_state=True,
        )

    invoke(cu_seqlens)
    capturing = True
    invoke(cu_seqlens)
    assert preparations == ["prepare"]
    assert dynamic_launches == ["launch"]

    with pytest.raises(RuntimeError, match="same unchanged tensors"):
        invoke(cu_seqlens.clone())
    cu_seqlens.add_(0)
    with pytest.raises(RuntimeError, match="same unchanged tensors"):
        invoke(cu_seqlens)


@pytest.mark.parametrize(
    ("use_cp", "heuristic_matches", "expected_route"),
    [
        ("auto", True, "gdn_cp"),
        ("auto", False, "non_cp"),
        (True, False, "gdn_cp"),
        (False, True, "non_cp"),
    ],
)
def test_public_dispatch_preserves_auto_and_explicit_cp_routes(
    monkeypatch: pytest.MonkeyPatch,
    use_cp: str | bool,
    heuristic_matches: bool,
    expected_route: str,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "NVIDIA B200")
    monkeypatch.setattr(
        gdn_prefill,
        "should_use_cp_host",
        lambda *_args, **_kwargs: heuristic_matches,
    )
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")
    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_gdn_cp_sm100",
        lambda *_args, **_kwargs: calls.append("gdn_cp"),
    )
    monkeypatch.setattr(
        gdn_prefill,
        "chunk_gated_delta_rule_sm100",
        lambda *_args, **_kwargs: calls.append("non_cp"),
    )

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    output = gdn_prefill.chunk_gated_delta_rule(
        q,
        q,
        q,
        cu_seqlens=cu_seqlens,
        use_cp=use_cp,
    )

    assert output.shape == q.shape
    assert calls == [expected_route]


@pytest.mark.parametrize("cuda_version", ["12.8", "12.9"])
def test_public_dispatch_allows_gdn_cp_before_cuda_13(
    monkeypatch: pytest.MonkeyPatch,
    cuda_version: str,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "NVIDIA B200")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", cuda_version)
    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_gdn_cp_sm100",
        lambda *_args, **_kwargs: calls.append("gdn_cp"),
    )

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    output = gdn_prefill.chunk_gated_delta_rule(
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        use_cp=True,
    )

    assert output.shape == q.shape
    assert calls == ["gdn_cp"]


@pytest.mark.parametrize("cuda_version", ["11.8", "12.7"])
def test_public_dispatch_rejects_gdn_cp_before_cuda_12_8(
    monkeypatch: pytest.MonkeyPatch,
    cuda_version: str,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "NVIDIA B200")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", cuda_version)
    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_gdn_cp_sm100",
        lambda *_args, **_kwargs: calls.append("gdn_cp"),
    )

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    with pytest.raises(ValueError, match="GDN CP SM100 kernel requires CUDA 12.8"):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            use_cp=True,
        )

    assert calls == []


@pytest.mark.parametrize("cuda_version", ["12.8", "12.9"])
def test_public_dispatch_keeps_cuda_13_gate_for_sm100_dsl(
    monkeypatch: pytest.MonkeyPatch,
    cuda_version: str,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "NVIDIA B200")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", cuda_version)
    monkeypatch.setattr(
        gdn_prefill,
        "cp_delta_rule_dsl_sm100",
        lambda *_args, **_kwargs: calls.append("cute_cp"),
    )

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    initial_state = torch.empty((1, 1, 128, 128), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="SM100 DSL kernel requires CUDA 13"):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            q,
            q,
            initial_state=initial_state,
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            use_cp=True,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("extension", "expected_route"),
    [
        ("checkpoint", "gdn_cp"),
        ("fp8_state", "cute_cp"),
        ("cp_chunk_len", "gdn_cp"),
    ],
)
def test_public_dispatch_preserves_upstream_sm100_cp_extensions(
    monkeypatch: pytest.MonkeyPatch,
    extension: str,
    expected_route: str,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "NVIDIA B200")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")
    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_gdn_cp_sm100",
        lambda *_args, **_kwargs: calls.append("gdn_cp"),
    )
    monkeypatch.setattr(
        gdn_prefill,
        "cp_delta_rule_dsl_sm100",
        lambda *_args, **_kwargs: calls.append("cute_cp"),
    )

    total = 64 if extension == "checkpoint" else 2
    q = torch.zeros((total, 1, 128), dtype=torch.float16)
    kwargs: dict[str, object] = {}
    if extension == "checkpoint":
        kwargs.update(
            state_checkpoints=torch.empty((1, 1, 128, 128), dtype=torch.float32),
            checkpoint_cu_starts=torch.tensor([0, 1], dtype=torch.int64),
            checkpoint_every_n_tokens=64,
        )
    elif extension == "fp8_state":
        kwargs["initial_state"] = torch.empty(
            (1, 1, 128, 128), dtype=torch.float8_e4m3fn
        )
    else:
        kwargs["_cp_chunk_len"] = 64

    output = gdn_prefill.chunk_gated_delta_rule(
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, total], dtype=torch.int32),
        use_cp=True,
        **kwargs,
    )

    assert output.shape == q.shape
    assert calls == [expected_route]


def test_public_dispatch_fails_closed_when_gdn_cp_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "NVIDIA B200")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")
    monkeypatch.setattr(gdn_prefill, "_chunk_gated_delta_rule_gdn_cp_sm100", None)

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    with pytest.raises(ValueError, match="GDN CP SM100 kernel"):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            use_cp=True,
        )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_read_seq_lens_uses_adjacent_offsets(dtype: torch.dtype) -> None:
    cu_seqlens = torch.tensor([0, 2, 5], dtype=dtype)

    assert gdn_cp._read_seq_lens(cu_seqlens, total_tokens=5, expected=(2, 3)) == (2, 3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="requires an exact SM100a or SM103a GPU",
)
@pytest.mark.parametrize(
    ("total", "hq", "hv"),
    [
        (2048, 2, 8),
        (65536, 16, 48),
        (65536, 16, 64),
        (65536, 32, 32),
    ],
)
def test_frozen_graph_matches_pr4078_and_preserves_inputs(
    total: int,
    hq: int,
    hv: int,
) -> None:
    from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import (
        cp_delta_rule_dsl_sm100,
    )

    torch.manual_seed(4078 + hq * 100 + hv)
    device = torch.device("cuda")
    seq_lens = (total,)
    dim = 128
    q = torch.randn((total, hq, dim), dtype=torch.float16, device=device)
    k = torch.nn.functional.normalize(
        torch.randn((total, hq, dim), dtype=torch.float32, device=device),
        p=2.0,
        dim=-1,
    ).to(torch.float16)
    v = torch.randn((total, hv, dim), dtype=torch.float16, device=device)
    alpha = 1.0 - torch.rand((total, hv), dtype=torch.float32, device=device) / total
    beta = torch.rand((total, hv), dtype=torch.float32, device=device).sigmoid()
    cu_seqlens = torch.tensor([0, total], dtype=torch.int64, device=device)
    initial_state = torch.randn((1, hv, dim, dim), dtype=torch.float32, device=device)
    output_state = torch.empty_like(initial_state)
    snapshots = tuple(
        tensor.clone() for tensor in (q, k, v, alpha, beta, cu_seqlens, initial_state)
    )

    expected_output = torch.empty((total, hv, dim), dtype=torch.float16, device=device)
    expected_state = torch.empty_like(initial_state)
    cp_delta_rule_dsl_sm100(
        expected_output,
        expected_state,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        1.0 / dim**0.5,
        initial_state=initial_state,
        max_seqlen=total,
    )
    prepared = gdn_cp.prepare_gdn_cp_prefill(
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        initial_state,
        seq_lens=seq_lens,
        output_state=output_state,
    )
    output, final_state = prepared.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final_state, expected_state, atol=1e-2, rtol=1e-2)
    for observed, before in zip(
        (q, k, v, alpha, beta, cu_seqlens, initial_state), snapshots, strict=True
    ):
        assert torch.equal(observed, before)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="requires an exact SM100a or SM103a GPU",
)
def test_public_checkpoint_matches_oracle_on_caller_stream_and_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(4436)
    seq_lens = (128, 256)
    interval = 128
    total = sum(seq_lens)
    hq, hk, hv = 4, 1, 1
    state_heads = max(hq, hv)
    q = torch.randn((total, hq, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.nn.functional.normalize(
        torch.randn((total, hk, 128), dtype=torch.float32, device="cuda"),
        p=2.0,
        dim=-1,
    ).to(torch.bfloat16)
    v = torch.randn((total, hv, 128), dtype=torch.bfloat16, device="cuda")
    alpha = torch.rand((total, state_heads), dtype=torch.float32, device="cuda")
    beta = torch.rand((total, state_heads), dtype=torch.float32, device="cuda")
    cu_seqlens = torch.tensor([0, 128, total], dtype=torch.int64, device="cuda")
    checkpoint_cu_starts = torch.tensor([0, 1, 3], dtype=torch.int64, device="cuda")
    output = torch.empty((total, state_heads, 128), dtype=q.dtype, device="cuda")
    output_state = torch.empty(
        (len(seq_lens), state_heads, 128, 128),
        dtype=torch.float32,
        device="cuda",
    )
    checkpoints = torch.empty(
        (3, state_heads, 128, 128), dtype=torch.float32, device="cuda"
    )

    q_semantic = q.float()
    q_semantic *= torch.rsqrt(
        q_semantic.square().sum(dim=-1, keepdim=True) + 1.0e-6
    )
    k_semantic = k.float()
    k_semantic *= torch.rsqrt(
        k_semantic.square().sum(dim=-1, keepdim=True) + 1.0e-6
    )
    q_kernel = q_semantic.to(q.dtype)
    k_kernel = k_semantic.to(k.dtype)
    expected_output_fp32, _ = delta_rule(
        q_kernel,
        k_kernel,
        v,
        list(seq_lens),
        alpha=alpha,
        beta=beta,
        scale_factor=1.0 / 128**0.5,
        state_dtype=torch.float32,
    )
    expected_output = expected_output_fp32.to(q.dtype)
    _, expected_state_hkv = delta_rule(
        q_semantic,
        k_semantic,
        v,
        list(seq_lens),
        alpha=alpha,
        beta=beta,
        scale_factor=1.0 / 128**0.5,
        state_dtype=torch.float32,
    )
    expected_state = expected_state_hkv.transpose(-1, -2).contiguous()
    expected_checkpoints = []
    token_start = 0
    for seq_len in seq_lens:
        for prefix_len in range(interval, seq_len + 1, interval):
            _, prefix_state_hkv = delta_rule(
                q_kernel[token_start : token_start + prefix_len].contiguous(),
                k_kernel[token_start : token_start + prefix_len].contiguous(),
                v[token_start : token_start + prefix_len].contiguous(),
                [prefix_len],
                alpha=alpha[token_start : token_start + prefix_len].contiguous(),
                beta=beta[token_start : token_start + prefix_len].contiguous(),
                scale_factor=1.0 / 128**0.5,
                state_dtype=torch.float32,
            )
            expected_checkpoints.append(
                prefix_state_hkv[0].transpose(-1, -2).contiguous()
            )
        token_start += seq_len
    expected_checkpoints_tensor = torch.stack(expected_checkpoints)

    def forbidden_external(*_args, **_kwargs):
        raise AssertionError("checkpoint route left the GDN CP backend")

    monkeypatch.setattr(gdn_prefill, "cp_delta_rule_dsl_sm100", forbidden_external)
    gdn_cp._public_key = None
    gdn_cp._public_prepared = None

    def invoke() -> None:
        result = gdn_prefill.chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            1.0 / 128**0.5,
            None,
            True,
            cu_seqlens,
            True,
            output=output,
            output_state=output_state,
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=interval,
            use_cp=True,
            _cp_chunk_len=interval,
        )
        assert result[0] is output
        assert result[1] is output_state

    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        invoke()
    stream.synchronize()
    direct_output = output.clone()
    direct_state = output_state.clone()
    direct_checkpoints = checkpoints.clone()

    assert gdn_cp._public_prepared is not None
    assert gdn_cp._public_prepared.plan.checkpoint_count == 3
    assert gdn_cp._public_prepared._checkpoint is not None
    _assert_oracle_close(direct_output, expected_output)
    _assert_oracle_close(direct_state, expected_state)
    _assert_oracle_close(direct_checkpoints, expected_checkpoints_tensor)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        invoke()
    torch.cuda.synchronize()
    output.fill_(float("nan"))
    output_state.fill_(float("nan"))
    checkpoints.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    _assert_oracle_close(output, direct_output)
    _assert_oracle_close(output_state, direct_state)
    _assert_oracle_close(checkpoints, direct_checkpoints)


def _allocate_state_pool(
    rows: int,
    heads: int,
    *,
    dtype: torch.dtype,
    padding: int,
) -> torch.Tensor:
    if padding == 0:
        return torch.empty((rows, heads, 128, 128), dtype=dtype, device="cuda")
    row_stride = heads * 128 * 128 + padding
    storage = torch.empty((rows * row_stride,), dtype=dtype, device="cuda")
    return storage.as_strided(
        (rows, heads, 128, 128),
        (row_stride, 128 * 128, 128, 1),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="requires an SM100a-compatible GPU",
)
@pytest.mark.parametrize(
    "case",
    [
        {
            "label": "bf16_gva_tail",
            "seq_lens": (65,),
            "heads": (1, 1, 2),
            "io_dtype": torch.bfloat16,
            "cu_dtype": torch.int64,
            "initial_dtype": torch.float16,
            "output_state_dtype": None,
            "indexed": False,
            "inplace": False,
            "padding": 0,
            "alpha": False,
            "beta": False,
            "normalize_k": True,
            "scale": 0.125,
            "seed": 50003,
        },
        {
            "label": "fp16_indexed_padded_state",
            "seq_lens": (511,),
            "heads": (2, 2, 8),
            "io_dtype": torch.float16,
            "cu_dtype": torch.int32,
            "initial_dtype": torch.float16,
            "output_state_dtype": torch.bfloat16,
            "indexed": True,
            "index_dtype": torch.int64,
            "inplace": False,
            "padding": 96,
            "alpha": True,
            "beta": False,
            "normalize_k": True,
            "scale": 0.125,
            "seed": 50027,
        },
        {
            "label": "bf16_gqa_indexed_inplace",
            "seq_lens": (8,) * 32,
            "heads": (4, 1, 1),
            "io_dtype": torch.bfloat16,
            "cu_dtype": torch.int64,
            "initial_dtype": torch.bfloat16,
            "output_state_dtype": torch.bfloat16,
            "indexed": True,
            "index_dtype": torch.int32,
            "inplace": True,
            "padding": 0,
            "alpha": False,
            "beta": False,
            "normalize_k": False,
            "scale": 1.0 / 128**0.5,
            "seed": 50028,
        },
    ],
    ids=lambda case: case["label"],
)
def test_generic_backend_matches_pr4078_state_and_lifecycle(
    case: dict[str, object],
) -> None:
    from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import (
        cp_delta_rule_dsl_sm100,
    )

    torch.manual_seed(int(case["seed"]))
    seq_lens = tuple(int(value) for value in case["seq_lens"])
    total = sum(seq_lens)
    num_seqs = len(seq_lens)
    hq, hk, hv = (int(value) for value in case["heads"])
    state_heads = max(hq, hv)
    io_dtype = case["io_dtype"]
    q = torch.randn((total, hq, 128), dtype=io_dtype, device="cuda")
    k_fp32 = torch.randn((total, hk, 128), dtype=torch.float32, device="cuda")
    if case["normalize_k"]:
        k_fp32 = torch.nn.functional.normalize(k_fp32, p=2.0, dim=-1)
    k = k_fp32.to(io_dtype)
    v = torch.randn((total, hv, 128), dtype=io_dtype, device="cuda")
    alpha = (
        torch.rand((total, state_heads), dtype=torch.float32, device="cuda")
        if case["alpha"]
        else None
    )
    beta = (
        torch.rand((total, state_heads), dtype=torch.float32, device="cuda").sigmoid()
        if case["beta"]
        else None
    )
    cu_values = [0]
    for length in seq_lens:
        cu_values.append(cu_values[-1] + length)
    cu_seqlens = torch.tensor(cu_values, dtype=case["cu_dtype"], device="cuda")
    pool_rows = num_seqs + 5 if case["indexed"] else num_seqs
    state_values = torch.randn(
        (pool_rows, state_heads, 128, 128),
        dtype=case["initial_dtype"],
        device="cuda",
    )
    candidate_initial = _allocate_state_pool(
        pool_rows,
        state_heads,
        dtype=case["initial_dtype"],
        padding=int(case["padding"]),
    )
    reference_initial = _allocate_state_pool(
        pool_rows,
        state_heads,
        dtype=case["initial_dtype"],
        padding=int(case["padding"]),
    )
    candidate_initial.copy_(state_values)
    reference_initial.copy_(state_values)
    state_indices = (
        torch.tensor(
            [(index + 2) % pool_rows for index in range(num_seqs)],
            dtype=case.get("index_dtype", torch.int32),
            device="cuda",
        )
        if case["indexed"]
        else None
    )
    output_state_dtype = case["output_state_dtype"]
    if output_state_dtype is None:
        candidate_state = reference_state = None
    elif case["inplace"]:
        candidate_state = candidate_initial
        reference_state = reference_initial
    else:
        candidate_state = _allocate_state_pool(
            pool_rows,
            state_heads,
            dtype=output_state_dtype,
            padding=int(case["padding"]),
        )
        reference_state = _allocate_state_pool(
            pool_rows,
            state_heads,
            dtype=output_state_dtype,
            padding=int(case["padding"]),
        )
        candidate_state.zero_()
        reference_state.zero_()
    output = torch.empty((total, state_heads, 128), dtype=io_dtype, device="cuda")
    expected_output = torch.empty_like(output)
    read_only = tuple(
        tensor.clone()
        for tensor in (q, k, v, alpha, beta, cu_seqlens)
        if tensor is not None
    )
    candidate_initial_before = candidate_initial.clone()
    candidate_state_before = (
        candidate_state.clone() if candidate_state is not None else None
    )
    baseline_alpha = (
        alpha
        if alpha is not None
        else torch.ones((total, state_heads), dtype=torch.float32, device="cuda")
    )
    baseline_beta = (
        beta
        if beta is not None
        else torch.ones((total, state_heads), dtype=torch.float32, device="cuda")
    )

    cp_delta_rule_dsl_sm100(
        expected_output,
        reference_state,
        q,
        k,
        v,
        baseline_alpha,
        baseline_beta,
        cu_seqlens,
        float(case["scale"]),
        initial_state=reference_initial,
        state_indices=state_indices,
        max_seqlen=total,
    )
    prepared = gdn_cp.prepare_gdn_cp_prefill(
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        candidate_initial,
        seq_lens=seq_lens,
        output=output,
        output_state=candidate_state,
        state_indices=state_indices,
        scale=float(case["scale"]),
        output_final_state=output_state_dtype is not None,
    )
    actual_output, actual_state = prepared.replay()
    torch.cuda.synchronize()

    _assert_oracle_close(actual_output, expected_output)
    if reference_state is None:
        assert actual_state is None
    else:
        assert actual_state is candidate_state
        _assert_oracle_close(actual_state, reference_state)
    observed_read_only = tuple(
        tensor for tensor in (q, k, v, alpha, beta, cu_seqlens) if tensor is not None
    )
    for observed, before in zip(observed_read_only, read_only, strict=True):
        assert torch.equal(observed, before)
    if not case["inplace"]:
        assert torch.equal(candidate_initial, candidate_initial_before)
    if state_indices is not None and candidate_state is not None:
        selected = set(int(value) for value in state_indices.cpu().tolist())
        assert candidate_state_before is not None
        for row in range(pool_rows):
            if row not in selected:
                assert torch.equal(candidate_state[row], candidate_state_before[row])


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="requires an SM100a-compatible GPU",
)
@pytest.mark.parametrize(
    ("seq_lens", "cu_dtype"),
    [
        ((0, 0, 0), torch.int64),
        ((0, 64, 0), torch.int32),
    ],
)
def test_public_dispatcher_preserves_zero_length_sequences_and_state_pool(
    monkeypatch: pytest.MonkeyPatch,
    seq_lens: tuple[int, ...],
    cu_dtype: torch.dtype,
) -> None:
    """Keep empty rows semantic while staying exclusively on the GDN CP route."""

    torch.manual_seed(504079 + sum(seq_lens))
    total = sum(seq_lens)
    q = torch.randn((total, 1, 128), dtype=torch.float16, device="cuda")
    k = torch.nn.functional.normalize(
        torch.randn((total, 1, 128), dtype=torch.float32, device="cuda"),
        p=2.0,
        dim=-1,
    ).to(torch.float16)
    v = torch.randn((total, 1, 128), dtype=torch.float16, device="cuda")
    cu_values = [0]
    for length in seq_lens:
        cu_values.append(cu_values[-1] + length)
    cu_seqlens = torch.tensor(cu_values, dtype=cu_dtype, device="cuda")
    state_indices = torch.tensor([5, 1, 3], dtype=torch.int64, device="cuda")
    initial_state = _allocate_state_pool(
        7,
        1,
        dtype=torch.float32,
        padding=96,
    )
    initial_state.copy_(torch.randn_like(initial_state))
    output_state = _allocate_state_pool(
        7,
        1,
        dtype=torch.float32,
        padding=160,
    )
    output_state.copy_(torch.randn_like(output_state))
    expected_state = output_state.clone()
    expected_output = torch.empty_like(q)
    token_start = 0
    for sequence_index, seq_len in enumerate(seq_lens):
        pool_row = int(state_indices[sequence_index])
        state_hkv = initial_state[pool_row].transpose(-1, -2).clone()
        for token in range(token_start, token_start + seq_len):
            previous_value = torch.einsum("hk,hkv->hv", k[token].float(), state_hkv)
            residual = v[token].float() - previous_value
            state_hkv += k[token].float().unsqueeze(-1) * residual.unsqueeze(-2)
            expected_output[token] = (
                0.125
                * torch.einsum("hk,hkv->hv", q[token].float(), state_hkv)
            ).to(expected_output.dtype)
        expected_state[pool_row].copy_(state_hkv.transpose(-1, -2))
        token_start += seq_len

    immutable = tuple(
        tensor.clone() for tensor in (q, k, v, cu_seqlens, state_indices, initial_state)
    )
    output_state_before = output_state.clone()
    route_calls: list[str] = []
    real_gdn_cp = gdn_prefill._chunk_gated_delta_rule_gdn_cp_sm100

    def observed_gdn_cp(*args, **kwargs):
        route_calls.append("gdn_cp")
        return real_gdn_cp(*args, **kwargs)

    def forbidden_external(*_args, **_kwargs):
        raise AssertionError("zero-length public route left GDN CP")

    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_gdn_cp_sm100",
        observed_gdn_cp,
    )
    monkeypatch.setattr(
        gdn_prefill,
        "chunk_gated_delta_rule_sm100",
        forbidden_external,
    )
    monkeypatch.setattr(
        gdn_prefill,
        "cp_delta_rule_dsl_sm100",
        forbidden_external,
        raising=False,
    )
    gdn_cp._reset_gdn_cp_prefill_cache()
    output = torch.empty_like(q)
    actual_output, actual_state = gdn_prefill.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=None,
        beta=None,
        scale=0.125,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=output,
        output_state=output_state,
        use_cp=True,
        state_indices=state_indices,
    )
    torch.cuda.synchronize()

    assert route_calls == ["gdn_cp"]
    assert actual_output is output
    assert actual_state is output_state
    _assert_oracle_close(actual_output, expected_output)
    _assert_oracle_close(actual_state, expected_state)
    for observed, before in zip(
        (q, k, v, cu_seqlens, state_indices, initial_state), immutable, strict=True
    ):
        assert torch.equal(observed, before)
    selected = set(int(value) for value in state_indices.cpu().tolist())
    for row in range(output_state.shape[0]):
        if row not in selected:
            assert torch.equal(output_state[row], output_state_before[row])


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="requires an SM100a-compatible GPU",
)
def test_public_dispatcher_uses_only_gdn_cp_for_indexed_inplace_gqa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the GDN CP public use_cp=True route with no external arm."""

    torch.manual_seed(504078)
    seq_lens = (64, 129)
    total = sum(seq_lens)
    hq, hk, hv = 4, 1, 1
    state_heads = max(hq, hv)
    q = torch.randn((total, hq, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.nn.functional.normalize(
        torch.randn((total, hk, 128), dtype=torch.float32, device="cuda"),
        p=2.0,
        dim=-1,
    ).to(torch.bfloat16)
    v = torch.randn((total, hv, 128), dtype=torch.bfloat16, device="cuda")
    cu_seqlens = torch.tensor([0, 64, total], dtype=torch.int32, device="cuda")
    state_indices = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
    candidate_state = _allocate_state_pool(
        7,
        state_heads,
        dtype=torch.bfloat16,
        padding=96,
    )
    candidate_state.copy_(torch.randn_like(candidate_state))
    reference_state = _allocate_state_pool(
        7,
        state_heads,
        dtype=torch.bfloat16,
        padding=96,
    )
    reference_state.copy_(candidate_state)
    expected_output_fp32 = torch.empty(
        (total, state_heads, 128), dtype=torch.float32, device="cuda"
    )
    q_reference = q.float()
    k_reference = k.float().repeat_interleave(hq // hk, dim=1)
    v_reference = v.float().repeat_interleave(hq // hv, dim=1)
    token_start = 0
    for sequence_index, seq_len in enumerate(seq_lens):
        pool_row = int(state_indices[sequence_index])
        state_hkv = reference_state[pool_row].transpose(-1, -2).float()
        for token in range(token_start, token_start + seq_len):
            previous_value = torch.einsum(
                "hk,hkv->hv", k_reference[token], state_hkv
            )
            residual = v_reference[token] - previous_value
            state_hkv += k_reference[token].unsqueeze(-1) * residual.unsqueeze(-2)
            expected_output_fp32[token] = 0.125 * torch.einsum(
                "hk,hkv->hv", q_reference[token], state_hkv
            )
        reference_state[pool_row].copy_(
            state_hkv.transpose(-1, -2).to(reference_state.dtype)
        )
        token_start += seq_len
    expected_output = expected_output_fp32.to(torch.bfloat16)

    immutable = tuple(tensor.clone() for tensor in (q, k, v, cu_seqlens, state_indices))
    state_before = candidate_state.clone()
    gdn_cp_calls: list[str] = []
    real_gdn_cp = gdn_prefill._chunk_gated_delta_rule_gdn_cp_sm100

    def observed_gdn_cp(*args, **kwargs):
        gdn_cp_calls.append("gdn_cp")
        return real_gdn_cp(*args, **kwargs)

    def forbidden_external(*_args, **_kwargs):
        raise AssertionError("public SM100/SM103 use_cp=True route left GDN CP")

    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_gdn_cp_sm100",
        observed_gdn_cp,
    )
    monkeypatch.setattr(
        gdn_prefill,
        "chunk_gated_delta_rule_sm100",
        forbidden_external,
    )
    monkeypatch.setattr(
        gdn_prefill,
        "cp_delta_rule_dsl_sm100",
        forbidden_external,
        raising=False,
    )
    gdn_cp._public_key = None
    gdn_cp._public_prepared = None
    output = torch.empty_like(expected_output)
    actual_output, actual_state = gdn_prefill.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=None,
        beta=None,
        scale=0.125,
        initial_state=candidate_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=output,
        output_state=candidate_state,
        use_cp=True,
        state_indices=state_indices,
    )
    torch.cuda.synchronize()

    assert gdn_cp_calls == ["gdn_cp"]
    assert actual_output is output
    assert actual_state is candidate_state
    torch.testing.assert_close(actual_output, expected_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_state, reference_state, atol=1e-2, rtol=1e-2)
    for observed, before in zip(
        (q, k, v, cu_seqlens, state_indices), immutable, strict=True
    ):
        assert torch.equal(observed, before)
    selected = set(int(value) for value in state_indices.cpu().tolist())
    for row in range(candidate_state.shape[0]):
        if row not in selected:
            assert torch.equal(candidate_state[row], state_before[row])
