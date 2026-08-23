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

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flashinfer
import flashinfer.gdn_prefill as gdn_prefill
from flashinfer.gdn_kernels.blackwell import cake_gdn_cp_prefill as cake
from flashinfer.jit import cake_gdn_cp_prefill as cake_jit


def _cake_sm100_toolchain_available() -> bool:
    if not torch.cuda.is_available():
        return False
    minimum = {(10, 0): (12, 8), (10, 3): (12, 9)}.get(
        torch.cuda.get_device_capability()
    )
    if minimum is None or gdn_prefill._cake_gdn_cp_nvcc_version is None:
        return False
    try:
        return gdn_prefill._cake_gdn_cp_nvcc_version() >= minimum
    except RuntimeError:
        return False


_CAKE_SM100_AVAILABLE = _cake_sm100_toolchain_available()


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "csrc" / "gdn" / "cake"


def _assert_oracle_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Require exact non-finite masks and the public tolerance elsewhere."""

    _assert_oracle_output_written(expected)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2, equal_nan=True)


def _assert_oracle_output_written(output: torch.Tensor) -> None:
    nonfinite = ~torch.isfinite(output)
    if bool(nonfinite.any().item()):
        first = torch.nonzero(nonfinite, as_tuple=False)[0].tolist()
        pytest.fail(
            "reference oracle left poisoned output storage unwritten: "
            f"count={int(nonfinite.sum().item())}, first_index={first}"
        )


def _run_fresh_gdn_oracle(
    *,
    tmp_path: Path,
    stem: str,
    payload: dict[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_path = tmp_path / f"{stem}-input.pt"
    output_path = tmp_path / f"{stem}-output.pt"
    oracle_home = tmp_path / f"{stem}-home"
    oracle_workspace = tmp_path / f"{stem}-workspace"
    oracle_home.mkdir()
    oracle_workspace.mkdir()
    torch.save(payload, input_path)
    worker = Path(__file__).with_name("_cake_gdn_prefill_checkpoint_oracle.py")
    environment = os.environ.copy()
    environment.update(
        {
            "FLASHINFER_WORKSPACE_BASE": str(oracle_workspace),
            "HOME": str(oracle_home),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(worker), str(input_path), str(output_path)],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"fresh {stem} oracle failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    values = torch.load(output_path, map_location="cpu", weights_only=True)
    return {name: tensor.to(device=device) for name, tensor in values.items()}


def _fresh_checkpoint_oracle(
    *,
    tmp_path: Path,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: tuple[int, ...],
    interval: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = _run_fresh_gdn_oracle(
        tmp_path=tmp_path,
        stem="checkpoint",
        payload={
            "mode": "checkpoint_per_sequence",
            "q": q.cpu(),
            "k": k.cpu(),
            "v": v.cpu(),
            "alpha": alpha.cpu(),
            "beta": beta.cpu(),
            "seq_lens": seq_lens,
            "interval": interval,
        },
        device=q.device,
    )
    return tuple(values[name] for name in ("output", "final_state", "checkpoints"))


def _fresh_batched_semantic_oracle(
    *,
    tmp_path: Path,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor | None,
    scale: float,
    output_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_lens = tuple(
        int(value)
        for value in (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    )
    values = _run_fresh_gdn_oracle(
        tmp_path=tmp_path,
        stem="batched-semantic",
        payload={
            "mode": "batched",
            "q": q.cpu(),
            "k": k.cpu(),
            "v": v.cpu(),
            "alpha": alpha.cpu(),
            "beta": beta.cpu(),
            "seq_lens": seq_lens,
            "interval": 0,
            "scale": scale,
            "cu_seqlens": cu_seqlens.cpu(),
            "initial_state": initial_state.cpu(),
            "state_indices": (
                None if state_indices is None else state_indices.cpu()
            ),
            "output_state": (
                None if output_state is None else output_state.cpu()
            ),
        },
        device=q.device,
    )
    return values["output"], values["final_state"]


def test_generated_source_inventory_and_hashes() -> None:
    root = _source_root()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "flashinfer-pr4078-cake-only-standalone-export-v3"
    assert "generator_commit" not in manifest
    assert "measured_commit" not in manifest
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
    assert manifest["launch_order"] == [
        "t_precompute",
        "mn_precompute",
        "state_fixup",
        "cp_prefill",
    ]
    assert manifest["launch_policy"]["tensor_map_abi"].startswith("grid_constant")
    assert len(manifest["cuda_headers"]) == 1
    assert manifest["cuda_headers"][0]["path"] == "cuda/cake_common.cuh"
    assert manifest["cuda_headers"][0]["sha256"] == (
        "d61494318fda829af229b2c507af4c83bf5f5a7f1a58dea1baa1c8226fe95e03"
    )
    assert [record["name"] for record in manifest["kernels"]] == [
        "t_precompute",
        "t_precompute_bf16",
        "t_precompute_gb300_hv48_min6",
        "mn_precompute",
        "mn_precompute_bf16",
        "state_fixup_simt_row4",
        "state_gather_fp32",
        "state_gather_fp16",
        "state_gather_bf16",
        "state_scatter_fp32",
        "state_scatter_fp16",
        "state_scatter_bf16",
        "state_fixup_utcmma64",
        "state_fixup_utcmma128",
        "cp_prefill",
        "cp_prefill_equal_head",
        "cp_prefill_equal_head_h32",
        "cp_prefill_bf16",
        "cp_prefill_generic",
        "cp_prefill_generic_bf16",
    ]
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
    cake_jit._manifest.cache_clear()
    assert cake_jit._manifest()["schema"] == (
        "flashinfer-pr4078-cake-only-standalone-export-v3"
    )


@pytest.mark.parametrize(
    ("arch", "hq", "hv", "expected_t", "expected_chunk"),
    [
        ("sm_100a", 16, 48, "t_precompute", None),
        ("sm_103a", 16, 48, "t_precompute_gb300_hv48_min6", None),
        ("sm_103a", 16, 64, "t_precompute", 4096),
        ("sm_103a", 32, 32, "t_precompute", None),
    ],
)
def test_long_row_dispatch_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    arch: str,
    hq: int,
    hv: int,
    expected_t: str,
    expected_chunk: int | None,
) -> None:
    monkeypatch.setattr(cake, "_arch_for", lambda _device: arch)
    monkeypatch.setattr(
        cake.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    q = SimpleNamespace(
        shape=(65536, hq, 128), device=SimpleNamespace(), dtype=torch.float16
    )
    k = SimpleNamespace(shape=(65536, hq, 128))
    v = SimpleNamespace(shape=(65536, hv, 128))
    plan = cake._build_plan(q, k, v, (65536,))
    assert plan.t_kernel == expected_t
    if expected_chunk is not None:
        assert plan.cp_chunk_len == expected_chunk
        assert plan.source_cp_chunk_len == 32768


@pytest.mark.parametrize(
    ("arch", "seq_lens", "dtype", "heads", "expected_prefill"),
    [
        ("sm_100a", (128,), torch.float16, (1, 1, 1), "cp_prefill_equal_head"),
        ("sm_100a", (128,), torch.float16, (32, 32, 32), "cp_prefill_equal_head"),
        ("sm_103a", (128,), torch.float16, (32, 32, 32), "cp_prefill_equal_head_h32"),
        ("sm_100a", (128,), torch.bfloat16, (1, 1, 1), "cp_prefill_bf16"),
        ("sm_103a", (384,), torch.float16, (2, 2, 8), "cp_prefill_generic"),
        ("sm_100a", (65,), torch.bfloat16, (1, 1, 2), "cp_prefill_generic_bf16"),
        ("sm_100a", (128, 129), torch.float16, (4, 2, 2), "cp_prefill_generic"),
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
    monkeypatch.setattr(cake, "_arch_for", lambda _device: arch)
    monkeypatch.setattr(
        cake.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    total = sum(seq_lens)
    hq, hk, hv = heads
    device = SimpleNamespace()
    q = SimpleNamespace(shape=(total, hq, 128), device=device, dtype=dtype)
    k = SimpleNamespace(shape=(total, hk, 128))
    v = SimpleNamespace(shape=(total, hv, 128))

    plan = cake._build_plan(q, k, v, seq_lens)

    suffix = "_bf16" if dtype == torch.bfloat16 else ""
    assert plan.t_kernel == f"t_precompute{suffix}"
    assert plan.mn_kernel == f"mn_precompute{suffix}"
    assert plan.prefill_kernel == expected_prefill
    assert plan.num_sab_heads == max(hq, hv)


def test_zero_length_plan_uses_simt_state_fixup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cake, "_arch_for", lambda _device: "sm_103a")
    monkeypatch.setattr(
        cake.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    device = SimpleNamespace()
    q = SimpleNamespace(shape=(129, 4, 128), device=device, dtype=torch.bfloat16)
    k = SimpleNamespace(shape=(129, 1, 128))
    v = SimpleNamespace(shape=(129, 1, 128))

    plan = cake._build_plan(q, k, v, (0, 64, 65, 0))

    assert cake._choose_fixup_kind(16, 148) == "state_fixup_utcmma64"
    assert plan.fixup_kernel == "state_fixup_simt_row4"


def test_checkpoint_interval_becomes_cp_chunk_and_maps_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cake, "_arch_for", lambda _device: "sm_103a")
    monkeypatch.setattr(
        cake.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    device = torch.device("cpu")
    q = SimpleNamespace(shape=(640, 1, 128), device=device, dtype=torch.float16)
    k = SimpleNamespace(shape=(640, 1, 128))
    v = SimpleNamespace(shape=(640, 1, 128))

    plan = cake._build_plan(
        q,
        k,
        v,
        (256, 384),
        checkpoint_every_n_tokens=128,
    )

    assert plan.cp_chunk_len == 128
    assert plan.source_cp_chunk_len == 256
    assert plan.checkpoint_count == 5
    assert cake._checkpoint_fixed_state_indices(plan, device).tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_all_contract_plans_match_frozen_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cake.torch.cuda,
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
            monkeypatch.setattr(cake, "_arch_for", lambda _device, value=arch: value)
            plan = cake._build_plan(q, k, v, tuple(shape["seq_lens"]))
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


def test_cake_prepared_launcher_is_not_a_new_public_api() -> None:
    assert not hasattr(gdn_prefill, "prepare_cake_gdn_cp_prefill")
    assert not hasattr(gdn_prefill, "CakeGDNCPPrefill")
    assert not hasattr(gdn_prefill, "CakeGDNCPPrefillPlan")
    assert not hasattr(flashinfer, "prepare_cake_gdn_cp_prefill")
    assert not hasattr(flashinfer, "CakeGDNCPPrefill")
    assert not hasattr(flashinfer, "CakeGDNCPPrefillPlan")


def test_indexed_state_rows_must_not_overlap() -> None:
    inner = 128 * 128
    storage = torch.empty(inner * 2, dtype=torch.float32)
    overlapping = storage.as_strided(
        (2, 1, 128, 128),
        (inner - 1, inner, 128, 1),
    )
    plan = SimpleNamespace(num_sab_heads=1, num_seqs=2)

    with pytest.raises(ValueError, match="rows overlap"):
        cake._validate_state(
            overlapping,
            name="initial_state",
            plan=plan,
            device=torch.device("cpu"),
            indexed=True,
        )


def test_indexed_state_preserves_positive_inner_strides() -> None:
    heads = 2
    inner_stride = 2
    strides = (
        heads * 128 * 128 * inner_stride + 96,
        128 * 128 * inner_stride,
        128 * inner_stride,
        inner_stride,
    )
    span = 1 + sum(
        (size - 1) * stride
        for size, stride in zip((3, heads, 128, 128), strides, strict=True)
    )
    state = torch.empty(span, dtype=torch.float32).as_strided(
        (3, heads, 128, 128),
        strides,
    )
    plan = SimpleNamespace(num_sab_heads=heads, num_seqs=1)

    cake._validate_state(
        state,
        name="initial_state",
        plan=plan,
        device=torch.device("cpu"),
        indexed=True,
    )
    assert cake._state_carrier(state).numel() == span


def test_sequence_lengths_allow_empty_rows_but_not_decreasing_offsets() -> None:
    cu_seqlens = torch.tensor([0, 4, 4], dtype=torch.int64)
    assert cake._read_seq_lens(
        cu_seqlens,
        total_tokens=4,
        expected=None,
    ) == (4, 0)

    with pytest.raises(ValueError, match="nonnegative"):
        cake._read_seq_lens(
            torch.tensor([0, 4, 3], dtype=torch.int64),
            total_tokens=3,
            expected=None,
        )


def test_metadata_version_accepts_inference_tensors() -> None:
    with torch.inference_mode():
        metadata = torch.tensor([0, 4], dtype=torch.int64)
        assert torch.is_inference(metadata)
        assert cake._metadata_version(metadata) is None


def test_public_cake_cache_reuses_equal_metadata_and_rebinds_tensor_addresses(
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

    monkeypatch.setattr(cake, "prepare_cake_gdn_cp_prefill", fake_prepare)
    monkeypatch.setattr(
        cake.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(cake.torch.cuda, "is_current_stream_capturing", lambda: False)
    cake._public_key = None
    cake._public_metadata_binding = None
    cake._public_prepared = None

    cu_seqlens = torch.tensor([0, 2], dtype=torch.int64)
    alpha = torch.ones((2, 1), dtype=torch.float32)
    beta = torch.ones((2, 1), dtype=torch.float32)
    initial_state = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
    for _ in range(2):
        q = torch.zeros((2, 1, 128), dtype=torch.float16)
        output = torch.empty_like(q)
        output_state = torch.empty_like(initial_state)
        cake.chunk_gated_delta_rule_cake_sm100(
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

    cake.chunk_gated_delta_rule_cake_sm100(
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

    cake._reset_cake_gdn_cp_prefill_cache()
    assert cake._public_key is None
    assert cake._public_metadata_binding is None
    assert cake._public_prepared is None

    cake.chunk_gated_delta_rule_cake_sm100(
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


def test_public_cake_cache_detects_metadata_writes_without_version_bump(
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

    monkeypatch.setattr(cake, "prepare_cake_gdn_cp_prefill", fake_prepare)
    monkeypatch.setattr(
        cake.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(cake.torch.cuda, "is_current_stream_capturing", lambda: False)
    cake._reset_cake_gdn_cp_prefill_cache()

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
        cake.chunk_gated_delta_rule_cake_sm100(
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


def test_public_cake_cache_requires_warmed_metadata_during_graph_capture(
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

    monkeypatch.setattr(cake, "prepare_cake_gdn_cp_prefill", fake_prepare)
    monkeypatch.setattr(
        cake.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(
        cake.torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing,
    )
    cake._reset_cake_gdn_cp_prefill_cache()

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    output = torch.empty_like(q)
    alpha = torch.ones((2, 1), dtype=torch.float32)
    beta = torch.ones((2, 1), dtype=torch.float32)
    state = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
    output_state = torch.empty_like(state)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int64)

    def invoke(metadata: torch.Tensor) -> None:
        cake.chunk_gated_delta_rule_cake_sm100(
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
        ("auto", True, "cake"),
        ("auto", False, "non_cp"),
        (True, False, "cake"),
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
    monkeypatch.setattr(gdn_prefill, "_cake_gdn_cp_nvcc_version", lambda: (13, 0))
    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_cake_sm100",
        lambda *_args, **_kwargs: calls.append("cake"),
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


@pytest.mark.parametrize(
    ("capability", "torch_cuda", "nvcc_version", "expected_error"),
    [
        ((10, 0), "12.0", (12, 8), None),
        ((10, 0), "13.0", (12, 7), "requires nvcc 12.8"),
        ((10, 3), "12.9", (12, 9), None),
        ((10, 3), "13.0", (12, 8), "requires nvcc 12.9"),
    ],
)
def test_public_dispatch_gates_cake_with_its_jit_nvcc(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    torch_cuda: str,
    nvcc_version: tuple[int, int],
    expected_error: str | None,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(
        gdn_prefill, "get_compute_capability", lambda _device: capability
    )
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "Blackwell")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", torch_cuda)
    monkeypatch.setattr(gdn_prefill, "_cake_gdn_cp_nvcc_version", lambda: nvcc_version)
    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_cake_sm100",
        lambda *_args, **_kwargs: calls.append("cake"),
    )

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    kwargs = dict(
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        use_cp=True,
    )
    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            gdn_prefill.chunk_gated_delta_rule(q, q, q, **kwargs)
        assert calls == []
        return

    output = gdn_prefill.chunk_gated_delta_rule(q, q, q, **kwargs)
    assert output.shape == q.shape
    assert calls == ["cake"]


def test_public_dispatch_preserves_cute_cp_cuda13_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 3))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "NVIDIA GB300")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "12.9")
    monkeypatch.setattr(gdn_prefill, "_cake_gdn_cp_nvcc_version", lambda: (12, 9))
    monkeypatch.setattr(
        gdn_prefill, "cp_delta_rule_dsl_sm100", lambda *_args, **_kwargs: None
    )

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    fp8_state = torch.empty((1, 1, 128, 128), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="DSL kernel requires CUDA 13"):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            initial_state=fp8_state,
            use_cp=True,
        )


@pytest.mark.parametrize(
    ("extension", "expected_route"),
    [
        ("checkpoint", "cake"),
        ("fp8_state", "cute_cp"),
        ("cp_chunk_len", "cute_cp"),
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
    monkeypatch.setattr(gdn_prefill, "_cake_gdn_cp_nvcc_version", lambda: (13, 0))
    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_cake_sm100",
        lambda *_args, **_kwargs: calls.append("cake"),
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


def test_public_dispatch_fails_closed_when_cake_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "NVIDIA B200")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")
    monkeypatch.setattr(gdn_prefill, "_cake_gdn_cp_nvcc_version", lambda: (13, 0))
    monkeypatch.setattr(gdn_prefill, "_chunk_gated_delta_rule_cake_sm100", None)

    q = torch.zeros((2, 1, 128), dtype=torch.float16)
    with pytest.raises(ValueError, match="Cake-only CP delta rule SM100"):
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

    assert cake._read_seq_lens(cu_seqlens, total_tokens=5, expected=(2, 3)) == (2, 3)


@pytest.mark.skipif(
    not _CAKE_SM100_AVAILABLE,
    reason="requires an SM100a or SM103a GPU with a supported Cake nvcc toolchain",
)
def test_odd_cp_chunk_uses_generic_kernel_and_matches_cute() -> None:
    from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import (
        cp_delta_rule_dsl_sm100,
    )

    torch.manual_seed(4278)
    total, hq, hv, dim = 384, 2, 8, 128
    q = torch.randn((total, hq, dim), dtype=torch.float16, device="cuda")
    k = torch.nn.functional.normalize(
        torch.randn((total, hq, dim), dtype=torch.float32, device="cuda"),
        p=2.0,
        dim=-1,
    ).to(torch.float16)
    v = torch.randn((total, hv, dim), dtype=torch.float16, device="cuda")
    alpha = 1.0 - torch.rand((total, hv), dtype=torch.float32, device="cuda") / total
    beta = torch.rand((total, hv), dtype=torch.float32, device="cuda").sigmoid()
    cu_seqlens = torch.tensor([0, total], dtype=torch.int64, device="cuda")
    expected_output = torch.full(
        (total, hv, dim), float("nan"), dtype=q.dtype, device="cuda"
    )
    expected_state = torch.empty((1, hv, dim, dim), dtype=torch.float32, device="cuda")
    cp_delta_rule_dsl_sm100(
        expected_output,
        expected_state,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        dim**-0.5,
        max_seqlen=total,
        cp_chunk_len=192,
    )
    _assert_oracle_output_written(expected_output)

    output_state = torch.empty_like(expected_state)
    prepared = cake.prepare_cake_gdn_cp_prefill(
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        None,
        seq_lens=(total,),
        output_state=output_state,
    )
    assert prepared.plan.cp_chunk_len == 192
    assert prepared.plan.prefill_kernel == "cp_prefill_generic"
    output, final_state = prepared.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final_state, expected_state, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not _CAKE_SM100_AVAILABLE,
    reason="requires an SM100a or SM103a GPU with a supported Cake nvcc toolchain",
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

    expected_output = torch.full(
        (total, hv, dim), float("nan"), dtype=torch.float16, device=device
    )
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
    _assert_oracle_output_written(expected_output)
    prepared = cake.prepare_cake_gdn_cp_prefill(
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
    not _CAKE_SM100_AVAILABLE,
    reason="requires an SM100a or SM103a GPU with a supported Cake nvcc toolchain",
)
def test_public_checkpoint_matches_cute_on_caller_stream_and_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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

    # The pinned source extension is process-global and can change behavior
    # after unrelated kernel compilations in the same pytest session. Keep
    # the numeric oracle in a fresh process; candidate execution stays here.
    expected_output, expected_state, expected_checkpoints_tensor = (
        _fresh_checkpoint_oracle(
            tmp_path=tmp_path,
            q=q,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
            seq_lens=seq_lens,
            interval=interval,
        )
    )

    def forbidden_external(*_args, **_kwargs):
        raise AssertionError("checkpoint route left the Cake backend")

    monkeypatch.setattr(gdn_prefill, "cp_delta_rule_dsl_sm100", forbidden_external)
    cake._public_key = None
    cake._public_prepared = None

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

    assert cake._public_prepared is not None
    assert cake._public_prepared.plan.checkpoint_count == 3
    assert cake._public_prepared._checkpoint is not None
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
    inner_stride: int = 1,
) -> torch.Tensor:
    if padding == 0 and inner_stride == 1:
        return torch.empty((rows, heads, 128, 128), dtype=dtype, device="cuda")
    shape = (rows, heads, 128, 128)
    strides = (
        heads * 128 * 128 * inner_stride + padding,
        128 * 128 * inner_stride,
        128 * inner_stride,
        inner_stride,
    )
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(shape, strides, strict=True)
    )
    return torch.empty(span, dtype=dtype, device="cuda").as_strided(shape, strides)


@pytest.mark.skipif(
    not _CAKE_SM100_AVAILABLE,
    reason="requires an SM100a or SM103a GPU with a supported Cake nvcc toolchain",
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
    tmp_path: Path,
) -> None:
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
    else:
        # Keep the raw-K path distinct from L2 normalization while bounding
        # the recurrence so NaN output poison only detects unwritten storage.
        k_fp32.mul_(1.0 / 128)
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
            dtype=torch.int32,
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

    expected_output, semantic_state = _fresh_batched_semantic_oracle(
        tmp_path=tmp_path,
        q=q,
        k=k,
        v=v,
        alpha=baseline_alpha,
        beta=baseline_beta,
        cu_seqlens=cu_seqlens,
        initial_state=reference_initial,
        state_indices=state_indices,
        scale=float(case["scale"]),
        output_state=reference_state,
    )
    _assert_oracle_output_written(expected_output)
    if reference_state is not None:
        reference_state.copy_(semantic_state)
    prepared = cake.prepare_cake_gdn_cp_prefill(
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
    not _CAKE_SM100_AVAILABLE,
    reason="requires an SM100a or SM103a GPU with a supported Cake nvcc toolchain",
)
def test_public_dispatcher_uses_only_cake_for_indexed_inplace_gqa(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the Cake public use_cp=True route with no external arm."""

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
    ones = torch.ones((total, state_heads), dtype=torch.float32, device="cuda")
    expected_output, reference_state = _fresh_batched_semantic_oracle(
        tmp_path=tmp_path,
        q=q,
        k=k,
        v=v,
        alpha=ones,
        beta=ones,
        cu_seqlens=cu_seqlens,
        initial_state=reference_state,
        state_indices=state_indices,
        scale=0.125,
    )
    _assert_oracle_output_written(expected_output)

    immutable = tuple(tensor.clone() for tensor in (q, k, v, cu_seqlens, state_indices))
    state_before = candidate_state.clone()
    cake_calls: list[str] = []
    real_cake = gdn_prefill._chunk_gated_delta_rule_cake_sm100

    def observed_cake(*args, **kwargs):
        cake_calls.append("cake")
        return real_cake(*args, **kwargs)

    def forbidden_external(*_args, **_kwargs):
        raise AssertionError("public SM100/SM103 use_cp=True route left Cake")

    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_cake_sm100",
        observed_cake,
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
    cake._public_key = None
    cake._public_prepared = None
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

    assert cake_calls == ["cake"]
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


@pytest.mark.skipif(
    not _CAKE_SM100_AVAILABLE,
    reason="requires an SM100a or SM103a GPU with a supported Cake nvcc toolchain",
)
def test_public_cake_inference_empty_int64_inner_strided_cache_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover public metadata caching and arbitrary positive state strides."""

    from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import (
        cp_delta_rule_dsl_sm100,
    )

    torch.manual_seed(504539)
    seq_lens = (0, 64, 65, 0)
    total = sum(seq_lens)
    hq, hk, hv = 4, 1, 1
    state_heads = max(hq, hv)
    pool_rows = 7
    state_slots = (4, 1, 6, 2)

    q_values = torch.randn((total, hq, 128), dtype=torch.bfloat16, device="cuda")
    k_values = torch.nn.functional.normalize(
        torch.randn((total, hk, 128), dtype=torch.float32, device="cuda"),
        p=2.0,
        dim=-1,
    ).to(torch.bfloat16)
    v_values = torch.randn((total, hv, 128), dtype=torch.bfloat16, device="cuda")
    state_values = torch.randn(
        (pool_rows, state_heads, 128, 128),
        dtype=torch.bfloat16,
        device="cuda",
    )
    with torch.inference_mode():
        cu_seqlens = torch.tensor(
            [0, 0, 64, 129, 129], dtype=torch.int64, device="cuda"
        )
        state_indices = torch.tensor(state_slots, dtype=torch.int64, device="cuda")
    assert torch.is_inference(cu_seqlens)
    assert torch.is_inference(state_indices)

    reference_cu = torch.tensor([0, 0, 64, 129, 129], dtype=torch.int32, device="cuda")
    reference_indices = torch.tensor(state_slots, dtype=torch.int32, device="cuda")
    reference_initial = state_values.clone()
    reference_state = torch.full_like(reference_initial, -7.0)
    expected_output = torch.full(
        (total, state_heads, 128),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    ones = torch.ones((total, state_heads), dtype=torch.float32, device="cuda")
    cp_delta_rule_dsl_sm100(
        expected_output,
        reference_state,
        q_values,
        k_values,
        v_values,
        ones,
        ones,
        reference_cu,
        0.125,
        initial_state=reference_initial,
        state_indices=reference_indices,
        max_seqlen=max(seq_lens),
    )
    _assert_oracle_output_written(expected_output)
    # The pinned CuTe oracle launches no state writer for a zero-token
    # sequence.  The public varlen contract publishes its unchanged initial
    # state instead, which the Cake route implements and this test verifies.
    for seq_len, state_slot in zip(seq_lens, state_slots, strict=True):
        if seq_len == 0:
            reference_state[state_slot].copy_(reference_initial[state_slot])

    candidates = []
    for _ in range(2):
        q = q_values.clone()
        k = k_values.clone()
        v = v_values.clone()
        initial = _allocate_state_pool(
            pool_rows,
            state_heads,
            dtype=torch.bfloat16,
            padding=96,
            inner_stride=2,
        )
        state = _allocate_state_pool(
            pool_rows,
            state_heads,
            dtype=torch.bfloat16,
            padding=96,
            inner_stride=2,
        )
        initial.copy_(state_values)
        state.fill_(-7.0)
        output = torch.empty_like(expected_output)
        candidates.append((q, k, v, initial, state, output))

    immutable_metadata = (cu_seqlens.clone(), state_indices.clone())
    immutable_inputs = tuple(
        tuple(tensor.clone() for tensor in candidate[:4]) for candidate in candidates
    )
    state_before = tuple(candidate[4].clone() for candidate in candidates)
    real_prepare = cake.prepare_cake_gdn_cp_prefill
    prepare_calls = 0

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(cake, "prepare_cake_gdn_cp_prefill", counted_prepare)
    cake._reset_cake_gdn_cp_prefill_cache()
    try:
        for q, k, v, initial, state, output in candidates:
            with torch.inference_mode():
                actual_output, actual_state = gdn_prefill.chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g=None,
                    beta=None,
                    scale=0.125,
                    initial_state=initial,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=False,
                    output=output,
                    output_state=state,
                    use_cp=True,
                    state_indices=state_indices,
                )
            torch.cuda.synchronize()
            assert actual_output is output
            assert actual_state is state
            _assert_oracle_close(actual_output, expected_output)
            _assert_oracle_close(actual_state, reference_state)
    finally:
        cake._reset_cake_gdn_cp_prefill_cache()

    assert prepare_calls == 1
    assert torch.equal(cu_seqlens, immutable_metadata[0])
    assert torch.equal(state_indices, immutable_metadata[1])
    selected = set(state_slots)
    for candidate, before_inputs, before_state in zip(
        candidates, immutable_inputs, state_before, strict=True
    ):
        for observed, before in zip(candidate[:4], before_inputs, strict=True):
            assert torch.equal(observed, before)
        state = candidate[4]
        for row in range(pool_rows):
            if row not in selected:
                assert torch.equal(state[row], before_state[row])
        for empty_slot in (state_slots[0], state_slots[-1]):
            assert torch.equal(state[empty_slot], candidate[3][empty_slot])
