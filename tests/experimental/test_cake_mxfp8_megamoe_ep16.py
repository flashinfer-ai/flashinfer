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

# Tests for the experimental Cake MXFP8 MegaMoE EP16 backend.

import ast
import copy
import inspect
import json
import os
import runpy
import subprocess
import sys
import textwrap
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import flashinfer.experimental.cake_mxfp8_megamoe_ep16.jit as _jit
import pytest
import torch
import torch.distributed as dist
from flashinfer.experimental.cake_mxfp8_megamoe_ep16.backend import (
    _MAX_LAUNCH_EPOCH,
    _interleave_gate_up_16,
    _pack_scale_n128_k128,
    _quantize_mxfp8_block32,
    _validate_backend,
    _validate_gathered_routing_capacity,
    _validate_launch_epoch,
)
from flashinfer.experimental.cake_mxfp8_megamoe_ep16.backend import (
    CakeMxfp8MegaMoeEp16 as _CakeMxfp8MegaMoeEp16Session,
)
from flashinfer.experimental.cake_mxfp8_megamoe_ep16.jit import _read_manifest
from flashinfer.moe_ep import (
    CakeMxfp8MegaMoeEp16,
    preprocess_cake_mxfp8_megamoe_ep16_weights,
)


@pytest.fixture(scope="module")
def example_device_index():
    example = (
        Path(__file__).resolve().parents[2]
        / "examples/experimental/cake_mxfp8_megamoe_ep16.py"
    )
    return runpy.run_path(str(example))["_select_device_index"]


@pytest.mark.parametrize(
    "local_rank,local_world_size,visible_devices,expected",
    (
        (0, 16, 16, 0),
        (15, 16, 16, 15),
        (3, 4, 4, 3),
        (3, 4, 8, 3),
        (0, 1, 1, 0),
        (3, 4, 1, 0),
        (15, 16, 1, 0),
    ),
)
def test_example_device_selection(
    example_device_index, local_rank, local_world_size, visible_devices, expected
) -> None:
    environ = {
        "WORLD_SIZE": "16",
        "LOCAL_WORLD_SIZE": str(local_world_size),
        "LOCAL_RANK": str(local_rank),
    }
    assert example_device_index(environ, visible_devices) == expected


@pytest.mark.parametrize(
    "world_size,local_world_size,local_rank,visible_devices",
    (
        (8, 4, 0, 4),
        (32, 4, 0, 4),
        (16, 0, 0, 1),
        (16, -1, 0, 1),
        (16, 17, 0, 1),
        (16, 4, -1, 4),
        (16, 4, 4, 4),
        (16, 4, -1, 1),
        (16, 4, 4, 1),
        (16, 4, 0, 0),
        (16, 4, 0, 2),
    ),
)
def test_example_device_selection_rejects_invalid_layout(
    example_device_index, world_size, local_world_size, local_rank, visible_devices
) -> None:
    environ = {
        "WORLD_SIZE": str(world_size),
        "LOCAL_WORLD_SIZE": str(local_world_size),
        "LOCAL_RANK": str(local_rank),
    }
    with pytest.raises(ValueError):
        example_device_index(environ, visible_devices)


@pytest.mark.parametrize("key", ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "LOCAL_RANK"))
@pytest.mark.parametrize("value", (None, "invalid"))
def test_example_device_selection_requires_torchrun_metadata(
    example_device_index, key, value
) -> None:
    environ = {"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "4", "LOCAL_RANK": "0"}
    if value is None:
        del environ[key]
    else:
        environ[key] = value
    with pytest.raises(ValueError, match="Launch with torchrun"):
        example_device_index(environ, 4)


def test_public_entry_points_are_experimental() -> None:
    assert CakeMxfp8MegaMoeEp16.is_experimental
    assert preprocess_cake_mxfp8_megamoe_ep16_weights.is_experimental


def test_backend_keyword_defaults_to_cuda() -> None:
    parameter = inspect.signature(CakeMxfp8MegaMoeEp16).parameters["backend"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "cuda"
    _validate_backend("cuda")
    _validate_backend("cute_dsl")


@pytest.mark.parametrize("backend", (None, True, 0, "auto", "CUDA", "cutedsl", []))
def test_invalid_backend_is_rejected_before_session_setup(backend) -> None:
    # Invalid selection fails before accessing the deliberately absent inputs.
    with pytest.raises(ValueError, match="backend must be 'cuda' or 'cute_dsl'"):
        CakeMxfp8MegaMoeEp16(None, None, backend=backend)


def test_public_import_and_default_selection_do_not_import_cute_kernels() -> None:
    code = """
import sys
from flashinfer.moe_ep import CakeMxfp8MegaMoeEp16

prefix = "flashinfer.experimental.cake_mxfp8_megamoe_ep16"
assert prefix not in sys.modules
for kwargs in ({}, {"backend": "cuda"}):
    try:
        CakeMxfp8MegaMoeEp16(None, None, **kwargs)
    except RuntimeError as error:
        assert "torch.distributed must be initialized" in str(error)
    else:
        raise AssertionError("uninitialized session was accepted")
    assert prefix + ".cute_dsl" not in sys.modules
    assert not any(name.startswith(prefix + ".kernels") for name in sys.modules)
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH")))
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=120)


def test_cute_peer_arguments_preserve_all_address_bits() -> None:
    from flashinfer.experimental.cake_mxfp8_megamoe_ep16.cute_dsl import (
        _peer_arguments,
    )

    active = (1, 2**63 - 1, 2**63, 2**64 - 1, *range(5, 17))
    pointers = active + (0,) * 16
    packed = _peer_arguments(SimpleNamespace(peer_pointers=pointers))
    assert packed == (1, 2**63 - 1, -(2**63), -1, *range(5, 17), *((0,) * 16))
    assert pointers == active + (0,) * 16


@pytest.mark.parametrize(
    "pointers",
    (
        None,
        [1] * 16 + [0] * 16,
        (1,) * 16 + (0,) * 15,
        (1,) * 16 + (0,) * 17,
        (0,) + (1,) * 15 + (0,) * 16,
        (-1,) + (1,) * 15 + (0,) * 16,
        (2**64,) + (1,) * 15 + (0,) * 16,
        (True,) + (1,) * 15 + (0,) * 16,
        ("1",) + (1,) * 15 + (0,) * 16,
        (1,) * 17 + (0,) * 15,
        (1,) * 16 + (False,) + (0,) * 15,
    ),
)
def test_cute_peer_arguments_reject_invalid_tables(pointers) -> None:
    from flashinfer.experimental.cake_mxfp8_megamoe_ep16.cute_dsl import (
        _peer_arguments,
    )

    with pytest.raises(ValueError):
        _peer_arguments(SimpleNamespace(peer_pointers=pointers))


def test_cute_source_key_covers_kernel_and_adapter_dependencies() -> None:
    from flashinfer.experimental.cake_mxfp8_megamoe_ep16 import cute_dsl
    from flashinfer.jit import cute_dsl_core

    package = Path(cute_dsl.__file__).resolve().parent
    flashinfer_root = package.parent.parent
    expected = {
        package / "kernels" / "fused_cta0.py",
        package / "kernels" / "fused_all_ctas.py",
        package / "kernels" / "topk_reduce.py",
        package / "cute_dsl.py",
        package / "backend.py",
        package / "__init__.py",
        flashinfer_root / "moe_ep" / "cake_mxfp8_megamoe_ep16.py",
        Path(cute_dsl_core.__file__).resolve(),
    }
    kernels_init = package / "kernels" / "__init__.py"
    if kernels_init.is_file():
        expected.add(kernels_init)
    actual = tuple(Path(path) for path in cute_dsl._source_files())
    assert len(actual) == len(set(actual))
    assert set(actual) == expected
    assert all(path.is_file() for path in actual)


def test_cute_source_key_changes_for_each_dependency(tmp_path) -> None:
    from flashinfer.experimental.cake_mxfp8_megamoe_ep16 import cute_dsl
    from flashinfer.jit.cute_dsl_core import _hash_source_files

    flashinfer_root = Path(cute_dsl.__file__).resolve().parents[2]
    sources = cute_dsl._source_files()
    original_key = _hash_source_files(sources)
    copies = []
    for name in sources:
        source = Path(name)
        copied = tmp_path / source.relative_to(flashinfer_root)
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(source.read_bytes())
        copies.append(copied)
    copied_paths = tuple(str(path) for path in copies)
    baseline_key = _hash_source_files(copied_paths)
    # Paths themselves participate in this hash. Keep the temporary path set
    # fixed and change only one file's bytes for each invalidation check.
    for copied in copies:
        original = copied.read_bytes()
        try:
            copied.write_bytes(original + b"\n# source-key invalidation test\n")
            assert _hash_source_files(copied_paths) != baseline_key
        finally:
            copied.write_bytes(original)
        assert _hash_source_files(copied_paths) == baseline_key
    assert _hash_source_files(sources) == original_key


def test_generated_source_closure() -> None:
    _, manifest = _read_manifest()
    sequence = manifest["sequences"][0]
    assert sequence["arch"] == "sm_103a"
    assert len(sequence["translation_units"]["devices"]) == 3
    assert sequence["setup_ffi_entry"] == "setup_tma"
    assert sequence["launches_per_call"] == 2
    assert sequence["setup_launches"] == 1
    assert sequence["max_launch_epoch"] == _MAX_LAUNCH_EPOCH


def test_jit_resolves_flashinfer_headers() -> None:
    header_dirs = _jit._get_flashinfer_header_dirs()
    assert any((path / "tvm_ffi_utils.h").is_file() for path in header_dirs)
    assert any((path / "flashinfer" / "layout.cuh").is_file() for path in header_dirs)


def _write_test_manifest(tmp_path, manifest: dict) -> None:
    operator_dir = tmp_path / "cake_mxfp8_megamoe_ep16"
    operator_dir.mkdir(parents=True)
    (operator_dir / "cake_mxfp8_megamoe_ep16_manifest.json").write_text(
        json.dumps(manifest)
    )


def test_manifest_rejects_aggregate_identity_drift(tmp_path, monkeypatch) -> None:
    _, manifest = _read_manifest()
    mutated = copy.deepcopy(manifest)
    mutated["sequences"][0]["max_launch_epoch"] += 1
    _write_test_manifest(tmp_path, mutated)
    monkeypatch.setattr(_jit, "_get_csrc_root", lambda: tmp_path)
    with pytest.raises(
        RuntimeError, match="aggregate source-closure identity mismatch"
    ):
        _read_manifest()


def test_manifest_requires_translation_units_to_equal_closure(
    tmp_path, monkeypatch
) -> None:
    _, manifest = _read_manifest()
    mutated = copy.deepcopy(manifest)
    mutated["sequences"][0]["translation_units"]["devices"].pop()
    _write_test_manifest(tmp_path, mutated)
    monkeypatch.setattr(_jit, "_get_csrc_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="translation units must exactly equal"):
        _read_manifest()


def test_host_binding_argument_count() -> None:
    source = textwrap.dedent(inspect.getsource(_CakeMxfp8MegaMoeEp16Session.run))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_module"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 35


def test_tma_setup_argument_count() -> None:
    source = textwrap.dedent(inspect.getsource(_CakeMxfp8MegaMoeEp16Session.__init__))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setup_tma"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_module"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 7


def test_host_calls_bridge_the_torch_stream() -> None:
    for method, module_entry in (
        (_CakeMxfp8MegaMoeEp16Session.__init__, "setup_tma"),
        (_CakeMxfp8MegaMoeEp16Session.run, "run"),
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        bridged_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            context_names = {
                item.context_expr.func.attr
                for item in node.items
                if isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
            }
            if {"device", "use_torch_stream"} - context_names:
                continue
            bridged_calls.extend(
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == module_entry
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "_module"
            )
        assert len(bridged_calls) == 1


def test_launch_epoch_guard_prevents_grid_counter_overflow() -> None:
    _validate_launch_epoch(0)
    _validate_launch_epoch(_MAX_LAUNCH_EPOCH)
    with pytest.raises(RuntimeError, match="launch epoch is exhausted"):
        _validate_launch_epoch(-1)
    with pytest.raises(RuntimeError, match="launch epoch is exhausted"):
        _validate_launch_epoch(_MAX_LAUNCH_EPOCH + 1)


def test_interleave_gate_up_16() -> None:
    w13 = torch.arange(64, dtype=torch.float32).view(1, 64, 1)
    interleaved = _interleave_gate_up_16(w13).view(1, 2, 2, 16, 1)
    assert torch.equal(
        interleaved[0, 0, 0, :, 0], torch.arange(0, 16, dtype=torch.float32)
    )
    assert torch.equal(
        interleaved[0, 0, 1, :, 0], torch.arange(32, 48, dtype=torch.float32)
    )
    assert torch.equal(
        interleaved[0, 1, 0, :, 0], torch.arange(16, 32, dtype=torch.float32)
    )
    assert torch.equal(
        interleaved[0, 1, 1, :, 0], torch.arange(48, 64, dtype=torch.float32)
    )


def test_quantize_zero_block_uses_zero_scale_code() -> None:
    weight = torch.zeros((1, 1, 32), dtype=torch.float32)
    quantized, scales = _quantize_mxfp8_block32(weight)
    assert torch.count_nonzero(quantized.float()) == 0
    assert torch.count_nonzero(scales) == 0


def test_routing_capacity_counts_duplicate_slots() -> None:
    at_capacity = torch.zeros((8, 8), dtype=torch.int64)
    _validate_gathered_routing_capacity(at_capacity)

    over_capacity = torch.ones((9, 8), dtype=torch.int64)
    over_capacity.reshape(-1)[:65] = 0
    with pytest.raises(ValueError, match="maximum load is 65, capacity is 64"):
        _validate_gathered_routing_capacity(over_capacity)


def test_routing_capacity_rejects_out_of_range_ids() -> None:
    with pytest.raises(ValueError, match=r"expert IDs in \[0, 512\)"):
        _validate_gathered_routing_capacity(torch.tensor([[-1]], dtype=torch.int64))
    with pytest.raises(ValueError, match=r"expert IDs in \[0, 512\)"):
        _validate_gathered_routing_capacity(torch.tensor([[512]], dtype=torch.int64))


def test_pack_scale_n128_k128() -> None:
    scales = torch.arange(256 * 8, dtype=torch.int32).view(1, 256, 8)
    packed = _pack_scale_n128_k128(scales)
    for row in range(256):
        for block_column in range(8):
            tile_k, u = divmod(block_column, 4)
            row_block, row_in_block = divmod(row, 128)
            a, row32 = divmod(row_in_block, 32)
            offset = row_block * 1024 + tile_k * 512 + row32 * 16 + a * 4 + u
            assert packed[offset] == scales[0, row, block_column]


def _expert_coefficients(
    global_experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = torch.exp2((global_experts % 8).float() - 3.0)
    up = torch.exp2(((global_experts // 8) % 8).float() - 4.0)
    fc2 = torch.exp2(((global_experts // 64) % 8).float() - 3.0)
    return gate, up, fc2


def _make_sparse_expert_weights(
    rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    local_experts = torch.arange(32, dtype=torch.int64, device=device)
    global_experts = rank * 32 + local_experts
    gate, up, fc2 = _expert_coefficients(global_experts)
    w13 = torch.zeros((32, 10240, 3072), dtype=torch.bfloat16, device=device)
    w2 = torch.zeros((32, 3072, 5120), dtype=torch.bfloat16, device=device)
    w13[local_experts, 0, 0] = gate.to(torch.bfloat16)
    w13[local_experts, 5120, 0] = up.to(torch.bfloat16)
    w2[local_experts, 0, 0] = fc2.to(torch.bfloat16)
    return w13, w2


def _analytical_sparse_reference(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    gate_coefficient, up_coefficient, fc2_coefficient = _expert_coefficients(topk_ids)
    hidden = hidden_states[:, :1].float()
    gate = hidden * gate_coefficient
    up = hidden * up_coefficient
    swiglu = (up * gate) * torch.sigmoid(gate)
    fc1_bf16 = (swiglu * topk_weights).to(torch.bfloat16)
    route_terms = (fc1_bf16.float() * fc2_coefficient).to(torch.bfloat16)
    ordered = route_terms[:, 0].float()
    for route_slot in range(1, 8):
        ordered = ordered + route_terms[:, route_slot].float()
    expected = torch.zeros_like(hidden_states)
    expected[:, 0] = ordered.to(torch.bfloat16)
    return expected


_HOT_EXPERT = 19


def _mixed_width_tail_routing(
    tokens: int, rank: int, device: torch.device
) -> torch.Tensor:
    local_tokens = torch.arange(tokens, dtype=torch.int64, device=device)
    global_tokens = rank * tokens + local_tokens
    route_slots = (
        global_tokens[:, None] * 8
        + torch.arange(8, dtype=torch.int64, device=device)[None, :]
    )
    non_hot = (route_slots * 127 + 23) % 511
    non_hot = non_hot + (non_hot >= _HOT_EXPERT)
    hot_block = route_slots // (2 * tokens)
    hot_mask = route_slots % (2 * tokens) == hot_block % 8
    return torch.where(hot_mask, _HOT_EXPERT, non_hot)


def test_sparse_reference_routing_exercises_mixed_width_tail() -> None:
    for tokens in (16, 32, 64):
        routing = torch.cat(
            [
                _mixed_width_tail_routing(tokens, rank, torch.device("cpu"))
                for rank in range(16)
            ]
        )
        counts = torch.bincount(routing.reshape(-1), minlength=512)
        assert int(counts.max()) == 64
        assert int(counts[_HOT_EXPERT]) == 64
        hot_rows, hot_slots = torch.nonzero(routing == _HOT_EXPERT, as_tuple=True)
        hot_hidden = torch.exp2((torch.arange(16 * tokens)[hot_rows] % 7).float() - 3.0)
        assert hot_hidden.unique().numel() > 1
        assert hot_slots.unique().numel() == 8
        _validate_gathered_routing_capacity(routing)


@pytest.mark.solo
@pytest.mark.parametrize(
    "backend", (None, "cute_dsl"), ids=("default_cuda", "cute_dsl")
)
def test_ep16_sm103_sparse_reference_and_repeated_result(backend) -> None:
    """Run under 16-rank torchrun with ``-m pytest <this-file> -k sparse_reference``."""

    if int(os.environ.get("WORLD_SIZE", "0")) != 16:
        pytest.skip("requires torchrun with exactly 16 ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    visible_devices = torch.cuda.device_count()
    if visible_devices == 1:
        # A per-rank CUDA_VISIBLE_DEVICES binding keeps the physical local rank
        # in the environment but exposes its assigned GPU as logical device 0.
        device_index = 0
    else:
        if local_world_size > visible_devices:
            pytest.skip("requires one visible CUDA device per local rank")
        if not 0 <= local_rank < visible_devices:
            pytest.skip("requires a valid local rank among visible CUDA devices")
        device_index = local_rank
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (10, 3):
        pytest.skip("requires exact SM103")

    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    if dist.get_world_size() != 16:
        pytest.skip("requires an EP16 process group")
    rank = dist.get_rank()

    try:
        # Use the actual collective, not a mocked agreement result. Mismatched
        # choices must fail before inspecting weights or allocating a workspace.
        with pytest.raises(
            ValueError, match="all EP16 ranks must select the same backend"
        ):
            CakeMxfp8MegaMoeEp16(None, None, backend="cuda" if rank % 2 else "cute_dsl")
        dist.barrier()
        w13, w2 = _make_sparse_expert_weights(rank, device)
        weights = preprocess_cake_mxfp8_megamoe_ep16_weights(w13, w2)
        del w13, w2
        torch.cuda.empty_cache()

        slot_weights = torch.tensor(
            (0.5, 0.25, 0.125, 0.0625, 0.5, 0.25, 0.125, 0.0625),
            dtype=torch.float32,
            device=device,
        )
        for tokens in (16, 32, 64):
            local_tokens = torch.arange(tokens, dtype=torch.int64, device=device)
            global_tokens = rank * tokens + local_tokens
            topk_ids = _mixed_width_tail_routing(tokens, rank, device)
            assert bool(torch.any(topk_ids // 32 != rank).item())
            topk_weights = slot_weights.expand(tokens, 8).contiguous()
            hidden_states = torch.zeros(
                (tokens, 3072), dtype=torch.bfloat16, device=device
            )
            base_hidden = torch.exp2((global_tokens % 7).float() - 3.0).to(
                torch.bfloat16
            )
            if backend is None:
                session = CakeMxfp8MegaMoeEp16(weights, topk_ids)
            else:
                session = CakeMxfp8MegaMoeEp16(weights, topk_ids, backend=backend)
            workspace = session._workspace
            assert workspace.flags.tensor.numel() == (66 if backend else 2)
            for symm in (
                workspace.flags,
                workspace.published_hidden,
                workspace.published_topk_ids,
                workspace.published_topk_weights,
                workspace.route_terms,
            ):
                if backend == "cute_dsl":
                    assert symm.peers is None
                    assert len(symm.peer_pointers) == 32
                    assert all(pointer > 0 for pointer in symm.peer_pointers[:16])
                    assert symm.peer_pointers[16:] == (0,) * 16
                else:
                    assert symm.peer_pointers is None
                    assert symm.peers.dtype == torch.int64
                    assert symm.peers.numel() == 16

            first_copy = None
            for repeat in range(32):
                # Identical pairs retain the repeated-result check, while
                # alternating pairs expose stale payloads when banks are reused.
                factor = 1.0 if (repeat // 2) % 2 == 0 else 0.5
                hidden_states[:, 0].copy_(base_hidden * factor)
                expected = _analytical_sparse_reference(
                    hidden_states, topk_ids, topk_weights
                )
                output = session.run(
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    out=session.workspace_output,
                )
                torch.cuda.synchronize(device)
                assert bool(torch.isfinite(output).all().item())
                torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
                if repeat % 2 == 0:
                    first_copy = output.clone()
                else:
                    assert torch.equal(first_copy, output)
            del session, workspace, symm, first_copy, expected, output
            torch.cuda.empty_cache()
            dist.barrier()
    finally:
        if initialized_here:
            dist.destroy_process_group()
