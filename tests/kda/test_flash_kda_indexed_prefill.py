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

Tests for the source-only generated indexed FlashKDA registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

import flashinfer.kda as kda_api
from flashinfer.jit import flash_kda_indexed as loader


_TOOLCHAIN_IDENTITY = "sha256:" + "7" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_record(root: Path, relative: str, payload: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": relative,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }


def _manifest(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.glob("flashkda_generated_indexed_*")):
        if path.is_file() and path.name != loader._RECEIPT_NAME:
            payload = path.read_bytes()
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(payload),
                    "size_bytes": len(payload),
                }
            )
    return records


def _write_receipt(root: Path, catalog_payload: bytes) -> None:
    outputs = _manifest(root)
    assert len(outputs) == 79
    receipt = {
        "kind": "flashinfer.generated_kda_indexed_prefill.import_receipt",
        "schema_version": 1,
        "catalog_sha256": _sha256(catalog_payload),
        "inputs": [
            {
                "target": target,
                "archive_sha256": _sha256(f"archive-{target}".encode()),
            }
            for target in ("sm100a", "sm103a")
        ],
        "outputs": outputs,
        "passed": True,
    }
    (root / loader._RECEIPT_NAME).write_text(json.dumps(receipt), encoding="utf-8")


def _catalog_tree(root: Path) -> tuple[loader._TargetRecord, ...]:
    targets = []
    for target, architecture in (("sm100a", "sm_100a"), ("sm103a", "sm_103a")):
        module_ids = tuple(
            f"{target}-module-{index:03d}"
            for index in range(loader._EXPECTED_MODULE_COUNTS[target])
        )
        dispatcher = (
            f"FLASHINFER_MODULE_IDS = {module_ids!r}\n"
            "def bind_loaded_modules(modules):\n"
            "    def select_fp32_indexed_schedule_route(**kwargs):\n"
            "        return 'direct_m128'\n"
            "    def prepare_fwd(**kwargs):\n"
            "        return modules[FLASHINFER_MODULE_IDS[0]]('lazy-call')\n"
            "    return {\n"
            "        'select_fp32_indexed_schedule_route': select_fp32_indexed_schedule_route,\n"
            "        'prepare_fwd': prepare_fwd,\n"
            "    }\n"
        ).encode()
        dispatcher_record = _write_record(
            root,
            f"flashkda_generated_indexed_{target}_dispatcher.py",
            dispatcher,
        )
        build_recipe = (
            f"EXPECTED_TOOLCHAIN_IDENTITY = {_TOOLCHAIN_IDENTITY!r}\n"
        ).encode()
        build_recipe_record = _write_record(
            root,
            f"flashkda_generated_indexed_{target}_build.py",
            build_recipe,
        )
        modules = []
        for index, module_id in enumerate(module_ids):
            cuda = f"// generated CUDA {target}/{index}\n".encode()
            host = f"// generated host {target}/{index}\n".encode()
            modules.append(
                {
                    "id": module_id,
                    "module_ident": f"sealed_kda_{target}_{index:03d}",
                    "entry_point": "run",
                    "kernel_name": f"kernel_generated_kda_{target}_{index:03d}",
                    "compile_options": ["--use_fast_math"],
                    "cooperative": False,
                    "tma_abi": "pointer",
                    "use_pdl": False,
                    "cuda_source": _write_record(
                        root,
                        f"flashkda_generated_indexed_sealed_kda_"
                        f"{target}_{index:03d}.cu",
                        cuda,
                    ),
                    "host_source": _write_record(
                        root,
                        f"flashkda_generated_indexed_sealed_kda_"
                        f"{target}_{index:03d}_binding.cpp",
                        host,
                    ),
                    "expected_cubin": {
                        "sha256": _sha256(b"exact cubin"),
                        "size_bytes": len(b"exact cubin"),
                    },
                }
            )
        targets.append(
            {
                "target": target,
                "architecture": architecture,
                "input_archive_sha256": _sha256(f"archive-{target}".encode()),
                "input_fragment_sha256": _sha256(f"fragment-{target}".encode()),
                "route_denominator_sha256": _sha256(f"routes-{target}".encode()),
                "dispatcher_seed_identity": "sha256:" + _sha256(dispatcher),
                "toolchain_identity": _TOOLCHAIN_IDENTITY,
                "contract": dict(loader._EXPECTED_RUNTIME_CONTRACT),
                "build_recipe": build_recipe_record,
                "dispatcher": dispatcher_record,
                "modules": modules,
            }
        )
    catalog = {
        "kind": "flashinfer.generated_kda_indexed_prefill.source_catalog",
        "schema_version": 2,
        "targets": targets,
    }
    catalog_payload = (
        json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    (root / loader._CATALOG_NAME).write_bytes(catalog_payload)
    _write_receipt(root, catalog_payload)
    assert sum(path.is_file() for path in root.rglob("*")) == 80
    return loader._read_catalog(root)


def test_source_catalog_verifies_complete_target_module_and_source_closure(
    tmp_path: Path,
) -> None:
    targets = _catalog_tree(tmp_path)

    assert [target.target for target in targets] == ["sm100a", "sm103a"]
    assert [len(target.modules) for target in targets] == [18, 19]
    assert all(
        module.cuda_source.path.name.startswith("flashkda_generated_indexed_")
        and module.host_source.path.name.startswith("flashkda_generated_indexed_")
        for target in targets
        for module in target.modules
    )

    targets[0].modules[0].cuda_source.path.write_bytes(b"drift")
    with pytest.raises(loader.GeneratedKDAIndexedPrefillError, match="source closure"):
        loader._read_catalog(tmp_path)


def test_checked_in_registry_uses_semantic_deduplicated_kda_layout() -> None:
    root = loader._source_root()
    targets = loader._read_catalog(root)
    modules = [module for target in targets for module in target.modules]
    cuda_paths = {module.cuda_source.path for module in modules}
    host_paths = {module.host_source.path for module in modules}

    assert root.name == "kda"
    assert len(cuda_paths) == 21
    assert len(host_paths) == 21
    for module in modules:
        semantic = module.module_ident.removeprefix("flashkda_")
        assert module.cuda_source.path.name == (
            f"flashkda_generated_indexed_{semantic}.cu"
        )
        assert module.host_source.path.name == (
            f"flashkda_generated_indexed_{semantic}_binding.cpp"
        )
        assert "module_" not in module.cuda_source.path.name
        assert "module_" not in module.host_source.path.name


def test_source_catalog_rejects_files_outside_the_import_receipt(
    tmp_path: Path,
) -> None:
    _catalog_tree(tmp_path)
    (tmp_path / "flashkda_generated_indexed_unbound_source.cu").write_text(
        "// unbound\n"
    )

    with pytest.raises(loader.GeneratedKDAIndexedPrefillError, match="source closure"):
        loader._read_catalog(tmp_path)


def test_source_catalog_rejects_path_capable_module_ident(tmp_path: Path) -> None:
    _catalog_tree(tmp_path)
    catalog_path = tmp_path / loader._CATALOG_NAME
    catalog = json.loads(catalog_path.read_text())
    catalog["targets"][0]["modules"][0]["module_ident"] = "../escape"
    catalog_payload = (
        json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    catalog_path.write_bytes(catalog_payload)
    _write_receipt(tmp_path, catalog_payload)

    with pytest.raises(loader.GeneratedKDAIndexedPrefillError, match="C identifier"):
        loader._read_catalog(tmp_path)


def test_source_catalog_binds_each_archive_to_its_receipt(tmp_path: Path) -> None:
    _catalog_tree(tmp_path)
    receipt_path = tmp_path / loader._RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"][0]["archive_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        loader.GeneratedKDAIndexedPrefillError, match="differs from the import receipt"
    ):
        loader._read_catalog(tmp_path)


def test_dispatcher_binding_keeps_module_loading_lazy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = _catalog_tree(tmp_path)[0]
    loads = []

    def load_entrypoint(target_name: str, module_id: str):
        loads.append((target_name, module_id))
        return lambda *args: args

    monkeypatch.setattr(loader, "_load_entrypoint", load_entrypoint)
    dispatcher = loader._load_dispatcher_source(target)

    assert loads == []
    assert dispatcher["prepare_fwd"]() == ("lazy-call",)
    assert loads == [("sm100a", target.modules[0].id)]


def test_cached_cubin_identity_is_checked_before_host_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = _catalog_tree(tmp_path)[0]
    module = target.modules[0]
    build = tmp_path / "generated_build"
    build.mkdir()
    (build / f"{module.module_ident}.cubin").write_bytes(b"wrong cubin")
    monkeypatch.setattr(loader, "_module_build_directory", lambda *_args: build)

    with pytest.raises(
        loader.GeneratedKDAIndexedPrefillError, match="cached cubin identity"
    ):
        loader._exact_cubin(module, target)


def test_exact_cubin_compilation_uses_isolated_toolchain_checked_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = _catalog_tree(tmp_path / "catalog")[0]
    module = target.modules[0]
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"exact cubin")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(loader, "_cuda_include_dirs", lambda: (tmp_path,))
    monkeypatch.setattr(loader.subprocess, "run", run)

    assert loader._compile_exact_cubin(module, target) == b"exact cubin"
    command = observed["command"]
    assert command[1] == "-I"
    assert Path(command[2]).name == "flash_kda_indexed_nvrtc.py"
    assert command[command.index("--toolchain-identity") + 1] == _TOOLCHAIN_IDENTITY
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": loader._NVRTC_TIMEOUT_SECONDS,
    }


def _public_call_tensors() -> dict[str, torch.Tensor]:
    q = torch.empty((1, 2, 1, 128), dtype=torch.bfloat16)
    return {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty_like(q),
        "beta": torch.empty((1, 2, 1), dtype=torch.bfloat16),
        "A_log": torch.empty((1,), dtype=torch.float32),
        "dt_bias": torch.empty((1, 128), dtype=torch.float32),
        "initial_state": torch.empty((2, 1, 128, 128), dtype=torch.float32),
        "ssm_state_indices": torch.zeros((1,), dtype=torch.int32),
    }


def test_explicit_backend_routes_exact_indexed_prefill_before_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = (object(), object())
    observed = {}
    monkeypatch.setattr(
        kda_api._flash_kda_indexed,
        "flash_kda_indexed_prefill_is_eligible",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        kda_api._flash_kda_indexed,
        "_run_flash_kda_indexed_prefill",
        lambda **kwargs: observed.update(kwargs) or sentinel,
    )
    monkeypatch.setattr(
        kda_api._kda_prefill,
        "_flash_kda_prefill_is_eligible",
        lambda **_kwargs: pytest.fail(
            "indexed source route must precede legacy prefill"
        ),
    )

    result = kda_api.recurrent_kda(
        **_public_call_tensors(),
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
        backend="cake",
    )

    assert result is sentinel
    assert observed["initial_state"].dtype == torch.float32
    assert observed["state_indices"].dtype == torch.int32


def test_source_backend_rejects_output_alias_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensors = _public_call_tensors()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        loader,
        "_target_for_device",
        lambda _device: pytest.fail("alias validation must precede dispatch"),
    )

    with pytest.raises(ValueError, match="output must not overlap q"):
        loader._run_flash_kda_indexed_prefill(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            g=tensors["g"],
            beta=tensors["beta"],
            A_log=tensors["A_log"],
            dt_bias=tensors["dt_bias"],
            scale=128**-0.5,
            initial_state=tensors["initial_state"],
            output_final_state=True,
            lower_bound=-5.0,
            cu_seqlens=None,
            output=tensors["q"].view_as(tensors["q"]),
            state_indices=tensors["ssm_state_indices"],
        )


@pytest.mark.gpu
@pytest.mark.arch_blackwell
def test_indexed_fp32_source_backend_matches_flash_kda_reference() -> None:
    reference_root_value = os.environ.get("FLASH_KDA_REFERENCE_ROOT")
    if reference_root_value is None:
        pytest.skip(
            "FLASH_KDA_REFERENCE_ROOT must identify the pinned reference checkout"
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("the generated source package targets SM100a and SM103a")
    flash_kda = pytest.importorskip("flash_kda")
    flash_kda_C = pytest.importorskip("flash_kda_C")
    reference_root = Path(reference_root_value).resolve(strict=True)
    expected_reference_commit = os.environ.get("FLASH_KDA_REFERENCE_COMMIT")
    if expected_reference_commit is None:
        pytest.skip("FLASH_KDA_REFERENCE_COMMIT must pin the reference commit")
    assert len(expected_reference_commit) == 40 and set(
        expected_reference_commit
    ) <= set("0123456789abcdef")
    reference_commit = subprocess.check_output(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD^{commit}"],
        text=True,
    ).strip()
    assert reference_commit == expected_reference_commit
    Path(flash_kda.__file__).resolve(strict=True).relative_to(reference_root)
    Path(flash_kda_C.__file__).resolve(strict=True).relative_to(reference_root)

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(13008)
    batch_size, tokens, heads, head_dim = 1, 63, 6, 128
    shape = (batch_size, tokens, heads, head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    g = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    beta = torch.randn(
        (batch_size, tokens, heads),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    A_log = torch.rand(
        (heads,), dtype=torch.float32, device=device, generator=generator
    )
    dt_bias = torch.rand(
        (heads, head_dim), dtype=torch.float32, device=device, generator=generator
    )
    state_pool = 0.25 * torch.randn(
        (loader._EXPECTED_STATE_POOL_CAPACITY, heads, head_dim, head_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    state_before = state_pool.clone()
    state_index = (13008 * 17 + 11) % loader._EXPECTED_STATE_POOL_CAPACITY
    assert state_index == 127
    state_indices = torch.tensor([state_index], dtype=torch.int32, device=device)
    compact_initial = state_before.index_select(0, state_indices.long()).contiguous()
    compact_final = torch.empty_like(compact_initial)
    expected_output = torch.empty_like(q)
    scale = head_dim**-0.5
    workspace = torch.empty(
        flash_kda.get_workspace_size(tokens, heads, batch_size),
        dtype=torch.uint8,
        device=device,
    )
    flash_kda._fwd_raw(
        q,
        k,
        v,
        g,
        beta,
        scale,
        expected_output,
        workspace,
        A_log,
        dt_bias,
        -5.0,
        initial_state=compact_initial,
        final_state=compact_final,
        cu_seqlens=None,
    )
    expected_state = state_before.clone()
    expected_state.index_copy_(0, state_indices.long(), compact_final)

    output = torch.empty_like(q)
    actual_output, actual_state = kda_api.recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_pool,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        ssm_state_indices=state_indices,
        output=output,
        beta_is_logit=True,
        backend="cake",
    )
    torch.cuda.synchronize()

    assert actual_output is output
    assert actual_state is state_pool
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    unselected = torch.tensor([0, 1, 256], dtype=torch.int64, device=device)
    assert torch.equal(
        actual_state.index_select(0, unselected),
        state_before.index_select(0, unselected),
    )
