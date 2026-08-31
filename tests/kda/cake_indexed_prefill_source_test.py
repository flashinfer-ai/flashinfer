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

Design-level tests for the source-only indexed Cake KDA loader.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

import flashinfer.cake_kda_indexed_prefill as loader
import flashinfer.kda as kda_api


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


def _catalog_tree(root: Path) -> tuple[loader._TargetRecord, ...]:
    targets = []
    for target, architecture in (("sm100a", "sm_100a"), ("sm103a", "sm_103a")):
        module_ids = tuple(f"{target}-module-{index:03d}" for index in range(18))
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
            f"generated/{target}/cake_kda_dispatcher.py",
            dispatcher,
        )
        modules = []
        for index, module_id in enumerate(module_ids):
            cuda = f"// cake CUDA {target}/{index}\n".encode()
            host = f"// cake host {target}/{index}\n".encode()
            modules.append(
                {
                    "id": module_id,
                    "module_ident": f"sealed_kda_{target}_{index:03d}",
                    "entry_point": "run",
                    "kernel_name": f"kernel_cake_kda_{target}_{index:03d}",
                    "compile_options": ["--use_fast_math"],
                    "cooperative": False,
                    "tma_abi": {},
                    "use_pdl": False,
                    "cuda_source": _write_record(
                        root,
                        f"generated/{target}/cake_kda_module_{index:03d}/"
                        f"cake_kda_module_{index:03d}_kernel.cu",
                        cuda,
                    ),
                    "host_source": _write_record(
                        root,
                        f"generated/{target}/cake_kda_module_{index:03d}/"
                        f"cake_kda_module_{index:03d}_host.cpp",
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
                "contract": dict(loader._EXPECTED_RUNTIME_CONTRACT),
                "dispatcher": dispatcher_record,
                "modules": modules,
            }
        )
    catalog = {
        "kind": "flashinfer.cake_kda_indexed_prefill.source_catalog",
        "schema_version": 1,
        "targets": targets,
    }
    catalog_payload = json.dumps(
        catalog, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    (root / "cake_kda_source_catalog.json").write_bytes(catalog_payload)
    receipt = {
        "kind": "flashinfer.cake_kda_indexed_prefill.import_receipt",
        "schema_version": 1,
        "catalog_sha256": _sha256(catalog_payload),
        "inputs": [],
        "outputs": [],
        "passed": True,
    }
    (root / "cake_kda_import_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return loader._read_catalog(root)


def test_source_catalog_verifies_complete_target_module_and_source_closure(
    tmp_path: Path,
) -> None:
    targets = _catalog_tree(tmp_path)

    assert [target.target for target in targets] == ["sm100a", "sm103a"]
    assert all(len(target.modules) == 18 for target in targets)
    assert all(
        module.cuda_source.path.name.startswith("cake_")
        and module.host_source.path.name.startswith("cake_")
        for target in targets
        for module in target.modules
    )

    targets[0].modules[0].cuda_source.path.write_bytes(b"drift")
    with pytest.raises(loader.CakeKDAIndexedPrefillError, match="content identity"):
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
    build = tmp_path / "cake_build"
    build.mkdir()
    (build / f"{module.module_ident}.cubin").write_bytes(b"wrong cubin")
    monkeypatch.setattr(loader, "_module_build_directory", lambda *_args: build)

    with pytest.raises(loader.CakeKDAIndexedPrefillError, match="cached cubin identity"):
        loader._exact_cubin(module, target)


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


def test_explicit_cake_backend_routes_exact_indexed_prefill_before_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = (object(), object())
    observed = {}
    monkeypatch.setattr(
        kda_api._cake_kda_indexed_prefill,
        "cake_kda_indexed_prefill_is_eligible",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        kda_api._cake_kda_indexed_prefill,
        "run_cake_kda_indexed_prefill",
        lambda **kwargs: observed.update(kwargs) or sentinel,
    )
    monkeypatch.setattr(
        kda_api._kda_prefill,
        "_flash_kda_prefill_is_eligible",
        lambda **_kwargs: pytest.fail("indexed source route must precede legacy Cake"),
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


def test_public_cake_wrapper_contains_literal_backend_and_forwards_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}
    monkeypatch.setattr(
        kda_api,
        "recurrent_kda",
        lambda **kwargs: observed.update(kwargs) or (object(), object()),
    )

    kda_api.recurrent_kda_cake(**_public_call_tensors())

    assert observed["backend"] == "cake"
    assert 'backend="cake"' in inspect.getsource(kda_api.recurrent_kda_cake)
