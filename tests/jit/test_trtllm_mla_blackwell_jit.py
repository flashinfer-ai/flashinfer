# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.jit import trtllm_mla_blackwell


def _write_source(root: Path, relative: str, payload: bytes) -> dict[str, str]:
    path = root / "generated" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}


def _write_synthetic_catalog(root: Path) -> dict[str, object]:
    domains: dict[str, object] = {}
    for domain, count in trtllm_mla_blackwell._DOMAIN_DEVICE_COUNTS.items():
        host = _write_source(
            root,
            f"host/{domain}.cpp",
            f"// host for {domain}\n".encode(),
        )
        devices_by_target = {}
        for target in trtllm_mla_blackwell._TARGET_ORDER:
            devices = []
            for index in range(count):
                module_ident = f"{domain}_kernel_{index}"
                device = _write_source(
                    root,
                    f"device/{target}/{module_ident}.cu",
                    f"// device {domain}/{target}/{index}\n".encode(),
                )
                devices.append(
                    {
                        **device,
                        "module_ident": module_ident,
                        "compile_flags": ["--use_fast_math"],
                    }
                )
            devices_by_target[target] = devices
        domains[domain] = {
            "host_source": host,
            "device_sources": devices_by_target,
        }
    catalog = {
        "schema_version": 3,
        "target_order": list(trtllm_mla_blackwell._TARGET_ORDER),
        "targets": {
            target: {
                "arch": arch,
                "multi_processor_count": multi_processor_count,
            }
            for target, (arch, multi_processor_count) in (
                trtllm_mla_blackwell._TARGETS.items()
            )
        },
        "domain_order": list(trtllm_mla_blackwell._DOMAIN_ORDER),
        "domains": domains,
    }
    (root / "generated" / "source_catalog.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )
    return catalog


def _rewrite_catalog(root: Path, catalog: dict[str, object]) -> None:
    (root / "generated" / "source_catalog.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def _clear_loader_caches():
    trtllm_mla_blackwell._source_catalog.cache_clear()
    trtllm_mla_blackwell._load_domain_module.cache_clear()
    yield
    trtllm_mla_blackwell._source_catalog.cache_clear()
    trtllm_mla_blackwell._load_domain_module.cache_clear()


def test_catalog_accepts_exact_ordered_four_target_nine_domain_eighteen_source_topology(
    monkeypatch, tmp_path
):
    _write_synthetic_catalog(tmp_path)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    catalog = trtllm_mla_blackwell._source_catalog()

    assert catalog["domain_order"] == list(trtllm_mla_blackwell._DOMAIN_ORDER)
    assert catalog["target_order"] == list(trtllm_mla_blackwell._TARGET_ORDER)
    domains = catalog["domains"]
    assert isinstance(domains, dict)
    assert tuple(domains) == trtllm_mla_blackwell._DOMAIN_ORDER
    assert tuple(catalog["targets"]) == trtllm_mla_blackwell._TARGET_ORDER
    assert catalog["targets"] == {
        "sm_100a_148": {"arch": "sm_100a", "multi_processor_count": 148},
        "sm_100a_152": {"arch": "sm_100a", "multi_processor_count": 152},
        "sm_103a_148": {"arch": "sm_103a", "multi_processor_count": 148},
        "sm_103a_152": {"arch": "sm_103a", "multi_processor_count": 152},
    }
    for target in trtllm_mla_blackwell._TARGET_ORDER:
        assert (
            sum(
                len(profile["device_sources"][target])
                for profile in domains.values()
            )
            == 18
        )
        assert [
            len(domains[domain]["device_sources"][target])
            for domain in trtllm_mla_blackwell._DOMAIN_ORDER
        ] == [1, 1, 1, 8, 1, 1, 1, 2, 2]


def test_catalog_rejects_reordered_targets(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    targets = catalog["targets"]
    assert isinstance(targets, dict)
    catalog["targets"] = {
        target: targets[target]
        for target in reversed(trtllm_mla_blackwell._TARGET_ORDER)
    }
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="catalog identity is invalid"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_reordered_target_order(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    catalog["target_order"] = list(reversed(trtllm_mla_blackwell._TARGET_ORDER))
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="catalog identity is invalid"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_missing_target_order(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    del catalog["target_order"]
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="catalog schema is invalid"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_schema_version_drift(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    catalog["schema_version"] = 2
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="catalog identity is invalid"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_reordered_domains(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    order = list(trtllm_mla_blackwell._DOMAIN_ORDER)
    order[0], order[1] = order[1], order[0]
    catalog["domain_order"] = order
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="catalog identity is invalid"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_reordered_domain_object(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    domains = catalog["domains"]
    assert isinstance(domains, dict)
    catalog["domains"] = {
        domain: domains[domain]
        for domain in reversed(trtllm_mla_blackwell._DOMAIN_ORDER)
    }
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="domain topology is invalid"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_schema_extensions(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    catalog["unsealed_extension"] = True
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="catalog schema is invalid"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_domain_device_count_drift(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    domains = catalog["domains"]
    assert isinstance(domains, dict)
    profile = domains["mla_bf16_clc"]
    assert isinstance(profile, dict)
    devices = profile["device_sources"]["sm_100a_148"]
    assert isinstance(devices, list)
    devices.pop()
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="exactly 8 device sources"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_reordered_domain_target_inventory(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    domains = catalog["domains"]
    assert isinstance(domains, dict)
    profile = domains["mla_bf16_vquarter"]
    assert isinstance(profile, dict)
    devices = profile["device_sources"]
    assert isinstance(devices, dict)
    profile["device_sources"] = {
        target: devices[target]
        for target in reversed(trtllm_mla_blackwell._TARGET_ORDER)
    }
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="target inventory is invalid"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_cross_target_device_identity_order_drift(
    monkeypatch, tmp_path
):
    catalog = _write_synthetic_catalog(tmp_path)
    domains = catalog["domains"]
    assert isinstance(domains, dict)
    profile = domains["mla_bf16_clc"]
    assert isinstance(profile, dict)
    devices = profile["device_sources"]["sm_103a_152"]
    assert isinstance(devices, list)
    devices[0], devices[1] = devices[1], devices[0]
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="identity order differs"):
        trtllm_mla_blackwell._source_catalog()


@pytest.mark.parametrize("compile_flags", [[], ["--use_fast_math", "-lineinfo"]])
def test_catalog_rejects_compile_flag_drift(monkeypatch, tmp_path, compile_flags):
    catalog = _write_synthetic_catalog(tmp_path)
    domains = catalog["domains"]
    assert isinstance(domains, dict)
    profile = domains["mla_bf16_vquarter"]
    assert isinstance(profile, dict)
    devices = profile["device_sources"]["sm_103a_152"]
    assert isinstance(devices, list)
    devices[0]["compile_flags"] = compile_flags
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="compile flags differ"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_noncanonical_path(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    domains = catalog["domains"]
    assert isinstance(domains, dict)
    profile = domains["mla_bf16_vquarter"]
    assert isinstance(profile, dict)
    profile["host_source"]["path"] = "../outside.cpp"
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="path is noncanonical"):
        trtllm_mla_blackwell._source_catalog()


def test_catalog_rejects_device_path_target_drift(monkeypatch, tmp_path):
    catalog = _write_synthetic_catalog(tmp_path)
    domains = catalog["domains"]
    assert isinstance(domains, dict)
    profile = domains["mla_bf16_vquarter"]
    assert isinstance(profile, dict)
    devices = profile["device_sources"]["sm_100a_148"]
    assert isinstance(devices, list)
    devices[0]["path"] = "device/sm_100a_152/mla_bf16_vquarter_kernel_0.cu"
    _rewrite_catalog(tmp_path, catalog)
    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="device source paths are invalid"):
        trtllm_mla_blackwell._source_catalog()


def test_sealed_source_rejects_checksum_drift(tmp_path):
    record = _write_source(tmp_path, "device/source.cu", b"original")
    (tmp_path / "generated" / "device" / "source.cu").write_bytes(b"drift")

    with pytest.raises(RuntimeError, match="source identity drift"):
        trtllm_mla_blackwell._sealed_source_bytes(tmp_path, record)


def test_get_domain_module_rejects_unknown_domain_before_loading_catalog():
    with pytest.raises(ValueError, match="unknown TRT-LLM MLA domain"):
        trtllm_mla_blackwell.get_domain_module("not_a_domain")


@pytest.mark.parametrize(
    ("capability", "multi_processor_count"),
    [((10, 0), 150), ((10, 3), 150), ((9, 0), 132), ((12, 0), 170)],
)
def test_get_domain_module_rejects_unsupported_runtime_target(
    monkeypatch, capability, multi_processor_count
):
    monkeypatch.setattr(
        trtllm_mla_blackwell.torch.cuda,
        "get_device_capability",
        lambda device: capability,
    )
    monkeypatch.setattr(
        trtllm_mla_blackwell.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            multi_processor_count=multi_processor_count
        ),
    )

    with pytest.raises(
        ValueError,
        match=rf"requires one of the exact targets.*{multi_processor_count} SMs",
    ):
        trtllm_mla_blackwell.get_domain_module("mla_bf16_clc")


@pytest.mark.parametrize(
    ("capability", "multi_processor_count", "target", "arch"),
    [
        ((10, 0), 148, "sm_100a_148", "sm_100a"),
        ((10, 0), 152, "sm_100a_152", "sm_100a"),
        ((10, 3), 148, "sm_103a_148", "sm_103a"),
        ((10, 3), 152, "sm_103a_152", "sm_103a"),
    ],
)
def test_get_domain_module_compiles_and_embeds_exact_domain_cubin_set_once(
    monkeypatch, tmp_path, capability, multi_processor_count, target, arch
):
    _write_synthetic_catalog(tmp_path)
    build_root = tmp_path / "build"
    commands: list[list[str]] = []
    locks: list[tuple[Path, bool]] = []
    loads: list[dict[str, object]] = []
    loaded_module = object()

    class FakeLock:
        def __init__(self, path, *, thread_local):
            locks.append((Path(path), thread_local))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_run(command, *, text, capture_output):
        assert text is True
        assert capture_output is True
        command = list(command)
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        source = Path(command[-3])
        output.write_bytes(f"cubin:{source.stem}".encode())
        return SimpleNamespace(returncode=0, stderr="")

    def fake_load_inline(name, **kwargs):
        loads.append({"name": name, **kwargs})
        return loaded_module

    monkeypatch.setattr(trtllm_mla_blackwell, "_source_dir", lambda: tmp_path)
    monkeypatch.setattr(
        trtllm_mla_blackwell.jit_env, "FLASHINFER_JIT_DIR", build_root
    )
    monkeypatch.setattr(
        trtllm_mla_blackwell, "_nvcc", lambda: Path("/opt/cuda/bin/nvcc")
    )
    monkeypatch.setattr(trtllm_mla_blackwell, "FileLock", FakeLock)
    monkeypatch.setattr(trtllm_mla_blackwell.subprocess, "run", fake_run)
    monkeypatch.setattr(trtllm_mla_blackwell.cpp, "load_inline", fake_load_inline)
    capability_devices = []
    properties_devices = []

    def fake_get_device_capability(device):
        capability_devices.append(device)
        return capability

    def fake_get_device_properties(device):
        properties_devices.append(device)
        return SimpleNamespace(multi_processor_count=multi_processor_count)

    monkeypatch.setattr(
        trtllm_mla_blackwell.torch.cuda,
        "get_device_capability",
        fake_get_device_capability,
    )
    monkeypatch.setattr(
        trtllm_mla_blackwell.torch.cuda,
        "get_device_properties",
        fake_get_device_properties,
    )
    device = torch.device("cuda:1")

    first = trtllm_mla_blackwell.get_domain_module("mla_bf16_clc", device)
    second = trtllm_mla_blackwell.get_domain_module("mla_bf16_clc", device)

    assert first is loaded_module
    assert second is loaded_module
    assert capability_devices == [device, device]
    assert properties_devices == [device, device]
    assert len(commands) == 8
    assert all(f"-arch={arch}" in command for command in commands)
    assert all("--use_fast_math" in command for command in commands)
    assert all(
        Path(command[-3]).parent.name == target for command in commands
    )
    other_arch = "sm_103a" if arch == "sm_100a" else "sm_100a"
    assert all(f"-arch={other_arch}" not in command for command in commands)
    assert len(locks) == 1
    assert locks[0][1] is False
    assert len(loads) == 1
    assert f"_{target}_" in loads[0]["name"]
    assert loads[0]["cpp_sources"] == "// host for mla_bf16_clc\n"
    assert set(loads[0]["embed_cubin"]) == {
        "mla_bf16_clc_kernel_0",
        "mla_bf16_clc_kernel_1",
        "mla_bf16_clc_kernel_2",
        "mla_bf16_clc_kernel_3",
        "mla_bf16_clc_kernel_4",
        "mla_bf16_clc_kernel_5",
        "mla_bf16_clc_kernel_6",
        "mla_bf16_clc_kernel_7",
    }
    assert loads[0]["extra_ldflags"] == ["-lcuda"]
