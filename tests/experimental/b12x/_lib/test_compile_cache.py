"""Compile-cache rooting: the highest-risk mechanical edit of the restructure.

Wrong fingerprint root ⇒ silent stale disk-cache hits (running old kernels),
so these assertions are load-bearing.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

compiler = importlib.import_module("b12x._lib.compiler")


def test_package_root_is_the_b12x_package():
    root = compiler._PACKAGE_ROOT
    assert root.name == "b12x", root
    assert (root / "_lib" / "compiler.py").is_file()


def test_fingerprint_tracks_source_edits(tmp_path, monkeypatch):
    (tmp_path / "kernel.py").write_text("x = 1\n")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    monkeypatch.setattr(compiler, "_PACKAGE_ROOT", tmp_path)

    before = compiler._compute_b12x_package_fingerprint()

    (pycache / "kernel.cpython-312.pyc").write_bytes(b"ignored")
    assert compiler._compute_b12x_package_fingerprint() == before, (
        "__pycache__ must not affect the fingerprint"
    )

    (tmp_path / "kernel.py").write_text("x = 2\n")
    after = compiler._compute_b12x_package_fingerprint()
    assert after != before, "editing any source must change the fingerprint"


def test_cache_dir_resolution_order(monkeypatch):
    for name in ("B12X_COMPILE_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)

    assert compiler._cute_compile_cache_dir() == (
        Path.home() / ".cache" / "b12x" / "compile"
    )

    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert compiler._cute_compile_cache_dir() == Path("/xdg/b12x/compile")

    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", "/explicit")
    assert compiler._cute_compile_cache_dir() == Path("/explicit")


def test_disk_cache_key_includes_device_arch_and_forwards_ordinal(monkeypatch):
    compile_callable = object()
    seen = {"calls": 0}

    def _fake_device_arch_key(ordinal):
        seen["calls"] += 1
        seen["ordinal"] = ordinal
        return ("cuda", (12, 1), 48)

    monkeypatch.setattr(compiler, "_current_device_ordinal", lambda: 3)
    monkeypatch.setattr(compiler, "_device_arch_key", _fake_device_arch_key)
    monkeypatch.setattr(
        compiler,
        "_static_compile_cache_context",
        lambda _callable: (
            "package",
            "toolchain",
            (),
            (),
        ),
    )

    payload = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_cache_key_includes_device_arch_and_forwards_ordinal,
        (),
        {},
    )
    repeated_payload = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_cache_key_includes_device_arch_and_forwards_ordinal,
        (),
        {},
    )

    assert payload[0] == "b12x_cute_compile_cache_v4"
    assert payload[4] == ("cuda", (12, 1), 48)
    assert repeated_payload == payload
    assert seen["ordinal"] == 3
    assert seen["calls"] == 1


def test_device_arch_key_retries_after_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compiler, "_DEVICE_ARCH_KEYS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert compiler._device_arch_key() is None
    assert compiler._DEVICE_ARCH_KEYS == {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(major=12, minor=1, multi_processor_count=48),
    )
    expected = ("cuda", (12, 1), 48)
    assert compiler._device_arch_key() == expected
    assert {0: expected} == compiler._DEVICE_ARCH_KEYS


def test_device_arch_key_is_memoized_per_ordinal(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compiler, "_DEVICE_ARCH_KEYS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    boards = {0: (12, 0, 170), 1: (12, 1, 48)}
    probes = {0: 0, 1: 0}
    current = {"index": 0}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])

    def _properties(device):
        probes[device] += 1
        return SimpleNamespace(
            major=boards[device][0],
            minor=boards[device][1],
            multi_processor_count=boards[device][2],
        )

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        _properties,
    )

    assert compiler._device_arch_key() == ("cuda", (12, 0), 170)
    current["index"] = 1
    assert compiler._device_arch_key() == ("cuda", (12, 1), 48)
    assert compiler._device_arch_key(0) == ("cuda", (12, 0), 170)
    assert compiler._DEVICE_ARCH_KEYS == {
        0: ("cuda", (12, 0), 170),
        1: ("cuda", (12, 1), 48),
    }
    assert probes == {0: 1, 1: 1}
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(
            AssertionError("cached architecture was re-probed")
        ),
    )
    assert compiler._device_arch_key(0) == ("cuda", (12, 0), 170)


def test_disk_payload_retries_architecture_probe_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compiler, "_DEVICE_ARCH_KEYS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        compiler,
        "_static_compile_cache_context",
        lambda _callable: ("package", "toolchain", (), ()),
    )
    attempts = {"count": 0}

    def _properties(_device):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient CUDA probe failure")
        return SimpleNamespace(major=12, minor=0, multi_processor_count=170)

    monkeypatch.setattr(torch.cuda, "get_device_properties", _properties)
    compile_callable = object()

    first = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_payload_retries_architecture_probe_failure,
        (),
        {},
    )
    second = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_payload_retries_architecture_probe_failure,
        (),
        {},
    )
    third = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_payload_retries_architecture_probe_failure,
        (),
        {},
    )

    assert first[4] is None
    assert second[4] == ("cuda", (12, 0), 170)
    assert third == second
    assert attempts["count"] == 2


def test_explicit_cache_payload_includes_device_arch(monkeypatch):
    compile_callable = object()
    device_arch = ("cuda", (12, 1), 48)
    compile_options = ("--opt-level=2",)
    compile_environment = (("CUTE_DSL_ARCH", "sm_120"),)
    monkeypatch.setattr(compiler, "_current_device_ordinal", lambda: 7)
    monkeypatch.setattr(
        compiler,
        "_device_arch_key",
        lambda ordinal: device_arch if ordinal == 7 else None,
    )
    monkeypatch.setattr(
        compiler,
        "_static_compile_cache_context",
        lambda _callable: (
            "a" * 64,
            (("python", "cpython", (3, 12, 0)),),
            compile_options,
            compile_environment,
        ),
    )
    compile_spec = compiler.KernelCompileSpec.from_facts(
        "test.arch.cache",
        1,
        ("rows", 8),
    )
    kwargs = {"mode": "test"}
    kwargs_json, kwargs_hash = compiler._compile_kwargs_json_key(kwargs)

    payload = compiler._compile_disk_cache_payload(
        compile_callable,
        test_explicit_cache_payload_includes_device_arch,
        (),
        kwargs,
        compile_spec,
    )

    assert len(payload) == 11
    assert payload[0] == "b12x_cute_compile_cache_v7_explicit_spec"
    assert payload[4] == device_arch
    assert payload[5:11] == (
        compile_spec.hash_key,
        compile_spec.json_key,
        kwargs_hash,
        kwargs_json,
        compile_options,
        compile_environment,
    )


def test_architecture_unavailable_disables_disk_cache(monkeypatch):
    monkeypatch.setenv("B12X_COMPILE_DISK_CACHE", "1")
    payload = (
        "b12x_cute_compile_cache_v4",
        ("function", "test", "kernel"),
        "package",
        "toolchain",
        None,
        (),
        (),
        (),
        (),
    )

    assert not compiler._cute_compile_disk_cache_enabled_for_payload(payload)


def test_explicit_memory_cache_hit_skips_freeze_and_disk_payload(monkeypatch):
    cute = pytest.importorskip("cutlass.cute")
    compile_callable = object()
    compiled = object()
    compile_spec = compiler.KernelCompileSpec.from_facts(
        "test.arch.hot_path",
        1,
        ("rows", 8),
    )
    monkeypatch.setattr(cute, "compile", compile_callable)
    monkeypatch.setattr(
        compiler,
        "_device_arch_key",
        lambda _ordinal: (_ for _ in ()).throw(
            AssertionError("memory hit probed the device architecture")
        ),
    )
    monkeypatch.setattr(compiler, "_memory_cache_get", lambda key: compiled)
    monkeypatch.setattr(
        compiler,
        "_compile_disk_cache_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("memory hit rebuilt the disk payload")
        ),
    )
    runtime_control = importlib.import_module("b12x._lib.runtime_control")

    with runtime_control.kernel_resolution_guard("cached compile remains launchable"):
        assert (
            compiler.compile(
                test_explicit_memory_cache_hit_skips_freeze_and_disk_payload,
                compile_spec=compile_spec,
            )
            is compiled
        )


def test_frozen_memory_miss_rejects_before_disk_cache_load(monkeypatch):
    cute = pytest.importorskip("cutlass.cute")
    runtime_control = importlib.import_module("b12x._lib.runtime_control")
    compile_spec = compiler.KernelCompileSpec.from_facts(
        "test.freeze.disk_hit",
        1,
        ("rows", 8),
    )
    monkeypatch.setattr(cute, "compile", object())
    monkeypatch.setattr(compiler, "_memory_cache_get", lambda _key: None)
    monkeypatch.setattr(
        compiler,
        "_compile_disk_cache_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen miss built a persistent-cache payload")
        ),
    )
    monkeypatch.setattr(
        compiler,
        "_load_cute_compile_from_disk",
        lambda _key: (_ for _ in ()).throw(
            AssertionError("frozen miss loaded a persistent module")
        ),
    )

    with runtime_control.kernel_resolution_guard("disk hits must not bypass freeze"):
        with pytest.raises(runtime_control.KernelResolutionFrozenError):
            compiler.compile(
                test_frozen_memory_miss_rejects_before_disk_cache_load,
                compile_spec=compile_spec,
            )


@pytest.mark.parametrize("explicit", [False, True])
def test_portable_manifest_records_architecture_without_uuid(monkeypatch, explicit):
    monkeypatch.setattr(compiler, "_current_device_ordinal", lambda: 0)
    monkeypatch.setattr(
        compiler, "_device_arch_key", lambda ordinal: ("cuda", (12, 1), 48)
    )
    spec = (
        compiler.KernelCompileSpec.from_facts("test.portable", 1, ("rows", 8))
        if explicit
        else None
    )
    payload = compiler._compile_disk_cache_payload(
        object(),
        test_portable_manifest_records_architecture_without_uuid,
        (),
        {},
        spec,
    )
    semantic = compiler._semantic_compile_manifest_payload(payload)
    assert semantic["device_arch"] == ["cuda", [12, 1], 48]
    assert "device_uuid" not in semantic
    assert ("compile_spec_hash" in semantic) == explicit


@pytest.mark.parametrize("explicit", [False, True])
def test_matching_silicon_shares_disk_artifacts_but_not_loaded_handles(
    monkeypatch,
    explicit,
):
    torch = pytest.importorskip("torch")
    current = {"ordinal": 0}
    monkeypatch.setattr(compiler, "_DEVICE_ARCH_KEYS", {})
    monkeypatch.setattr(compiler, "_DEVICE_COMPILE_CACHE_CONTEXTS", {})
    monkeypatch.setattr(compiler, "_current_device_ordinal", lambda: current["ordinal"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    boards = {
        0: SimpleNamespace(
            major=12,
            minor=1,
            multi_processor_count=48,
            uuid="first",
            name="A",
            total_memory=1,
        ),
        1: SimpleNamespace(
            major=12,
            minor=1,
            multi_processor_count=48,
            uuid="second",
            name="B",
            total_memory=2,
        ),
        2: SimpleNamespace(major=12, minor=0, multi_processor_count=48),
        3: SimpleNamespace(major=12, minor=1, multi_processor_count=40),
    }
    monkeypatch.setattr(torch.cuda, "get_device_properties", boards.__getitem__)
    callable_ = object()
    spec = (
        compiler.KernelCompileSpec.from_facts("test.portable", 1, ("rows", 8))
        if explicit
        else None
    )
    disk, memory = [], []
    for ordinal in boards:
        current["ordinal"] = ordinal
        args = (
            callable_,
            test_matching_silicon_shares_disk_artifacts_but_not_loaded_handles,
            (),
            {},
            spec,
        )
        disk.append(compiler._build_compile_disk_cache_key(*args))
        memory.append(compiler._compile_memory_cache_key(*args))
    assert disk[0] == disk[1]
    assert len(set(disk)) == 3
    assert len(set(memory)) == 4


def test_offline_compile_identity_matches_parent_without_cuda_probing(monkeypatch):
    monkeypatch.setattr(compiler, "_DEVICE_ARCH_KEYS", {})
    monkeypatch.setattr(compiler, "_OFFLINE_COMPILE_DEVICE_ORDINAL", None)
    monkeypatch.setattr(compiler, "_OFFLINE_CUTE_NO_JIT", False)
    compiler._configure_offline_compile_target(7, (12, 1), 48)
    assert compiler._device_arch_key(7) == ("cuda", (12, 1), 48)
    with pytest.raises(RuntimeError, match="cannot change"):
        compiler._configure_offline_compile_target(7, (12, 0), 48)
