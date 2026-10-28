import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def wheel_builder(monkeypatch):
    path = Path(__file__).parents[2] / "scripts" / "build_jit_cache_wheel.py"
    spec = importlib.util.spec_from_file_location("build_jit_cache_wheel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", Mock())
    return module


@pytest.mark.cpu_only
def test_dependency_install_recovers_from_disconnect(wheel_builder):
    failure = subprocess.CalledProcessError(
        1, ["python", "-m", "pip"], stderr=b"IncompleteRead(32768 bytes read)"
    )
    env = Mock()
    env.install.side_effect = [failure, None]
    requirements = {"torch==2.13.0+cu129"}

    wheel_builder.install_build_requirements(env, requirements)

    assert env.install.call_count == 2
    assert all(call.args == (requirements,) for call in env.install.call_args_list)
    wheel_builder.time.sleep.assert_called_once_with(10)


@pytest.mark.cpu_only
def test_dependency_install_stops_after_three_attempts(wheel_builder):
    failure = subprocess.CalledProcessError(
        1, ["python", "-m", "pip"], output="RemoteDisconnected"
    )
    env = Mock()
    env.install.side_effect = failure

    with pytest.raises(subprocess.CalledProcessError) as caught:
        wheel_builder.install_build_requirements(env, {"torch"})

    assert caught.value is failure
    assert env.install.call_count == 3
    assert [call.args[0] for call in wheel_builder.time.sleep.call_args_list] == [
        10,
        20,
    ]


@pytest.mark.cpu_only
def test_dependency_resolution_error_is_not_retried(wheel_builder):
    failure = subprocess.CalledProcessError(
        1, ["python", "-m", "pip"], stderr=b"No matching distribution found"
    )
    env = Mock()
    env.install.side_effect = failure

    with pytest.raises(subprocess.CalledProcessError):
        wheel_builder.install_build_requirements(env, {"unavailable-package"})

    env.install.assert_called_once()
    wheel_builder.time.sleep.assert_not_called()


def _fake_python(tmp_path, body):
    """A stand-in interpreter that records `-m pip install ...` invocations."""
    log = tmp_path / "pip.log"
    python = tmp_path / "python"
    python.write_text(f'#!/bin/bash\necho "$*" >> "{log}"\n{body}\n')
    python.chmod(0o755)
    return python, log


@pytest.mark.cpu_only
def test_current_interpreter_install_retries_interrupted_pip(wheel_builder, tmp_path):
    attempts = tmp_path / "attempts"
    python, log = _fake_python(
        tmp_path,
        f'echo x >> "{attempts}"\n'
        f'if [ "$(wc -l < "{attempts}")" -lt 2 ]; then\n'
        '  echo "ReadTimeoutError: read timed out" >&2; exit 1\n'
        "fi",
    )
    env = wheel_builder.CurrentInterpreterEnv()
    env.python_executable = str(python)

    wheel_builder.install_build_requirements(env, {"torch", "ninja"})

    assert log.read_text().splitlines() == ["-m pip install ninja torch"] * 2
    wheel_builder.time.sleep.assert_called_once_with(10)


@pytest.mark.cpu_only
def test_current_interpreter_install_skips_empty_requirements(wheel_builder, tmp_path):
    python, log = _fake_python(tmp_path, "exit 1")
    env = wheel_builder.CurrentInterpreterEnv()
    env.python_executable = str(python)

    wheel_builder.install_build_requirements(env, set())

    assert not log.exists()


@pytest.mark.cpu_only
def test_build_wheel_runs_in_current_interpreter(wheel_builder, monkeypatch, tmp_path):
    """The build must not create a PEP 517 isolated env: its random path would
    leak into compile commands and defeat sccache across runs."""
    calls = []

    class FakeBuilder:
        build_system_requires = {"setuptools>=77", "torch"}

        def __init__(self, source_dir, python_executable):
            calls.append(("init", source_dir, python_executable))

        def get_requires_for_build(self, distribution):
            calls.append(("requires", distribution))
            return {"nvidia-cutlass-dsl>=4.6.2a0"}

        def build(self, distribution, output_directory):
            calls.append(("build", distribution, output_directory))
            return "dist/provider.whl"

    build_module = type(sys)("build")
    build_module.ProjectBuilder = FakeBuilder
    monkeypatch.setitem(sys.modules, "build", build_module)
    installs = []
    monkeypatch.setattr(
        wheel_builder.CurrentInterpreterEnv,
        "install",
        lambda self, requirements: installs.append(set(requirements)),
    )
    monkeypatch.setattr(wheel_builder, "site_package_dirs", lambda: ["site"])
    monkeypatch.setattr(
        wheel_builder,
        "pin_header_mtimes",
        lambda roots: calls.append(("pin", roots, len(installs))) or 0,
    )
    monkeypatch.chdir(tmp_path)

    assert wheel_builder.build_wheel() == "dist/provider.whl"

    assert calls[0] == ("init", ".", sys.executable)
    assert installs == [{"setuptools>=77", "torch"}, {"nvidia-cutlass-dsl>=4.6.2a0"}]
    # Headers are pinned after the last install and before compilation.
    assert calls[-2:] == [
        ("pin", ["site"], 2),
        ("build", "wheel", str(tmp_path.resolve() / "dist")),
    ]


@pytest.mark.cpu_only
def test_pin_header_mtimes_only_touches_headers(wheel_builder, tmp_path):
    site = tmp_path / "site-packages"
    headers = [
        site / "torch/include/ATen/ops/a_native.h",
        site / "nvidia/cu12/include/cuda/std/atomic",
        site / "tvm_ffi/kernel.cuh",
    ]
    others = [
        site / "torch/__init__.py",
        site / "torch/lib/libtorch.so",
        site / "torch/include/ATen/stub.py",
    ]
    for index, path in enumerate(headers + others):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        os.utime(path, (1_700_000_000 + index, 1_700_000_000 + index))
    link = site / "torch/include/link.h"
    link.symlink_to(headers[0])

    assert wheel_builder.pin_header_mtimes([site]) == len(headers)

    pinned = wheel_builder.PINNED_HEADER_MTIME
    assert [path.stat().st_mtime for path in headers] == [pinned] * len(headers)
    assert [path.stat().st_mtime for path in others] == [
        1_700_000_000 + len(headers) + index for index in range(len(others))
    ]
    assert link.is_symlink()
    assert wheel_builder.pin_header_mtimes([site]) == 0


def _gnu_cxx():
    for compiler in ("g++", "c++"):
        try:
            version = subprocess.run(
                [compiler, "--version"], capture_output=True, text=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        if "Free Software Foundation" in version:
            return compiler
    return None


@pytest.mark.cpu_only
def test_pinned_mtimes_make_gcc_preprocessing_deterministic(wheel_builder, tmp_path):
    """Reproduce the torch-header cache misses with GCC's `#pragma once`.

    Two byte-identical `#pragma once` headers stand in for torch's duplicate
    `ATen/ops/*_native.h` stubs. Their mtimes model two pip installs of the same
    wheel: one wrote both copies within one second, the other did not.
    """
    compiler = _gnu_cxx()
    if compiler is None:
        pytest.skip("requires GCC, whose #pragma once compares mtime and contents")
    include = tmp_path / "site-packages" / "pkg" / "include"
    include.mkdir(parents=True)
    duplicates = [include / "first_native.h", include / "second_native.h"]
    for path in duplicates:
        path.write_text("#pragma once\nint declared_in_native_header();\n")
    source = tmp_path / "tu.cpp"
    source.write_text('#include "first_native.h"\n#include "second_native.h"\n')

    def preprocess(mtimes):
        for path, mtime in zip(duplicates, mtimes, strict=True):
            os.utime(path, (mtime, mtime))
        wheel_builder.pin_header_mtimes([tmp_path / "site-packages"])
        return subprocess.run(
            [compiler, "-E", "-P", f"-I{include}", str(source)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    same_second = (1_700_000_000, 1_700_000_000)
    different_seconds = (1_700_000_000, 1_700_000_001)
    assert preprocess(same_second) == preprocess(different_seconds)
    assert preprocess(same_second).count("declared_in_native_header") == 1
