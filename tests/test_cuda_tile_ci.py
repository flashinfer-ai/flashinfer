from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import build_backend
import build_utils
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

CUDA_TILE_COMPILE_REQUIREMENTS = [
    "nvidia-cuda-nvcc<13.4,>=13.2",
    "nvidia-cuda-tileiras<13.4,>=13.2",
    "nvidia-nvvm<13.4,>=13.2",
    "nvidia-nvjitlink<14,>=13.3",
    "nvidia-cuda-crt<13.4,>=13.2",
]


class FakeDistribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.version = version


def load_ci_image_module():
    path = REPO_ROOT / "docker" / "test_ci_image.py"
    spec = importlib.util.spec_from_file_location("test_ci_image_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_tile_compile_requirements_are_exact_and_return_a_copy() -> None:
    first = build_utils.get_cuda_tile_compile_dependency_requirements()
    second = build_utils.get_cuda_tile_compile_dependency_requirements()

    assert first == CUDA_TILE_COMPILE_REQUIREMENTS
    assert second == CUDA_TILE_COMPILE_REQUIREMENTS
    assert first is not second
    assert all("runtime" not in requirement for requirement in first)

    first.append("unexpected")
    assert second == CUDA_TILE_COMPILE_REQUIREMENTS


def test_ci_installer_uses_shared_cuda_tile_compile_requirements_for_cuda13() -> None:
    installer = (
        REPO_ROOT / "docker" / "install" / "install_python_packages.sh"
    ).read_text()

    assert (
        "from build_utils import get_cuda_tile_compile_dependency_requirements"
        in installer
    )
    assert "mapfile -t CUDA_TILE_COMPILE_DEPENDENCIES" in installer
    assert 'pip3 install --no-deps "${CUDA_TILE_COMPILE_DEPENDENCIES[@]}"' in installer
    assert "cuda-tile[tileiras]" not in installer
    assert '[[ "${CUDA_MAJOR}" == "13" ]] && (( 10#${CUDA_MINOR} < 4 ))' in installer
    assert "(( ${#CUDA_TILE_COMPILE_DEPENDENCIES[@]} > 0 ))" in installer
    assert '[[ "${CUDA_TAG}" == "cu134" ]]' in installer
    assert 'CUDA_PYTHON="cuda-python==13.4.1"' in installer
    assert 'CUDA_PYTHON="cuda-python==${CUDA_MAJOR}.${CUDA_MINOR}"' in installer

    resolver_install = installer.index(
        "pip3 install \\\n  -r /install/requirements.txt"
    )
    compiler_install = installer.index(
        'pip3 install --no-deps "${CUDA_TILE_COMPILE_DEPENDENCIES[@]}"'
    )
    cudnn_override = installer.index(
        'pip3 install --upgrade --no-deps "${CUDNN_PACKAGE}==${CUDNN_VERSION}"'
    )
    assert resolver_install < compiler_install < cudnn_override


def test_build_backend_installs_shared_cuda_tile_compile_requirements(
    monkeypatch,
) -> None:
    sentinel = "example-compile-dependency==1"
    commands: list[list[str]] = []

    monkeypatch.setattr(
        build_backend, "_system_cuda_tile_compiler_available", lambda: False
    )
    monkeypatch.setattr(build_backend, "_compile_deps_installed", lambda specs: False)
    monkeypatch.setattr(build_backend, "_no_pip_installs", lambda: False)
    monkeypatch.setattr(build_backend.shutil, "which", lambda executable: None)
    monkeypatch.setattr(
        build_backend,
        "get_cuda_tile_compile_dependency_requirements",
        lambda: [sentinel],
        raising=False,
    )
    monkeypatch.setattr(
        build_backend.subprocess,
        "run",
        lambda command, check: commands.append(command),
    )

    build_backend._install_cuda_tile_compile_deps()

    assert commands == [[sys.executable, "-m", "pip", "install", "--no-deps", sentinel]]


def test_build_backend_uses_system_cuda_tile_compiler(monkeypatch) -> None:
    monkeypatch.setattr(
        build_backend, "_system_cuda_tile_compiler_available", lambda: True
    )
    monkeypatch.setattr(
        build_backend,
        "get_cuda_tile_compile_dependency_requirements",
        lambda: pytest.fail("pip compile requirements should not be loaded"),
    )

    build_backend._install_cuda_tile_compile_deps()


@pytest.mark.parametrize(
    ("release", "expected"),
    [((13, 0), False), ((13, 4), True), ((13, 5), True), (None, False)],
)
def test_build_backend_system_cuda_tile_compiler_requires_cuda_13_4(
    monkeypatch, release, expected
) -> None:
    monkeypatch.setattr(build_backend, "_detect_cuda_release", lambda: release)
    monkeypatch.setattr(build_backend.Path, "is_file", lambda path: True)
    monkeypatch.setattr(build_backend.os, "access", lambda path, mode: True)

    assert build_backend._system_cuda_tile_compiler_available() is expected


def test_ci_image_validates_cuda_tile_versions_and_compiler(monkeypatch) -> None:
    ci_image = load_ci_image_module()
    compiler_path = "/opt/cuda-tile/bin/tileiras"
    compile_module = SimpleNamespace(
        _find_compiler_bin=lambda: SimpleNamespace(path=compiler_path)
    )
    commands = []

    def fake_import_module(name: str):
        if name == "cuda.tile.tune":
            return SimpleNamespace()
        if name == "cuda.tile._compile":
            return compile_module
        raise AssertionError(f"unexpected import: {name}")

    versions = {
        "cuda-tile": "1.4.2",
        "nvidia-cuda-tileiras": "13.3.0",
    }
    monkeypatch.setattr(ci_image.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        ci_image.importlib.metadata, "version", lambda name: versions[name]
    )
    monkeypatch.setattr(
        ci_image.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )

    assert ci_image._validate_cuda_tile_compiler() == (
        "1.4.2",
        "13.3.0",
        compiler_path,
    )
    assert commands == [
        (
            [compiler_path, "--help"],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "timeout": 30,
            },
        )
    ]


def test_ci_image_uses_system_tileiras_for_cuda_13_4(monkeypatch) -> None:
    ci_image = load_ci_image_module()
    compiler_path = "/usr/local/cuda/bin/tileiras"
    compile_module = SimpleNamespace(
        _find_compiler_bin=lambda: SimpleNamespace(path=compiler_path)
    )

    def fake_import_module(name: str):
        if name == "cuda.tile.tune":
            return SimpleNamespace()
        if name == "cuda.tile._compile":
            return compile_module
        raise AssertionError(f"unexpected import: {name}")

    def fake_version(name: str):
        if name == "cuda-tile":
            return "1.6.0"
        raise ci_image.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(ci_image.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(ci_image.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(
        ci_image.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout="Supported targets: sm_100 sm_107", stderr=""
        ),
    )

    assert ci_image._validate_cuda_tile_compiler("13.4") == (
        "1.6.0",
        "system",
        compiler_path,
    )


def test_ci_image_rejects_tileiras_wheel_for_cuda_13_4(monkeypatch) -> None:
    ci_image = load_ci_image_module()
    monkeypatch.setattr(
        ci_image.importlib, "import_module", lambda name: SimpleNamespace()
    )
    versions = {"cuda-tile": "1.6.0", "nvidia-cuda-tileiras": "13.4.92"}
    monkeypatch.setattr(
        ci_image.importlib.metadata, "version", lambda name: versions[name]
    )

    with pytest.raises(SystemExit, match="shadows the system compiler"):
        ci_image._validate_cuda_tile_compiler("13.4")


def test_ci_image_rejects_system_tileiras_without_sm107(monkeypatch) -> None:
    ci_image = load_ci_image_module()
    compiler_path = "/usr/local/cuda/bin/tileiras"

    def fake_import_module(name: str):
        if name == "cuda.tile.tune":
            return SimpleNamespace()
        if name == "cuda.tile._compile":
            return SimpleNamespace(
                _find_compiler_bin=lambda: SimpleNamespace(path=compiler_path)
            )
        raise AssertionError(f"unexpected import: {name}")

    def fake_version(name: str):
        if name == "cuda-tile":
            return "1.6.0"
        raise ci_image.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(ci_image.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(ci_image.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(
        ci_image.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout="sm_100", stderr=""),
    )

    with pytest.raises(SystemExit, match="does not advertise SM107"):
        ci_image._validate_cuda_tile_compiler("13.4")


@pytest.mark.parametrize(
    ("expected_cuda", "expected_result", "expected_calls"),
    [
        ("12.9", None, []),
        ("13.0", ("1.4.2", "13.3.0", "/opt/cuda-tile/bin/tileiras"), ["13.0"]),
        ("13.4", ("1.4.2", "13.3.0", "/opt/cuda-tile/bin/tileiras"), ["13.4"]),
    ],
)
def test_ci_image_requires_cuda_tile_compiler_only_for_cuda13(
    monkeypatch,
    expected_cuda: str,
    expected_result: tuple[str, str, str] | None,
    expected_calls: list[str],
) -> None:
    ci_image = load_ci_image_module()
    calls = []
    compiler_details = ("1.4.2", "13.3.0", "/opt/cuda-tile/bin/tileiras")

    def validate_compiler(expected_cuda_version):
        calls.append(expected_cuda_version)
        return compiler_details

    monkeypatch.setattr(ci_image, "_validate_cuda_tile_compiler", validate_compiler)

    assert ci_image._validate_cuda_tile_for_runtime(expected_cuda) == expected_result
    assert calls == expected_calls


def test_ci_image_routes_triton_blackwell_to_the_system_toolkit() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.ci").read_text()

    assert 'ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"' in dockerfile
    assert 'ENV TRITON_PTXAS_BLACKWELL_PATH="/usr/local/cuda/bin/ptxas"' in dockerfile


def test_ci_image_rejects_cuda_tile_tune_import_failure(monkeypatch) -> None:
    ci_image = load_ci_image_module()

    def fake_import_module(name: str):
        assert name == "cuda.tile.tune"
        raise ModuleNotFoundError("cuda.tile.tune is missing")

    monkeypatch.setattr(ci_image.importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit, match=r"could not import cuda\.tile\.tune"):
        ci_image._validate_cuda_tile_compiler()


def test_ci_image_rejects_cuda_tile_compiler_discovery_failure(monkeypatch) -> None:
    ci_image = load_ci_image_module()

    def find_compiler_bin():
        raise RuntimeError("compiler is unavailable")

    def fake_import_module(name: str):
        if name == "cuda.tile.tune":
            return SimpleNamespace()
        if name == "cuda.tile._compile":
            return SimpleNamespace(_find_compiler_bin=find_compiler_bin)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(ci_image.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(ci_image.importlib.metadata, "version", lambda name: "1.4.2")

    with pytest.raises(SystemExit, match="could not discover cuda-tile compiler"):
        ci_image._validate_cuda_tile_compiler()


def test_ci_image_rejects_nonzero_cuda_tile_compiler(monkeypatch) -> None:
    ci_image = load_ci_image_module()
    compiler_path = "/opt/cuda-tile/bin/tileiras"

    def fake_import_module(name: str):
        if name == "cuda.tile.tune":
            return SimpleNamespace()
        if name == "cuda.tile._compile":
            return SimpleNamespace(
                _find_compiler_bin=lambda: SimpleNamespace(path=compiler_path)
            )
        raise AssertionError(f"unexpected import: {name}")

    def fail_run(command, **kwargs):
        raise ci_image.subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(ci_image.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(ci_image.importlib.metadata, "version", lambda name: "1.4.2")
    monkeypatch.setattr(ci_image.subprocess, "run", fail_run)

    with pytest.raises(SystemExit, match="compiler --help exited with status 7"):
        ci_image._validate_cuda_tile_compiler()


@pytest.mark.parametrize(
    ("expected_cuda", "distribution_name", "version"),
    [
        ("12.9", "nvidia-cuda-runtime-cu12", "12.9.79"),
        ("13.0", "nvidia_cuda_runtime_cu13", "13.0.48"),
    ],
)
def test_ci_image_accepts_matching_cuda_runtime_major(
    monkeypatch,
    expected_cuda: str,
    distribution_name: str,
    version: str,
) -> None:
    ci_image = load_ci_image_module()
    distributions = [FakeDistribution(distribution_name, version)]
    monkeypatch.setattr(
        ci_image.importlib.metadata, "distributions", lambda: distributions
    )

    assert ci_image._validate_cuda_runtime_distributions(expected_cuda) == [
        (distribution_name, version)
    ]


def test_ci_image_rejects_mixed_cuda_runtime_majors(monkeypatch) -> None:
    ci_image = load_ci_image_module()
    distributions = [
        FakeDistribution("nvidia-cuda-runtime-cu12", "12.9.79"),
        FakeDistribution("nvidia-cuda-runtime", "13.3.0"),
    ]
    monkeypatch.setattr(
        ci_image.importlib.metadata, "distributions", lambda: distributions
    )

    with pytest.raises(
        SystemExit,
        match=r"nvidia-cuda-runtime==13\.3\.0 targets CUDA 13; expected CUDA 12",
    ):
        ci_image._validate_cuda_runtime_distributions("12.9")
