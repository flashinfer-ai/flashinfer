from __future__ import annotations

import importlib.util
import re
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

    cuda13_blocks = [
        match.group("body")
        for match in re.finditer(
            r'if \[\[ "\$\{CUDA_MAJOR\}" == "13" \]\]; then\n'
            r"(?P<body>.*?)\nfi",
            installer,
            flags=re.DOTALL,
        )
    ]
    assert len(cuda13_blocks) == 2
    assert any(
        "mapfile -t CUDA_TILE_COMPILE_DEPENDENCIES" in block for block in cuda13_blocks
    )
    assert any(
        'pip3 install --no-deps "${CUDA_TILE_COMPILE_DEPENDENCIES[@]}"' in block
        for block in cuda13_blocks
    )

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


@pytest.mark.parametrize(
    ("expected_cuda", "expected_result", "expected_calls"),
    [
        ("12.9", None, []),
        ("13.0", ("1.4.2", "13.3.0", "/opt/cuda-tile/bin/tileiras"), [True]),
    ],
)
def test_ci_image_requires_cuda_tile_compiler_only_for_cuda13(
    monkeypatch,
    expected_cuda: str,
    expected_result: tuple[str, str, str] | None,
    expected_calls: list[bool],
) -> None:
    ci_image = load_ci_image_module()
    calls = []
    compiler_details = ("1.4.2", "13.3.0", "/opt/cuda-tile/bin/tileiras")

    def validate_compiler():
        calls.append(True)
        return compiler_details

    monkeypatch.setattr(ci_image, "_validate_cuda_tile_compiler", validate_compiler)

    assert ci_image._validate_cuda_tile_for_runtime(expected_cuda) == expected_result
    assert calls == expected_calls


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
