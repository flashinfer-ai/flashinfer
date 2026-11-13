"""CPU tests for scripts/build_windows_arm64_wheels.py.

Every subprocess call goes through a fake runner, so these run on Linux, macOS
and Windows without a GPU, CUDA toolkit or MSVC. Builds are simulated by
writing a minimal wheel where the real build would.
"""

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"

requires_posix_bash = pytest.mark.skipif(
    sys.platform == "win32", reason="sources Linux CI bash scripts"
)


@pytest.fixture
def driver():
    name = "_test_build_windows_arm64_wheels"
    spec = importlib.util.spec_from_file_location(
        name, SCRIPTS_DIR / "build_windows_arm64_wheels.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop("jit_cache_provider_validation", None)


@pytest.fixture
def validation():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import jit_cache_provider_validation

        yield jit_cache_provider_validation
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        sys.modules.pop("jit_cache_provider_validation", None)


# --------------------------------------------------------------------------- #
# Fake wheels and runner
# --------------------------------------------------------------------------- #


def _write_wheel(path: Path, name: str, version: str, files: dict, requires=()):
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {req}\n" for req in requires)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        for member, content in files.items():
            archive.writestr(member.replace("{dist_info}", dist_info), content)
    return path


def make_provider_wheel(
    dist: Path,
    provider: str,
    version: str,
    platform_tag="win_arm64",
    library=lambda module: "module.dll",
):
    name = f"flashinfer-jit-cache-{provider}"
    prefix = f"flashinfer_jit_cache/providers/{provider}"
    modules = ["batch_decode", "norm"]
    manifest = {
        "schema_version": 1,
        "provider_id": provider,
        "distribution": name,
        "version": version,
        "cuda_architectures": [provider],
        "modules": modules,
    }
    files = {f"{prefix}/manifest.json": json.dumps(manifest)}
    for module in modules:
        files[f"{prefix}/jit_cache/{module}/{library(module)}"] = b"MZ"
    files["{dist_info}/entry_points.txt"] = (
        "[flashinfer.jit_cache.providers]\n"
        f"{provider} = flashinfer_jit_cache.providers.{provider}:get_provider\n"
    )
    filename = f"{name.replace('-', '_')}-{version}-cp39-abi3-{platform_tag}.whl"
    return _write_wheel(dist / filename, name, version, files)


def make_shim_wheel(dist: Path, version: str, providers, platform_tag="win_arm64"):
    filename = f"flashinfer_jit_cache-{version}-cp39-abi3-{platform_tag}.whl"
    requires = [f"flashinfer-jit-cache-{p}=={version}" for p in sorted(providers)]
    return _write_wheel(
        dist / filename,
        "flashinfer-jit-cache",
        version,
        {"flashinfer_jit_cache/__init__.py": ""},
        requires,
    )


class FakeRunner:
    """Record commands; answer probes; drop a wheel where a build would."""

    def __init__(self, *, nvcc="13.0", torch_cuda="13.0", platform_tag="win_arm64"):
        self.calls = []
        self.nvcc = nvcc
        self.torch_cuda = torch_cuda
        self.platform_tag = platform_tag

    def __call__(self, command, *, env, cwd=None, capture=False):
        command = [str(part) for part in command]
        self.calls.append({"command": command, "env": dict(env), "cwd": cwd})
        stdout = ""
        joined = " ".join(command)
        if command[:2] == ["nvcc", "--version"]:
            stdout = f"Cuda compilation tools, release {self.nvcc}, V{self.nvcc}.1\n"
        elif "-c" in command and "torch.version.cuda" in joined:
            if self.torch_cuda != command[-1]:
                raise subprocess.CalledProcessError(1, command)
        elif "-c" in command and "m.version" in joined:
            stdout = "torch==2.9.0+cu130\n"
        elif command[-1].endswith("build_jit_cache_wheel.py"):
            self._build_jit_cache_wheel(Path(cwd), env)
        elif command[1:3] == ["-m", "build"]:
            version = env["_TEST_VERSION"]
            providers = [
                f"sm{a.replace('.', '')}"
                for a in env["FLASHINFER_JIT_CACHE_PROVIDER_ARCHS"].split()
            ]
            make_shim_wheel(Path(cwd) / "dist", version, providers, self.platform_tag)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    def _build_jit_cache_wheel(self, cwd: Path, env):
        version = env["_TEST_VERSION"]
        if cwd.name == "flashinfer-jit-cache-provider":
            arch = env["FLASHINFER_JIT_CACHE_PROVIDER_ARCH"].replace(".", "")
            make_provider_wheel(cwd / "dist", f"sm{arch}", version, self.platform_tag)
        else:
            _write_wheel(
                cwd / "dist" / "flashinfer_python-0.6.16-py3-none-any.whl",
                "flashinfer-python",
                "0.6.16",
                {},
            )

    def commands(self):
        return [call["command"] for call in self.calls]


@pytest.fixture
def fake_repo(tmp_path):
    """A repository root with the files the driver reads and writes."""
    repo = tmp_path / "repo"
    (repo / "ci").mkdir(parents=True)
    (repo / "ci" / "cuda-versions.json").write_text(
        (REPO_ROOT / "ci" / "cuda-versions.json").read_text(encoding="utf-8")
    )
    (repo / "version.txt").write_text("0.6.16\n")
    for package in ("flashinfer-jit-cache-provider", "flashinfer-jit-cache"):
        (repo / package).mkdir()
    return repo


def _which_with_msvc(name, path=None):
    return f"C:\\VS\\{name}" if name in ("cl.exe", "link.exe") else None


# --------------------------------------------------------------------------- #
# Host, configuration and toolchain
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_host_check_accepts_only_native_arm64_interpreters(driver):
    driver.check_host("win-arm64")
    for platform_name in ("win-amd64", "linux-aarch64", "macosx-14.0-arm64"):
        with pytest.raises(driver.BuildError, match="native Windows arm64 Python"):
            driver.check_host(platform_name)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("architecture", "expected"),
    [("12.0f", "sm120f"), ("10.3a", "sm103a"), ("sm_90a", "sm90a"), ("8.0", "sm80")],
)
def test_provider_tag_matches_linux_spelling(driver, architecture, expected):
    assert driver.provider_tag(architecture) == expected


@pytest.mark.cpu_only
def test_provider_tag_rejects_garbage(driver):
    with pytest.raises(driver.BuildError, match="Invalid provider architecture"):
        driver.provider_tag("blackwell")


@pytest.mark.cpu_only
def test_cuda_config_comes_from_the_shared_matrix(driver):
    config = driver.load_cuda_config("cu134", REPO_ROOT / "ci" / "cuda-versions.json")
    assert (config.version, config.nightly) == ("13.4", True)
    assert config.pytorch_index_url == "https://download.pytorch.org/whl/nightly/cu134"
    with pytest.raises(driver.BuildError, match="Unknown JIT-cache CUDA label"):
        driver.load_cuda_config("cu999", REPO_ROOT / "ci" / "cuda-versions.json")


@pytest.mark.cpu_only
def test_package_version_matches_provider_backend(driver, fake_repo):
    assert driver.package_version(fake_repo, "20261113", "cu130") == (
        "0.6.16.dev20261113+cu130"
    )
    assert driver.package_version(fake_repo, "", "cu130") == "0.6.16+cu130"


@pytest.mark.cpu_only
def test_cuda_toolchain_must_match_matrix(driver):
    driver.validate_cuda_toolchain(FakeRunner(nvcc="13.0"), {}, "13.0")
    with pytest.raises(driver.BuildError, match="nvcc reports CUDA 12.9"):
        driver.validate_cuda_toolchain(FakeRunner(nvcc="12.9"), {}, "13.0")


@pytest.mark.cpu_only
def test_msvc_on_path_is_used_as_is(driver):
    runner = FakeRunner()
    env = {"PATH": "C:\\VS"}
    assert driver.msvc_environment(runner, env, which=_which_with_msvc) == env
    assert runner.calls == []


@pytest.mark.cpu_only
def test_msvc_is_loaded_from_vcvarsall_when_missing(driver, tmp_path):
    vswhere = tmp_path / "vswhere.exe"
    vswhere.touch()
    install = tmp_path / "VS"
    vcvarsall = install / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    vcvarsall.parent.mkdir(parents=True)
    vcvarsall.touch()
    calls = []

    def runner(command, *, env, cwd=None, capture=False):
        calls.append([str(part) for part in command])
        if str(command[0]) == "cmd.exe":
            calls[-1].append(Path(command[-1]).read_bytes().decode())
        if str(command[0]) == str(vswhere):
            stdout = f"{install}\n"
        else:
            stdout = (
                "=C:=C:\\work\nPath=C:\\VS\\arm64;C:\\Windows\nINCLUDE=C:\\VS\\inc\n"
            )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    def which(name, path=None):
        return f"{path}\\{name}" if path and "arm64" in path else None

    env = driver.msvc_environment(
        runner, {"PATH": "C:\\Windows", "KEEP": "1"}, which=which, vswhere=vswhere
    )

    assert env["Path"] == "C:\\VS\\arm64;C:\\Windows"
    assert env["INCLUDE"] == "C:\\VS\\inc" and env["KEEP"] == "1"
    assert "=C:" not in env
    assert "Microsoft.VisualStudio.Component.VC.Tools.ARM64" in calls[0]
    assert calls[1][:3] == ["cmd.exe", "/d", "/c"]
    assert calls[1][3].endswith("vcvars.bat")
    assert calls[1][4] == f'@call "{vcvarsall}" arm64 >nul || exit /b 1\r\n@set\r\n'


@pytest.mark.cpu_only
def test_msvc_missing_without_vswhere_explains_the_fix(driver, tmp_path):
    with pytest.raises(driver.BuildError, match="ARM64 Native Tools"):
        driver.msvc_environment(
            FakeRunner(),
            {"PATH": ""},
            which=lambda name, path=None: None,
            vswhere=tmp_path / "missing.exe",
        )


# --------------------------------------------------------------------------- #
# Parallelism
# --------------------------------------------------------------------------- #


PARALLELISM_CASES = [
    # (available GB, CPUs, env) -> (MAX_JOBS, NVCC_THREADS, MEM_PER_JOB)
    (128, 72, {}, (16, 1, 8)),
    (4, 72, {}, (1, 1, 8)),
    (512, 16, {}, (16, 1, 8)),
    (512, 72, {"AOT_MAX_JOBS_CAP": "4"}, (4, 1, 8)),
    (512, 72, {"FLASHINFER_NVCC_THREADS": "12"}, (9, 8, 16)),
    (
        512,
        72,
        {"FLASHINFER_NVCC_THREADS": "x", "AOT_MAX_JOBS_MEMORY_GB": "0"},
        (64, 1, 8),
    ),
    (512, 72, {"AOT_MAX_JOBS_MEMORY_GB": "32"}, (16, 1, 32)),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(("memory", "cpus", "env", "expected"), PARALLELISM_CASES)
def test_parallelism(driver, memory, cpus, env, expected):
    result = driver.compute_parallelism(memory, cpus, env)
    assert (result.max_jobs, result.nvcc_threads, result.mem_per_job_gb) == expected


@pytest.mark.cpu_only
@requires_posix_bash
@pytest.mark.parametrize(("memory", "cpus", "env", "expected"), PARALLELISM_CASES)
def test_parallelism_matches_linux_helper(memory, cpus, env, expected):
    """The Python port and compute_jit_cache_parallelism agree on every case."""
    script = f"""
set -euo pipefail
source "{SCRIPTS_DIR / "jit_cache_build_common.sh"}"
free() {{ echo "Mem: 0 0 0 0 0 {memory}"; }}
nproc() {{ echo {cpus}; }}
compute_jit_cache_parallelism >/dev/null
echo "$MAX_JOBS $FLASHINFER_NVCC_THREADS $MEM_PER_JOB"
"""
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in ("FLASHINFER_NVCC_THREADS", "AOT_MAX_JOBS_MEMORY_GB", "AOT_MAX_JOBS_CAP")
    }
    result = subprocess.run(
        ["bash", "-c", script],
        env={**clean_env, **env},
        check=True,
        capture_output=True,
        text=True,
    )
    assert tuple(int(v) for v in result.stdout.split()) == expected


# --------------------------------------------------------------------------- #
# Python build environment (pip constraints)
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_build_environment_pins_preinstalled_torch(driver):
    runner = FakeRunner(torch_cuda="13.0")
    cuda = driver.CudaConfig("cu130", "13.0", "cu130")
    outer = {"PIP_CONSTRAINT": "/outer", "PIP_EXTRA_INDEX_URL": "https://mirror"}

    with driver.python_build_environment(runner, "py", cuda, outer) as env:
        constraint = Path(env["PIP_CONSTRAINT"])
        assert constraint.read_text().strip() == "torch==2.9.0+cu130"
        assert env["PIP_BUILD_CONSTRAINT"] == env["PIP_CONSTRAINT"]
        assert env["PIP_EXTRA_INDEX_URL"] == (
            "https://mirror https://download.pytorch.org/whl/cu130"
        )
        assert env["PIP_RETRIES"] == "10" and env["PIP_DEFAULT_TIMEOUT"] == "120"
        assert "PIP_PRE" not in env

    assert not constraint.exists()
    assert outer == {
        "PIP_CONSTRAINT": "/outer",
        "PIP_EXTRA_INDEX_URL": "https://mirror",
    }
    bootstrap = runner.calls[0]
    assert bootstrap["command"] == [
        "py",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip>=26.2.1",
        "build",
    ]
    assert "PIP_CONSTRAINT" not in bootstrap["env"]
    assert not any("--index-url" in call for call in runner.commands())


@pytest.mark.cpu_only
def test_build_environment_installs_nightly_torch(driver):
    commands = []
    installed = []

    def runner(command, *, env, cwd=None, capture=False):
        command = [str(part) for part in command]
        commands.append(command)
        if "torch.version.cuda" in " ".join(command) and not installed:
            raise subprocess.CalledProcessError(1, command)
        if "--index-url" in command:
            installed.append(True)
        return subprocess.CompletedProcess(command, 0, "torch==2.10.0.dev+cu134", "")

    cuda = driver.CudaConfig("cu134", "13.4", "nightly/cu134")
    with driver.python_build_environment(runner, "py", cuda, {}) as env:
        assert env["PIP_PRE"] == "1"

    install = next(c for c in commands if "--index-url" in c)
    assert install == [
        "py",
        "-m",
        "pip",
        "install",
        "--pre",
        "--upgrade",
        "torch",
        "--index-url",
        "https://download.pytorch.org/whl/nightly/cu134",
    ]


@pytest.mark.cpu_only
def test_build_environment_reports_missing_torch(driver):
    def runner(command, **kwargs):
        if "torch.version.cuda" in " ".join(map(str, command)):
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, "", "")

    cuda = driver.CudaConfig("cu130", "13.0", "cu130")
    with (
        pytest.raises(driver.BuildError, match="preinstall a win_arm64"),
        driver.python_build_environment(runner, "py", cuda, {}),
    ):
        pass


# --------------------------------------------------------------------------- #
# Wheel names and contents
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_wheel_tags_parse_local_versions(driver):
    assert driver.wheel_tags(
        "flashinfer_jit_cache_sm120f-0.6.16.dev20261113+cu130-cp39-abi3-win_arm64.whl"
    ) == ("cp39", "abi3", "win_arm64")
    with pytest.raises(driver.BuildError, match="manylinux_2_28_aarch64"):
        driver.check_wheel_tags(
            Path(
                "flashinfer_jit_cache-0.6.16+cu130-cp39-abi3-manylinux_2_28_aarch64.whl"
            ),
            [driver.JIT_CACHE_WHEEL_TAGS],
        )


@pytest.mark.cpu_only
def test_provider_validation_accepts_module_dll_layout(driver, tmp_path):
    wheel = make_provider_wheel(tmp_path, "sm120f", "0.6.16+cu130")
    driver.validate_provider_wheel(wheel, "sm120f", "0.6.16+cu130")


@pytest.mark.cpu_only
def test_provider_validation_rejects_linux_layout_on_windows(driver, tmp_path):
    wheel = make_provider_wheel(
        tmp_path, "sm120f", "0.6.16+cu130", library=lambda module: f"{module}.so"
    )
    with pytest.raises(driver.BuildError, match="no shared libraries"):
        driver.validate_provider_wheel(wheel, "sm120f", "0.6.16+cu130")


@pytest.mark.cpu_only
def test_linux_provider_validation_is_unchanged(validation, tmp_path):
    wheel = make_provider_wheel(
        tmp_path,
        "sm90a",
        "0.6.16+cu130",
        platform_tag="manylinux_2_28_x86_64",
        library=lambda module: f"{module}.so",
    )
    _, modules = validation.validate_provider(
        validation.Wheel.open(wheel), "sm90a", "0.6.16+cu130", "manylinux_2_28_x86_64"
    )
    assert set(modules) == {"batch_decode", "norm"}
    dll_wheel = make_provider_wheel(tmp_path / "dll", "sm90a", "0.6.16+cu130")
    with pytest.raises(ValueError, match="no shared libraries"):
        validation.validate_provider(
            validation.Wheel.open(dll_wheel), "sm90a", "0.6.16+cu130"
        )


@pytest.mark.cpu_only
def test_shim_validation_rejects_bundled_dlls(validation, tmp_path):
    wheel = _write_wheel(
        tmp_path / "flashinfer_jit_cache-0.6.16+cu130-cp39-abi3-win_arm64.whl",
        "flashinfer-jit-cache",
        "0.6.16+cu130",
        {"flashinfer_jit_cache/jit_cache/norm/module.dll": b"MZ"},
    )
    with pytest.raises(ValueError, match="must not contain shared libraries"):
        validation.validate_shim(validation.Wheel.open(wheel), "0.6.16+cu130", set())


@pytest.mark.cpu_only
def test_wheel_index_recognizes_windows_provider_names():
    spec = importlib.util.spec_from_file_location(
        "_test_update_whl_index_win", SCRIPTS_DIR / "update_whl_index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    filename = "flashinfer_jit_cache_sm120f-0.6.16+cu130-cp39-abi3-win_arm64.whl"
    assert module.get_package_info(Path(filename)) == {
        "package": "flashinfer-jit-cache-sm120f",
        "version": "0.6.16+cu130",
        "cuda": "130",
    }


# --------------------------------------------------------------------------- #
# End to end with a fake runner
# --------------------------------------------------------------------------- #


def _run_main(driver, fake_repo, argv, runner, extra_env=None):
    environ = {
        "PATH": "C:\\VS",
        "_TEST_VERSION": "0.6.16.dev20261113+cu130",
        "FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG": "manylinux_2_28_aarch64",
        "FLASHINFER_NVCC_LAUNCHER": "sccache",
        **(extra_env or {}),
    }
    return driver.main(
        [
            "--skip-host-check",
            "--cuda-label",
            "cu130",
            "--dev-suffix",
            "20261113",
            "--output-dir",
            str(fake_repo / "wheelhouse"),
            "--python",
            "py",
            *argv,
        ],
        runner=runner,
        environ=environ,
        repo_root=fake_repo,
        memory_gb=lambda: 256,
        which=_which_with_msvc,
    )


@pytest.mark.cpu_only
def test_main_builds_all_targets_into_output_dir(driver, fake_repo):
    runner = FakeRunner()
    summary = _run_main(
        driver,
        fake_repo,
        ["--provider-arch", "12.0f", "--provider-arch", "10.3a"],
        runner,
    )

    version = "0.6.16.dev20261113+cu130"
    assert summary["wheels"] == [
        "flashinfer_python-0.6.16-py3-none-any.whl",
        f"flashinfer_jit_cache_sm120f-{version}-cp39-abi3-win_arm64.whl",
        f"flashinfer_jit_cache_sm103a-{version}-cp39-abi3-win_arm64.whl",
        f"flashinfer_jit_cache-{version}-cp39-abi3-win_arm64.whl",
    ]
    assert summary["sccache"]["enabled"] is False
    wheelhouse = fake_repo / "wheelhouse"
    assert sorted(p.name for p in wheelhouse.iterdir()) == sorted(
        [*summary["wheels"], "build-summary.json"]
    )
    assert json.loads((wheelhouse / "build-summary.json").read_text()) == summary

    builds = [
        c for c in runner.calls if c["command"][-1].endswith("build_jit_cache_wheel.py")
    ]
    python_build, *provider_builds = builds
    assert python_build["cwd"] == fake_repo
    assert "FLASHINFER_LOCAL_VERSION" not in python_build["env"]
    for call, arch in zip(provider_builds, ["12.0f", "10.3a"], strict=True):
        assert call["cwd"] == fake_repo / "flashinfer-jit-cache-provider"
        assert call["env"]["FLASHINFER_JIT_CACHE_PROVIDER_ARCH"] == arch
        assert call["env"]["FLASHINFER_LOCAL_VERSION"] == "cu130"
        assert call["env"]["FLASHINFER_DEV_RELEASE_SUFFIX"] == "20261113"
        assert Path(call["env"]["PIP_CONSTRAINT"]).name == "torch-constraint.txt"
        expected_jobs = driver.compute_parallelism(256, os.cpu_count() or 1, {})
        assert call["env"]["MAX_JOBS"] == str(expected_jobs.max_jobs)
        # Linux-only settings never reach the Windows backends.
        assert "FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG" not in call["env"]
        assert "FLASHINFER_NVCC_LAUNCHER" not in call["env"]

    shim_build = next(c for c in runner.calls if c["command"][1:3] == ["-m", "build"])
    assert shim_build["env"]["FLASHINFER_JIT_CACHE_PROVIDER_ARCHS"] == "12.0f 10.3a"
    assert "PIP_CONSTRAINT" not in shim_build["env"]


@pytest.mark.cpu_only
def test_main_shim_only_needs_no_toolchain(driver, fake_repo):
    runner = FakeRunner()
    _run_main(
        driver, fake_repo, ["--target", "shim", "--provider-arch", "12.0f"], runner
    )
    commands = [" ".join(c) for c in runner.commands()]
    assert not any("nvcc" in c or "torch" in c for c in commands)


@pytest.mark.cpu_only
def test_main_rejects_wrongly_tagged_provider(driver, fake_repo):
    runner = FakeRunner(platform_tag="win_amd64")
    with pytest.raises(driver.BuildError, match="tagged cp39-abi3-win_amd64"):
        _run_main(
            driver,
            fake_repo,
            ["--target", "provider", "--provider-arch", "12.0f"],
            runner,
        )


@pytest.mark.cpu_only
def test_main_requires_provider_architectures(driver, fake_repo):
    with pytest.raises(SystemExit):
        _run_main(driver, fake_repo, ["--target", "provider"], FakeRunner())
