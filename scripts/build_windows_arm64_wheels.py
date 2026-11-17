#!/usr/bin/env python3
"""Build FlashInfer wheels and JIT-cache packages on native Windows arm64.

This is the Windows counterpart of the Linux release flow
(``scripts/jit_cache_build_common.sh`` with ``build_flashinfer_jit_cache_whl.sh``
and ``build_jit_cache_shim_wheel.sh``). The Linux flow runs in a manylinux
container and leans on POSIX tools; on a Windows arm64 host (a DGX Station or a
future Windows CI runner) the toolchain is MSVC plus a native CUDA toolkit, so
this driver does the same steps in Python:

========================================  =========================================
Linux (bash)                              This driver
========================================  =========================================
manylinux image, ``uname -m``             host check: ``sysconfig`` says win-arm64
(compiler preinstalled in the image)      MSVC: ``cl.exe`` on PATH, else vcvarsall
``validate_jit_cache_cuda_toolchain``     ``validate_cuda_toolchain``
``compute_jit_cache_parallelism``         ``compute_parallelism``
``setup_jit_cache_python_build``          ``python_build_environment``
``setup_sccache`` (musl tarball)          skipped; see ``SCCACHE_SKIP_REASON``
``run_with_output_watchdog.py``           not used: it reads ``/proc``
``build_jit_cache_wheel.py``              reused unchanged (python + provider)
``build_jit_cache_shim_wheel.sh``         ``build_shim`` (throwaway venv)
``verify_jit_cache_provider_artifact``    ``validate_provider`` with module.dll
``pip install`` + entry-point check       read from the wheel, nothing installed
``cp dist/*.whl $OUTPUT_DIR``             copy + ``build-summary.json``
========================================  =========================================

The provider wheels need the Windows build port (flashinfer-ai/flashinfer#8431):
it teaches the JIT to drive MSVC and names each module library ``module.dll``.
The driver itself only orchestrates and imports nothing from ``flashinfer``.

Example (from an "ARM64 Native Tools" prompt or any shell; vcvarsall is found
with vswhere when ``cl.exe`` is not on PATH)::

    python scripts/build_windows_arm64_wheels.py --cuda-label cu134 ^
        --target provider --target shim ^
        --provider-arch 10.3a --provider-arch 12.0f --output-dir wheelhouse
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

WINDOWS_ARM64_PLATFORM = "win-arm64"
WHEEL_PLATFORM_TAG = "win_arm64"
# Provider and shim wheels carry the stable ABI tag on every platform.
JIT_CACHE_WHEEL_TAGS = ("cp39", "abi3", WHEEL_PLATFORM_TAG)
TARGETS = ("python", "provider", "shim")
# CUDA 13.4 is the first toolkit with Windows on Arm support, so the older
# jit_cache labels in ci/cuda-versions.json cannot produce win_arm64 wheels.
WINDOWS_ARM64_CUDA_LABELS = ("cu134",)
# Outside every directory a build cleans (dist/ and the package build dirs).
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "wheelhouse-win_arm64"

# pip 26.2.1 fixed truncated downloads bypassing resume (see the Linux helper).
PIP_BOOTSTRAP_REQUIREMENTS = ("pip>=26.2.1", "build")
SHIM_BUILD_REQUIREMENTS = ("setuptools>=77", "wheel", "build")
PIP_NETWORK_DEFAULTS = {
    "PIP_RETRIES": "10",
    "PIP_RESUME_RETRIES": "10",
    "PIP_DEFAULT_TIMEOUT": "120",
}

# The Linux flow wraps nvcc and the host compiler with sccache through
# FLASHINFER_NVCC_LAUNCHER / FLASHINFER_CXX_LAUNCHER. mozilla/sccache publishes
# an aarch64-pc-windows-msvc build, but the Windows ninja rules of the build port
# call cl.exe and nvcc directly and ignore both launchers, so installing it would
# only add a download. Revisit once the Windows rules honour the launchers.
SCCACHE_SKIP_REASON = (
    "the Windows JIT ninja rules (flashinfer-ai/flashinfer#8431) do not use "
    "FLASHINFER_NVCC_LAUNCHER/FLASHINFER_CXX_LAUNCHER, so sccache would not be "
    "invoked"
)

# Variables the Linux flow sets that would break or mislabel a Windows build.
# The provider backend rejects a platform-tag override outside Linux glibc, and
# the sccache launchers are not honoured (see above).
ENV_TO_CLEAR = (
    "FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG",
    "FLASHINFER_NVCC_LAUNCHER",
    "FLASHINFER_CXX_LAUNCHER",
)

VSWHERE_DEFAULT = (
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    / "Microsoft Visual Studio"
    / "Installer"
    / "vswhere.exe"
)


class BuildError(RuntimeError):
    """A precondition or post-build check failed."""


Runner = Callable[..., subprocess.CompletedProcess]


def run(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Optional[Path] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``command`` and raise on failure; the default ``Runner``."""
    print(f"+ {subprocess.list2cmdline([str(part) for part in command])}", flush=True)
    return subprocess.run(
        [str(part) for part in command],
        env=dict(env),
        cwd=None if cwd is None else str(cwd),
        check=True,
        capture_output=capture,
        text=True,
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class CudaConfig:
    label: str
    version: str
    pytorch_index: str

    @property
    def pytorch_index_url(self) -> str:
        return f"https://download.pytorch.org/whl/{self.pytorch_index}"

    @property
    def nightly(self) -> bool:
        return self.pytorch_index.startswith("nightly/")


def load_cuda_config(label: str, config_path: Path) -> CudaConfig:
    """Read the ``jit_cache`` entry for ``label`` from ``ci/cuda-versions.json``."""
    entries = json.loads(config_path.read_text(encoding="utf-8"))["jit_cache"]
    for entry in entries:
        if entry["label"] == label:
            return CudaConfig(label, entry["version"], entry["pytorch_index"])
    labels = ", ".join(entry["label"] for entry in entries)
    raise BuildError(
        f"Unknown JIT-cache CUDA label {label!r}; expected one of: {labels}"
    )


def provider_tag(architecture: str) -> str:
    """``12.0f`` -> ``sm120f``; same spelling rules as the Linux scripts."""
    normalized = architecture.strip().lower()
    for prefix in ("compute_", "sm_", "sm"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    normalized = normalized.replace(".", "").replace("_", "")
    if not re.fullmatch(r"\d{2,3}[af]?", normalized):
        raise BuildError(f"Invalid provider architecture: {architecture!r}")
    return f"sm{normalized}"


def package_version(repo_root: Path, dev_suffix: str, local_version: str) -> str:
    version = (repo_root / "version.txt").read_text(encoding="utf-8").strip()
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"
    return f"{version}+{local_version}" if local_version else version


# --------------------------------------------------------------------------- #
# Host and toolchain
# --------------------------------------------------------------------------- #


def check_host(platform_name: Optional[str] = None) -> None:
    """Refuse to run unless the interpreter builds win-arm64 wheels.

    The wheel tag comes from the interpreter, so an x64 Python under emulation
    on an arm64 machine would produce win_amd64 wheels: check the interpreter's
    platform, not the machine's.
    """
    platform_name = platform_name or sysconfig.get_platform()
    if platform_name != WINDOWS_ARM64_PLATFORM:
        raise BuildError(
            f"This driver builds {WHEEL_PLATFORM_TAG} wheels and needs a native "
            f"Windows arm64 Python; this interpreter is {platform_name}"
        )


def find_vcvarsall(runner: Runner, env: Mapping[str, str], vswhere: Path) -> Path:
    """Locate vcvarsall.bat of the newest Visual Studio with the arm64 tools."""
    if not vswhere.is_file():
        raise BuildError(
            "cl.exe is not on PATH and vswhere.exe was not found at "
            f"{vswhere}; run from an 'ARM64 Native Tools' prompt or install "
            "Visual Studio Build Tools with the ARM64 C++ tools"
        )
    result = runner(
        [
            vswhere,
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.ARM64",
            "-property",
            "installationPath",
        ],
        env=env,
        capture=True,
    )
    installation = result.stdout.strip().splitlines()
    if not installation:
        raise BuildError("vswhere found no Visual Studio with the ARM64 C++ tools")
    vcvarsall = Path(installation[0]) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall.is_file():
        raise BuildError(f"vcvarsall.bat not found at {vcvarsall}")
    return vcvarsall


def parse_environment_dump(output: str) -> dict[str, str]:
    """Parse the output of ``cmd /c set`` into a mapping."""
    environment = {}
    for line in output.splitlines():
        name, separator, value = line.partition("=")
        # cmd lists per-drive working directories as "=C:=C:\\..."; skip them.
        if separator and name:
            environment[name] = value
    return environment


def msvc_environment(
    runner: Runner,
    env: Mapping[str, str],
    which: Callable[..., Optional[str]] = shutil.which,
    vswhere: Path = VSWHERE_DEFAULT,
) -> dict[str, str]:
    """Return ``env`` with the MSVC arm64 toolchain on PATH.

    The build port expects ``cl.exe`` and ``link.exe`` on PATH and does not
    locate Visual Studio itself. An environment that already has them (a
    developer prompt, a CI step that ran vcvarsall) is used as is.
    """
    path = env.get("PATH", "")
    if which("cl.exe", path=path) and which("link.exe", path=path):
        return dict(env)
    vcvarsall = find_vcvarsall(runner, env, vswhere)
    # A batch file sidesteps quoting: subprocess escapes embedded quotes with
    # backslashes, which cmd.exe does not understand.
    with tempfile.TemporaryDirectory(prefix="flashinfer-vcvars-") as temp_dir:
        script = Path(temp_dir) / "vcvars.bat"
        script.write_bytes(
            f'@call "{vcvarsall}" arm64 >nul || exit /b 1\r\n@set\r\n'.encode()
        )
        result = runner(["cmd.exe", "/d", "/c", script], env=env, capture=True)
    loaded = parse_environment_dump(result.stdout)
    new_path = loaded.get("PATH") or loaded.get("Path", "")
    if not which("cl.exe", path=new_path):
        raise BuildError(f"{vcvarsall} arm64 did not put cl.exe on PATH")
    return {**env, **loaded}


def nvcc_release(output: str) -> Optional[str]:
    match = re.search(r"release (\d+\.\d+)", output)
    return match.group(1) if match else None


def validate_cuda_toolchain(
    runner: Runner, env: Mapping[str, str], expected_version: str
) -> str:
    """Check that nvcc on PATH matches the CUDA version of the matrix entry."""
    result = runner(["nvcc", "--version"], env=env, capture=True)
    found = nvcc_release(result.stdout)
    if found != expected_version:
        raise BuildError(
            f"nvcc reports CUDA {found or 'unknown'}; expected {expected_version}"
        )
    print(f"nvcc CUDA version check passed: {found}")
    return found


def cuda_home(env: Mapping[str, str]) -> Optional[Path]:
    root = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    return Path(root) if root else None


# --------------------------------------------------------------------------- #
# Parallelism
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Parallelism:
    max_jobs: int
    nvcc_threads: int
    mem_per_job_gb: int


def _positive_int(value: Optional[str], default: int, name: str) -> int:
    if value is None or value == "":
        return default
    if re.fullmatch(r"[1-9]\d*", value):
        return int(value)
    print(f"Invalid {name}={value}; using {default}")
    return default


def compute_parallelism(
    mem_available_gb: int, cpu_count: int, env: Mapping[str, str]
) -> Parallelism:
    """Port of ``compute_jit_cache_parallelism`` with the same inputs and clamps."""
    nvcc_threads = _positive_int(
        env.get("FLASHINFER_NVCC_THREADS"), 1, "FLASHINFER_NVCC_THREADS"
    )
    nvcc_threads = max(1, min(nvcc_threads, 8, cpu_count))
    default_mem_per_job = max(8, nvcc_threads * 2)
    mem_per_job = _positive_int(
        env.get("AOT_MAX_JOBS_MEMORY_GB"), default_mem_per_job, "AOT_MAX_JOBS_MEMORY_GB"
    )
    cap_value = env.get("AOT_MAX_JOBS_CAP", "0")
    if re.fullmatch(r"\d+", cap_value):
        max_jobs_cap = int(cap_value)
    else:
        print(f"Invalid AOT_MAX_JOBS_CAP={cap_value}; disabling cap")
        max_jobs_cap = 0

    max_jobs = max(1, min(mem_available_gb // mem_per_job, cpu_count))
    if max_jobs_cap > 0:
        max_jobs = min(max_jobs, max_jobs_cap)
    if max_jobs * nvcc_threads > cpu_count:
        max_jobs = max(1, cpu_count // nvcc_threads)
    return Parallelism(max_jobs, nvcc_threads, mem_per_job)


def available_memory_gb() -> int:
    """Available physical memory in whole GiB (Windows API, else /proc)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return status.ullAvailPhys // 2**30
    with open("/proc/meminfo", encoding="ascii") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 2**20
    raise BuildError("Could not read available memory")


# --------------------------------------------------------------------------- #
# Python build environment
# --------------------------------------------------------------------------- #


TORCH_CUDA_PROBE = (
    "import sys\n"
    "try:\n"
    "    import torch\n"
    "except ImportError:\n"
    "    raise SystemExit(1)\n"
    "raise SystemExit(0 if torch.version.cuda == sys.argv[1] else 1)\n"
)
TORCH_VERSION_PROBE = (
    'import importlib.metadata as m; print("torch==" + m.version("torch"))'
)


@contextlib.contextmanager
def python_build_environment(
    runner: Runner,
    python: str,
    cuda: CudaConfig,
    env: Mapping[str, str],
) -> Iterator[dict[str, str]]:
    """Port of ``setup_jit_cache_python_build``.

    Installs or reuses a PyTorch build for ``cuda`` and yields an environment
    whose ``PIP_CONSTRAINT``/``PIP_BUILD_CONSTRAINT`` pin that exact torch, so
    no later install (including the wheel's build requirements) replaces it.
    The caller's environment is never modified, so no restore step is needed.
    """
    build_env = dict(env)
    for name, value in PIP_NETWORK_DEFAULTS.items():
        build_env.setdefault(name, value)

    bootstrap_env = {
        name: value
        for name, value in build_env.items()
        if name not in ("PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT")
    }
    runner(
        [python, "-m", "pip", "install", "--upgrade", *PIP_BOOTSTRAP_REQUIREMENTS],
        env=bootstrap_env,
    )

    try:
        runner([python, "-c", TORCH_CUDA_PROBE, cuda.version], env=build_env)
        print(f"Using preinstalled PyTorch for CUDA {cuda.version}")
    except subprocess.CalledProcessError:
        install = [python, "-m", "pip", "install", "--upgrade", "torch"]
        install += ["--index-url", cuda.pytorch_index_url]
        if cuda.nightly:
            install.insert(4, "--pre")
        runner(install, env=build_env)
        try:
            runner([python, "-c", TORCH_CUDA_PROBE, cuda.version], env=build_env)
        except subprocess.CalledProcessError:
            raise BuildError(
                f"No PyTorch for CUDA {cuda.version} from {cuda.pytorch_index_url} "
                "could be installed for this interpreter; preinstall a win_arm64 "
                "CUDA build of torch"
            ) from None

    pinned = runner(
        [python, "-c", TORCH_VERSION_PROBE], env=build_env, capture=True
    ).stdout.strip()
    print(f"PyTorch distribution: {pinned}")

    with tempfile.TemporaryDirectory(prefix="flashinfer-win-build-") as temp_dir:
        constraint = Path(temp_dir) / "torch-constraint.txt"
        constraint.write_text(pinned + "\n", encoding="utf-8")
        build_env["PIP_CONSTRAINT"] = str(constraint)
        build_env["PIP_BUILD_CONSTRAINT"] = str(constraint)
        extra = build_env.get("PIP_EXTRA_INDEX_URL", "").split()
        if cuda.pytorch_index_url not in extra:
            build_env["PIP_EXTRA_INDEX_URL"] = " ".join(
                [*extra, cuda.pytorch_index_url]
            )
        if cuda.nightly:
            build_env["PIP_PRE"] = "1"
        yield build_env


# --------------------------------------------------------------------------- #
# Builds
# --------------------------------------------------------------------------- #


def clean_package_dir(package_dir: Path, generated: Sequence[str]) -> None:
    """Remove outputs of a previous build, as the Linux scripts do."""
    for name in ("dist", "build", *generated):
        path = package_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for egg_info in package_dir.glob("*.egg-info"):
        shutil.rmtree(egg_info)


PROVIDER_GENERATED = (
    "flashinfer_jit_cache_provider/jit_cache",
    "flashinfer_jit_cache_provider/manifest.json",
    "flashinfer_jit_cache_provider/_build_meta.py",
)
SHIM_GENERATED = (
    "flashinfer_jit_cache/jit_cache",
    "flashinfer_jit_cache/_build_meta.py",
    "flashinfer_jit_cache/_provider_requirements.txt",
)


def single_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise BuildError(f"Expected one wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def wheel_tags(filename: str) -> tuple[str, str, str]:
    """Return the (python, abi, platform) tags of a wheel file name."""
    parts = filename.removesuffix(".whl").split("-")
    if not filename.endswith(".whl") or len(parts) not in (5, 6):
        raise BuildError(f"Not a wheel file name: {filename}")
    return parts[-3], parts[-2], parts[-1]


def check_wheel_tags(
    wheel: Path, allowed: Sequence[tuple[str, str, str]]
) -> tuple[str, str, str]:
    tags = wheel_tags(wheel.name)
    if tags not in allowed:
        expected = " or ".join("-".join(tag) for tag in allowed)
        raise BuildError(
            f"{wheel.name} is tagged {'-'.join(tags)}; expected {expected}"
        )
    return tags


def build_python(
    runner: Runner, python: str, env: Mapping[str, str], repo_root: Path
) -> Path:
    """Build the flashinfer-python wheel (the sdist comes from the Linux job)."""
    dist = repo_root / "dist"
    if dist.is_dir():
        shutil.rmtree(dist)
    # The release job builds flashinfer-python without a local version label;
    # build_backend.py would otherwise append +<cuda label> to its version.
    python_env = {k: v for k, v in env.items() if k != "FLASHINFER_LOCAL_VERSION"}
    runner(
        [python, SCRIPT_DIR / "build_jit_cache_wheel.py"], env=python_env, cwd=repo_root
    )
    wheel = single_wheel(dist)
    # flashinfer-python is pure Python on Linux; a Windows build that compiles
    # native extensions must still be tagged for this platform, never linux_*.
    tags = wheel_tags(wheel.name)
    if tags[2] not in ("any", WHEEL_PLATFORM_TAG):
        raise BuildError(f"{wheel.name} has platform tag {tags[2]}")
    return wheel


def build_provider(
    runner: Runner,
    python: str,
    env: Mapping[str, str],
    repo_root: Path,
    architecture: str,
    version: str,
    cuobjdump: Optional[Path],
    cuda_architecture_policy: str,
) -> Path:
    """Build and validate one ``flashinfer-jit-cache-sm*`` provider wheel."""
    package_dir = repo_root / "flashinfer-jit-cache-provider"
    clean_package_dir(package_dir, PROVIDER_GENERATED)
    provider_env = dict(env)
    provider_env["FLASHINFER_JIT_CACHE_PROVIDER_ARCH"] = architecture
    runner(
        [python, SCRIPT_DIR / "build_jit_cache_wheel.py"],
        env=provider_env,
        cwd=package_dir,
    )
    wheel = single_wheel(package_dir / "dist")
    check_wheel_tags(wheel, [JIT_CACHE_WHEEL_TAGS])
    validate_provider_wheel(wheel, provider_tag(architecture), version)
    if cuobjdump is None:
        print(
            "Skipping the cuobjdump architecture inventory: cuobjdump.exe not found "
            "under CUDA_HOME/CUDA_PATH"
        )
    else:
        runner(
            [
                python,
                SCRIPT_DIR / "verify_jit_cache_provider_artifact.py",
                "--artifact-dir",
                wheel.parent,
                "--provider",
                provider_tag(architecture),
                "--version",
                version,
                "--provider-platform-tag",
                WHEEL_PLATFORM_TAG,
                "--library-layout",
                "windows",
                "--cuobjdump",
                cuobjdump,
                "--cuda-architecture-policy",
                cuda_architecture_policy,
            ],
            env=env,
        )
    return wheel


def validate_provider_wheel(wheel: Path, provider: str, version: str) -> None:
    """Check name, version, tag, manifest, module.dll layout and entry point.

    Reads the wheel instead of installing it, so a persistent Windows runner's
    interpreter does not accumulate provider packages between builds.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import jit_cache_provider_validation as validation
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    try:
        manifest, modules = validation.validate_provider(
            validation.Wheel.open(wheel),
            provider,
            version,
            WHEEL_PLATFORM_TAG,
            library_name=validation.windows_library_name,
        )
    except ValueError as error:
        raise BuildError(str(error)) from None
    print(f"Validated {wheel.name}: {len(modules)} modules")


def build_shim(
    runner: Runner,
    python: str,
    env: Mapping[str, str],
    repo_root: Path,
    architectures: Sequence[str],
    version: str,
) -> Path:
    """Port of ``build_jit_cache_shim_wheel.sh``: metadata only, no torch."""
    package_dir = repo_root / "flashinfer-jit-cache"
    clean_package_dir(package_dir, SHIM_GENERATED)
    shim_env = dict(env)
    shim_env["FLASHINFER_JIT_CACHE_PROVIDER_ARCHS"] = " ".join(architectures)
    shim_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    for name in ("PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT"):
        shim_env.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="flashinfer-shim-venv-") as venv_root:
        venv = Path(venv_root) / "venv"
        runner([python, "-m", "venv", venv], env=shim_env)
        venv_python = venv / "Scripts" / "python.exe"
        if sys.platform != "win32":
            venv_python = venv / "bin" / "python"
        runner(
            [venv_python, "-m", "pip", "install", "--quiet", *SHIM_BUILD_REQUIREMENTS],
            env=shim_env,
        )
        runner(
            [
                venv_python,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                "dist",
                ".",
            ],
            env=shim_env,
            cwd=package_dir,
        )
    wheel = single_wheel(package_dir / "dist")
    check_wheel_tags(wheel, [JIT_CACHE_WHEEL_TAGS])
    validate_shim_wheel(wheel, version, {provider_tag(a) for a in architectures})
    return wheel


def validate_shim_wheel(wheel: Path, version: str, providers: set[str]) -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import jit_cache_provider_validation as validation
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    try:
        validation.validate_shim(
            validation.Wheel.open(wheel), version, providers, WHEEL_PLATFORM_TAG
        )
    except ValueError as error:
        raise BuildError(str(error)) from None
    print(f"Validated {wheel.name}: requires {', '.join(sorted(providers))}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        help="What to build; repeat for several (default: all, in this order: "
        "python, provider, shim)",
    )
    parser.add_argument(
        "--cuda-label",
        default=os.environ.get("FLASHINFER_LOCAL_VERSION", ""),
        help="jit_cache label in ci/cuda-versions.json; only cu134 supports "
        "Windows on Arm "
        "(default: $FLASHINFER_LOCAL_VERSION)",
    )
    parser.add_argument(
        "--provider-arch",
        action="append",
        default=[],
        help="CUDA architecture of a provider wheel, e.g. 12.0f; repeat for "
        "several. ci/cuda-versions.json has no Windows arm64 list yet.",
    )
    parser.add_argument(
        "--dev-suffix",
        default=os.environ.get("FLASHINFER_DEV_RELEASE_SUFFIX", ""),
        help="Nightly .devN suffix (default: $FLASHINFER_DEV_RELEASE_SUFFIX)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where the wheels and build-summary.json go (default: %(default)s)",
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Interpreter that runs the builds"
    )
    parser.add_argument(
        "--cuda-architecture-policy",
        choices=("strict", "report"),
        default=os.environ.get("CUDA_ARCHITECTURE_POLICY", "strict"),
    )
    parser.add_argument(
        "--skip-host-check",
        action="store_true",
        help="Do not require a win-arm64 interpreter (for testing the driver)",
    )
    args = parser.parse_args(argv)
    args.target = [t for t in TARGETS if t in (args.target or TARGETS)]
    if not args.cuda_label:
        parser.error("--cuda-label (or FLASHINFER_LOCAL_VERSION) is required")
    if {"provider", "shim"} & set(args.target) and not args.provider_arch:
        parser.error("provider and shim builds need at least one --provider-arch")
    return args


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    runner: Runner = run,
    environ: Optional[Mapping[str, str]] = None,
    repo_root: Path = REPO_ROOT,
    memory_gb: Callable[[], int] = available_memory_gb,
    which: Callable[..., Optional[str]] = shutil.which,
) -> dict:
    args = parse_args(argv)
    if not args.skip_host_check:
        check_host()
    if args.cuda_label not in WINDOWS_ARM64_CUDA_LABELS:
        raise BuildError(
            f"CUDA label {args.cuda_label!r} cannot target Windows arm64; CUDA 13.4 "
            f"is the first toolkit with Windows on Arm support (supported: "
            f"{', '.join(WINDOWS_ARM64_CUDA_LABELS)})"
        )
    cuda = load_cuda_config(args.cuda_label, repo_root / "ci" / "cuda-versions.json")
    version = package_version(repo_root, args.dev_suffix, cuda.label)
    for architecture in args.provider_arch:
        provider_tag(architecture)  # Fail on a bad spelling before any build.

    env = {k: v for k, v in (os.environ if environ is None else environ).items()}
    for name in ENV_TO_CLEAR:
        env.pop(name, None)
    env["FLASHINFER_LOCAL_VERSION"] = cuda.label
    env["FLASHINFER_DEV_RELEASE_SUFFIX"] = args.dev_suffix
    print(f"sccache: disabled ({SCCACHE_SKIP_REASON})")

    native = {"python", "provider"} & set(args.target)
    if native:
        env = msvc_environment(runner, env, which=which)
        validate_cuda_toolchain(runner, env, cuda.version)
        parallelism = compute_parallelism(memory_gb(), os.cpu_count() or 1, env)
        env["MAX_JOBS"] = str(parallelism.max_jobs)
        env["FLASHINFER_NVCC_THREADS"] = str(parallelism.nvcc_threads)
        print(
            f"MAX_JOBS={parallelism.max_jobs} "
            f"NVCC_THREADS={parallelism.nvcc_threads} "
            f"memory/job={parallelism.mem_per_job_gb} GB"
        )

    root = cuda_home(env)
    cuobjdump = root / "bin" / "cuobjdump.exe" if root else None
    if cuobjdump is not None and not cuobjdump.is_file():
        cuobjdump = None

    built: list[Path] = []
    if native:
        with python_build_environment(runner, args.python, cuda, env) as build_env:
            if "python" in args.target:
                wheel = build_python(runner, args.python, build_env, repo_root)
                built.append(stash(wheel, args.output_dir))
            if "provider" in args.target:
                for architecture in args.provider_arch:
                    wheel = build_provider(
                        runner,
                        args.python,
                        build_env,
                        repo_root,
                        architecture,
                        version,
                        cuobjdump,
                        args.cuda_architecture_policy,
                    )
                    built.append(stash(wheel, args.output_dir))
    if "shim" in args.target:
        wheel = build_shim(
            runner, args.python, env, repo_root, args.provider_arch, version
        )
        built.append(stash(wheel, args.output_dir))

    summary = {
        "platform_tag": WHEEL_PLATFORM_TAG,
        "cuda_label": cuda.label,
        "cuda_version": cuda.version,
        "version": version,
        "targets": args.target,
        "provider_architectures": [provider_tag(a) for a in args.provider_arch],
        "wheels": [wheel.name for wheel in built],
        "sccache": {"enabled": False, "reason": SCCACHE_SKIP_REASON},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Built wheels:")
    for wheel in built:
        print(f"  {wheel}")
    return summary


def stash(wheel: Path, output_dir: Path) -> Path:
    """Copy ``wheel`` to ``output_dir`` before the next build cleans ``dist``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / wheel.name
    shutil.copy2(wheel, destination)
    return destination


if __name__ == "__main__":
    try:
        main()
    except (BuildError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
