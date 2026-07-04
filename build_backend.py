"""
Copyright (c) 2023 by FlashInfer team.

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

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from setuptools import build_meta as orig
from build_utils import get_git_version

_root = Path(__file__).parent.resolve()
_data_dir = _root / "flashinfer" / "data"


# moe_ep build infra. Both EP backends are ON BY DEFAULT since the moe_ep
# runtime deps moved into the base dependencies (`pip install .` is enough):
#   NCCL-EP  — provided by the `nccl4py>=0.3.1` wheel (a base dep now); NO
#              in-tree build.
#   NIXL-EP  — built in-tree from 3rdparty/nixl (meson). Missing build deps
#              (meson/ninja/nvcc/UCX/...) skip the backend with a warning
#              instead of failing the install (best-effort).
#
# Env switches (tri-state; unset means "default on, best-effort"):
#   BUILD_NIXL_EP=0   → skip the NIXL-EP submodule build
#   BUILD_NIXL_EP=1   → strict: a missing build dep FAILS the install
#   BUILD_NCCL_EP=0/1 → same idea for NCCL-EP (no build step; only affects
#                       the informational logging)
#   BUILD_NVEP=0      → legacy alias: turns BOTH off
#   BUILD_NVEP=1      → legacy alias: both on, best-effort (back-compat)
def _flag(name: str) -> bool:
    v = os.environ.get(name, "")
    return v == "1" or v.lower() in ("true", "yes", "on")


def _tri_flag(name: str) -> bool | None:
    """Tri-state env flag: True / False when set, None when unset/empty."""
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    return v == "1" or v.lower() in ("true", "yes", "on")


@contextmanager
def _time_phase(label: str):
    """Emit a wall-clock duration line for a build phase.

    Docker buffers the entire `RUN` layer's output, so without these markers
    you can't tell from a finished build log how long each backend took. The
    lines are flushed explicitly so they survive a SIGKILL on the parent.
    """
    print(f"[BUILD_NVEP] {label}: start", flush=True)
    t0 = time.monotonic()
    try:
        yield
    finally:
        dt = time.monotonic() - t0
        print(f"[BUILD_NVEP] {label}: done in {dt:.1f}s", flush=True)


# Resolution order per backend: explicit BUILD_{NIXL,NCCL}_EP, then the
# legacy BUILD_NVEP alias, then the default (ON).
_BUILD_NVEP = _tri_flag("BUILD_NVEP")


def _backend_enabled(name: str) -> bool:
    explicit = _tri_flag(name)
    if explicit is not None:
        return explicit
    if _BUILD_NVEP is not None:
        return _BUILD_NVEP
    return True


_BUILD_NCCL_EP = _backend_enabled("BUILD_NCCL_EP")
_BUILD_NIXL_EP = _backend_enabled("BUILD_NIXL_EP")

# Missing build-time deps skip the backend with a warning instead of aborting
# the install — EXCEPT when the user explicitly asked for the NIXL-EP build
# with BUILD_NIXL_EP=1; then a missing dep is a hard error. Only NIXL-EP goes
# through _gate_backend (NCCL-EP has no build step), so strictness is keyed
# solely off the NIXL-EP flag — an explicit BUILD_NCCL_EP=1 must not force
# NIXL-EP into strict mode. The default-on install and the legacy
# BUILD_NVEP=1 alias are both best-effort.
_BUILD_NVEP_BEST_EFFORT = _tri_flag("BUILD_NIXL_EP") is not True

_nvep_build_root = _root / "build_nvep"
_moe_ep_pkg = _root / "flashinfer" / "moe_ep"


def _in_isolated_build_env() -> bool:
    """Heuristic: are we running inside a PEP 517 isolated build env?

    pip's isolated build envs live in a ``pip-build-env-*`` temp dir injected
    on sys.path; uv's ephemeral build envs live under a ``builds-v0`` cache
    dir. In such an env, wheels installed by this hook (nixl-cu13) vanish
    when the build finishes and never reach the user's target environment —
    and the env usually has no ``pip`` module at all, so the installs fail
    outright. The moe_ep build path therefore needs --no-build-isolation.
    """
    markers = ("pip-build-env-", f"{os.sep}builds-v0{os.sep}")
    paths = [sys.prefix, *sys.path]
    return any(m in p for m in markers for p in paths)


def _detect_cuda_major() -> int:
    """Best-effort detection of the CUDA major version on the host."""
    try:
        out = subprocess.check_output(["nvcc", "--version"]).decode()
        for line in out.splitlines():
            if "release" in line:
                # e.g. "Cuda compilation tools, release 13.0, V13.0.48"
                token = line.split("release", 1)[1].split(",", 1)[0].strip()
                return int(token.split(".")[0])
    except Exception:
        pass
    return 13  # default — pyproject's nvep extras pin cu13 packages


def _apply_patches(submodule_dir: Path, patches_dir: Path) -> None:
    """Apply every *.patch in patches_dir to the submodule working tree.

    No-ops if patches_dir doesn't exist. Idempotent: skips patches that are
    already applied by checking `git apply --reverse --check` first.
    """
    if not patches_dir.is_dir():
        return
    for patch in sorted(patches_dir.glob("*.patch")):
        # Already applied?
        already = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=submodule_dir,
            capture_output=True,
        )
        if already.returncode == 0:
            print(f"[BUILD_NVEP] patch already applied, skipping: {patch.name}")
            continue
        # Check we *can* apply, then apply.
        subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=submodule_dir,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(patch)],
            cwd=submodule_dir,
            check=True,
        )
        print(f"[BUILD_NVEP] applied patch: {patch.name}")


def _find_nixl_wheel_lib_dir() -> Path | None:
    """Locate the nixl-cu* pip wheel's libnixl.so directory.

    Layout of the current nixl-cu13 wheel (meson-python packaging):
        <site-packages>/nixl_cu13/                  — importable python module
        <site-packages>/.nixl_cu13.mesonpy.libs/    — libnixl.so + sibling libs
        <site-packages>/nixl_cu13.libs/             — auditwheel deps + plugins

    We probe by importing `nixl_cu13` / `nixl_cu12` / `nixl` (legacy), then
    look for `.{name}.mesonpy.libs/libnixl.so` next to it.

    `<machine>-linux-gnu` follows the Debian multiarch convention used by
    auditwheel-tagged wheels — `x86_64-linux-gnu` on x86_64, and
    `aarch64-linux-gnu` on ARM64 (e.g. NVIDIA Grace / AWS Graviton hosts).
    """
    import platform

    multiarch = f"{platform.machine()}-linux-gnu"
    for pkg_name in ("nixl_cu13", "nixl_cu12", "nixl"):
        try:
            mod = __import__(pkg_name)
        except ImportError:
            continue
        try:
            pkg_root = Path(mod.__path__[0])
        except Exception:
            continue
        site_packages = pkg_root.parent
        # Prefer the meson-python sibling layout.
        for candidate in (
            site_packages / f".{pkg_name}.mesonpy.libs",
            pkg_root / "lib" / multiarch,
            pkg_root / "lib",
            pkg_root,
        ):
            if (candidate / "libnixl.so").exists():
                return candidate
    # Last resort: a glob-walk of site-packages for any .nixl_*.mesonpy.libs/.
    try:
        sp = Path(__import__("site").getsitepackages()[0])  # type: ignore[no-untyped-call]
    except Exception:
        return None
    for candidate in sp.glob(".nixl*.mesonpy.libs"):
        if (candidate / "libnixl.so").exists():
            return candidate
    return None


def _build_nixl_ep() -> None:
    src = _root / "3rdparty" / "nixl"
    build = _nvep_build_root / "nixl"
    prefix = _nvep_build_root / "nixl_install"
    _apply_patches(src, _root / "3rdparty_patches" / "nixl")

    # Default path: skip the parent libnixl build and link the EP example
    # against the libnixl.so shipped by the nixl-cu13 pip wheel — mirrors the
    # contrib/nccl_ep wheel-driven path (Section 9 of the integration plan).
    # The hermetic env var falls back to the full parent build for hosts
    # without the wheel pre-installed.
    hermetic = _flag("BUILD_NIXL_EP_HERMETIC")
    setup_args = [
        "meson",
        "setup",
        str(build),
        str(src),
        "-Dbuild_nixl_ep=true",
        f"-Dprefix={prefix}",
        "--buildtype=release",
    ]
    if hermetic:
        print("[BUILD_NVEP] BUILD_NIXL_EP_HERMETIC=1 — building full NIXL tree")
        setup_args.append("-Dbuild_examples=true")
    else:
        wheel_lib_dir = _find_nixl_wheel_lib_dir()
        if wheel_lib_dir is None:
            raise RuntimeError(
                "The NIXL-EP build requires the nixl-cu13 wheel (the build "
                "hook normally pre-installs it; see _ensure_nixl_wheel).\n"
                "Run: uv pip install --no-deps 'nixl-cu13>=1.0.1'\n"
                "Or set BUILD_NIXL_EP_HERMETIC=1 to build the full NIXL tree."
            )
        setup_args += [
            "-Dbuild_examples=false",
            "-Dnixl_ep_only=true",
            f"-Dnixl_wheel_lib_dir={wheel_lib_dir}",
        ]
        print(
            f"[BUILD_NVEP] nixl_ep_only=true; linking against wheel libnixl.so "
            f"at {wheel_lib_dir}. Set BUILD_NIXL_EP_HERMETIC=1 to opt out."
        )

    # Re-run meson setup with --reconfigure when the build dir already
    # exists, so patch / option changes (e.g. a different wheel path,
    # flipping HERMETIC mode) take effect on subsequent installs without
    # requiring the user to `rm -rf build_nvep/nixl` manually.
    if build.exists():
        setup_args.append("--reconfigure")
    subprocess.run(setup_args, check=True)

    # `install` only makes sense in hermetic mode (it populates `prefix/` with
    # libnixl + headers + plugins). In ep_only mode there's nothing to install
    # — we just compile and pluck nixl_ep_cpp.so out of the build tree.
    ninja_cmd = ["ninja", "-C", str(build)]
    if hermetic:
        ninja_cmd.append("install")
    subprocess.run(ninja_cmd, check=True)

    dst = _moe_ep_pkg / "nixl_ep" / "_libs"
    dst.mkdir(parents=True, exist_ok=True)

    # We do NOT stage the base NIXL libraries (libnixl.so, libnixl_capi.so,
    # libserdes.so, etc.) — they come from the `nixl-cu13` pip wheel installed
    # by _install_nvep_runtime_wheels(). The runtime loader in
    # flashinfer/moe_ep/nixl_ep/__init__.py ctypes-preloads them via the wheel's
    # site-packages path before loading nixl_ep_cpp.so. This keeps the
    # FlashInfer wheel small.

    # The torch extension lands either in build/ or build/examples/device/ep/
    for cand in (build / "examples/device/ep").glob("nixl_ep_cpp*.so"):
        shutil.copy(cand, dst / cand.name)
        print(f"[BUILD_NVEP] staged: {cand.name}")

    # Vendor the python wrapper sources so we can import from
    # flashinfer.moe_ep.nixl_ep._vendored (Step B5).
    vendored_src = src / "examples/device/ep/nixl_ep"
    if vendored_src.exists():
        shutil.copytree(
            vendored_src,
            _moe_ep_pkg / "nixl_ep" / "_vendored",
            dirs_exist_ok=True,
        )


def _find_nccl_wheel_root() -> Path | None:
    """Locate the nvidia-nccl-cu13 pip wheel's nvidia/nccl/ directory.

    Returns the resolved path or None if the wheel isn't installed in the
    Python environment used by this build hook.
    """
    try:
        import nvidia.nccl  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return Path(nvidia.nccl.__path__[0])
    except Exception:
        return None


def _fix_rpaths() -> None:
    """Rewrite RPATHs on staged .so files so they find siblings without LD_LIBRARY_PATH.

    Since we no longer stage the base libs (libnccl.so.2, libnixl.so) inside
    the package, the RPATH only needs to cover $ORIGIN and $ORIGIN/_libs for
    co-located plugin files. The base libs are loaded explicitly at Python
    import time via the runtime preloaders in
    flashinfer/moe_ep/{nccl,nixl}_ep/__init__.py.
    """
    patchelf_ok = shutil.which("patchelf") is not None
    if not patchelf_ok:
        print("[BUILD_NVEP] patchelf not found; skipping RPATH fix-up")
        return
    rpath = "$ORIGIN:$ORIGIN/_libs"
    for so in _moe_ep_pkg.rglob("*.so*"):
        # Skip symlinks
        if so.is_symlink():
            continue
        # Use check=False because patchelf legitimately exits nonzero on
        # files that already have the desired RPATH or that aren't ELFs
        # we care about. Surface anything else as a warning so a real
        # failure (e.g. binary lacks .dynamic section) isn't silently lost.
        r = subprocess.run(
            ["patchelf", "--set-rpath", rpath, str(so)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()
            print(f"[BUILD_NVEP] WARNING: patchelf failed on {so.name}: {err}")


def _ensure_nixl_wheel() -> None:
    """Pre-install the nixl-cu* wheel the NIXL-EP build links against.

    The default (non-hermetic) NIXL-EP build links nixl_ep_cpp.so against the
    libnixl.so shipped by the `nixl-cu13` pip wheel. Since the EP build now
    runs by default on `pip install .`, install that wheel up front instead of
    requiring users to pre-install it. `--no-deps` for the same reason as
    _install_nvep_runtime_wheels: the wheel's transitive constraints downgrade
    torch. Best-effort: on failure the _nixl_buildable probe reports the
    missing wheel and the backend is gated as usual (skip or hard error).
    """
    if _find_nixl_wheel_lib_dir() is not None:
        return
    cuda_major = _detect_cuda_major()
    wheel = f"nixl-cu{cuda_major}>=1.0.1"
    print(f"[BUILD_NVEP] pre-installing NIXL wheel --no-deps: {wheel}")

    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [uv_bin, "pip", "install", "--python", sys.executable, "--no-deps", wheel]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", wheel]
    print(f"[BUILD_NVEP] $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"[BUILD_NVEP] WARNING: could not pre-install {wheel} ({e}); "
            "the NIXL-EP pre-flight probe will decide whether to skip or fail."
        )


def _ensure_nccl_floor() -> None:
    """Best-effort upgrade of nvidia-nccl-cu13 to the B200 EP floor (>=2.30.7).

    Deliberately NOT a base dependency: torch's cu13 wheels pin
    nvidia-nccl-cu13 EXACTLY (e.g. ==2.29.7), so declaring a >=2.30.7 floor
    in package metadata makes pip's resolver evict torch — on aarch64 it
    backtracks to the CPU-only torch wheel. Installing here with --no-deps
    (mirroring the nixl-cu13 pattern) upgrades the wheel without ever
    entering the resolver. Failures only warn: torch's own NCCL is
    sufficient everywhere except NCCL-EP group-create on B200, and
    moe_ep/_validators.py enforces the floor at runtime with an actionable
    error where it actually matters.
    """
    cuda_major = _detect_cuda_major()
    if cuda_major < 13:
        return  # EP is CUDA-13-only; nothing to upgrade on cu12 hosts.
    wheel = "nvidia-nccl-cu13>=2.30.7"
    print(f"[BUILD_NVEP] ensuring NCCL-EP floor --no-deps: {wheel}")

    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [uv_bin, "pip", "install", "--python", sys.executable, "--no-deps", wheel]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", wheel]
    print(f"[BUILD_NVEP] $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"[BUILD_NVEP] WARNING: could not install {wheel} ({e}). "
            "NCCL-EP on B200 needs NCCL >= 2.30.7 (group-create fails on "
            "older releases); the runtime validator will raise there. "
            "Install manually if needed: pip install --no-deps "
            f"'{wheel}'"
        )


def _nixl_buildable() -> tuple[bool, str]:
    """Probe for hard NIXL-EP build-time deps. Returns (ok, reason_if_not).

    In the default (wheel-driven) flow, the EP example links against the
    nixl-cu13 pip wheel — so it must be importable. In hermetic mode
    (BUILD_NIXL_EP_HERMETIC=1), we build libnixl from source and the wheel
    isn't required.
    """
    if not shutil.which("meson"):
        return False, "meson not on PATH (apt install meson)"
    if not shutil.which("ninja"):
        return False, "ninja not on PATH (apt install ninja-build)"
    if not shutil.which("nvcc"):
        return False, (
            "nvcc not on PATH (install CUDA toolkit and put "
            "/usr/local/cuda/bin on $PATH); needed for nixl_ep CUDA kernels"
        )
    if not shutil.which("git"):
        return False, "git not on PATH; needed for `git apply` of patch overlays"
    pkgconfig = shutil.which("pkg-config")
    if not pkgconfig:
        return False, "pkg-config not on PATH (apt install pkg-config)"
    r = subprocess.run([pkgconfig, "--exists", "ucx"], capture_output=True)
    if r.returncode:
        return False, (
            "UCX not found via pkg-config (no ucx.pc); "
            "build UCX from source or set PKG_CONFIG_PATH"
        )
    # libibverbs is needed by NIXL's UCX + EP transports.
    r = subprocess.run([pkgconfig, "--exists", "libibverbs"], capture_output=True)
    if r.returncode:
        return False, "libibverbs not found via pkg-config (apt install libibverbs-dev)"
    if not _flag("BUILD_NIXL_EP_HERMETIC"):
        if _find_nixl_wheel_lib_dir() is None:
            return False, (
                "nixl pip wheel not importable (or libnixl.so missing); install with "
                "`uv pip install --no-deps 'nixl-cu13>=1.0.1'` "
                "or set BUILD_NIXL_EP_HERMETIC=1 to build the full NIXL tree"
            )
    return True, ""


def _install_nvep_runtime_wheels(built_nixl: bool) -> None:
    """Install the NIXL runtime wheel with --no-deps when NIXL-EP was built.

    The wheel supplies the BASE libraries (libnixl.so + siblings) that the
    nixl_ep_cpp.so plugin dynamically loads at runtime. We do NOT stage the
    base libs into the FlashInfer package tree — relying on the pip wheel
    keeps the wheel small and avoids the duplication. (NCCL-EP's libnccl
    comes from torch's own nvidia-nccl-cu13 pin; see _ensure_nccl_floor.)

    The wheel carries transitive constraints (e.g. an nvidia-nccl-cu12 pin via
    the `nixl` meta-package) that conflict with a recent torch and force a
    downgrade when resolved normally. SGLang's Dockerfile mirrors this with
    `pip install nixl nixl-cu13 --no-deps`; we do the same.

    Two pip backends are tried, in order:
      1. `uv pip install` — works in venvs created by `uv venv` (which have
         no pip module). This is the path most users hit.
      2. `python -m pip install` — for venvs with pip seeded.

    Gated on what was ACTUALLY built (not what was requested), so `pip list`
    stays honest when the backend was skipped due to missing build-time deps
    in best-effort mode.

    This step is FATAL on failure. Since we no longer stage the base libs, a
    half-installed env where the wheel failed to install would leave the EP
    plugin unable to load at runtime. Better to fail loudly at install time.
    """
    cuda_major = _detect_cuda_major()
    wheels: list[str] = []
    if built_nixl:
        wheels.append(f"nixl-cu{cuda_major}>=1.0.1")
    if not wheels:
        return

    print(f"[BUILD_NVEP] installing runtime wheels --no-deps: {' '.join(wheels)}")

    # Prefer uv (works in a uv-created venv that has no pip module).
    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [
            uv_bin,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            *wheels,
        ]
        print(f"[BUILD_NVEP] $ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return

    # Fall back to python -m pip (requires pip in the venv).
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", *wheels]
    print(f"[BUILD_NVEP] $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Failed to install moe_ep runtime wheels and `uv` is not on "
            "PATH. Either install uv (https://docs.astral.sh/uv/) so the "
            "build hook can use `uv pip install`, or `--seed` your venv so "
            "it has a pip module. The wheels we tried to install: "
            f"{wheels}"
        ) from e


def _install_cuda_tile_compile_deps() -> None:
    """Install cuda-tile's compile chain with ``--no-deps`` to dodge libcudart.so.13.

    Background: ``cuda-tile[tileiras]>=1.4.0`` transitively pulls
    ``cuda-toolkit[nvcc,nvvm,tileiras]<13.4,>=13.2``, which in turn pulls
    ``nvidia-cuda-runtime==13.3.*`` (libcudart.so.13). That conflicts at
    *cudnn import* time with ``nvidia-cuda-runtime-cu12`` (libcudart.so.12)
    shipped by torch — cudnn's helper raises ``Multiple libcudart libraries
    found: libcudart.so.12 and libcudart.so.13`` and every cudnn_decode /
    cudnn_prefill test in the suite fails.

    To work around this we drop the ``[tileiras]`` extra from
    ``requirements.txt`` and instead install the compile-side wheels here
    with ``--no-deps``. The chain (nvcc + tileiras + nvvm + nvjitlink + crt)
    doesn't need libcudart at *compile* time — nvcc emits PTX/cubin; cubins
    run against whatever libcudart torch ships (cu12) at *test* time. So
    skipping the transitive ``nvidia-cuda-runtime`` (cu13) install is safe.

    Mirrors ``_install_nvep_runtime_wheels`` for the uv-first / pip-fallback
    path.

    Best-effort — PEP 517 build isolation limits what we can install here.
    When ``pip install`` runs without ``--no-build-isolation`` (the default),
    pip creates a **clean isolated build environment** that contains only the
    declared build dependencies (such as ``setuptools`` and ``packaging``) and
    does *not* include ``pip`` itself or ``uv``.  As a result, we cannot invoke
    ``pip install`` from within that environment to resolve the
    ``nvidia-cuda-runtime`` version conflict described above.

    In such isolated builds (e.g. the AOT Build Import workflow) the compile
    chain is already present on flashinfer-ci images, so we *warn and continue*
    instead of blocking the build.  A clean PyPI install on a system that lacks
    both ``uv`` and the compile chain will surface a clear ``ImportError`` the
    first time the user calls a cuTile kernel — a better failure mode than
    aborting the install entirely.
    """
    wheels = [
        "nvidia-cuda-nvcc<13.4,>=13.2",
        "nvidia-cuda-tileiras<13.4,>=13.2",
        "nvidia-nvvm<13.4,>=13.2",
        "nvidia-nvjitlink<14,>=13.3",
        "nvidia-cuda-crt<13.4,>=13.2",
    ]
    print(f"[BUILD] cuda-tile compile deps (--no-deps): {' '.join(wheels)}", flush=True)

    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [
            uv_bin,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            *wheels,
        ]
        print(f"[BUILD] $ {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"[BUILD] WARNING: uv pip install of cuda-tile compile deps "
                f"failed ({e}); continuing — wheels may already be present.",
                flush=True,
            )
        return

    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", *wheels]
    print(f"[BUILD] $ {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # PEP 517 isolated build envs (the default when ``pip install -e .`` is
        # invoked without ``--no-build-isolation``) have no ``pip`` module and
        # no ``uv``. Don't block the build — log and continue. The compile
        # chain is preinstalled in flashinfer-ci images, and a clean PyPI
        # install would surface a clear ImportError at first cuTile use.
        print(
            f"[BUILD] WARNING: could not install cuda-tile compile deps "
            f"(no `uv` on PATH and no `pip` module in this venv: {e}). "
            f"Skipping — install these wheels manually if cuTile JIT compile "
            f"fails at runtime: {wheels}",
            flush=True,
        )


def _gate_backend(name: str, requested: bool, probe) -> bool:
    """Decide whether to actually build `name` given its requested flag.

    Returns True if the backend should be built, False if it should be
    skipped. Raises RuntimeError if the user explicitly asked for this
    backend (not via the legacy BUILD_NVEP=1 alias) and a build-time dep
    is missing.
    """
    if not requested:
        return False
    ok, reason = probe()
    if ok:
        return True
    msg = f"[BUILD_NVEP] {name}: build skipped — {reason}"
    if _BUILD_NVEP_BEST_EFFORT:
        print(msg)
        return False
    # User opted in explicitly (BUILD_NCCL_EP=1 or BUILD_NIXL_EP=1) — fail hard.
    raise RuntimeError(
        f"{name} build requested but a hard dep is missing: {reason}. "
        "Either install the missing dependency, unset the BUILD_*_EP flag "
        "(the default build is best-effort and skips this backend with a "
        "warning), or set it to 0 to skip the backend entirely."
    )


def _build_nvep_if_enabled() -> None:
    if not (_BUILD_NCCL_EP or _BUILD_NIXL_EP):
        return

    requested = [
        b
        for b, is_enabled in (("NIXL-EP", _BUILD_NIXL_EP), ("NCCL-EP", _BUILD_NCCL_EP))
        if is_enabled
    ]
    mode = "best-effort" if _BUILD_NVEP_BEST_EFFORT else "strict"
    print(f"[BUILD_NVEP] requested: {', '.join(requested)} (mode: {mode})")

    if _BUILD_NIXL_EP and _in_isolated_build_env():
        print(
            "[BUILD_NVEP] WARNING: PEP 517 build isolation detected. Wheels "
            "installed by this hook (nixl-cu13) land in the throwaway build "
            "env — the NIXL-EP build will most likely be skipped, and even "
            "if it succeeds its runtime wheel will NOT persist into the "
            "target environment. To enable NIXL-EP when installing from "
            "source, disable isolation:\n"
            "    pip install --no-build-isolation .\n"
            "If NIXL-EP libs were still staged, install the runtime wheel "
            "manually afterwards: pip install --no-deps 'nixl-cu13>=1.0.1'.",
            flush=True,
        )

    # NCCL-EP is not built from source — it is provided by the released
    # `nccl4py` wheel (>=0.3.1, the `nccl.ep` API + bundled libnccl_ep.so),
    # which is a base dependency now. So BUILD_NCCL_EP requires no in-tree
    # build step; we only note it here.
    if _BUILD_NCCL_EP:
        print(
            "[BUILD_NVEP] NCCL-EP is provided by the nccl4py wheel (>=0.3.1), "
            "a base dependency of flashinfer-python; no in-tree build."
        )
        # torch's cu13 wheels pin nvidia-nccl-cu13 exactly (< the B200 EP
        # floor), so upgrade it out-of-band; best-effort by design.
        _ensure_nccl_floor()

    # The default (non-hermetic) NIXL-EP build links against the nixl-cu13
    # wheel's libnixl.so — install it up front so plain `pip install .` works
    # without a manual pre-install step.
    if _BUILD_NIXL_EP and not _flag("BUILD_NIXL_EP_HERMETIC"):
        _ensure_nixl_wheel()

    # Pre-flight gating — probe the NIXL-EP build-time deps (NCCL-EP needs none).
    will_build_nixl = _gate_backend("NIXL-EP", _BUILD_NIXL_EP, _nixl_buildable)

    if not will_build_nixl:
        print("[BUILD_NVEP] no submodule backend to build after pre-flight; skipping")
        return

    # Make sure each backend's submodule is initialized. Only fetch what we
    # actually need — saves ~300MB and a network round-trip per backend.
    # Guarded on `.git` because an sdist install has no git metadata: the
    # submodule trees travel inside the sdist as plain directories and
    # `git submodule update` would fail with "not a git repository".
    in_git_repo = (_root / ".git").exists()
    if will_build_nixl and not (_root / "3rdparty/nixl/meson.build").exists():
        if not in_git_repo:
            raise RuntimeError(
                "3rdparty/nixl/meson.build is missing and this is not a git "
                "checkout (likely an sdist install where the submodule wasn't "
                "packaged). Either install from a git clone or fetch the "
                "submodule tree manually into 3rdparty/nixl."
            )
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive", "3rdparty/nixl"],
            cwd=_root,
            check=True,
        )

    # Actual builds. If best-effort and the build raises despite the probe
    # passing, swallow the error so the other backend still has a chance.
    overall_t0 = time.monotonic()
    built_nixl = False
    if will_build_nixl:
        try:
            with _time_phase("_build_nixl_ep"):
                _build_nixl_ep()
            built_nixl = True
        except Exception as e:
            if not _BUILD_NVEP_BEST_EFFORT:
                raise
            print(f"[BUILD_NVEP] NIXL-EP build failed in best-effort mode: {e}")

    if built_nixl:
        with _time_phase("_fix_rpaths"):
            _fix_rpaths()
        with _time_phase("_install_nvep_runtime_wheels"):
            _install_nvep_runtime_wheels(built_nixl=built_nixl)

    print(
        f"[BUILD_NVEP] total build phase wall time: "
        f"{time.monotonic() - overall_t0:.1f}s",
        flush=True,
    )

    built = [b for b, is_enabled in (("NIXL-EP", built_nixl),) if is_enabled]
    if _BUILD_NCCL_EP:
        built.append("NCCL-EP (via nccl4py wheel)")
    print(f"[BUILD_NVEP] done — built: {', '.join(built) if built else 'nothing'}")


def _create_build_metadata():
    """Create build metadata file with version information."""
    version_file = _root / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0+unknown"

    # Add dev suffix if specified
    dev_suffix = os.environ.get("FLASHINFER_DEV_RELEASE_SUFFIX", "")
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    # Get git version
    git_version = get_git_version(cwd=_root)

    # Append local version suffix if available
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")
    if local_version:
        # Use + to create a local version identifier that will appear in wheel name
        version = f"{version}+{local_version}"

    # Create build metadata in the source tree
    package_dir = Path(__file__).parent / "flashinfer"
    build_meta_file = package_dir / "_build_meta.py"

    # Check if we're in a git repository
    git_dir = Path(__file__).parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _create_data_dir(use_symlinks=True):
    _data_dir.mkdir(parents=True, exist_ok=True)

    def ln(source: str, target: str) -> None:
        src = _root / source
        dst = _data_dir / target
        if dst.exists():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if use_symlinks:
            dst.symlink_to(src, target_is_directory=True)
        else:
            # For wheel/sdist, copy actual files instead of symlinks
            if src.exists():
                shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)

    ln("3rdparty/cutlass", "cutlass")
    ln("3rdparty/spdlog", "spdlog")
    ln("3rdparty/cccl", "cccl")
    ln("csrc", "csrc")
    ln("include", "include")


def _prepare_for_wheel():
    # For wheel, copy actual files instead of symlinks so they are included in the wheel
    _install_cuda_tile_compile_deps()
    _build_nvep_if_enabled()
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=False)

    # Copy license files from licenses/ to root to avoid nested path in wheel
    licenses_dir = _root / "licenses"
    if licenses_dir.exists():
        for license_file in licenses_dir.glob("*.txt"):
            shutil.copy2(
                license_file,
                _root / f"LICENSE.{license_file.stem.removeprefix('LICENSE.')}.txt",
            )


def _prepare_for_editable():
    # For editable install, use symlinks so changes are reflected immediately
    _install_cuda_tile_compile_deps()
    _build_nvep_if_enabled()
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=True)


def _prepare_for_sdist():
    # For sdist, copy actual files instead of symlinks so they are included in the tarball
    # NOTE: do NOT build moe_ep here — submodules + patches travel in the sdist
    # itself and get built during the *install* of the sdist.
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=False)


def get_requires_for_build_wheel(config_settings=None):
    _prepare_for_wheel()
    return []


def get_requires_for_build_sdist(config_settings=None):
    _prepare_for_sdist()
    return []


def get_requires_for_build_editable(config_settings=None):
    _prepare_for_editable()
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _prepare_for_wheel()
    return orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _prepare_for_editable()
    return orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_editable()
    return orig.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _prepare_for_sdist()
    return orig.build_sdist(sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_wheel()
    return orig.build_wheel(wheel_directory, config_settings, metadata_directory)
