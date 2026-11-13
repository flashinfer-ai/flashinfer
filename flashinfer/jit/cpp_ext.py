# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import functools
import logging
import os
import platform
import re
import subprocess
import sys
import sysconfig
import threading
import uuid
from packaging.version import Version
from pathlib import Path
from typing import List, Mapping, Optional

import tvm_ffi
import torch

from . import env as jit_env
from .spdlog_patch import SPDLOG_FORMAT_HEADER, patch_bundled_format_h
from ..compilation_context import CompilationContext

is_windows = platform.system() == "Windows"
logger = logging.getLogger(__name__)

_CCCL_TCGEN05_HEADER = Path("cuda/__ptx/instructions/generated/tcgen05_ld.h")
_EXPECTED_CCCL_OUT_TOKENS = 6736


def parse_env_flags(env_var_name) -> List[str]:
    env_flags = os.environ.get(env_var_name)
    if env_flags:
        try:
            import shlex

            return shlex.split(env_flags)
        except ValueError as e:
            logger.warning(
                "Could not parse %s with shlex: %s. Falling back to simple split.",
                env_var_name,
                e,
            )
            return env_flags.split()
    return []


def _get_glibcxx_abi_build_flags() -> List[str]:
    if is_windows:
        return []
    glibcxx_abi_cflags = [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]
    return glibcxx_abi_cflags


@functools.cache
def get_cuda_path() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home
    # get output of "which nvcc"
    nvcc_path = subprocess.run(["which", "nvcc"], capture_output=True)
    if nvcc_path.returncode == 0:
        cuda_home = os.path.dirname(
            os.path.dirname(nvcc_path.stdout.decode("utf-8").strip())
        )
    else:
        cuda_home = "/usr/local/cuda"  # This default value is from: https://github.com/pytorch/pytorch/blob/ceb11a584d6b3fdc600358577d9bf2644f88def9/torch/utils/cpp_extension.py#L115
        if not os.path.exists(cuda_home):
            raise RuntimeError(
                f"Could not find nvcc and default {cuda_home=} doesn't exist"
            )
    return cuda_home


@functools.cache
def get_cuda_version() -> Version:
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        cuda_home = get_cuda_path()
        nvcc = os.path.join(cuda_home, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        if not matches:
            raise RuntimeError(
                f"Could not parse CUDA version from nvcc --version output: {txt}"
            )
        return Version(matches[0])
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        # NOTE(Zihao): when nvcc is unavailable, fall back to torch.version.cuda
        if torch.version.cuda is None:
            raise RuntimeError(
                "nvcc not found and PyTorch is not built with CUDA support. "
                "Could not determine CUDA version."
            ) from e
        return Version(torch.version.cuda)


def is_cuda_version_at_least(version_str: str) -> bool:
    return get_cuda_version() >= Version(version_str)


@functools.cache
def get_cuda_include_overlay(cuda_home: str) -> Optional[Path]:
    """Create a Windows CUDA 13 CUtensorMap alignment overlay when needed."""
    if not is_windows or get_cuda_version() < Version("13.0"):
        return None

    source_header = Path(cuda_home) / "include" / "cuda.h"
    if not source_header.is_file():
        # A CUDA home without headers (a placeholder path in unit tests): no overlay.
        return None
    content = source_header.read_bytes()
    struct_start = content.find(b"typedef struct CUtensorMap_st {")
    struct_end = content.find(b"} CUtensorMap;", struct_start)
    if struct_start < 0 or struct_end < 0:
        raise RuntimeError(f"CUtensorMap declaration not found in {source_header}")
    struct_end += len(b"} CUtensorMap;")
    declaration = content[struct_start:struct_end]

    if b"alignas(64)" in declaration and b"_Alignas(64)" in declaration:
        return None
    if (
        declaration.count(b"alignas(128)") != 1
        or declaration.count(b"_Alignas(128)") != 1
    ):
        raise RuntimeError(
            f"Unexpected CUtensorMap alignment declaration in {source_header}"
        )

    patched_declaration = declaration.replace(b"alignas(128)", b"alignas(64)").replace(
        b"_Alignas(128)", b"_Alignas(64)"
    )
    patched_content = (
        content[:struct_start] + patched_declaration + content[struct_end:]
    )

    overlay_dir = (
        jit_env.FLASHINFER_CACHE_DIR / "cuda_include_overlay" / str(get_cuda_version())
    )
    overlay_header = overlay_dir / "cuda.h"
    if not overlay_header.exists() or overlay_header.read_bytes() != patched_content:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        temporary_header = overlay_header.with_name(
            f"{overlay_header.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary_header.write_bytes(patched_content)
            os.replace(temporary_header, overlay_header)
        finally:
            temporary_header.unlink(missing_ok=True)
    return overlay_dir


@functools.cache
def _get_cccl_include_overlay(source_header: Path, overlay_dir: Path) -> Path:
    original = source_header.read_bytes()
    content = original.decode("utf-8")
    if re.search(r"^\s*#\s*define\s+__out\b", content, re.MULTILINE):
        raise RuntimeError(f"Unexpected __out macro definition in {source_header}")
    if re.search(r"\b__cccl_out\b", content):
        raise RuntimeError(f"Replacement identifier already exists in {source_header}")

    count = len(re.findall(r"\b__out\b", content))
    if count != _EXPECTED_CCCL_OUT_TOKENS:
        raise RuntimeError(
            f"Expected {_EXPECTED_CCCL_OUT_TOKENS} __out tokens in {source_header}, "
            f"found {count}"
        )
    patched = re.sub(r"\b__out\b", "__cccl_out", content).encode("utf-8")
    overlay_header = overlay_dir / _CCCL_TCGEN05_HEADER
    if not overlay_header.exists() or overlay_header.read_bytes() != patched:
        overlay_header.parent.mkdir(parents=True, exist_ok=True)
        temporary_header = overlay_header.with_name(
            f"{overlay_header.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary_header.write_bytes(patched)
            os.replace(temporary_header, overlay_header)
        finally:
            temporary_header.unlink(missing_ok=True)
    return overlay_dir


def get_cccl_include_overlay() -> Optional[Path]:
    """Create a writable Windows overlay for CCCL's SAL-conflicting header."""
    if not is_windows:
        return None

    source_header = next(
        (
            include_dir / _CCCL_TCGEN05_HEADER
            for include_dir in jit_env.CCCL_INCLUDE_DIRS
            if (include_dir / _CCCL_TCGEN05_HEADER).is_file()
        ),
        None,
    )
    if source_header is None:
        raise RuntimeError(
            f"CCCL header {_CCCL_TCGEN05_HEADER} was not found under "
            f"{jit_env.CCCL_INCLUDE_DIRS}"
        )

    overlay_dir = jit_env.FLASHINFER_GEN_SRC_DIR / "_cccl_include_overlay"
    return _get_cccl_include_overlay(source_header.resolve(), overlay_dir.resolve())


def _write_if_changed(target: Path, data: bytes) -> None:
    if target.exists() and target.read_bytes() == data:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_bytes(data)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@functools.cache
def _get_spdlog_include_overlay(source_dir: Path, overlay_dir: Path) -> Path:
    target_dir = overlay_dir / SPDLOG_FORMAT_HEADER.parent
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        data = source.read_bytes()
        if source.name == SPDLOG_FORMAT_HEADER.name:
            data = patch_bundled_format_h(data.decode("utf-8")).encode("utf-8")
        _write_if_changed(target_dir / source.name, data)
    return overlay_dir


def get_spdlog_include_overlay() -> Optional[Path]:
    """Create a Windows overlay of spdlog's bundled fmt with the MSVC-safe literal.

    nvcc with the Visual Studio host compiler rejects the char32_t literal in
    bundled/format.h (fixed in fmt 12.2, not yet in spdlog's bundled copy).  The
    overlay holds a copy of the bundled directory with the two lines rewritten
    and is passed with -I ahead of the spdlog include directory, so the quoted
    sibling includes of format.h resolve inside the overlay as well.

    The overlay lives under the current FLASHINFER_GEN_SRC_DIR: an AOT build that
    moves to another build directory gets its own copy, and a copy that was
    deleted is written again.
    """
    if not is_windows:
        return None
    source_dir = jit_env.SPDLOG_INCLUDE_DIR / SPDLOG_FORMAT_HEADER.parent
    if not source_dir.is_dir():
        raise RuntimeError(f"spdlog bundled fmt directory not found: {source_dir}")
    overlay_dir = jit_env.FLASHINFER_GEN_SRC_DIR / "_spdlog_include_overlay"
    if not (overlay_dir / SPDLOG_FORMAT_HEADER).is_file():
        _get_spdlog_include_overlay.cache_clear()
    return _get_spdlog_include_overlay(source_dir.resolve(), overlay_dir.resolve())


def get_nvcc_parallelism_flags() -> List[str]:
    """Build nvcc flags controlled by FlashInfer parallelism environment variables."""
    env_var_name = "FLASHINFER_NVCC_THREADS"
    default = 1
    value = os.environ.get(env_var_name, str(default))

    try:
        threads = int(value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %s.", env_var_name, value, default
        )
        threads = default

    if threads < 1:
        logger.warning("Ignoring %s=%r; value must be >= 1.", env_var_name, value)
        threads = default

    return [f"--threads={threads}"]


@functools.lru_cache(maxsize=1)
def host_compiler_is_gcc() -> bool:
    """Whether the compiler nvcc forwards -Xcompiler flags to is GCC.

    Cached because the underlying torch helper shells out to the compiler.
    """
    from torch.utils.cpp_extension import check_compiler_is_gcc, get_cxx_compiler

    host_compiler = os.environ.get("CC", get_cxx_compiler())
    return check_compiler_is_gcc(host_compiler)


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def get_cccl_includes() -> List:
    """Get vendored CCCL include directories (added with -I for CTK override precedence)."""
    return [p.resolve() for p in jit_env.CCCL_INCLUDE_DIRS]


def get_system_includes(cuda_home: str) -> List:
    """Get list of system include directories."""
    system_includes = [
        sysconfig.get_path("include"),
        "$cuda_home/include",
        tvm_ffi.libinfo.find_include_path(),
        tvm_ffi.libinfo.find_dlpack_include_path(),
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]
    system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]
    system_includes.append(jit_env.SPDLOG_INCLUDE_DIR.resolve())

    if cuda_home == "/usr":
        # NOTE: this will resolve to /usr/include, which will mess up includes. See #1793
        system_includes.remove("$cuda_home/include")

    return system_includes


def build_common_cflags(
    cuda_home: str,
    extra_include_dirs: Optional[List[Path]] = None,
) -> List[str]:
    """Build common compilation flags."""
    cccl_includes = get_cccl_includes()
    system_includes = get_system_includes(cuda_home)

    common_cflags = []
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        common_cflags.append("-DPy_LIMITED_API=0x03090000")
    common_cflags += _get_glibcxx_abi_build_flags()
    cuda_include_overlay = get_cuda_include_overlay(cuda_home)
    if cuda_include_overlay is not None:
        common_cflags.append(f'-I"{cuda_include_overlay}"')
    cccl_include_overlay = get_cccl_include_overlay()
    if cccl_include_overlay is not None:
        common_cflags.append(f'-I"{cccl_include_overlay}"')
    spdlog_include_overlay = get_spdlog_include_overlay()
    if spdlog_include_overlay is not None:
        common_cflags.append(f'-I"{spdlog_include_overlay}"')
    if extra_include_dirs is not None:
        for extra_dir in extra_include_dirs:
            if is_windows:
                common_cflags.append(f'-I"{extra_dir.resolve()}"')
            else:
                common_cflags.append(f"-I{extra_dir.resolve()}")
    # Vendored CCCL headers use -I (not -isystem) so they take precedence
    # over the CTK-bundled copy. CCCL headers use #pragma system_header
    # internally to suppress warnings. See https://github.com/NVIDIA/cccl/issues/527
    if is_windows:
        # MSVC has no -isystem; quoted -I keeps paths with spaces whole.
        for cccl_dir in cccl_includes:
            common_cflags.append(f'-I"{str(cccl_dir)}"')
        for sys_dir in system_includes:
            common_cflags.append(f'-I"{str(sys_dir)}"')
    else:
        for cccl_dir in cccl_includes:
            common_cflags.append(f"-I{cccl_dir}")
        for sys_dir in system_includes:
            common_cflags.append(f"-isystem {sys_dir}")

    return common_cflags


def build_cflags(
    common_cflags: List[str],
    extra_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build C++ compilation flags."""
    cflags = [
        "$common_cflags",
    ]
    if is_windows:
        cflags += ["/std:c++20", "/DNOMINMAX", "/Zc:preprocessor", "/bigobj"]
    else:
        cflags.append("-fPIC")
    if extra_cflags is not None:
        cflags += extra_cflags

    env_extra_cflags = parse_env_flags("FLASHINFER_EXTRA_CFLAGS")
    if env_extra_cflags is not None:
        cflags += env_extra_cflags

    if is_windows:
        # cl.exe warns on repeated options; keep the first occurrence of each flag.
        cflags = list(dict.fromkeys(cflags))
    return cflags


def build_common_cuda_flags(common_cflags: List[str]) -> List[str]:
    """Common flags for nvcc; on Windows the MSVC host compiler needs its own set."""
    if not is_windows:
        return list(common_cflags)
    # The Py_LIMITED_API define is dropped for nvcc on Windows, as in the Windows
    # port. The filter matches the flag by value: build_common_cflags emits the
    # define only when Py_GIL_DISABLED is unset, so on a free-threaded Python the
    # first entry is an include flag that nvcc must keep. The MSVC conformance
    # switches are passed through -Xcompiler.
    return [
        "-DTORCH_EXTENSION_NAME=$name",
        "--std=c++20",
        "-Xcompiler /Zc:__cplusplus",
        "-Xcompiler /Zc:preprocessor",
        "-Xcompiler=/bigobj",
    ] + [flag for flag in common_cflags if not flag.startswith("-DPy_LIMITED_API")]


def build_cuda_cflags(
    common_cflags: List[str],
    extra_cuda_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build CUDA compilation flags."""
    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]
    cuda_cflags += [
        "$common_cuda_flags",
        "--expt-relaxed-constexpr",
    ]
    if not is_windows:
        cuda_cflags.append("--compiler-options=-fPIC")
    cuda_version = get_cuda_version()
    # enable -static-global-template-stub when cuda version >= 12.8
    if cuda_version >= Version("12.8"):
        cuda_cflags += [
            "-static-global-template-stub=false",
        ]

    cpp_ext_initial_compilation_context = CompilationContext()
    global_flags = cpp_ext_initial_compilation_context.get_nvcc_flags_list(
        map_sm107_to_100f=True
    )
    if extra_cuda_cflags is not None:
        # Check if module provides architecture flags
        module_has_gencode = any(
            flag.startswith("-gencode=") for flag in extra_cuda_cflags
        )

        if module_has_gencode:
            # Use module's architecture flags, but keep global non-architecture flags
            global_non_arch_flags = [
                flag for flag in global_flags if not flag.startswith("-gencode=")
            ]
            cuda_cflags += global_non_arch_flags + extra_cuda_cflags
        else:
            # No module architecture flags, use both global and module flags
            cuda_cflags += global_flags + extra_cuda_cflags
    else:
        # No module flags, use global flags
        cuda_cflags += global_flags

    env_extra_cuda_cflags = parse_env_flags("FLASHINFER_EXTRA_CUDAFLAGS")
    if env_extra_cuda_cflags is not None:
        cuda_cflags += env_extra_cuda_cflags

    return cuda_cflags


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
    needs_device_linking: bool = False,
    embedded_cubins: Optional[Mapping[str, Path]] = None,
    extra_cuda_cflags_by_source: Optional[Mapping[Path, List[str]]] = None,
) -> str:
    cuda_home = get_cuda_path()
    common_cflags = build_common_cflags(cuda_home, extra_include_dirs)
    cflags = build_cflags(common_cflags, extra_cflags)
    common_cuda_flags = build_common_cuda_flags(common_cflags)
    cuda_cflags = build_cuda_cflags(common_cflags, extra_cuda_cflags)
    cuda_arch_flags = [flag for flag in cuda_cflags if flag.startswith("-gencode=")]
    cuda_cflags_by_source = {
        Path(source).resolve(): build_cuda_cflags(common_cflags, flags)
        for source, flags in (extra_cuda_cflags_by_source or {}).items()
    }

    if is_windows:
        python_path = os.path.dirname(sys.executable)
        if python_path.endswith("\\Scripts"):
            python_path = os.path.dirname(python_path)
        python_lib_path = os.path.join(sys.base_exec_prefix, "libs")
        cuda_lib_arch = (
            "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
        )
        ldflags = [
            f'"/LIBPATH:{python_lib_path}"',
            f'"/LIBPATH:$cuda_home\\lib\\{cuda_lib_arch}"',
            f'"/LIBPATH:{python_path}\\Lib\\site-packages\\tvm_ffi\\lib"',
            f'"/LIBPATH:{python_path}\\Lib\\site-packages\\torch\\lib"',
            "c10.lib",
            "c10_cuda.lib",
            "torch.lib",
            "torch_cuda.lib",
            "cudart.lib",
            "cuda.lib",
            "tvm_ffi.lib",
            "torch_python.lib",
        ]
    else:
        ldflags = [
            "-shared",
            "-L$cuda_home/lib64",
            "-L$cuda_home/lib64/stubs",
            "-lcudart",
            "-lcuda",
        ]

    env_extra_ldflags = parse_env_flags("FLASHINFER_EXTRA_LDFLAGS")
    if env_extra_ldflags is not None:
        ldflags += env_extra_ldflags

    if extra_ldflags is not None:
        if is_windows:
            for ldflag in extra_ldflags:
                if ldflag.startswith("-l"):
                    ldflag = ldflag[2:] + ".lib"
                ldflags.append(ldflag)
        else:
            ldflags += extra_ldflags

    cxx = os.environ.get("CXX", "c++")
    nvcc = os.environ.get("FLASHINFER_NVCC", "$cuda_home/bin/nvcc")
    if is_windows:
        nvcc = f'"{nvcc}"'
    # Compiler launchers (e.g., sccache, ccache) — empty string when unset
    cxx_launcher = os.environ.get("FLASHINFER_CXX_LAUNCHER", "")
    nvcc_launcher = os.environ.get("FLASHINFER_NVCC_LAUNCHER", "")
    output_dir = jit_env.FLASHINFER_JIT_DIR / name

    if is_windows:
        link_rsp = str((output_dir / "link.rsp").resolve()).replace(":\\", "$:\\")
        rule_compile = [
            "rule compile",
            "  command = cl.exe /showIncludes $cflags -c $in /Fo$out $post_cflags",
            "  deps = msvc",
        ]
        rule_cuda_compile = [
            "rule cuda_compile",
            "  command = $nvcc --generate-dependencies-with-compile -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
        ]
        rule_link = [
            "rule link",
            f'  command = link.exe @"{link_rsp}"',
            f"  rspfile = {link_rsp}",
            "  rspfile_content = /DLL $in /nologo $ldflags /out:$out",
        ]
    else:
        rule_compile = [
            "rule compile",
            "  command = $cxx_launcher $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
        ]
        rule_cuda_compile = [
            "rule cuda_compile",
            "  command = $nvcc_launcher $nvcc --generate-dependencies-with-compile -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
        ]
        rule_link = [
            "rule link",
            "  command = $cxx $in $ldflags -o $out",
        ]

    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
        f"cuda_home = {cuda_home}",
        f"cxx = {cxx}",
        f"nvcc = {nvcc}",
        f"cxx_launcher = {cxx_launcher}",
        f"nvcc_launcher = {nvcc_launcher}",
        "",
        "common_cflags = " + join_multiline(common_cflags),
        "common_cuda_flags = " + join_multiline(common_cuda_flags),
        "cflags = " + join_multiline(cflags),
        "post_cflags =",
        "cuda_cflags = " + join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "cuda_arch_flags = " + join_multiline(cuda_arch_flags),
        "ldflags = " + join_multiline(ldflags),
        "",
        *rule_compile,
        "",
        *rule_cuda_compile,
        "",
    ]

    if embedded_cubins:
        lines.extend(
            [
                "rule embed_cubin",
                f"  command = {sys.executable} -m tvm_ffi.utils.embed_cubin "
                "--output-obj $out --input-obj $in --cubin $cubin --name $cubin_name",
                "",
            ]
        )

    # Add nvcc linking rule for device code
    if needs_device_linking:
        lines.extend(
            [
                "rule nvcc_link",
                "  command = $nvcc -shared $cuda_arch_flags $in $ldflags -o $out",
                "",
            ]
        )
    else:
        lines.extend(
            [
                *rule_link,
                "",
            ]
        )

    # Use absolute paths for outputs so ninja files work with any workdir
    # This enables isolated workdirs for runtime JIT (avoiding .ninja_log races)
    # while still supporting subninja for parallel AOT builds
    output_dir = jit_env.FLASHINFER_JIT_DIR / name

    def ninja_path(path) -> str:
        # ninja reads "C:" as a variable reference; escape the drive colon on Windows.
        text = str(path)
        if not is_windows:
            return text
        return text.replace("$", "$$").replace(" ", "$ ").replace(":\\", "$:\\")

    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        object_suffix = ".cuda.o" if is_cuda else ".o"
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = f"{source.parent.name}_{source.stem}{object_suffix}"
        obj = ninja_path((output_dir / obj_name).resolve())
        objects.append(obj)
        lines.append(f"build {obj}: {cmd} {ninja_path(source.resolve())}")
        if source.resolve() in cuda_cflags_by_source:
            lines.append(
                "  cuda_cflags = "
                + join_multiline(cuda_cflags_by_source[source.resolve()])
            )

    if embedded_cubins:
        if not objects:
            raise ValueError("embedded cubins require at least one host object")
        current_obj = objects[0]
        for index, (cubin_name, cubin_path) in enumerate(
            sorted(embedded_cubins.items())
        ):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cubin_name) is None:
                raise ValueError(f"invalid embedded cubin identifier: {cubin_name!r}")
            resolved_cubin = ninja_path(Path(cubin_path).resolve(strict=True))
            embedded_obj = ninja_path(
                (output_dir / f"embedded_{index}_{Path(current_obj).name}").resolve()
            )
            lines.extend(
                [
                    f"build {embedded_obj}: embed_cubin {current_obj} | {resolved_cubin}",
                    f"  cubin = {resolved_cubin}",
                    f"  cubin_name = {cubin_name}",
                ]
            )
            current_obj = embedded_obj
        objects[0] = current_obj

    lines.append("")
    link_rule = "nvcc_link" if needs_device_linking else "link"
    output_so = ninja_path((output_dir / jit_env.jit_library_name(name)).resolve())
    lines.append(f"build {output_so}: {link_rule} " + " ".join(objects))
    lines.append(f"default {output_so}")
    lines.append("")

    return "\n".join(lines)


def _get_num_workers() -> Optional[int]:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        return int(max_jobs)
    return None


def run_ninja(
    workdir: Path,
    ninja_file: Path,
    verbose: bool,
    max_jobs: Optional[int] = None,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = max_jobs if max_jobs is not None else _get_num_workers()
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e
