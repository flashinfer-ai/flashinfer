#!/usr/bin/env python3
"""Build a wheel in the current interpreter, retrying interrupted dependency downloads.

The build is deliberately not isolated. The CI callers run it inside a
throwaway container whose interpreter already holds the pinned torch, so a
PEP 517 isolated environment adds no hermeticity. It would only install torch
and the other build requirements a second time, into a randomly named
``/tmp/build-env-*`` directory. That directory then appears as an include path
(tvm_ffi) in every compile command, so each run gave every translation unit
that includes tvm_ffi headers a new sccache key.

Before compiling, the build also pins the mtime of every installed header (see
``pin_header_mtimes``), because GCC's ``#pragma once`` otherwise makes the
preprocessed output of torch-including translation units depend on when pip
wrote each header.
"""

import logging
import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

# 1980-01-01T00:00:00Z, the earliest timestamp a ZIP archive can record.
PINNED_HEADER_MTIME = 315532800
HEADER_SUFFIXES = frozenset(
    {".h", ".hh", ".hpp", ".hxx", ".cuh", ".inl", ".inc", ".ipp", ".tcc"}
)


class CurrentInterpreterEnv:
    """Install build requirements into the interpreter running this script."""

    python_executable = sys.executable

    def install(self, requirements):
        if not requirements:
            return
        # Captured so install_build_requirements can classify a failure; echoed
        # so the build log still shows what pip did.
        result = subprocess.run(
            [self.python_executable, "-m", "pip", "install", *sorted(requirements)],
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )


def install_build_requirements(env, requirements):
    """Restart pip when a network error escapes pip's own request retries."""
    for attempt in range(1, 4):
        try:
            env.install(requirements)
            return
        except subprocess.CalledProcessError as error:
            output = "\n".join(
                stream.decode(errors="replace") if isinstance(stream, bytes) else stream
                for stream in (error.stdout, error.stderr)
                if stream
            )
            interrupted = any(
                marker in output
                for marker in (
                    "IncompleteRead",
                    "RemoteDisconnected",
                    "ConnectionResetError",
                    "ReadTimeoutError",
                )
            )
            if not interrupted or attempt == 3:
                raise
            delay = 10 * attempt
            logging.warning(
                "Build dependency download interrupted; retrying pip in %ss "
                "(attempt %s/3)",
                delay,
                attempt + 1,
            )
            time.sleep(delay)


def site_package_dirs():
    """Return the install directories of the interpreter that runs the build."""
    paths = sysconfig.get_paths()
    return sorted({Path(paths["purelib"]), Path(paths["platlib"])})


def pin_header_mtimes(roots):
    """Give every installed C/C++ header one fixed mtime; return how many changed.

    GCC's ``#pragma once`` treats two different files as the same file when
    their size, contents and mtime (whole seconds) all match, and then skips the
    second one. pip stamps each installed file with the time it wrote it, and
    torch ships byte-identical ``#pragma once`` headers at several paths (e.g.
    the empty ``ATen/ops/*_native.h`` stubs that ``<ATen/ATen.h>`` includes).
    Which of those GCC keeps therefore depends on which copies pip happened to
    write within the same second, so two builds of one commit preprocess the
    same translation unit differently. sccache keys cudafe++ on that output, and
    with CUDA 12.8+ a cudafe++ miss also misses cicc and ptxas, which costs
    minutes per translation unit. With one shared mtime GCC always keeps
    exactly one copy of each duplicate, whatever the install timing.

    A file counts as a header when it has a header suffix or sits below an
    ``include`` directory (for extension-less headers such as ``cuda/std/*``).
    Python sources are never touched, so bytecode caches stay valid.
    """
    changed = 0
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            in_include = "include" in Path(dirpath).relative_to(root).parts
            for name in filenames:
                path = Path(dirpath, name)
                if path.suffix in (".py", ".pyc"):
                    continue
                if not in_include and path.suffix not in HEADER_SUFFIXES:
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                if path.stat().st_mtime_ns == PINNED_HEADER_MTIME * 10**9:
                    continue
                os.utime(path, (PINNED_HEADER_MTIME, PINNED_HEADER_MTIME))
                changed += 1
    return changed


def build_wheel():
    from build import ProjectBuilder

    env = CurrentInterpreterEnv()
    builder = ProjectBuilder(".", python_executable=env.python_executable)
    install_build_requirements(env, builder.build_system_requires)
    install_build_requirements(env, builder.get_requires_for_build("wheel"))
    # After the last install, so headers pip wrote for the build requirements
    # are pinned as well.
    logging.info(
        "Pinned the mtime of %d installed headers",
        pin_header_mtimes(site_package_dirs()),
    )
    # Backend hooks and compilation execute once; only dependency installs retry.
    return builder.build("wheel", str(Path("dist").resolve()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Successfully built {build_wheel()}", flush=True)
