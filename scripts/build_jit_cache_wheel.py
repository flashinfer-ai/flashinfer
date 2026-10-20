#!/usr/bin/env python3
"""Build a wheel in the current interpreter, retrying interrupted dependency downloads.

The build is deliberately not isolated. The CI callers run it inside a
throwaway container whose interpreter already holds the pinned torch, so a
PEP 517 isolated environment adds no hermeticity. It would only install torch
and the other build requirements a second time, into a randomly named
``/tmp/build-env-*`` directory. That directory then appears as an include path
(tvm_ffi) in every compile command, so each run gave every translation unit
that includes tvm_ffi headers a new sccache key.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path


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


def build_wheel():
    from build import ProjectBuilder

    env = CurrentInterpreterEnv()
    builder = ProjectBuilder(".", python_executable=env.python_executable)
    install_build_requirements(env, builder.build_system_requires)
    install_build_requirements(env, builder.get_requires_for_build("wheel"))
    # Backend hooks and compilation execute once; only dependency installs retry.
    return builder.build("wheel", str(Path("dist").resolve()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Successfully built {build_wheel()}", flush=True)
