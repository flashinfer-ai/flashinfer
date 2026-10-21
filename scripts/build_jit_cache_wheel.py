#!/usr/bin/env python3
"""Build an isolated wheel, retrying interrupted dependency downloads."""

import logging
import subprocess
import time
from pathlib import Path


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
    from build.env import DefaultIsolatedEnv

    with DefaultIsolatedEnv() as env:
        builder = ProjectBuilder.from_isolated_env(env, ".")
        install_build_requirements(env, builder.build_system_requires)
        install_build_requirements(env, builder.get_requires_for_build("wheel"))
        # Backend hooks and compilation execute once; only dependency installs retry.
        return builder.build("wheel", str(Path("dist").resolve()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Successfully built {build_wheel()}", flush=True)
