#!/usr/bin/env python3
"""
Copyright (c) 2025 by FlashInfer team.

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
import subprocess
import sys
from pathlib import Path


def build_wheel():
    """Build the flashinfer-cubin wheel."""

    # Change to the flashinfer-cubin directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("Building flashinfer-cubin wheel...")
    print(f"Working directory: {script_dir}")

    # Clean previous builds
    dist_dir = script_dir / "dist"
    build_dir = script_dir / "build"
    egg_info_dir = script_dir / "flashinfer_cubin.egg-info"

    for dir_to_clean in [dist_dir, build_dir, egg_info_dir]:
        if dir_to_clean.exists():
            print(f"Cleaning {dir_to_clean}")
            import shutil

            shutil.rmtree(dir_to_clean)

    # Build wheel
    try:
        subprocess.run([sys.executable, "setup.py", "bdist_wheel"], check=True)

        print("Wheel built successfully!")

        # List built wheels
        if dist_dir.exists():
            wheels = list(dist_dir.glob("*.whl"))
            if wheels:
                print(f"Built wheel: {wheels[0]}")
            else:
                print("No wheel files found in dist/")

    except subprocess.CalledProcessError as e:
        print(f"Failed to build wheel: {e}")
        sys.exit(1)


if __name__ == "__main__":
    build_wheel()
