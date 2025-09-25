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
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import flashinfer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from flashinfer.artifacts import download_artifacts
from flashinfer.jit.cubin_loader import FLASHINFER_CUBINS_REPOSITORY


def main():
    parser = argparse.ArgumentParser(
        description="Download FlashInfer cubins from artifactory"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="flashinfer_cubin/cubins",
        help="Output directory for cubins (default: flashinfer_cubin/cubins)",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=4,
        help="Number of download threads (default: 4)",
    )
    parser.add_argument(
        "--repository",
        "-r",
        type=str,
        default=None,
        help="Override the cubins repository URL",
    )

    args = parser.parse_args()

    # Set environment variables to control download behavior
    if args.repository:
        os.environ["FLASHINFER_CUBINS_REPOSITORY"] = args.repository

    os.environ["FLASHINFER_CUBIN_DIR"] = str(Path(args.output_dir).absolute())
    os.environ["FLASHINFER_CUBIN_DOWNLOAD_THREADS"] = str(args.threads)

    print(f"Downloading cubins to {args.output_dir}")
    print(
        f"Repository: {os.environ.get('FLASHINFER_CUBINS_REPOSITORY', FLASHINFER_CUBINS_REPOSITORY)}"
    )

    # Use the existing download_artifacts function
    try:
        download_artifacts()
        print("Download complete!")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
