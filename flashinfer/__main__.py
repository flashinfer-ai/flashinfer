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

# flashinfer-cli
import argparse

from .artifacts import download_artifacts

if __name__ == "__main__":
    parser = argparse.ArgumentParser("FlashInfer CLI")
    parser.add_argument(
        "--download-cubin", action="store_true", help="Download artifacts"
    )

    args = parser.parse_args()

    if args.download_cubin:
        if download_artifacts():
            print("✅ All cubin download tasks completed successfully.")
        else:
            print("❌ Some cubin download tasks failed.")
