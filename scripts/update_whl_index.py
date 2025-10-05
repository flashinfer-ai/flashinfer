"""
Update wheel index for flashinfer packages.

This script generates PEP 503 compatible simple repository index pages for:
- flashinfer-python (no CUDA suffix in version)
- flashinfer-cubin (no CUDA suffix in version)
- flashinfer-jit-cache (has CUDA suffix like +cu130)

The index is organized by CUDA version for jit-cache, and flat for others.
"""

import hashlib
import pathlib
import re
import argparse
import sys
from typing import Optional


def get_cuda_version(wheel_name: str) -> Optional[str]:
    """Extract CUDA version from wheel filename."""
    # Match patterns like +cu128, +cu129, +cu130
    match = re.search(r"\+cu(\d+)", wheel_name)
    if match:
        return match.group(1)
    return None


def get_package_info(wheel_path: pathlib.Path) -> Optional[dict]:
    """Extract package information from wheel filename."""
    wheel_name = wheel_path.name

    # Try flashinfer-python pattern
    match = re.match(r"flashinfer_python-([0-9.]+(?:\.dev\d+)?)-", wheel_name)
    if match:
        version = match.group(1)
        return {
            "package": "flashinfer-python",
            "version": version,
            "cuda": None,
        }

    # Try flashinfer-cubin pattern
    match = re.match(r"flashinfer_cubin-([0-9.]+(?:\.dev\d+)?)-", wheel_name)
    if match:
        version = match.group(1)
        return {
            "package": "flashinfer-cubin",
            "version": version,
            "cuda": None,
        }

    # Try flashinfer-jit-cache pattern (has CUDA suffix in version)
    match = re.match(r"flashinfer_jit_cache-([0-9.]+(?:\.dev\d+)?\+cu\d+)-", wheel_name)
    if match:
        version = match.group(1)
        cuda_ver = get_cuda_version(wheel_name)
        return {
            "package": "flashinfer-jit-cache",
            "version": version,
            "cuda": cuda_ver,
        }

    return None


def compute_sha256(file_path: pathlib.Path) -> str:
    """Compute SHA256 hash of a file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def update_index(
    dist_dir: str = "dist",
    output_dir: str = "whl",
    base_url: str = "https://github.com/flashinfer-ai/flashinfer/releases/download",
    release_tag: Optional[str] = None,
):
    """
    Update wheel index from dist directory.

    Args:
        dist_dir: Directory containing wheel files
        output_dir: Output directory for index files
        base_url: Base URL for wheel downloads
        release_tag: GitHub release tag (e.g., 'nightly' or 'v0.3.1')
    """
    dist_path = pathlib.Path(dist_dir)
    if not dist_path.exists():
        print(f"Error: dist directory '{dist_dir}' does not exist")
        sys.exit(1)

    wheels = sorted(dist_path.glob("*.whl"))
    if not wheels:
        print(f"No wheel files found in '{dist_dir}'")
        sys.exit(1)

    print(f"Found {len(wheels)} wheel file(s)")

    for wheel_path in wheels:
        print(f"\nProcessing: {wheel_path.name}")

        # Extract package information
        info = get_package_info(wheel_path)
        if not info:
            print("  ⚠️  Skipping: Could not parse wheel filename")
            continue

        # Compute SHA256
        sha256 = compute_sha256(wheel_path)

        # Determine index directory
        package = info["package"]
        cuda = info["cuda"]

        if cuda:
            # CUDA-specific index for jit-cache: whl/cu130/flashinfer-jit-cache/
            index_dir = pathlib.Path(output_dir) / f"cu{cuda}" / package
        else:
            # No CUDA version for python/cubin: whl/flashinfer-python/
            index_dir = pathlib.Path(output_dir) / package

        index_dir.mkdir(parents=True, exist_ok=True)

        # Construct download URL
        tag = release_tag or f"v{info['version'].split('+')[0].split('.dev')[0]}"
        download_url = f"{base_url}/{tag}/{wheel_path.name}#sha256={sha256}"

        # Update index.html
        index_file = index_dir / "index.html"

        # Read existing content to avoid duplicates
        existing_links = set()
        if index_file.exists():
            with index_file.open("r") as f:
                existing_links = set(f.readlines())
        else:
            # Create new index file with HTML header
            with index_file.open("w") as f:
                f.write("<!DOCTYPE html>\n")
                f.write("<html>\n")
                f.write(f"<head><title>Links for {package}</title></head>\n")
                f.write("<body>\n")
                f.write(f"<h1>Links for {package}</h1>\n")
            print(f"  📝 Created new index file: {index_dir}/index.html")

        # Create new link
        new_link = f'<a href="{download_url}">{wheel_path.name}</a><br>\n'

        if new_link in existing_links:
            print(f"  ℹ️  Already in index: {index_dir}/index.html")
        else:
            with index_file.open("a") as f:
                f.write(new_link)
            print(f"  ✅ Added to index: {index_dir}/index.html")

        print(f"  📦 Package: {package}")
        print(f"  🔖 Version: {info['version']}")
        if cuda:
            print(f"  🎮 CUDA: cu{cuda}")
        print(f"  📍 URL: {download_url}")


def main():
    parser = argparse.ArgumentParser(
        description="Update wheel index for flashinfer packages"
    )
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="Directory containing wheel files (default: dist)",
    )
    parser.add_argument(
        "--output-dir",
        default="whl",
        help="Output directory for index files (default: whl)",
    )
    parser.add_argument(
        "--base-url",
        default="https://github.com/flashinfer-ai/flashinfer/releases/download",
        help="Base URL for wheel downloads",
    )
    parser.add_argument(
        "--release-tag",
        help="GitHub release tag (e.g., 'nightly' or 'v0.3.1'). If not specified, will be derived from version.",
    )

    args = parser.parse_args()

    update_index(
        dist_dir=args.dist_dir,
        output_dir=args.output_dir,
        base_url=args.base_url,
        release_tag=args.release_tag,
    )


if __name__ == "__main__":
    main()
