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
    # Supports PEP 440: base_version[{a|b|rc}N][.postN][.devN]
    match = re.match(
        r"flashinfer_python-([0-9.]+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?)-",
        wheel_name,
    )
    if match:
        version = match.group(1)
        return {
            "package": "flashinfer-python",
            "version": version,
            "cuda": None,
        }

    # Try flashinfer-cubin pattern
    # Supports PEP 440: base_version[{a|b|rc}N][.postN][.devN]
    match = re.match(
        r"flashinfer_cubin-([0-9.]+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?)-",
        wheel_name,
    )
    if match:
        version = match.group(1)
        return {
            "package": "flashinfer-cubin",
            "version": version,
            "cuda": None,
        }

    # Try flashinfer-jit-cache pattern (has CUDA suffix in version)
    # Supports PEP 440: base_version[{a|b|rc}N][.postN][.devN]+cuXXX
    match = re.match(
        r"flashinfer_jit_cache-([0-9.]+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?\+cu\d+)-",
        wheel_name,
    )
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


def generate_directory_index(directory: pathlib.Path):
    """Generate index.html for a directory listing its subdirectories."""
    # Get all subdirectories
    subdirs = sorted([d for d in directory.iterdir() if d.is_dir()])

    if not subdirs:
        return

    index_file = directory / "index.html"

    # Generate HTML for directory listing
    with index_file.open("w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n")
        f.write(f"<head><title>Index of {directory.name or 'root'}</title></head>\n")
        f.write("<body>\n")
        f.write(f"<h1>Index of {directory.name or 'root'}</h1>\n")

        for subdir in subdirs:
            f.write(f'<a href="{subdir.name}/">{subdir.name}/</a><br>\n')

        f.write("</body>\n")
        f.write("</html>\n")


def update_parent_indices(leaf_dir: pathlib.Path, root_dir: pathlib.Path):
    """Recursively update index.html for all parent directories."""
    current = leaf_dir.parent

    while current >= root_dir and current != current.parent:
        generate_directory_index(current)
        current = current.parent


def update_index(
    dist_dir: str = "dist",
    output_dir: str = "whl",
    base_url: str = "https://github.com/flashinfer-ai/flashinfer/releases/download",
    release_tag: Optional[str] = None,
    nightly: bool = False,
):
    """
    Update wheel index from dist directory.

    Args:
        dist_dir: Directory containing wheel files
        output_dir: Output directory for index files
        base_url: Base URL for wheel downloads
        release_tag: GitHub release tag (e.g., 'nightly' or 'v0.3.1')
        nightly: If True, update index to whl/nightly subdirectory for nightly releases
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

    # Track all directories that need parent index updates
    created_dirs = set()

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

        # Add nightly subdirectory if nightly flag is set
        base_output = pathlib.Path(output_dir)
        if nightly:
            base_output = base_output / "nightly"

        if cuda:
            # CUDA-specific index for jit-cache: whl/nightly/cu130/flashinfer-jit-cache/
            index_dir = base_output / f"cu{cuda}" / package
        else:
            # No CUDA version for python/cubin: whl/nightly/flashinfer-python/
            index_dir = base_output / package

        index_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.add(index_dir)

        # Construct download URL
        tag = release_tag or f"v{info['version'].split('+')[0].split('.dev')[0]}"
        download_url = f"{base_url}/{tag}/{wheel_path.name}#sha256={sha256}"

        # Update index.html
        index_file = index_dir / "index.html"

        # Read existing links to avoid duplicates
        links = set()
        if index_file.exists():
            with index_file.open("r") as f:
                content = f.read()
                # Simple regex to extract the <a> tags
                links.update(re.findall(r'<a href=".*?">.*?</a><br>\n', content))

        # Create and add new link
        new_link = f'<a href="{download_url}">{wheel_path.name}</a><br>\n'
        is_new = new_link not in links
        if is_new:
            links.add(new_link)

            # Write the complete, valid HTML file
            with index_file.open("w") as f:
                f.write("<!DOCTYPE html>\n")
                f.write("<html>\n")
                f.write(f"<head><title>Links for {package}</title></head>\n")
                f.write("<body>\n")
                f.write(f"<h1>Links for {package}</h1>\n")
                for link in sorted(list(links)):
                    f.write(link)
                f.write("</body>\n")
                f.write("</html>\n")
            print(f"  ✅ Added to index: {index_dir}/index.html")
        else:
            print(f"  ℹ️  Already in index: {index_dir}/index.html")

        print(f"  📦 Package: {package}")
        print(f"  🔖 Version: {info['version']}")
        if cuda:
            print(f"  🎮 CUDA: cu{cuda}")
        print(f"  📍 URL: {download_url}")

    # Update parent directory indices
    print("\n📂 Updating parent directory indices...")
    root_output = pathlib.Path(output_dir)
    for leaf_dir in created_dirs:
        update_parent_indices(leaf_dir, root_output)
    print("  ✅ Parent indices updated")


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
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Update index to whl/nightly subdirectory for nightly releases",
    )

    args = parser.parse_args()

    update_index(
        dist_dir=args.dist_dir,
        output_dir=args.output_dir,
        base_url=args.base_url,
        release_tag=args.release_tag,
        nightly=args.nightly,
    )


if __name__ == "__main__":
    main()
