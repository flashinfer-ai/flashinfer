import hashlib
import pathlib
import re

for path in sorted(pathlib.Path("dist").glob("*.whl")):
    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    # Extract version and CUDA version from wheel name
    # Example: flashinfer_jit_cache-1.2.3+cu128-cp39-abi3-manylinux_2_28_x86_64.whl
    # Example: flashinfer_jit_cache-1.2.3rc1+cu128-cp39-abi3-manylinux_2_28_x86_64.whl
    # Example: flashinfer_jit_cache-1.2.3.post1+cu128-cp39-abi3-manylinux_2_28_x86_64.whl
    match = re.search(
        r"flashinfer_jit_cache-([0-9]+\.[0-9]+\.[0-9]+[a-z0-9.]*)\+cu(\d+)-",
        path.name,
    )
    if not match:
        print(f"Warning: Could not parse wheel name: {path.name}")
        continue

    ver, cu = match.groups()

    # Create directory structure: cu{version}/flashinfer-jit-cache/
    # No torch subdirectory since we don't separate by torch version
    index_dir = pathlib.Path(f"flashinfer-whl/cu{cu}/flashinfer-jit-cache")
    index_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://github.com/flashinfer-ai/flashinfer/releases/download"
    full_url = f"{base_url}/v{ver}/{path.name}#sha256={sha256}"

    with (index_dir / "index.html").open("a") as f:
        f.write(f'<a href="{full_url}">{path.name}</a><br>\n')
