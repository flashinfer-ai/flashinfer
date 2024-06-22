import hashlib
import pathlib
import re

for path in sorted(pathlib.Path("python/dist").glob("*.whl")):
    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    ver, cu, torch = re.findall(
        r"flashinfer-([0-9.]+)\+cu(\d+)torch([0-9.]+)-", path.name
    )[0]
    index_dir = pathlib.Path(f"flashinfer-whl/cu{cu}/torch{torch}/flashinfer")
    index_dir.mkdir(exist_ok=True)
    index_path = index_dir / "index.html"
    with index_path.open("a") as f:
        f.write(
            f'<a href="https://github.com/flashinfer-ai/flashinfer/releases/download/v{ver}/{path.name}#sha256={sha256}">{path.name}</a><br>\n'
        )
