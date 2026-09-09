#!/usr/bin/env python3
"""Check a built flashinfer-jit-cache wheel against the tvm-ffi range we advertise.

The .so files in this wheel take their ABI from whichever apache-tvm-ffi was resolved
while building them, and nothing in the wheel records that. requirements.txt advertises
a range; the claim most likely to be wrong is the *floor*, because the artifact is built
against something newer. That is exactly how 0.6.16 shipped: its kernels referenced
``TVMFFIGetCustomAllocator``, a symbol libtvm_ffi did not export before 0.1.13, so every
environment on the advertised floor crashed at load time.

This resolves the floor (and the release pin, if different) from the declared constraint,
downloads those libtvm_ffi builds, and fails if any kernel references a symbol they do
not export.

Scope: this catches undefined-symbol breakage only. A change that keeps the symbol names
but alters a layout -- e.g. tvm-ffi 0.1.13 growing ``Optional<T>`` from 8 to 16 bytes and
with it the return slot of the ``ModuleObj::GetFunction`` we override -- is invisible to
nm and needs a GPU run with FLASHINFER_DISABLE_JIT=1.

Usage:
    python scripts/verify_jit_cache_abi.py flashinfer-jit-cache/dist/*.whl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPI = "https://pypi.org/pypi/apache-tvm-ffi/json"


def declared_constraint() -> str:
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line.startswith("apache-tvm-ffi"):
            return line
    sys.exit("no apache-tvm-ffi requirement found in requirements.txt")


def release_pin() -> str | None:
    path = REPO_ROOT / "scripts" / "build_constraints.txt"
    if not path.is_file():
        return None
    match = re.search(r"^apache-tvm-ffi==(\S+)", path.read_text(), re.M)
    return match.group(1) if match else None


def versions_to_check(constraint: str) -> list[str]:
    """The floor of the advertised range, plus the release pin if it differs."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    spec = SpecifierSet(constraint.split("apache-tvm-ffi", 1)[1])
    with urllib.request.urlopen(PYPI, timeout=60) as response:
        releases = json.load(response)["releases"]
    # A yanked release is still installable via an exact pin, so it stays in scope.
    available = [v for v, files in releases.items() if files and spec.contains(v)]
    if not available:
        sys.exit(f"no published apache-tvm-ffi satisfies {constraint!r}")
    checks = [min(available, key=Version)]
    pin = release_pin()
    if pin and pin not in checks and pin in available:
        checks.append(pin)
    return checks


def exported_symbols(version: str, workdir: Path) -> set[str]:
    """TVMFFI* symbols that this apache-tvm-ffi's libtvm_ffi.so defines."""
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/apache-tvm-ffi/{version}/json"
    ) as r:
        urls = json.load(r)["urls"]
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheels = [
        u for u in urls if u["filename"].endswith(".whl") and "linux" in u["filename"]
    ]
    # abi3 wheels serve every interpreter; otherwise match this one.
    match = [
        u for u in wheels if "abi3" in u["filename"] or tag in u["filename"]
    ] or wheels
    if not match:
        sys.exit(f"apache-tvm-ffi {version} has no linux wheel to inspect")
    dest = workdir / match[0]["filename"]
    urllib.request.urlretrieve(match[0]["url"], dest)
    extracted = workdir / f"ffi-{version}"
    with zipfile.ZipFile(dest) as zf:
        lib = next(n for n in zf.namelist() if n.endswith("lib/libtvm_ffi.so"))
        zf.extract(lib, extracted)
    out = subprocess.run(
        ["nm", "-D", "--defined-only", str(extracted / lib)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line.split()[-1] for line in out.splitlines() if " T TVMFFI" in line}


def referenced_symbols(wheel: Path, workdir: Path) -> dict[str, set[str]]:
    """TVMFFI* symbols each kernel .so in the wheel needs from libtvm_ffi."""
    needed: dict[str, set[str]] = {}
    scratch = workdir / "so"
    scratch.mkdir(exist_ok=True)
    with zipfile.ZipFile(wheel) as zf:
        members = [i for i in zf.infolist() if i.filename.endswith(".so")]
        if not members:
            sys.exit(f"{wheel.name} contains no .so files")
        print(f"scanning {len(members)} kernel libraries in {wheel.name}")
        target = scratch / "current.so"
        for info in members:
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out = subprocess.run(
                ["nm", "-D", "--undefined-only", str(target)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            syms = {line.split()[-1] for line in out.splitlines() if "TVMFFI" in line}
            if syms:
                needed[Path(info.filename).name] = syms
    return needed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="built flashinfer-jit-cache wheel")
    args = parser.parse_args()
    if subprocess.run(["which", "nm"], capture_output=True).returncode != 0:
        sys.exit("nm (binutils) is required")

    constraint = declared_constraint()
    print(f"declared constraint: {constraint}")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        needed = referenced_symbols(args.wheel, workdir)
        used = set().union(*needed.values()) if needed else set()
        print(f"kernels reference {len(used)} TVMFFI symbols: {sorted(used)}")

        failed = False
        for version in versions_to_check(constraint):
            exported = exported_symbols(version, workdir)
            missing = used - exported
            if missing:
                failed = True
                print(
                    f"\nFAIL apache-tvm-ffi {version} does not export {sorted(missing)}"
                )
                for name, syms in sorted(needed.items()):
                    if syms & missing:
                        print(f"       needed by {name}")
                        break  # one example is enough to locate the build
            else:
                print(
                    f"OK   apache-tvm-ffi {version} exports everything the kernels need"
                )

    if failed:
        print(
            "\nThis wheel was built against a newer apache-tvm-ffi than it advertises. "
            "Either raise the floor in requirements.txt or build against it "
            "(scripts/build_constraints.txt)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
