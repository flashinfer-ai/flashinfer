"""
Copyright (c) 2026 by FlashInfer team.

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

# Vendored upstream source trees of the MOSCA kernels (see VENDOR.md).
#
# Each tree is a verbatim, byte-identical copy of an upstream source subset at
# csrc/vendor/<tree>/, described by vendor/<tree>.json: upstream repository and
# commit, and the sha256 of every file. No vendored file carries a MOSCA change.
# csrc/vendor/ is the include root, so sources include ``<deep_gemm/...>``
# exactly as upstream does.
#
# MOSCA's changes live in files this package owns. The manifest lists the ones
# copied from an upstream file and then modified (``derived_sources``, with the
# upstream hash and the line regions of each MOSCA macro) and the upstream host
# files that the package re-implements (``ported_host_sources``).
#
# csrc/ sits directly under this package on purpose: release wheels drop
# ``flashinfer/experimental/<package>/csrc/**`` (build_backend.py), and a
# deeper csrc/ would not match that rule.
#
# Check the manifest against a checkout of the upstream repository at the
# manifest's commit, and optionally print each derived file's diff from upstream:
#   python -m flashinfer.experimental.mosca._vendor deep_gemm <upstream checkout> [--diff]

from __future__ import annotations

import difflib
import functools
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
CSRC_DIR = PACKAGE_DIR / "csrc"
#: Include root of every vendored tree.
VENDOR_INCLUDE_DIR = CSRC_DIR / "vendor"
MANIFEST_DIR = PACKAGE_DIR / "vendor"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class VendoredTree:
    """One vendored source tree and its manifest."""

    name: str
    manifest: dict[str, Any]
    #: Directory that ``derived_sources`` paths are relative to.
    package_dir: Path = PACKAGE_DIR

    @property
    def root(self) -> Path:
        return VENDOR_INCLUDE_DIR / self.name

    @property
    def files(self) -> tuple[Path, ...]:
        """Every file of the tree, from the manifest, in a stable order."""
        return tuple(VENDOR_INCLUDE_DIR / rel for rel in sorted(self.manifest["files"]))

    @property
    def include_root(self) -> str:
        """Upstream directory that the tree's paths are relative to."""
        return str(self.manifest["upstream"]["include_root"])

    @property
    def derived(self) -> dict[str, Any]:
        """Package files derived from an upstream file, with their record."""
        return dict(self.manifest.get("derived_sources", {}).get("files", {}))

    @property
    def ported(self) -> dict[str, Any]:
        """Upstream host files the package re-implements, with their record."""
        return dict(self.manifest.get("ported_host_sources", {}).get("files", {}))

    def installed(self) -> bool:
        """Whether the sources are present (release wheels exclude them)."""
        return self.root.is_dir()

    def problems(self) -> list[str]:
        """Differences between the installed tree and its manifest (empty if none)."""
        if not self.installed():
            return [f"{self.root} is missing"]
        out: list[str] = []
        expected = set(self.manifest["files"])
        on_disk = {
            p.relative_to(VENDOR_INCLUDE_DIR).as_posix()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        out += [f"unexpected file {rel}" for rel in sorted(on_disk - expected)]
        out += [f"missing file {rel}" for rel in sorted(expected - on_disk)]
        for rel in sorted(expected & on_disk):
            if _sha256(VENDOR_INCLUDE_DIR / rel) != self.manifest["files"][rel]:
                out.append(f"sha256 mismatch {rel}")
        prefix = self.include_root + "/"
        for rel, info in sorted(self.derived.items()):
            if not (self.package_dir / rel).is_file():
                out.append(f"derived source {rel} is missing")
            if info["upstream"].removeprefix(prefix) in expected:
                out.append(f"{info['upstream']} is both vendored and derived ({rel})")
        return out

    def compare_with_upstream(self, upstream_root: Path) -> list[str]:
        """Check the manifest against an upstream checkout at the manifest's commit.

        Every vendored file must be byte-identical to upstream; the upstream file
        of every derived source and every ported host file must still have its
        recorded hash. An empty result proves the vendored tree carries no local
        change and that nothing the package derived from moved.
        """
        out: list[str] = []
        root = Path(upstream_root)
        for rel, digest in sorted(self.manifest["files"].items()):
            path = root / self.include_root / rel
            if not path.is_file():
                out.append(f"{rel} is not in the upstream checkout")
            elif _sha256(path) != digest:
                out.append(f"{rel} is not byte-identical to upstream")
        for rel, info in sorted(self.derived.items()):
            path = root / info["upstream"]
            if not path.is_file():
                out.append(
                    f"{info['upstream']} (derived: {rel}) is not in the checkout"
                )
            elif _sha256(path) != info["upstream_sha256"]:
                out.append(f"{info['upstream']} changed upstream: merge it into {rel}")
        for rel, info in sorted(self.ported.items()):
            path = root / rel
            if not path.is_file():
                out.append(f"ported source {rel} is not in the checkout")
            elif _sha256(path) != info["sha256"]:
                out.append(f"ported source {rel} changed (port: {info['ported_to']})")
        return out

    def diff_derived(self, upstream_root: Path) -> str:
        """Unified diff of every derived source against its upstream file."""
        chunks: list[str] = []
        for rel, info in sorted(self.derived.items()):
            upstream = (Path(upstream_root) / info["upstream"]).read_text()
            ours = (self.package_dir / rel).read_text()
            chunks += difflib.unified_diff(
                upstream.splitlines(keepends=True),
                ours.splitlines(keepends=True),
                fromfile=f"upstream/{info['upstream']}",
                tofile=f"mosca/{rel}",
            )
        return "".join(chunks)


@functools.cache
def vendored_tree(name: str) -> VendoredTree:
    manifest = json.loads((MANIFEST_DIR / f"{name}.json").read_text())
    if manifest.get("tree") != name:
        raise ValueError(f"vendor/{name}.json describes tree {manifest.get('tree')!r}")
    return VendoredTree(name=name, manifest=manifest)


def _main(argv: list[str]) -> int:
    show_diff = "--diff" in argv
    args = [a for a in argv if a != "--diff"]
    if len(args) != 2:
        print(
            "usage: python -m flashinfer.experimental.mosca._vendor TREE "
            "UPSTREAM_CHECKOUT [--diff]"
        )
        return 2
    tree = vendored_tree(args[0])
    upstream = Path(args[1])
    issues = tree.problems() + tree.compare_with_upstream(upstream)
    if show_diff and not issues:
        sys.stdout.write(tree.diff_derived(upstream))
    for issue in issues:
        print(issue)
    print(f"{tree.name}: {'OK' if not issues else f'{len(issues)} problem(s)'}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "CSRC_DIR",
    "MANIFEST_DIR",
    "PACKAGE_DIR",
    "VENDOR_INCLUDE_DIR",
    "VendoredTree",
    "vendored_tree",
]
