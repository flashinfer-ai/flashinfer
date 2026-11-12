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

# The vendored source trees of the MOSCA kernels and how they ship: the
# installed tree matches its manifest and carries no MOSCA change, the record of
# the MOSCA kernel derived from upstream matches the file (macro regions
# included, and the README's copy of them), the upstream re-check logic, and
# the packaging rules (package data, release-wheel exclusion, formatter
# exclusion). No GPU needed.

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import tomllib
from pathlib import Path

import pytest

from flashinfer.experimental import mosca
from flashinfer.experimental.mosca import _vendor
from flashinfer.experimental.mosca.megamoe import jit as mosca_jit

PACKAGE = Path(mosca.__file__).resolve().parent
REPO = PACKAGE.parents[2]
TREE = _vendor.vendored_tree("deep_gemm")
KERNEL = "csrc/megamoe/sm100_fp8_fp4_mega_moe.cuh"
DEEPGEMM_PORTIONS = (
    "Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek, MIT licensed"
)
MOSCA_MACRO = re.compile(r"\bDGX_\w+|\bDG_MEGA_PHASE_TIMER\b|\bDG_MPT\b")


def _needs_sources() -> None:
    if not TREE.installed():
        pytest.skip("experimental kernel sources are not installed (release wheel)")


def test_installed_tree_matches_its_manifest() -> None:
    _needs_sources()
    assert TREE.problems() == []
    assert len(TREE.manifest["files"]) == 17
    assert (
        TREE.manifest["upstream"]["commit"]
        == "1ff3932f0f9d9187b92afb476da4e3ade10a6575"
    )
    # The changes are code of this package, not a patch series on the tree.
    assert "patches" not in TREE.manifest
    assert "mosca_changes" not in TREE.manifest


def test_vendored_tree_carries_no_mosca_change() -> None:
    _needs_sources()
    for path in TREE.files:
        assert not MOSCA_MACRO.search(path.read_text()), path


def _conditional_blocks(text: str) -> dict[str, list[str]]:
    """``macro -> ['first-last', ...]`` for every #ifdef/#ifndef block, 1-based lines."""
    blocks: dict[str, list[str]] = {}
    stack: list[tuple[int, str | None]] = []
    for number, line in enumerate(text.split("\n"), 1):
        m = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b\s*(\w*)", line)
        if not m:
            continue
        directive, name = m.groups()
        if directive in ("ifdef", "ifndef"):
            stack.append((number, name))
        elif directive == "if":
            stack.append((number, None))
        elif directive == "endif":
            start, macro = stack.pop()
            if macro:
                blocks.setdefault(macro, []).append(f"{start}-{number}")
    return blocks


def test_derived_kernel_record_matches_the_file() -> None:
    _needs_sources()
    assert set(TREE.derived) == {KERNEL}
    info = TREE.derived[KERNEL]
    text = (PACKAGE / KERNEL).read_text()
    digest = hashlib.sha256((PACKAGE / KERNEL).read_bytes()).hexdigest()
    assert digest != info["upstream_sha256"]
    blocks = _conditional_blocks(text)
    assert set(info["macros"]) == set(info["regions"])
    for macro, regions in info["regions"].items():
        assert blocks.get(macro) == regions, macro
    # Every MOSCA macro the kernel branches on is a switch the JIT emits, and
    # nothing else (no diagnostics, no timer).
    branched = {m for m in blocks if MOSCA_MACRO.fullmatch(m)}
    assert branched == set(info["macros"]) == set(mosca_jit.FEATURE_MACROS.values())
    assert not re.search(r"\bDG_MEGA_PHASE_TIMER\b|\bDG_MPT\b|NOFENCE", text)
    # FlashInfer's license header with its line for derived code, then the
    # upstream file and commit.
    head = text[: text.index("#pragma once")]
    assert "Copyright (c) 2026 by FlashInfer team." in head
    assert DEEPGEMM_PORTIONS in head and "LICENSE.DeepGEMM" in head
    assert info["upstream"] in head
    assert TREE.manifest["upstream"]["commit"] in head


def test_derived_and_ported_files_carry_the_deepgemm_notice() -> None:
    # FlashInfer's convention for code derived from DeepGEMM: the Apache header
    # of the file plus the "Portions derived from DeepGEMM" line.
    targets = set(TREE.derived)
    for info in TREE.ported.values():
        targets.update(info["ported_to"].split(", "))
    assert "megamoe/staging.py" in targets
    for rel in sorted(targets):
        path = PACKAGE / rel
        if rel.startswith("csrc/") and not TREE.installed():
            continue
        head = path.read_text()[:2000]
        assert DEEPGEMM_PORTIONS in head, rel
        assert "LICENSE.DeepGEMM" in head, rel


def test_readme_shows_the_recorded_regions() -> None:
    readme = (PACKAGE / "README.md").read_text()
    for info in TREE.derived.values():
        for macro, regions in info["regions"].items():
            for region in regions:
                assert region in readme, (macro, region)


def test_upstream_recheck_logic(tmp_path) -> None:
    """compare_with_upstream on a synthetic tree: vendored, derived and ported files."""
    vendored, kernel, host = b"same\n", b"upstream kernel\n", b"host\n"
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    package = tmp_path / "pkg"
    (package / "csrc").mkdir(parents=True)
    (package / "csrc/k.cuh").write_bytes(b"upstream kernel\nwith a change\n")
    tree = _vendor.VendoredTree(
        name="t",
        manifest={
            "upstream": {"include_root": "inc"},
            "files": {"t/a.h": sha(vendored)},
            "derived_sources": {
                "files": {
                    "csrc/k.cuh": {
                        "upstream": "inc/t/k.cuh",
                        "upstream_sha256": sha(kernel),
                    }
                }
            },
            "ported_host_sources": {
                "files": {"src/h.hpp": {"sha256": sha(host), "ported_to": "csrc/x.cu"}}
            },
        },
        package_dir=package,
    )
    upstream = tmp_path / "upstream"
    (upstream / "inc/t").mkdir(parents=True)
    (upstream / "src").mkdir()
    (upstream / "inc/t/a.h").write_bytes(vendored)
    (upstream / "inc/t/k.cuh").write_bytes(kernel)
    (upstream / "src/h.hpp").write_bytes(host)
    assert tree.compare_with_upstream(upstream) == []
    diff = tree.diff_derived(upstream)
    assert "+with a change" in diff and "upstream/inc/t/k.cuh" in diff
    (upstream / "inc/t/a.h").write_bytes(b"drifted\n")
    (upstream / "inc/t/k.cuh").write_bytes(b"new upstream kernel\n")
    (upstream / "src/h.hpp").unlink()
    problems = tree.compare_with_upstream(upstream)
    assert any("t/a.h is not byte-identical" in p for p in problems)
    assert any("merge it into csrc/k.cuh" in p for p in problems)
    assert any("ported source src/h.hpp is not in the checkout" in p for p in problems)


def _repo_file(rel: str) -> Path:
    path = REPO / rel
    if not path.is_file():
        pytest.skip(f"{rel} is only in a source checkout")
    return path


def test_package_data_ships_sources_manifests_and_license() -> None:
    data = tomllib.loads(_repo_file("pyproject.toml").read_text())
    globs = data["tool"]["setuptools"]["package-data"]["flashinfer.experimental.mosca"]
    for required in (
        "csrc/**",
        "vendor/*.json",
        "LICENSE.DeepGEMM",
        "README.md",
        "VENDOR.md",
    ):
        assert required in globs, required
    assert (PACKAGE / "LICENSE.DeepGEMM").is_file()


def test_release_wheels_drop_the_sources_but_keep_the_manifests() -> None:
    path = _repo_file("build_backend.py")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "flashinfer_build_backend_mosca", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rule = module._EXPERIMENTAL_SOURCES_RE
    assert rule.match("flashinfer/experimental/mosca/csrc/vendor/deep_gemm/impls/x.cuh")
    assert rule.match(
        "flashinfer/experimental/mosca/csrc/megamoe/sm100_fp8_fp4_launcher.cu"
    )
    assert rule.match(f"flashinfer/experimental/mosca/{KERNEL}")
    assert not rule.match("flashinfer/experimental/mosca/vendor/deep_gemm.json")
    assert not rule.match("flashinfer/experimental/mosca/megamoe/backend.py")


def test_formatters_skip_the_vendored_trees() -> None:
    config = _repo_file(".pre-commit-config.yaml").read_text()
    top_level = re.search(r"^exclude: (.+)$", config, re.M)
    assert top_level, "the global pre-commit exclude moved"
    pattern = re.compile(top_level.group(1).strip())
    vendored = "flashinfer/experimental/mosca/csrc/vendor/deep_gemm/common/math.cuh"
    assert pattern.match(vendored)
    # The derived kernel keeps upstream's layout so upstream changes merge cleanly.
    assert pattern.match(f"flashinfer/experimental/mosca/{KERNEL}")
    # FlashInfer-side code of the package stays formatted.
    assert not pattern.match(
        "flashinfer/experimental/mosca/csrc/megamoe/sm100_fp8_fp4_launcher.cu"
    )


def test_ported_host_sources_are_pinned() -> None:
    ported = TREE.manifest["ported_host_sources"]["files"]
    assert set(ported) >= {
        "csrc/apis/mega_moe.hpp",
        "csrc/jit_kernels/heuristics/mega_moe.hpp",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "deep_gemm/mega/__init__.py",
    }
    for rel, info in ported.items():
        assert re.fullmatch(r"[0-9a-f]{64}", info["sha256"]), rel
        for target in info["ported_to"].split(", "):
            if target.startswith("csrc/") and not TREE.installed():
                continue
            assert (PACKAGE / target).is_file(), (rel, target)
    toolchain = TREE.manifest["toolchain"]
    assert (
        toolchain["deep_gemm_cutlass_submodule"]
        != toolchain["flashinfer_cutlass_submodule"]
    )
