# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant GEMM modules' JIT disk-cache keys and build path, without a device.

The key closure of every GEMM module is its kernel sources and nothing of the mode's other surfaces (an edit to the
sampling, GDN, KDA, quantization or attention classes, or to the GEMM dispatcher, must keep every compiled GEMM object
valid).
"""

import pytest

import flashinfer.batch_invariant.jit as J

pytestmark = pytest.mark.cpu_only

#: Every GEMM cache module with the kernel sources its objects are built from (the compile roots, the bodies they import,
#: the specification and generator): all of them must be in the module's key.
GEMM_MODULES = {
    "batch_invariant_gemm": {
        "gemm/batch_invariant/blocks/compile.py",
        "gemm/batch_invariant/blocks/kernel.py",
        "gemm/batch_invariant/blocks/mainloop.py",
        "gemm/batch_invariant/blocks/producer.py",
        "gemm/batch_invariant/blocks/epilogue.py",
        "gemm/batch_invariant/blocks/operands.py",
        "gemm/batch_invariant/fixup.py",
    },
    "batch_invariant_gemm_blockscaled": {
        "gemm/batch_invariant/blocks/compile_blockscaled.py",
        "gemm/batch_invariant/blocks/blockscaled.py",
        "gemm/batch_invariant/blocks/epilogue.py",
        "gemm/batch_invariant/blocks/operands.py",
    },
    "batch_invariant_gemm_blockscaled_mixed": {
        "gemm/batch_invariant/blocks/compile_blockscaled_mixed.py",
        "gemm/batch_invariant/blocks/blockscaled_mx_mixed.py",
        "gemm/batch_invariant/blocks/compile_blockscaled.py",
        "gemm/batch_invariant/blocks/blockscaled.py",
        "gemm/batch_invariant/blocks/epilogue.py",
        "gemm/batch_invariant/blocks/operands.py",
    },
    "batch_invariant_gemm_fp8_blockscale": {
        "gemm/batch_invariant/blocks/compile_fp8bs.py",
        "gemm/batch_invariant/blocks/blockscale_fp8.py",
        "gemm/batch_invariant/blocks/mainloop.py",
        "gemm/batch_invariant/blocks/epilogue.py",
        "gemm/batch_invariant/blocks/operands.py",
    },
}

#: What every GEMM kernel composes besides its own blocks: the generator and specification, the engine's canonical plan
#: and floating-point interface, the vendored tile scheduler, and the key's own module (with the ``__init__`` files on the way).
SHARED_KERNEL_SOURCES = {
    "gemm/batch_invariant/generator.py",
    "gemm/batch_invariant/spec.py",
    "batch_invariant/canon.py",
    "batch_invariant/fp_interface/cute.py",
    "batch_invariant/fp_interface/term.py",
    "batch_invariant/jit.py",
    "cute_dsl/attention/fmha/helpers/static_persistent_tile_scheduler.py",
}

#: Where a GEMM key file may live; anything else in a key is a surface a GEMM kernel never enters.
KEY_PREFIXES = (
    "gemm/batch_invariant/",
    "batch_invariant/__init__.py",
    "batch_invariant/jit.py",
    "batch_invariant/canon.py",
    "batch_invariant/fp_interface/",
    "cute_dsl/attention/fmha/helpers/static_persistent_tile_scheduler.py",
)

#: Host-side code of the GEMM family: it selects or wraps kernels and never enters one.
GEMM_HOST_SOURCES = (
    "gemm/batch_invariant/api.py",
    "gemm/batch_invariant/schedule.py",
    "gemm/batch_invariant/heuristic.py",
    "gemm/batch_invariant/tuning.py",
)


@pytest.mark.parametrize("module", sorted(GEMM_MODULES))
def test_key_closure_is_the_kernel_sources(module):
    srcs = J.relative_sources(module)
    print(f"{module}: {len(srcs)} key files")
    for s in srcs:
        print(f"  {s}")
    assert GEMM_MODULES[module] | SHARED_KERNEL_SOURCES <= set(srcs)
    outside = [s for s in srcs if not s.startswith(KEY_PREFIXES)]
    assert not outside, outside  # no sampling / GDN / KDA / quantization / attention / MoE file
    host = [s for s in srcs if s in GEMM_HOST_SOURCES]
    assert not host, host


def test_package_inits_enter_the_key_unfollowed():
    """``from ....batch_invariant import jit`` names ``jit.py``; the package's ``__init__`` (which imports the mode's
    other surfaces) is in the key as a file on the way, and nothing it imports is."""
    srcs = set(J.relative_sources("batch_invariant_gemm"))
    assert {"batch_invariant/__init__.py", "batch_invariant/jit.py"} <= srcs
    assert "batch_invariant/support.py" not in srcs  # imported by the __init__ only


def test_import_resolution_names_the_file(tmp_path, monkeypatch):
    """The walk's rule on a small package: a submodule import names the submodule, a symbol import names the module
    defining it (a package's ``__init__`` only for its own symbols), and nothing else."""
    pkg = tmp_path / "mode"
    files = {
        "__init__.py": "from .surface import api\n",
        "surface.py": "api = 1\n",
        "defined.py": "SYMBOL = 1\n",
        "sub/__init__.py": "from .impl import THING\nfrom .. import surface\n",
        "sub/impl.py": "THING = 1\n",
        "sub/leaf.py": "LEAF = 1\n",
        "kernel.py": (
            "from . import defined\n"
            "from .sub import leaf\n"
            "from .sub import THING\n"
            "def body():\n"
            "    from .defined import SYMBOL\n"
        ),
    }
    for rel, text in files.items():
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(J, "MODE_DIRS", (pkg.resolve(),))
    got = {str(p.relative_to(pkg.resolve())) for p in J._relative_imports(pkg / "kernel.py")}
    assert got == {"defined.py", "sub/leaf.py", "sub/__init__.py"}
    # the package __init__ was named for THING, so its own imports are followed from it; the mode's top-level __init__
    # is only on the way and its surface import stays out
    got = {str(p.relative_to(pkg.resolve())) for p in J._relative_imports(pkg / "sub" / "__init__.py")}
    assert got == {"sub/impl.py", "surface.py"}
