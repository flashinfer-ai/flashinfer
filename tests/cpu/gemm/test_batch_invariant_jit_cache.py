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
        "gemm/batch_invariant/blocks/scheduler.py",
        "gemm/batch_invariant/fixup.py",
    },
    "batch_invariant_gemm_blockscaled": {
        "gemm/batch_invariant/blocks/compile_blockscaled.py",
        "gemm/batch_invariant/blocks/blockscaled.py",
        "gemm/batch_invariant/blocks/epilogue.py",
        "gemm/batch_invariant/blocks/operands.py",
        "gemm/batch_invariant/blocks/scheduler.py",
    },
    "batch_invariant_gemm_blockscaled_mixed": {
        "gemm/batch_invariant/blocks/compile_blockscaled_mixed.py",
        "gemm/batch_invariant/blocks/blockscaled_mx_mixed.py",
        "gemm/batch_invariant/blocks/compile_blockscaled.py",
        "gemm/batch_invariant/blocks/blockscaled.py",
        "gemm/batch_invariant/blocks/epilogue.py",
        "gemm/batch_invariant/blocks/operands.py",
        "gemm/batch_invariant/blocks/scheduler.py",
    },
    "batch_invariant_gemm_fp8_blockscale": {
        "gemm/batch_invariant/blocks/compile_fp8bs.py",
        "gemm/batch_invariant/blocks/blockscale_fp8.py",
        "gemm/batch_invariant/blocks/mainloop.py",
        "gemm/batch_invariant/blocks/epilogue.py",
        "gemm/batch_invariant/blocks/operands.py",
        "gemm/batch_invariant/blocks/scheduler.py",
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
    assert not outside, (
        outside
    )  # no sampling / GDN / KDA / quantization / attention / MoE file
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
    got = {
        str(p.relative_to(pkg.resolve()))
        for p in J._relative_imports(pkg / "kernel.py")
    }
    assert got == {"defined.py", "sub/leaf.py", "sub/__init__.py"}
    # the package __init__ was named for THING, so its own imports are followed from it; the mode's top-level __init__
    # is only on the way and its surface import stays out
    got = {
        str(p.relative_to(pkg.resolve()))
        for p in J._relative_imports(pkg / "sub" / "__init__.py")
    }
    assert got == {"sub/impl.py", "surface.py"}


def test_build_and_load_compiles_the_exported_object_without_the_engine(monkeypatch):
    """A compile closure forwarding the DSL's keyword options gets the engine-less compile for the exported object; one
    without them keeps the engine (today's path)."""
    got = {}

    def fake_build_and_load(module, kernel, compile_fn, extra_key_files=(), **kw):
        got.update(kw, module=module, kernel=kernel, files=tuple(extra_key_files))
        return "loaded"

    monkeypatch.setattr(J, "build_and_load_cute_dsl_kernel", fake_build_and_load)
    seen = []

    def compile_fn(**compile_options):
        seen.append(compile_options)
        return "compiled"

    assert J.build_and_load("batch_invariant_gemm", "k", compile_fn) == "loaded"
    assert got["files"] == tuple(
        str(p) for p in J.module_sources("batch_invariant_gemm")
    )
    assert got["source_sha256"] == J.module_source_sha256("batch_invariant_gemm")
    assert got["export_compile_fn"]() == "compiled" and seen == [
        {"no_jit_engine": True}
    ]
    assert J.build_and_load("batch_invariant_gemm", "k", lambda: "compiled") == "loaded"
    assert got["export_compile_fn"] is None


def test_key_hash_is_computed_once_per_process(monkeypatch):
    """The key files are read and hashed once per module and process, not once per kernel; a different file list is a
    different hash."""
    calls = []
    real = J._hash_source_files

    def counting(paths):
        calls.append(paths)
        return real(paths)

    monkeypatch.setattr(J, "_hash_source_files", counting)
    J.module_source_sha256.cache_clear()
    try:
        hashes = {J.module_source_sha256("batch_invariant_gemm") for _ in range(8)}
        assert len(hashes) == 1 and len(calls) == 1
        assert J.module_source_sha256("batch_invariant_gemm_blockscaled") not in hashes
        assert len(calls) == 2
    finally:
        J.module_source_sha256.cache_clear()


#: The other kernel modules the walk keys (their keys shrink with the same resolution rule): the compile roots and bodies
#: that must stay in the key, and the surfaces of the mode that must not be in it.
OTHER_MODULES = {
    "batch_invariant_attention": (
        {
            "attention/batch_invariant/blocks/compile.py",
            "attention/batch_invariant/blocks/kernel.py",
            "attention/batch_invariant/blocks/producer.py",
            "attention/batch_invariant/blocks/mma.py",
            "attention/batch_invariant/blocks/softmax.py",
            "attention/batch_invariant/blocks/correction.py",
            "attention/batch_invariant/swaps/compile.py",
            "attention/batch_invariant/swaps/kernel.py",
            "attention/batch_invariant/fixup.py",
            "attention/batch_invariant/plan.py",
            "attention/batch_invariant/spec.py",
            "batch_invariant/canon.py",
            "batch_invariant/fp_interface/cute.py",
            "batch_invariant/jit.py",
        },
        (
            "batch_invariant/sampling/",
            "batch_invariant/gdn/",
            "batch_invariant/kda/",
            "batch_invariant/quantization/",
            "fused_moe/",
        ),
    ),
    "batch_invariant_moe": (
        {
            "fused_moe/batch_invariant/blocks/compile.py",
            "fused_moe/batch_invariant/blocks/kernel.py",
            "fused_moe/batch_invariant/blocks/epilogue.py",
            "fused_moe/batch_invariant/blocks/operands.py",
            "fused_moe/batch_invariant/blocks/scheduler.py",
            "fused_moe/batch_invariant/blocks/swap.py",
            "fused_moe/batch_invariant/blocks/epilogue_mx.py",
            "fused_moe/batch_invariant/blocks/epilogue_fp8bs.py",
            "fused_moe/batch_invariant/blocks/fp8bs_small.py",
            "fused_moe/batch_invariant/route.py",
            "fused_moe/batch_invariant/route_plan.py",
            "fused_moe/batch_invariant/permute.py",
            "fused_moe/batch_invariant/permute_nvfp4.py",
            "fused_moe/batch_invariant/mx_permute.py",
            "fused_moe/batch_invariant/combine.py",
            "fused_moe/batch_invariant/spec.py",
            "fused_moe/batch_invariant/spec_fp8.py",
            "fused_moe/batch_invariant/spec_nvfp4.py",
            "fused_moe/batch_invariant/spec_mxfp8.py",
            "fused_moe/batch_invariant/spec_mxfp4w8a.py",
            "fused_moe/batch_invariant/spec_fp8bs.py",
            "gemm/batch_invariant/blocks/blockscaled.py",
            "gemm/batch_invariant/blocks/blockscaled_mx_mixed.py",
            "gemm/batch_invariant/blocks/blockscale_fp8.py",
            "gemm/batch_invariant/blocks/mainloop.py",
            "gemm/batch_invariant/blocks/operands.py",
            "batch_invariant/canon.py",
            "batch_invariant/fp_interface/cute.py",
            "batch_invariant/jit.py",
        },
        ("batch_invariant/gdn/", "batch_invariant/kda/", "attention/"),
    ),
    # the NVFP4 MoE classes' row-mapped quantizer: the quantization kernel sources and nothing of the GEMM or MoE bodies
    "batch_invariant_moe_nvfp4_quant": (
        {
            "fused_moe/batch_invariant/blocks/quant_nvfp4.py",
            "batch_invariant/quantization/nvfp4_kernel.py",
            "batch_invariant/quantization/nvfp4_spec.py",
            "batch_invariant/quantization/blocks.py",
            "batch_invariant/canon.py",
            "batch_invariant/fp_interface/cute.py",
            "batch_invariant/jit.py",
        },
        (
            "batch_invariant/gdn/",
            "batch_invariant/kda/",
            "batch_invariant/sampling/",
            "attention/",
            "gemm/",
            "fused_moe/batch_invariant/blocks/kernel.py",
            "fused_moe/batch_invariant/blocks/swap.py",
        ),
    ),
}


@pytest.mark.parametrize("module", sorted(OTHER_MODULES))
def test_other_kernel_modules_keep_their_sources_and_drop_the_surfaces(module):
    """The attention and MoE keys shrink under the same rule: their kernel sources stay, the mode's other surfaces go."""
    required, forbidden = OTHER_MODULES[module]
    srcs = set(J.relative_sources(module))
    missing = required - srcs
    assert not missing, missing
    outside = sorted(s for s in srcs if s.startswith(forbidden))
    assert not outside, outside
    host = [s for s in srcs if s in GEMM_HOST_SOURCES]
    assert not host, host
