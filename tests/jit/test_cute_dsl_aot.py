"""Runtime cache compatibility and numerical FP4 provider-loading tests."""

import json

import pytest
import torch

from flashinfer.jit import cute_dsl_core as cache
from flashinfer.jit import env as jit_env
from flashinfer.jit.cute_dsl_aot import _compile_aot_kernel
from flashinfer.jit.gemm.cute_dsl_fp4 import MM_FP4_OPERATION


def test_source_fingerprint_survives_install_relocation(tmp_path):
    trees = []
    for name in ("checkout", "site-packages"):
        tree = tmp_path / name
        (tree / "kernels").mkdir(parents=True)
        (tree / "entry.py").write_text("entry")
        (tree / "kernels" / "kernel.py").write_text("kernel")
        trees.append((tree / "entry.py", tree / "kernels" / "kernel.py"))
    assert cache._hash_source_files(trees[0]) == cache._hash_source_files(
        trees[1][::-1]
    )
    trees[1][1].write_text("changed")
    assert cache._hash_source_files(trees[0]) != cache._hash_source_files(trees[1])


@pytest.mark.parametrize(
    "mismatch", [None, "source_sha256", "cute_dsl_version", "arch", "corrupt"]
)
def test_provider_objects_are_loaded_read_only(monkeypatch, tmp_path, mismatch):
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "writable")
    spec = cache.JitSpecCuteDsl(
        "mm_fp4", "test", lambda: pytest.fail("compiled"), "source"
    )
    provider_dir = tmp_path / "installed"
    module_dir = provider_dir / spec.module_dir_name
    module_dir.mkdir(parents=True)
    obj = module_dir / spec.object_path.name
    obj.write_bytes(b"object")
    meta = dict(spec.expected_meta)
    if mismatch in meta:
        meta[mismatch] = "incompatible"
    (module_dir / "meta.json").write_text(json.dumps(meta))
    provider = jit_env.AOTProvider(
        "sm100a",
        "provider",
        "test",
        provider_dir,
        frozenset({"sm100a"}),
        frozenset({spec.module_dir_name}),
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", (provider,))

    def load(path):
        assert path == obj
        if mismatch == "corrupt":
            raise ValueError("invalid object")
        return "loaded"

    monkeypatch.setattr(spec, "_load_object", load)
    if mismatch:
        from flashinfer.jit import MissingJITCacheError

        with pytest.raises(MissingJITCacheError):
            spec.build_and_load()
    else:
        assert spec.is_compiled
        assert spec.build_and_load() == "loaded"
    assert not spec.module_dir.exists()
    assert obj.read_bytes() == b"object"


@pytest.mark.parametrize("use_nvfp4", [True, False])
@pytest.mark.parametrize("per_token", [False, True])
def test_fp4_runs_from_provider_with_jit_disabled(
    monkeypatch, tmp_path, use_nvfp4, per_token
):
    if not torch.cuda.is_available():
        pytest.skip("Requires SM100/103/107 GPU")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Requires SM100/103/107 GPU")
    if minor == 7 and per_token:
        pytest.skip("Public per-token alpha is not supported on Rubin")
    import cutlass.cute as cute
    from flashinfer.gemm import gemm_base
    from flashinfer.gemm.gemm_mm_fp4_cute_dsl import (
        _blockscaled_kernel_disk_name,
        _mm_fp4_cache_key,
    )

    arch = f"sm{major}{minor}a"
    monkeypatch.setenv("CUTE_DSL_ARCH", arch.replace("sm", "sm_", 1))
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "build")
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", ())
    tactic = ((128, 128), (1, 1), False, False, "sm100", None)

    config = _mm_fp4_cache_key(
        16 if use_nvfp4 else 32,
        tactic,
        False,
        torch.bfloat16,
        "m" if per_token else None,
        "b_ef",
    )
    name = _blockscaled_kernel_disk_name(config, 1, None)
    _compile_aot_kernel(MM_FP4_OPERATION, name, config, arch, tmp_path / "build")
    module_name = f"mm_fp4_{arch}_cute_dsl"
    provider = jit_env.AOTProvider(
        arch,
        "test-provider",
        "test",
        tmp_path / "build",
        frozenset({arch}),
        frozenset({module_name}),
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", (provider,))
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "empty")
    monkeypatch.setattr(gemm_base, "_CUTE_DSL_MM_FP4_KERNEL_CACHE", {})
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    monkeypatch.setattr(
        cute, "compile", lambda *args, **kwargs: pytest.fail("JIT compiled")
    )
    # FP4 code 2 is 1.0; E4M3 code 56 and E8M0 code 127 are both 1.0.
    a = torch.full((128, 256), 0x22, dtype=torch.uint8, device="cuda")
    b = a.clone()
    scales = torch.full(
        (128, 32), 56 if use_nvfp4 else 127, dtype=torch.uint8, device="cuda"
    )
    alpha = torch.linspace(0.5, 1.0, 128 if per_token else 1, device="cuda")
    out = torch.empty((128, 128), dtype=torch.bfloat16, device="cuda")
    runner = gemm_base._cute_dsl_gemm_fp4_runner(
        major, minor, False, out.dtype, use_nvfp4
    )
    runner.forward(
        (a, b.T, scales, scales.T, alpha, None, out, None, None, None), tactic=tactic
    )
    expected = (
        (512 * alpha).expand(128, 128).T if per_token else torch.full_like(out, 256)
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.01, atol=0)
    assert not (tmp_path / "empty" / module_name).exists()
