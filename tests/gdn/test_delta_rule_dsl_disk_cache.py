"""Delta-rule DSL kernels compiled with TVM-FFI go through the CuTe-DSL disk cache."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

cute = pytest.importorskip("cutlass.cute")

from flashinfer.gdn_kernels.delta_rule_dsl import custom_compile_cache as ccc  # noqa: E402
from flashinfer.jit import cute_dsl_core  # noqa: E402
from flashinfer.utils import is_sm12x_supported, is_sm90a_supported  # noqa: E402


class _Kernel(ccc.KeyedCompileMixin):
    def __init__(self, tile):
        """Kernel stub keyed on ``tile``."""
        self.tile = tile
        self.manual_cache_key("tile")


SM12X = (cute.EnableTVMFFI(True), cute.GPUArch("sm_121a"))


@pytest.fixture(autouse=True)
def _clear_mem_cache():
    """Isolate each test from kernels cached by the others."""
    ccc._in_mem_compile_cache.clear()
    yield
    ccc._in_mem_compile_cache.clear()


def _key(kernel):
    """Compile key as built by cached_compile without compile options."""
    return (kernel._get_compile_key(), None)


def test_disk_kernel_name_is_stable_and_specific():
    """Equal configurations share a name; different ones do not."""
    name = ccc._disk_kernel_name(_Kernel(64), _key(_Kernel(64)))
    assert name == ccc._disk_kernel_name(_Kernel(64), _key(_Kernel(64)))
    assert name != ccc._disk_kernel_name(_Kernel(128), _key(_Kernel(128)))


def test_disk_kernel_name_rejects_address_dependent_keys():
    """Keys whose repr carries an object address are not persisted."""
    assert ccc._disk_kernel_name(_Kernel(64), ((object(),), None)) is None


@pytest.mark.parametrize(
    "options, uses_disk",
    [
        (SM12X, True),
        ((cute.EnableTVMFFI(False), cute.GPUArch("sm_121a")), False),
        ((cute.EnableTVMFFI(True), cute.GPUArch("sm_90a")), True),
        ((cute.EnableTVMFFI(True), cute.GPUArch("sm_80")), True),
    ],
)
def test_disk_cache_only_with_tvm_ffi(monkeypatch, options, uses_disk):
    """Only TVM-FFI compiles go through the disk cache, once per key."""
    calls = []

    def fake_build_and_load(module, name, compile_fn, key_files):
        """Record the call and compile in place."""
        calls.append((module, name, key_files))
        return compile_fn()

    monkeypatch.setattr(
        cute_dsl_core, "build_and_load_cute_dsl_kernel", fake_build_and_load
    )
    monkeypatch.setattr(ccc, "_compile_uncached", lambda *a, **k: "compiled")
    for _ in range(2):
        assert ccc.cached_compile(_Kernel(64), compile_options=options) == "compiled"
    if not uses_disk:
        assert calls == []
        return
    assert len(calls) == 1
    module, name, key_files = calls[0]
    assert module == ccc._DISK_CACHE_MODULE and name.startswith("_Kernel_")
    assert any(f.endswith("delta_rule_sm120.py") for f in key_files)


def test_disk_module_follows_the_kernel_class_hierarchy():
    """Subclasses outside delta_rule_dsl key on their sources, in their own module."""
    from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill

    module, files = ccc._disk_module(
        object.__new__(gdn_cp_prefill.CPDeltaRuleFixupSm100)
    )
    assert module == ccc._DISK_CACHE_MODULE + "_blackwell"
    assert gdn_cp_prefill.__file__ in files
    assert any(f.endswith("delta_rule_cp_sm120.py") for f in files)
    base_module, base_files = ccc._disk_module(_Kernel(64))
    assert base_module == ccc._DISK_CACHE_MODULE
    assert gdn_cp_prefill.__file__ not in base_files


def _use_tmp_jit_dir(monkeypatch, tmp_path):
    """Point the CuTe-DSL disk cache and its locks at ``tmp_path``."""
    monkeypatch.setattr(cute_dsl_core.jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    monkeypatch.setattr(cute_dsl_core, "get_tmpdir", lambda: tmp_path)
    # JitSpecCuteDsl names the module after the arch; keep these tests GPU-free.
    monkeypatch.setattr(cute_dsl_core, "_get_compile_arch", lambda: "sm121a")


def test_disabled_disk_cache_compiles_without_artifacts(monkeypatch, tmp_path):
    """FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 compiles but writes nothing."""
    _use_tmp_jit_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    monkeypatch.setattr(ccc, "_compile_uncached", lambda *a, **k: "compiled")
    assert ccc.cached_compile(_Kernel(64), compile_options=SM12X) == "compiled"
    assert not any(tmp_path.rglob("*.o"))


def test_failed_compile_or_ptx_check_exports_nothing(monkeypatch, tmp_path):
    """A failing compile or PTX check propagates and exports nothing."""
    _use_tmp_jit_dir(monkeypatch, tmp_path)
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)

    def reject(*a, **k):
        """Stand-in for a compile rejected by the SM12x PTX check."""
        raise RuntimeError("unsupported cluster-scoped TMA loads")

    monkeypatch.setattr(ccc, "_compile_uncached", reject)
    with pytest.raises(RuntimeError, match="cluster-scoped TMA"):
        ccc.cached_compile(_Kernel(64), compile_options=SM12X)
    assert not any(tmp_path.rglob("*.o"))
    assert ccc._in_mem_compile_cache == {}


_PREFILL_SCRIPT = textwrap.dedent(
    """
    import sys, torch
    if sys.argv[2] == "forbid_compile":
        import cutlass.cute as cute

        class _NoCompile:
            def __getitem__(self, _):
                return self

            def __call__(self, *a, **k):
                raise AssertionError("kernel was recompiled")

        cute.compile = _NoCompile()
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    torch.manual_seed(0)
    seq_lens, heads, d = [61, 123], 4, 128
    t = sum(seq_lens)
    with torch.device("cuda"):
        q, k, v = (torch.randn(t, heads, d, dtype=torch.bfloat16) for _ in range(3))
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        alpha, beta = torch.rand(t, heads), torch.rand(t, heads)
        cu = torch.tensor([0, 61, 184], dtype=torch.int64)
        o = torch.empty(t, heads, d, dtype=torch.bfloat16)
        s = torch.empty(2, heads, d, d, dtype=torch.float32)
    chunk_gated_delta_rule(
        q, k, v, alpha, beta, d**-0.5, None, True, cu,
        use_qk_l2norm_in_kernel=False, output=o, output_state=s, max_seqlen=123,
    )
    torch.cuda.synchronize()
    torch.save({"o": o.cpu(), "s": s.cpu()}, sys.argv[1])
    """
)


def _delta_rule_dsl_device() -> bool:
    """Whether the current GPU runs one of the delta-rule DSL paths."""
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    major = torch.cuda.get_device_capability(device)[0]
    return is_sm12x_supported(device) or is_sm90a_supported(device) or major == 8


@pytest.mark.skipif(
    not _delta_rule_dsl_device(), reason="Requires an SM8x, SM90 or SM12x GPU"
)
def test_second_process_reloads_without_compiling(tmp_path):
    """A second process reloads the exported kernel without compiling, bitwise equal."""
    env = {**os.environ, "FLASHINFER_WORKSPACE_BASE": str(tmp_path / "ws")}
    env.pop("FLASHINFER_CUTE_DSL_DISABLE_CACHE", None)
    script = tmp_path / "prefill.py"
    script.write_text(_PREFILL_SCRIPT)
    out = [tmp_path / "first.pt", tmp_path / "second.pt"]
    subprocess.run([sys.executable, script, out[0], "compile"], env=env, check=True)
    artifacts = list((tmp_path / "ws").rglob("gdn_delta_rule_dsl_*/*.o"))
    assert artifacts, "no SM12x delta-rule kernel was exported"
    subprocess.run(
        [sys.executable, script, out[1], "forbid_compile"], env=env, check=True
    )
    first, second = (torch.load(p) for p in out)
    assert torch.equal(first["o"], second["o"])
    assert torch.equal(first["s"], second["s"])
