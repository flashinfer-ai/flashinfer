"""Linear-attention CuTe-DSL kernels persist through the CuTe-DSL disk cache."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

from flashinfer.gdn_kernels import cute_dsl_cache_naming as naming  # noqa: E402
from flashinfer.jit import cute_dsl_core  # noqa: E402


def test_name_accepts_cutlass_dtypes():
    """Cutlass numeric classes (as passed by the Mamba SSD factory) format to names."""
    name = naming.make_kernel_name("k", cutlass.BFloat16, cutlass.Float32, 128, None)
    assert name == "k_BFloat16_Float32_128_none"


def test_persist_names_kernel_by_every_argument(monkeypatch):
    """The wrapped factory runs only inside the disk cache, named by all arguments."""
    seen = []

    def fake_build(module, name, compile_fn, extra_key_files=()):
        seen.append((module, name, tuple(extra_key_files)))
        return compile_fn()

    monkeypatch.setattr(cute_dsl_core, "build_and_load_cute_dsl_kernel", fake_build)

    @naming.persist_cute_dsl_compile("mod", ("a.py",))
    def _get_thing(rows, flag=False, dtype=torch.bfloat16):
        return ("compiled", rows, flag, dtype)

    assert _get_thing(16) == ("compiled", 16, False, torch.bfloat16)
    assert _get_thing(16, dtype=torch.float32, flag=True)[3] is torch.float32
    assert seen == [
        ("mod", "get_thing_16_0_bfloat16", ("a.py",)),
        ("mod", "get_thing_16_1_float32", ("a.py",)),
    ]


_RECURRENT_KDA_SCRIPT = textwrap.dedent(
    """
    import importlib
    import sys
    import cutlass.cute as cute

    rk = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")

    if sys.argv[1] == "forbid_compile":
        def _boom(*args, **kwargs):
            raise AssertionError("compiled instead of reloading from disk")
        cute.compile = _boom
    fn = rk._get_compiled_kernel(
        128, 1, 1, 1, 0, 0, 0, 0, 1, 16, rk.DOT_REDUCTION_DUAL_ACCUM, 0, 0, 0
    )
    assert callable(fn)
    """
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires an SM100-family GPU",
)
def test_second_process_reloads_recurrent_kda_without_compiling(tmp_path):
    """A second process loads the exported register-tile kernel without compiling."""
    env = {**os.environ, "FLASHINFER_WORKSPACE_BASE": str(tmp_path / "ws")}
    env.pop("FLASHINFER_CUTE_DSL_DISABLE_CACHE", None)
    script = tmp_path / "recurrent_kda_factory.py"
    script.write_text(_RECURRENT_KDA_SCRIPT)
    subprocess.run([sys.executable, script, "compile"], env=env, check=True)
    assert list((tmp_path / "ws").rglob("recurrent_kda_*/*.o")), "nothing exported"
    subprocess.run([sys.executable, script, "forbid_compile"], env=env, check=True)
