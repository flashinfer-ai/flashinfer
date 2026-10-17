"""Cache identities describe resolved codegen settings, not configuration sources."""

import importlib

import pytest

from b12x._lib.compile_options import clear_options_cache, resolve_compile_options


@pytest.fixture
def dsl():
    pytest.importorskip("cutlass")
    from cutlass.cutlass_dsl import CuTeDSL

    clear_options_cache()
    yield CuTeDSL._get_dsl()
    clear_options_cache()


def test_environment_defaults_and_explicit_options_have_identical_keys(
    dsl, monkeypatch
):
    import cutlass.cute as cute
    from cutlass.base_dsl.compiler import OptLevel

    monkeypatch.setattr(dsl.envar, "compiler_opt", "opt-level=2")
    configured = resolve_compile_options(cute.compile)
    monkeypatch.setattr(dsl.envar, "compiler_opt", "")
    clear_options_cache()
    explicit = resolve_compile_options(cute.compile, OptLevel(2))
    textual = resolve_compile_options(cute.compile, None, "--opt-level 2")
    assert (
        configured._b12x_options_key
        == explicit._b12x_options_key
        == textual._b12x_options_key
    )


def test_explicit_default_overrides_configured_default(dsl, monkeypatch):
    import cutlass.cute as cute
    from cutlass.base_dsl.compiler import OptLevel

    monkeypatch.setattr(dsl.envar, "compiler_opt", "opt-level=2")
    for options in (
        resolve_compile_options(cute.compile, OptLevel(3)),
        resolve_compile_options(cute.compile, None, "--opt-level 3"),
    ):
        assert options._compile_options.options[OptLevel].value == 3
        options._compile_options.apply_envar_settings(dsl.envar, "test_kernel")
        assert options._compile_options.options[OptLevel].value == 3


@pytest.mark.parametrize(
    "name,value", [("lineinfo", True), ("enable_assertions", True), ("arch", "sm_121a")]
)
def test_effective_codegen_settings_change_identity(dsl, monkeypatch, name, value):
    import cutlass.cute as cute

    before = resolve_compile_options(cute.compile)._b12x_options_key
    monkeypatch.setattr(dsl.envar, name, value)
    clear_options_cache()
    assert before != resolve_compile_options(cute.compile)._b12x_options_key


def test_operational_settings_do_not_change_identity(dsl, monkeypatch, tmp_path):
    import cutlass.cute as cute

    before = resolve_compile_options(cute.compile)._b12x_options_key
    monkeypatch.setattr(dsl.envar, "dump_dir", str(tmp_path))
    monkeypatch.setattr(dsl.envar, "keep_ptx", True)
    clear_options_cache()
    assert before == resolve_compile_options(cute.compile)._b12x_options_key


def test_compiler_diagnostics_do_not_change_identity(dsl, monkeypatch):
    import cutlass.cute as cute

    before = resolve_compile_options(cute.compile)._b12x_options_key
    monkeypatch.setattr(dsl.envar, "compiler_opt", "warnings{nvvm,ptx}")
    clear_options_cache()
    assert before == resolve_compile_options(cute.compile)._b12x_options_key


def test_launch_checks_are_preserved_and_keyed(dsl, monkeypatch):
    import cutlass.cute as cute

    default = resolve_compile_options(cute.compile)
    monkeypatch.setattr(dsl.envar, "compiler_opt", "debug{launch-check}")
    clear_options_cache()
    checked = resolve_compile_options(cute.compile)
    assert checked._compile_options._debug_selectors == {"launch-check"}
    assert default._b12x_options_key != checked._b12x_options_key


def test_raw_environment_does_not_participate_in_disk_key(dsl, monkeypatch):
    import cutlass.cute as cute
    from b12x.norm.hyperconnection._cute import _GroupedRmsNorm

    compiler = importlib.import_module("b12x._lib.compiler")
    compile_callable = resolve_compile_options(cute.compile)
    target = _GroupedRmsNorm(4, 256)
    spec = compiler.KernelCompileSpec.from_facts("test.rmsnorm", 1, 4, 256)
    before = compiler._build_compile_disk_cache_key(
        compile_callable, target, (), {}, spec
    )
    for name in (
        "B12X_UNUSED_SETTING",
        "CUTE_UNUSED_SETTING",
        "CUTLASS_UNUSED_SETTING",
        "NVCC_APPEND_FLAGS",
        "CC",
        "CUDA_HOME",
    ):
        monkeypatch.setenv(name, "different")
    assert before == compiler._build_compile_disk_cache_key(
        compile_callable, target, (), {}, spec
    )
