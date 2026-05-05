from flashinfer.jit.core import gen_jit_spec
from flashinfer.jit.cpp_ext import get_cuda_path


def test_issue_3229(monkeypatch, tmp_path):
    """
    Generate two JitSpec objects with different settings.
    Assert that their `jit_library_path`s are different.

    https://github.com/flashinfer-ai/flashinfer/issues/3229
    """
    monkeypatch.setattr(
        "flashinfer.jit.core.check_cuda_arch", lambda: None, raising=True
    )

    cu = tmp_path / "k.cu"
    cu.write_text("// test\n", encoding="utf-8")

    monkeypatch.setenv("CUDA_HOME", "dummy1")
    get_cuda_path.cache_clear()
    jit_spec1 = gen_jit_spec("test_issue_3229", [cu])
    build_dir1 = jit_spec1.build_dir

    monkeypatch.setenv("CUDA_HOME", "dummy2")
    get_cuda_path.cache_clear()
    jit_spec2 = gen_jit_spec("test_issue_3229", [cu])
    build_dir2 = jit_spec2.build_dir

    assert build_dir1 != build_dir2

    get_cuda_path.cache_clear()
