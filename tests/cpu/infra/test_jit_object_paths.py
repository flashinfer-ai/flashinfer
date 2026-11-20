import pytest
from packaging.version import Version

from flashinfer.jit import core, cpp_ext


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "source_names",
    [
        ["generated/kernel.cu"],
        ["generated/binding.cpp"],
        ["generated/kernel.cu", "generated/binding.cc"],
        ["first/kernel.cu", "second/kernel.cu"],
    ],
)
def test_jit_spec_object_paths_match_ninja(monkeypatch, tmp_path, source_names):
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")
    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: Version("12.8"))
    monkeypatch.setattr(core.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "7.5")
    spec = core.JitSpecNvcc(
        name="test_module",
        sources=[tmp_path / name for name in source_names],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    ninja = cpp_ext.generate_ninja_build_for_op(
        name=spec.name,
        sources=spec.sources,
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    escape = cpp_ext._ninja_windows_path if cpp_ext.is_windows else str
    objects = [escape(obj.resolve()) for obj in spec.get_object_paths()]
    for source, obj in zip(spec.sources, objects, strict=True):
        rule = "cuda_compile" if source.suffix == ".cu" else "compile"
        assert f"build {obj}: {rule} {escape(source.resolve())}" in ninja
    assert f": link {' '.join(objects)}\n" in ninja
