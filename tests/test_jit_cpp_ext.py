import subprocess

from packaging.version import Version

from flashinfer.jit import core, cpp_ext


def test_nvcc_parallelism_flags_use_flashinfer_nvcc_threads(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")
    monkeypatch.setattr(cpp_ext, "should_use_ninja_jobserver", lambda: False)

    assert cpp_ext.get_nvcc_parallelism_flags(Version("13.0")) == ["--threads=4"]


def test_nvcc_parallelism_flags_add_jobserver_for_cuda_13_linux(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")
    monkeypatch.setattr(cpp_ext.sys, "platform", "linux")
    monkeypatch.setattr(cpp_ext, "should_use_ninja_jobserver", lambda: True)
    monkeypatch.delenv("FLASHINFER_NVCC_LAUNCHER", raising=False)

    assert cpp_ext.get_nvcc_parallelism_flags(Version("13.0")) == [
        "--threads=4",
        "--jobserver",
    ]


def test_nvcc_parallelism_flags_add_jobserver_for_sccache_launcher(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")
    monkeypatch.setenv("FLASHINFER_NVCC_LAUNCHER", "sccache")
    monkeypatch.setattr(cpp_ext.sys, "platform", "linux")
    monkeypatch.setattr(cpp_ext, "should_use_ninja_jobserver", lambda: False)

    assert cpp_ext.get_nvcc_parallelism_flags(Version("13.0")) == [
        "--threads=4",
        "--jobserver",
    ]


def test_nvcc_parallelism_flags_skip_jobserver_before_cuda_13(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")
    monkeypatch.setenv("FLASHINFER_NVCC_LAUNCHER", "sccache")
    monkeypatch.setattr(cpp_ext.sys, "platform", "linux")
    monkeypatch.setattr(cpp_ext, "should_use_ninja_jobserver", lambda: True)

    assert cpp_ext.get_nvcc_parallelism_flags(Version("12.9")) == ["--threads=4"]


def test_run_ninja_uses_max_jobs_without_jobserver(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setattr(cpp_ext, "should_use_ninja_jobserver", lambda: False)
    monkeypatch.setattr(cpp_ext.subprocess, "run", fake_run)

    cpp_ext.run_ninja(tmp_path, tmp_path / "build.ninja", verbose=False)

    assert commands == [
        [
            "ninja",
            "-v",
            "-C",
            str(tmp_path.resolve()),
            "-f",
            str((tmp_path / "build.ninja").resolve()),
            "-j",
            "8",
        ]
    ]


def test_run_ninja_omits_max_jobs_when_jobserver_is_available(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setattr(cpp_ext, "should_use_ninja_jobserver", lambda: True)
    monkeypatch.setattr(cpp_ext.subprocess, "run", fake_run)

    cpp_ext.run_ninja(tmp_path, tmp_path / "build.ninja", verbose=False)

    assert commands == [
        [
            "ninja",
            "-v",
            "-C",
            str(tmp_path.resolve()),
            "-f",
            str((tmp_path / "build.ninja").resolve()),
        ]
    ]


def test_jit_spec_build_rewrites_ninja_before_build(monkeypatch):
    writes = []
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)

    spec = core.JitSpec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    monkeypatch.setattr(spec, "write_ninja", lambda: writes.append(True))
    monkeypatch.setattr(core, "run_ninja", lambda *_args, **_kwargs: None)

    spec.build(verbose=False, need_lock=False)

    assert writes == [True]
