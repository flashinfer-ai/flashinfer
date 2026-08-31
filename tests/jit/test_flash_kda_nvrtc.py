import json

import pytest

from flashinfer.jit import flash_kda_nvrtc


def _selector(body_name: str) -> str:
    return f'''\
#define FLASHKDA_GENERATED_BODY_FILE "{body_name}"
#define FLASHKDA_GENERATED_KERNEL kernel_flashkda_test
'''


def test_prepare_generated_cubin_is_content_addressed_and_revalidates(
    monkeypatch, tmp_path
):
    selector = tmp_path / "selector.cu"
    body = tmp_path / "body.cu"
    selector.write_text(_selector(body.name))
    body.write_text('extern "C" __global__ void kernel_flashkda_test() {}\n')
    cuda_include = tmp_path / "cuda" / "include"
    cuda_include.mkdir(parents=True)
    monkeypatch.setattr(flash_kda_nvrtc, "_cuda_include_dirs", lambda: (cuda_include,))
    calls = []

    def fake_compile(source, *, source_name, options):
        calls.append((source, source_name, options))
        return b"exact-cubin"

    monkeypatch.setattr(flash_kda_nvrtc, "_compile_cubin", fake_compile)
    kwargs = {
        "selector_path": selector,
        "body_path": body,
        "module_ident": "flashkda_test_0123456789",
        "target": "sm103a",
    }

    first = flash_kda_nvrtc.prepare_generated_flash_kda_cubin(
        tmp_path / "jit" / "module", **kwargs
    )
    second = flash_kda_nvrtc.prepare_generated_flash_kda_cubin(
        tmp_path / "jit" / "module", **kwargs
    )

    assert first == second
    assert len(calls) == 1
    cubin = first["flashkda_test_0123456789"]
    receipt = json.loads(cubin.with_suffix(".json").read_text())
    assert receipt["inputs"]["arch"] == "sm_103a"
    assert receipt["inputs"]["compile_options"] == [
        "--gpu-architecture=sm_103a",
        "-std=c++17",
        "-default-device",
        f"-I{cuda_include}",
        "--use_fast_math",
    ]
    assert receipt["inputs"]["optimization_level_one_absent"] is True

    cubin.write_bytes(b"tampered")
    flash_kda_nvrtc.prepare_generated_flash_kda_cubin(
        tmp_path / "jit" / "module", **kwargs
    )
    assert len(calls) == 2
    assert cubin.read_bytes() == b"exact-cubin"


def test_prepare_generated_cubin_rejects_selector_body_mismatch(tmp_path):
    selector = tmp_path / "selector.cu"
    body = tmp_path / "body.cu"
    selector.write_text(_selector("other.cu"))
    body.write_text("kernel")

    with pytest.raises(ValueError, match="selector/body mismatch"):
        flash_kda_nvrtc.prepare_generated_flash_kda_cubin(
            tmp_path / "jit" / "module",
            selector_path=selector,
            body_path=body,
            module_ident="flashkda_test_0123456789",
            target="sm100a",
        )
