"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

import flashinfer
import flashinfer._wan_hybrid as wan_hybrid_impl
import flashinfer.jit.wan_hybrid as wan_hybrid_jit
import flashinfer.wan_hybrid as wan_hybrid


_EXACT_SHAPE = (1, 4800, 40, 128)
_STORAGE_PADDED_SEQUENCE = 5120
_WRITTEN_PADDED_SEQUENCE = 4864
_VALUE_ROWS = 40 * 128
_PHYSICAL_BLOCKS = _STORAGE_PADDED_SEQUENCE // 128
_SCALE_COLUMNS = _WRITTEN_PADDED_SEQUENCE // 16
_VALID_PACKED_COLUMNS = _EXACT_SHAPE[1] // 2
_VALID_SCALE_COLUMNS = _EXACT_SHAPE[1] // 16
_NVFP4_QUANT_ENV_VARS = (
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
    "FLASHINFER_NVFP4_4OVER6",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
)


def _meta_tensor(*, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(_EXACT_SHAPE, dtype=dtype, device="meta")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _uninitialized_workspace() -> wan_hybrid.WanHybridAttentionWorkspace:
    workspace = object.__new__(wan_hybrid.WanHybridAttentionWorkspace)
    workspace.device = torch.device("cuda:0")
    workspace._buffers = {}
    return workspace


def test_wan_hybrid_public_exports() -> None:
    assert (
        flashinfer.WanHybridAttentionWorkspace is wan_hybrid.WanHybridAttentionWorkspace
    )
    assert (
        flashinfer.is_wan_hybrid_attention_available
        is wan_hybrid.is_wan_hybrid_attention_available
    )
    assert flashinfer.wan_hybrid_attention is wan_hybrid.wan_hybrid_attention
    assert not hasattr(flashinfer, "wan_hybrid_quantize_value")


def test_wan_hybrid_public_signature_is_explicit() -> None:
    signature = inspect.signature(wan_hybrid.wan_hybrid_attention)
    assert tuple(signature.parameters) == (
        "q",
        "k",
        "v",
        "out",
        "workspace",
        "sm_scale",
        "qkv_layout",
        "causal",
    )
    assert signature.parameters["out"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["out"].default is inspect.Parameter.empty
    assert signature.parameters["workspace"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["workspace"].default is inspect.Parameter.empty
    assert signature.parameters["qkv_layout"].default == "NHD"
    assert signature.parameters["causal"].default is False


def test_wan_hybrid_unlinked_attention_is_explicitly_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(wan_hybrid, "_wan_hybrid_attention_impl", None)
    assert not wan_hybrid.is_wan_hybrid_attention_available()
    assert not wan_hybrid.is_wan_hybrid_attention_available("cpu")
    assert not wan_hybrid.is_wan_hybrid_attention_available("not-a-device")


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_wan_hybrid_supported_capabilities_remain_available(
    monkeypatch, capability
) -> None:
    monkeypatch.setattr(wan_hybrid, "_wan_hybrid_attention_impl", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)

    assert wan_hybrid.is_wan_hybrid_attention_available("cuda:0")


def test_wan_hybrid_jit_flags_are_target_specific(monkeypatch) -> None:
    calls = {}

    def fake_gen_jit_spec(name, sources, **kwargs):
        calls[name] = {"sources": sources, **kwargs}
        return object()

    monkeypatch.setattr(wan_hybrid_jit, "gen_jit_spec", fake_gen_jit_spec)
    monkeypatch.setattr(
        wan_hybrid_jit, "_wan_hybrid_csrc_dir", lambda: Path("/wan_hybrid")
    )
    wan_hybrid_jit.gen_wan_hybrid_quantization_module.cache_clear()
    wan_hybrid_jit.gen_wan_hybrid_attention_module.cache_clear()

    try:
        for target in ("sm100", "sm103"):
            wan_hybrid_jit.gen_wan_hybrid_quantization_module(target)
            wan_hybrid_jit.gen_wan_hybrid_attention_module(target)

        for target, arch in (("sm100", "100a"), ("sm103", "103a")):
            target_minor = "0" if target == "sm100" else "3"
            for component in ("quantization", "attention"):
                call = calls[f"wan_hybrid_{component}_{target}"]
                flags = call["extra_cuda_cflags"]
                assert flags.count(f"-gencode=arch=compute_{arch},code=sm_{arch}") == 1
                assert flags.count(
                    f"-DFLASHINFER_WAN_HYBRID_TARGET_MINOR={target_minor}"
                ) == 1
                assert len(call["sources"]) == 1
                assert call["sources"][0].name == (
                    f"wan_hybrid_{component}_binding.cu"
                )

        assert calls["wan_hybrid_quantization_sm100"]["use_fast_math"] is False
        assert calls["wan_hybrid_quantization_sm103"]["use_fast_math"] is True
        for target in ("sm100", "sm103"):
            assert "--ptxas-options=--opt-level=1" in calls[
                f"wan_hybrid_attention_{target}"
            ]["extra_cuda_cflags"]
    finally:
        wan_hybrid_jit.gen_wan_hybrid_quantization_module.cache_clear()
        wan_hybrid_jit.gen_wan_hybrid_attention_module.cache_clear()


def test_wan_hybrid_quantizer_binding_matches_frozen_device_abi() -> None:
    source_root = Path(__file__).resolve().parents[2] / "csrc" / "wan_hybrid"
    binding = (source_root / "wan_hybrid_quantization_binding.cu").read_text(
        encoding="utf-8"
    )
    for argument in (
        "value",
        "base",
        "residual",
        "base_scale_lo",
        "base_scale_hi",
        "residual_scale_lo",
        "residual_scale_hi",
    ):
        assert f"TensorView {argument}" in binding
    assert "kernel_wan_hybrid_quantize_value<<<" in binding
    assert "kHeads = 40" in binding
    assert "kSequence = 4800" in binding
    assert "kPaddedSequence = 5120" in binding
    assert "kLogicalBlocks = 38" in binding
    assert "kPhysicalBlocks = 40" in binding
    assert "SMEM_TOTAL == 32896" in binding
    assert "SMEM_TOTAL == 33280" in binding
    assert "const cudaStream_t stream = get_stream(value.device());" in binding
    assert binding.count('#include "device/wan_hybrid_quantize_value_sm') == 2
    assert _sha256(source_root / "device/wan_hybrid_quantize_value_sm100.cu") == (
        "808fa99c273e7b0902cf7938bfb0078e26a8a5ac49f58f2f9432ef17d858fcf5"
    )
    assert _sha256(source_root / "device/wan_hybrid_quantize_value_sm103.cu") == (
        "f2a92e9b3cb774673e5ca1192c793cfd782dc07f2c8c294a12466b3d26be7f3e"
    )


def test_wan_hybrid_attention_binding_matches_frozen_device_abi() -> None:
    source_root = Path(__file__).resolve().parents[2] / "csrc" / "wan_hybrid"
    binding = (source_root / "wan_hybrid_attention_binding.cu").read_text(
        encoding="utf-8"
    )
    for argument in (
        "q",
        "k",
        "vt",
        "sfvt_lo",
        "sfvt_hi",
        "out",
        "descriptor_storage",
    ):
        assert f"TensorView {argument}" in binding
    assert "cudaLaunchKernelEx(&config, kernel_wan_hybrid_attention" in binding
    assert "kSequence = 4800" in binding
    assert "kHeads = 40" in binding
    assert "kHeadDim = 128" in binding
    assert "kTensorMapCount = 6" in binding
    assert "kMaximumTiles = 147" in binding
    assert "kDynamicSmemBytes = 231'424" in binding
    assert 'EncodeNHD(k, 128, "cuTensorMapEncodeTiled(k)")' in binding
    assert "kPackedValueRows, 64, 128" in binding
    assert "physical_num_blocks = kPaddedSequence / 128" in binding
    assert "cudaLaunchAttributeClusterDimension" not in binding
    for removed in ("TensorView sfq", "TensorView sfk", "TensorView qk_correction"):
        assert removed not in binding
    assert binding.count('#include "device/wan_hybrid_attention_sm') == 2
    for target in ("sm100", "sm103"):
        assert _sha256(
            source_root / "device" / f"wan_hybrid_attention_{target}.cu"
        ) == "2b9d37f9cf9fa60d129c4b16edf8e5a2d792bcd2f1fa4a7d724079339fb30e30"


def test_wan_hybrid_attention_requires_prewarm_before_capture(monkeypatch) -> None:
    q = torch.empty((1,), device="cpu")
    k = torch.empty((2,), device="cpu")
    out = torch.empty((3,), device="cpu")
    workspace = object.__new__(wan_hybrid.WanHybridAttentionWorkspace)
    workspace._attention_views = wan_hybrid._WanHybridAttentionABIViews(
        *(torch.empty((index + 4,), device="cpu") for index in range(3))
    )
    workspace._descriptor_signature = None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="must be prewarmed"):
        wan_hybrid_impl.wan_hybrid_attention_impl(
            q, k, torch.empty((1,), device="cpu"), out, workspace, 0.125
        )


def test_wan_hybrid_quantizer_dispatches_cache_by_value_device(monkeypatch) -> None:
    targets = iter(("sm100", "sm103"))
    resolved_devices = []
    loaded_targets = []
    launched_targets = []

    def resolve_target(device) -> str:
        resolved_devices.append(device)
        return next(targets)

    class Module:
        def __init__(self, target: str) -> None:
            self.target = target

        def wan_hybrid_quantize_value(self, *args) -> None:
            launched_targets.append(self.target)

    def load_module(target: str) -> Module:
        loaded_targets.append(target)
        return Module(target)

    monkeypatch.setattr(wan_hybrid, "_wan_hybrid_quantization_target", resolve_target)
    monkeypatch.setattr(wan_hybrid, "_get_wan_hybrid_quantization_module", load_module)
    value = torch.empty((), dtype=torch.bfloat16, device="meta")
    outputs = [torch.empty((), dtype=torch.uint8, device="meta") for _ in range(6)]

    wan_hybrid._wan_hybrid_quantize_value_impl(value, *outputs)
    wan_hybrid._wan_hybrid_quantize_value_impl(value, *outputs)

    assert resolved_devices == [value.device, value.device]
    assert loaded_targets == ["sm100", "sm103"]
    assert launched_targets == ["sm100", "sm103"]


def test_wan_hybrid_capability_fails_closed_before_cuda_probe(monkeypatch) -> None:
    monkeypatch.setattr(wan_hybrid, "_wan_hybrid_attention_impl", None)

    def unexpected_cuda_probe() -> bool:
        pytest.fail("an unavailable implementation must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_probe)
    assert not wan_hybrid.is_wan_hybrid_attention_available()
    assert not wan_hybrid.is_wan_hybrid_attention_available("cuda:0")


def test_wan_hybrid_workspace_requires_cuda() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        wan_hybrid.WanHybridAttentionWorkspace("cpu")


@pytest.mark.parametrize("name", ["q", "k", "v", "out"])
def test_wan_hybrid_rejects_non_exact_shape(name: str) -> None:
    tensors = {key: _meta_tensor() for key in ("q", "k", "v", "out")}
    tensors[name] = torch.empty((1, 4799, 40, 128), device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=rf"{name} must have NHD shape"):
        wan_hybrid.wan_hybrid_attention(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            out=tensors["out"],
            workspace=_uninitialized_workspace(),
        )


@pytest.mark.parametrize("name", ["q", "k", "v", "out"])
def test_wan_hybrid_rejects_non_bf16_tensor(name: str) -> None:
    tensors = {key: _meta_tensor() for key in ("q", "k", "v", "out")}
    tensors[name] = _meta_tensor(dtype=torch.float16)
    with pytest.raises(ValueError, match=rf"{name} must have dtype torch.bfloat16"):
        wan_hybrid.wan_hybrid_attention(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            out=tensors["out"],
            workspace=_uninitialized_workspace(),
        )


def test_wan_hybrid_rejects_noncontiguous_nhd() -> None:
    q = torch.empty((1, 4800, 40, 256), device="meta", dtype=torch.bfloat16)[..., ::2]
    assert q.shape == _EXACT_SHAPE
    assert not q.is_contiguous()
    with pytest.raises(ValueError, match="q must be contiguous"):
        wan_hybrid.wan_hybrid_attention(
            q,
            _meta_tensor(),
            _meta_tensor(),
            out=_meta_tensor(),
            workspace=_uninitialized_workspace(),
        )


@pytest.mark.parametrize("name", ["q", "k", "v"])
def test_wan_hybrid_rejects_caller_output_alias(name: str) -> None:
    tensors = {key: _meta_tensor() for key in ("q", "k", "v")}
    with pytest.raises(ValueError, match=rf"caller-owned out must not alias {name}"):
        wan_hybrid.wan_hybrid_attention(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            out=tensors[name],
            workspace=_uninitialized_workspace(),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"qkv_layout": "HND"}, "qkv_layout must be 'NHD'"),
        ({"causal": True}, "only supports noncausal"),
        ({"sm_scale": float("nan")}, "sm_scale must be finite"),
    ],
)
def test_wan_hybrid_rejects_non_exact_options(kwargs, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        wan_hybrid.wan_hybrid_attention(
            _meta_tensor(),
            _meta_tensor(),
            _meta_tensor(),
            out=_meta_tensor(),
            workspace=_uninitialized_workspace(),
            **kwargs,
        )


def test_wan_hybrid_rejects_non_cuda_exact_contract() -> None:
    with pytest.raises(ValueError, match="q must be a CUDA tensor"):
        wan_hybrid.wan_hybrid_attention(
            _meta_tensor(),
            _meta_tensor(),
            _meta_tensor(),
            out=_meta_tensor(),
            workspace=_uninitialized_workspace(),
        )


def test_wan_hybrid_returns_the_caller_owned_output(monkeypatch) -> None:
    q, k, v, out = (_meta_tensor() for _ in range(4))
    workspace = object()
    calls = []

    monkeypatch.setattr(
        wan_hybrid,
        "_validate_wan_hybrid_attention_contract",
        lambda *args, **kwargs: 0.125,
    )
    monkeypatch.setattr(
        wan_hybrid,
        "is_wan_hybrid_attention_available",
        lambda device: True,
    )

    def quantize(*args) -> None:
        calls.append(("quantize", *args))

    def implementation(*args) -> None:
        calls.append(args)

    monkeypatch.setattr(wan_hybrid, "_quantize_wan_hybrid_value", quantize)
    monkeypatch.setattr(wan_hybrid, "_wan_hybrid_attention_impl", implementation)
    result = wan_hybrid.wan_hybrid_attention(
        q,
        k,
        v,
        out=out,
        workspace=workspace,
    )

    assert result is out
    assert calls == [
        ("quantize", v, workspace),
        (q, k, v, out, workspace, 0.125),
    ]


def test_wan_hybrid_out_is_required() -> None:
    with pytest.raises(
        TypeError, match="missing 1 required keyword-only argument: 'out'"
    ):
        wan_hybrid.wan_hybrid_attention(
            _meta_tensor(),
            _meta_tensor(),
            _meta_tensor(),
            workspace=_uninitialized_workspace(),
        )


def _require_wan_hybrid_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("wan_hybrid requires SM100 or SM103")


def _reference_logical_scales(scales: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(_VALUE_ROWS, device=scales.device)[:, None]
    columns = torch.arange(_SCALE_COLUMNS, device=scales.device)[None, :]
    offsets = (
        (rows // 128) * (_SCALE_COLUMNS // 4) * 512
        + (columns // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + columns % 4
    )
    return scales.view(torch.uint8).reshape(-1)[offsets]


def _split_logical_scales(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(_VALUE_ROWS, device=lo.device)[:, None]
    columns = torch.arange(_SCALE_COLUMNS, device=lo.device)[None, :]
    bh = rows // 128
    dim = rows % 128
    block = columns // 8
    group = columns % 8
    row_outer = dim // 32
    row_inner = dim % 32
    row_quad = row_inner // 8
    row_lane = row_inner % 8
    offsets = (
        (bh * _PHYSICAL_BLOCKS + block) * 512
        + ((row_quad * 8 + row_lane) * 4 + row_outer) * 4
        + group % 4
    )
    return torch.where(
        group < 4,
        lo.reshape(-1)[offsets],
        hi.reshape(-1)[offsets],
    )


def _reference_value_quantization(v: torch.Tensor):
    rows = torch.zeros(
        (_VALUE_ROWS, _WRITTEN_PADDED_SEQUENCE),
        dtype=torch.bfloat16,
        device=v.device,
    )
    rows[:, : _EXACT_SHAPE[1]].copy_(
        v.permute(0, 2, 3, 1).reshape(_VALUE_ROWS, _EXACT_SHAPE[1])
    )
    global_scale = torch.ones((1,), dtype=torch.float32, device=v.device)

    base, base_scales = flashinfer.fp4_quantize(rows, global_scale)
    return (
        base.reshape(_VALUE_ROWS, _WRITTEN_PADDED_SEQUENCE // 2),
        _reference_logical_scales(base_scales),
    )


def _decode_split_level(
    packed: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> torch.Tensor:
    packed = packed[:, :_VALID_PACKED_COLUMNS]
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(
        _VALUE_ROWS, _EXACT_SHAPE[1]
    )
    e2m1_lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    scales = _split_logical_scales(lo, hi)[:, :_VALID_SCALE_COLUMNS]
    scales = scales.contiguous().view(torch.float8_e4m3fn).float()
    return e2m1_lut[codes.long()] * scales.repeat_interleave(16, dim=1)


def test_wan_hybrid_workspace_uses_exact_reusable_quantizer_storage() -> None:
    _require_wan_hybrid_gpu()
    workspace = wan_hybrid.WanHybridAttentionWorkspace("cuda")
    expected = {
        "v_base": ((_VALUE_ROWS, _STORAGE_PADDED_SEQUENCE // 2), 13_107_200),
        "v_residual": ((_VALUE_ROWS, _STORAGE_PADDED_SEQUENCE // 2), 13_107_200),
        "v_scale_base_lo": ((25_600, 32), 819_200),
        "v_scale_base_hi": ((25_600, 32), 819_200),
        "v_scale_residual_lo": ((25_600, 32), 819_200),
        "v_scale_residual_hi": ((25_600, 32), 819_200),
    }
    assert set(workspace._buffers) == set(expected)
    assert workspace.device == torch.device("cuda", torch.cuda.current_device())
    pointers = set()
    for name, (shape, byte_count) in expected.items():
        tensor = workspace._buffers[name]
        assert tensor.dtype == torch.uint8
        assert tensor.device == workspace.device
        assert tensor.is_contiguous()
        assert tuple(tensor.shape) == shape
        assert tensor.numel() * tensor.element_size() == byte_count
        pointers.add(tensor.data_ptr())
    assert len(pointers) == len(expected)

    base = workspace._buffers["v_base"]
    residual = workspace._buffers["v_residual"]
    assert base.untyped_storage().data_ptr() == residual.untyped_storage().data_ptr()
    assert residual.data_ptr() == base.data_ptr() + base.numel() * base.element_size()

    for suffix in ("lo", "hi"):
        scale_base = workspace._buffers[f"v_scale_base_{suffix}"]
        scale_residual = workspace._buffers[f"v_scale_residual_{suffix}"]
        assert (
            scale_base.untyped_storage().data_ptr()
            == scale_residual.untyped_storage().data_ptr()
        )
        assert scale_residual.data_ptr() == (
            scale_base.data_ptr() + scale_base.numel() * scale_base.element_size()
        )

    views = workspace._attention_abi_views
    assert workspace._attention_abi_views is views
    assert tuple(views.vt.shape) == (2 * _VALUE_ROWS, 2560)
    assert tuple(views.sfvt_lo.shape) == (51_200, 32)
    assert tuple(views.sfvt_hi.shape) == (51_200, 32)
    assert views.vt.data_ptr() == base.data_ptr()
    assert views.sfvt_lo.data_ptr() == workspace._buffers[
        "v_scale_base_lo"
    ].data_ptr()
    assert views.sfvt_hi.data_ptr() == workspace._buffers[
        "v_scale_base_hi"
    ].data_ptr()
    assert workspace._descriptor_storage.dtype == torch.uint8
    assert workspace._descriptor_storage.device == workspace.device
    assert tuple(workspace._descriptor_storage.shape) == (6, 128)
    assert workspace._descriptor_storage.data_ptr() % 128 == 0
    assert workspace._descriptor_signature is None


def test_wan_hybrid_value_quantization_matches_reference_and_reuses_storage(
    monkeypatch,
) -> None:
    _require_wan_hybrid_gpu()
    for name in _NVFP4_QUANT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    torch.manual_seed(4254)
    v = torch.randn(_EXACT_SHAPE, dtype=torch.bfloat16, device="cuda")
    workspace = wan_hybrid.WanHybridAttentionWorkspace(v.device)
    expected_base, expected_base_scales = _reference_value_quantization(v)

    actual = wan_hybrid._quantize_wan_hybrid_value(v, workspace)
    pointers = tuple(tensor.data_ptr() for tensor in actual)
    attention_views = workspace._attention_abi_views
    attention_pointers = tuple(tensor.data_ptr() for tensor in attention_views)
    first = tuple(tensor.clone() for tensor in actual)
    repeated = wan_hybrid._quantize_wan_hybrid_value(v, workspace)

    assert tuple(tensor.data_ptr() for tensor in repeated) == pointers
    for actual_tensor, repeated_tensor, first_tensor in zip(
        actual, repeated, first, strict=True
    ):
        assert actual_tensor is repeated_tensor
        assert torch.equal(repeated_tensor, first_tensor)
    assert torch.equal(
        actual[0][:, :_VALID_PACKED_COLUMNS],
        expected_base[:, :_VALID_PACKED_COLUMNS],
    )
    assert torch.equal(
        _split_logical_scales(actual[2], actual[3])[:, :_VALID_SCALE_COLUMNS],
        expected_base_scales[:, :_VALID_SCALE_COLUMNS],
    )
    reconstructed = _decode_split_level(actual[0], actual[2], actual[3])
    reconstructed += _decode_split_level(actual[1], actual[4], actual[5])
    expected_values = (
        v.permute(0, 2, 3, 1).reshape(_VALUE_ROWS, _EXACT_SHAPE[1]).float()
    )
    delta = (reconstructed - expected_values).abs()
    cosine = torch.nn.functional.cosine_similarity(
        reconstructed.flatten(), expected_values.flatten(), dim=0
    )
    assert torch.isfinite(reconstructed).all()
    assert torch.allclose(reconstructed, expected_values, atol=1.0, rtol=0.1)
    assert cosine.item() >= 0.995
    assert delta.mean().item() <= 0.025

    def unexpected_python_allocation(*args, **kwargs):
        pytest.fail("reused value quantization called a Python tensor allocator")

    for function_name in (
        "empty",
        "empty_like",
        "zeros",
        "zeros_like",
        "full",
        "full_like",
    ):
        monkeypatch.setattr(torch, function_name, unexpected_python_allocation)
    allocated_before = torch.cuda.memory_allocated(v.device)
    for _ in range(10):
        wan_hybrid._quantize_wan_hybrid_value(v, workspace)
        assert workspace._attention_abi_views is attention_views
        assert (
            tuple(tensor.data_ptr() for tensor in workspace._attention_abi_views)
            == attention_pointers
        )
    torch.cuda.synchronize(v.device)
    assert torch.cuda.memory_allocated(v.device) == allocated_before


def test_wan_hybrid_value_quantization_cuda_graph_replay() -> None:
    _require_wan_hybrid_gpu()
    torch.manual_seed(4254)
    v = torch.randn(_EXACT_SHAPE, dtype=torch.bfloat16, device="cuda")
    workspace = wan_hybrid.WanHybridAttentionWorkspace(v.device)
    reference_workspace = wan_hybrid.WanHybridAttentionWorkspace(v.device)
    eager = wan_hybrid._quantize_wan_hybrid_value(v, workspace)
    attention_views = workspace._attention_abi_views
    attention_pointers = tuple(tensor.data_ptr() for tensor in attention_views)
    eager_snapshot = tuple(tensor.clone() for tensor in eager)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wan_hybrid._quantize_wan_hybrid_value(v, workspace)

    v.mul_(-0.5).add_(0.25)
    for tensor in captured:
        tensor.zero_()
    graph.replay()
    torch.cuda.synchronize(v.device)
    replayed = tuple(tensor.clone() for tensor in captured)

    assert workspace._attention_abi_views is attention_views
    assert (
        tuple(tensor.data_ptr() for tensor in workspace._attention_abi_views)
        == attention_pointers
    )

    for tensor in reference_workspace._buffers.values():
        tensor.zero_()
    reference = wan_hybrid._quantize_wan_hybrid_value(v, reference_workspace)

    assert any(
        not torch.equal(replayed_tensor, eager_tensor)
        for replayed_tensor, eager_tensor in zip(replayed, eager_snapshot, strict=True)
    )
    for replayed_tensor, reference_tensor in zip(replayed, reference, strict=True):
        assert torch.equal(replayed_tensor, reference_tensor)


def _wan_hybrid_bf16_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return (
        torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            scale=_EXACT_SHAPE[-1] ** -0.5,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _assert_wan_hybrid_quality(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    delta = (actual_f32 - expected_f32).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    quality = {
        "finite": bool(torch.isfinite(actual).all().item()),
        "allclose_atol_1_rtol_0_1": bool(
            torch.allclose(actual, expected, atol=1.0, rtol=0.1)
        ),
        "cosine": cosine.item(),
        "mae": delta.mean().item(),
        "max_abs": delta.max().item(),
    }
    assert quality["finite"]
    assert quality["allclose_atol_1_rtol_0_1"]
    assert quality["cosine"] >= 0.995
    assert quality["mae"] <= 0.025
    return quality


def test_wan_hybrid_attention_correctness_repeatability_and_reuse() -> None:
    _require_wan_hybrid_gpu()
    torch.manual_seed(4254)
    q, k, v = (
        torch.randn(_EXACT_SHAPE, dtype=torch.bfloat16, device="cuda") for _ in range(3)
    )
    out = torch.empty_like(q)
    workspace = wan_hybrid.WanHybridAttentionWorkspace(q.device)
    inputs_before = tuple(tensor.clone() for tensor in (q, k, v))
    workspace_pointers = tuple(
        tensor.data_ptr() for tensor in workspace._buffers.values()
    ) + (workspace._descriptor_storage.data_ptr(),)

    result = wan_hybrid.wan_hybrid_attention(q, k, v, out=out, workspace=workspace)
    torch.cuda.synchronize(q.device)
    assert result is out
    first = out.clone()
    descriptor_pointer = workspace._descriptor_storage.data_ptr()
    descriptor_signature = workspace._descriptor_signature
    allocated_before = torch.cuda.memory_allocated(q.device)

    wan_hybrid.wan_hybrid_attention(q, k, v, out=out, workspace=workspace)
    torch.cuda.synchronize(q.device)
    repeat_bitwise = bool(torch.equal(out, first))
    input_immutable = all(
        torch.equal(actual, expected)
        for actual, expected in zip((q, k, v), inputs_before, strict=True)
    )
    workspace_pointers_stable = workspace_pointers == tuple(
        tensor.data_ptr() for tensor in workspace._buffers.values()
    ) + (workspace._descriptor_storage.data_ptr(),)
    allocation_stable = torch.cuda.memory_allocated(q.device) == allocated_before
    assert repeat_bitwise
    assert input_immutable
    assert workspace_pointers_stable
    assert workspace._descriptor_storage.data_ptr() == descriptor_pointer
    assert workspace._descriptor_signature == descriptor_signature
    assert allocation_stable

    reference = _wan_hybrid_bf16_reference(q, k, v)
    quality = _assert_wan_hybrid_quality(out, reference)
    print(
        json.dumps(
            {
                "test": "correctness_repeatability_and_reuse",
                "quality": quality,
                "repeat_bitwise": repeat_bitwise,
                "input_immutable": input_immutable,
                "caller_output_same_object": result is out,
                "caller_output_pointer": out.data_ptr(),
                "workspace_pointers_stable": workspace_pointers_stable,
                "allocation_stable": allocation_stable,
            },
            sort_keys=True,
        )
    )


def test_wan_hybrid_attention_cuda_graph_replay() -> None:
    _require_wan_hybrid_gpu()
    torch.manual_seed(4254)
    q, k, v = (
        torch.randn(_EXACT_SHAPE, dtype=torch.bfloat16, device="cuda") for _ in range(3)
    )
    out = torch.empty_like(q)
    workspace = wan_hybrid.WanHybridAttentionWorkspace(q.device)

    wan_hybrid.wan_hybrid_attention(q, k, v, out=out, workspace=workspace)
    torch.cuda.synchronize(q.device)
    descriptor_signature = workspace._descriptor_signature
    output_pointer = out.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wan_hybrid.wan_hybrid_attention(
            q, k, v, out=out, workspace=workspace
        )
    prewarm_output = out.clone()

    q.mul_(0.75)
    k.mul_(-0.5)
    v.add_(0.125)
    updated_inputs = tuple(tensor.clone() for tensor in (q, k, v))
    out.zero_()
    graph.replay()
    torch.cuda.synchronize(q.device)

    input_immutable = all(
        torch.equal(actual, expected)
        for actual, expected in zip((q, k, v), updated_inputs, strict=True)
    )
    output_changed = not torch.equal(out, prewarm_output)
    assert captured is out
    assert out.data_ptr() == output_pointer
    assert workspace._descriptor_signature == descriptor_signature
    assert input_immutable
    assert output_changed
    reference = _wan_hybrid_bf16_reference(q, k, v)
    quality = _assert_wan_hybrid_quality(out, reference)
    print(
        json.dumps(
            {
                "test": "cuda_graph_replay",
                "quality": quality,
                "prewarmed_before_capture": descriptor_signature is not None,
                "output_changed_after_input_update": output_changed,
                "input_immutable_during_replay": input_immutable,
                "caller_output_pointer_stable": out.data_ptr() == output_pointer,
                "descriptor_signature_stable": (
                    workspace._descriptor_signature == descriptor_signature
                ),
            },
            sort_keys=True,
        )
    )
