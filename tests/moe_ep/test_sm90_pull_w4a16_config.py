"""Host-only unit tests for the sm90_bf16_nvfp4_bf16_pull_cutedsl (W4A16) wiring.

No GPU / no kernel compile: config defaults and the swap-AB-only rule,
registry resolution, public re-exports, the arch gate, forward-input
validation (runtime alphas), and the host weight layout (``augment_w4a16``
on CPU tensors).  Deliberately does NOT import
``flashinfer.moe_ep.kernel_src.sm90.*`` (see ``test_sm90_pull_fp8_config.py``);
kernel coverage lives in ``test_sm90_pull_w4a16_kernel_vs_reference.py``.
"""

from __future__ import annotations

import dataclasses
import sys

import pytest
import torch

from flashinfer.moe_ep import (
    FleetParams,
    MoEEpConfigError,
    MoEEpTensors,
    Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_nvfp4_bf16_pull_cutedsl import (
    Sm90PullW4A16MegaKernelBackend,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90.common.nvfp4 import (
    W4A16_PAIR_TILE_BYTES,
    augment_w4a16,
    dequantize_nvfp4,
)
from flashinfer.moe_ep.core.kernel.registry import (
    create_mega_kernel,
    is_mega_kernel_config,
)

_NAME = "sm90_bf16_nvfp4_bf16_pull_cutedsl"


def _config(**overrides) -> Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig:
    return Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
        **{"intermediate_size": 1024, "top_k": 4, **overrides}
    )


def test_defaults_and_swap_ab_only() -> None:
    cfg = _config()
    assert cfg.kernel_name == _NAME
    assert cfg.kind == "bf16" and cfg.weight_format == "nvfp4"
    assert cfg.fc1_alpha is None and cfg.fc2_alpha is None
    # Heuristic / knob-cache geometry stays unset (all W4A16 rows are swap-AB).
    assert cfg.swap_ab is None
    # Manual geometry defaults to swap-AB instead of the native layout.
    assert _config(mma_tiler_mnk=(256, 32, 128)).swap_ab is True
    with pytest.raises(MoEEpConfigError, match="swap-AB only"):
        _config(swap_ab=False)


def test_registry_and_reexports() -> None:
    import flashinfer.moe_ep as moe_ep

    assert is_mega_kernel_config(_config())
    backend = create_mega_kernel(_config())
    assert isinstance(backend, Sm90PullW4A16MegaKernelBackend)
    assert backend.kernel_name() == _NAME
    assert "Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig" in moe_ep.__all__
    assert "preprocess_sm90_pull_w4a16_mega_weights" in moe_ep.__all__
    assert callable(moe_ep.preprocess_sm90_pull_w4a16_mega_weights)
    # Registering / importing the backend must not load the kernel tree.
    assert not any(".kernel_src.sm90.pull_style" in name for name in sys.modules)


def test_arch_gate_names_the_backend(monkeypatch) -> None:
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.core.validation import common as vcommon

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vcommon, "_device_capability", lambda: (10, 0))
    backend = create_mega_kernel(_config())
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=64, token_hidden_size=1024)
    with pytest.raises(vcommon.MoEEpArchError, match=f"{_NAME}.*sm_90"):
        backend.validate_init(BootstrapConfig(rank=0, world_size=1), fleet)


def test_forward_validation_runtime_alpha() -> None:
    from flashinfer.moe_ep.config import BootstrapConfig

    backend = create_mega_kernel(_config(top_k=2))
    backend.bind_ep_bootstrap(BootstrapConfig(rank=0, world_size=1))
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=16, token_hidden_size=256)
    ok = MoEEpTensors(
        hidden_states=torch.zeros(4, 256, dtype=torch.bfloat16),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        topk_weights=torch.full((4, 2), 0.5),
    )
    backend.validate_forward(ok, fleet, quantize_input=True)
    alpha = torch.ones(8, dtype=torch.float32)
    backend.validate_forward(
        dataclasses.replace(ok, fc1_alpha=alpha, fc2_alpha=alpha),
        fleet,
        quantize_input=True,
    )
    for name, bad in (
        ("fc1_alpha", alpha.double()),
        ("fc2_alpha", torch.ones(4, dtype=torch.float32)),
    ):
        with pytest.raises(MoEEpConfigError, match=name):
            backend.validate_forward(
                dataclasses.replace(ok, **{name: bad}), fleet, quantize_input=True
            )


def _cpu_pack(experts=2, rows=4, k=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0, 256, (experts, rows, k // 2), generator=g, dtype=torch.uint8
    )
    scales = (torch.rand(experts, rows, k // 16, generator=g) + 0.1).to(
        torch.float8_e4m3fn
    )
    return packed, scales


def test_augment_w4a16_layout_and_canonicalization() -> None:
    """Pair tiles: 144 B per (row pair, 128-K tile); codes decode by the kernel rule."""
    packed, scales = _cpu_pack()
    s = scales.view(torch.uint8).clone()
    s[0, 0, 0] = 0x00  # zero scale: its codes become zero
    s[0, 1, 3] = 0x80  # negative zero
    s[1, 2, 5] |= 0x80  # negative scale: sign moves into the codes
    aug = augment_w4a16(packed, s)
    experts, rows, half_k = packed.shape
    k = 2 * half_k
    assert aug.dtype == torch.uint8
    assert aug.shape == (experts, rows // 2, k // 128 * W4A16_PAIR_TILE_BYTES)

    # Decode exactly as the kernel does: lane l / k16 block kb / half h live
    # in word l*4 + kb//2 of the row's payload, pair j = (kb % 2)*2 + h in
    # nibbles j (even k) and j + 4 (odd k); scales after both payloads.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    tiles = aug.view(experts, rows // 2, k // 128, 144)
    out = torch.empty(experts, rows, k)
    for row in range(rows):
        payload = tiles[:, row // 2, :, (row % 2) * 64 : (row % 2) * 64 + 64]
        scale_bytes = tiles[:, row // 2, :, 128 + (row % 2) * 8 : 136 + (row % 2) * 8]
        assert int(scale_bytes.max()) < 0x7F  # canonical: non-negative, finite
        block_scale = scale_bytes.contiguous().view(torch.float8_e4m3fn).float()
        for kb in range(8):
            for h in range(2):
                for lane in range(4):
                    byte = lane * 16 + (kb // 2) * 4
                    words = payload[..., byte : byte + 4].to(torch.int64)
                    q = (
                        words[..., 0]
                        | words[..., 1] << 8
                        | words[..., 2] << 16
                        | words[..., 3] << 24
                    )
                    j = (kb % 2) * 2 + h
                    for e in range(2):
                        code = (q >> (4 * (j + 4 * e))) & 0xF
                        col = kb * 16 + h * 8 + 2 * lane + e
                        for tile in range(k // 128):
                            out[:, row, tile * 128 + col] = (
                                lut[code[:, tile]] * block_scale[:, tile, kb]
                            )
    want = dequantize_nvfp4(packed, s)
    assert torch.equal(out, want)


def test_augment_w4a16_rejects_nan_scale() -> None:
    packed, scales = _cpu_pack()
    s = scales.view(torch.uint8).clone()
    s[1, 3, 0] = 0xFF
    with pytest.raises(ValueError, match="NaN"):
        augment_w4a16(packed, s)
