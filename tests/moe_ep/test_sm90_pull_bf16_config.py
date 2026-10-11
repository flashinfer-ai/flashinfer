"""Host-only unit tests for the sm90_bf16_bf16_bf16_pull_cutedsl mega-kernel wiring.

No GPU / no kernel compile: config defaults and fixed format attributes,
registry resolution, public re-exports, runtime requirements, the arch gate,
and forward-input validation.  Deliberately does NOT import
``flashinfer.moe_ep.kernel_src.sm90.*`` (see ``test_sm90_pull_fp8_config.py``);
kernel coverage lives in ``test_sm90_pull_bf16_kernel_vs_reference.py``.
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
    Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_pull_cutedsl import (
    Sm90PullBf16MegaKernelBackend,
)
from flashinfer.moe_ep.core.kernel.registry import (
    create_mega_kernel,
    is_mega_kernel_config,
)

_NAME = "sm90_bf16_bf16_bf16_pull_cutedsl"


def _config(**overrides) -> Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig:
    return Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig(
        **{"intermediate_size": 1024, "top_k": 4, **overrides}
    )


def test_defaults_and_fixed_format() -> None:
    cfg = _config()
    assert cfg.kernel_name == _NAME
    assert cfg.knobs is None
    assert cfg.swap_ab is None and cfg.mma_tiler_mnk is None
    # Fixed by the format (ClassVars, not constructor fields).
    assert cfg.kind == "bf16"
    assert cfg.fp8_scale_mode == "per_tensor"
    assert cfg.fp8_accum_mode == "1xacc"
    assert cfg.fc1_activation_dequant_scale == 1.0
    fields = {f.name for f in dataclasses.fields(cfg)}
    assert not fields & {"kind", "fp8_scale_mode", "fp8_accum_mode"}
    with pytest.raises(TypeError):
        _config(kind="fp8_e4m3")


def test_registry_and_reexports() -> None:
    import flashinfer.moe_ep as moe_ep

    assert is_mega_kernel_config(_config())
    backend = create_mega_kernel(_config())
    assert isinstance(backend, Sm90PullBf16MegaKernelBackend)
    assert backend.kernel_name() == _NAME
    assert "Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig" in moe_ep.__all__
    assert "preprocess_sm90_pull_bf16_mega_weights" in moe_ep.__all__
    assert callable(moe_ep.preprocess_sm90_pull_bf16_mega_weights)
    # Registering / importing the backend must not load the kernel tree.
    assert not any(".kernel_src.sm90.pull_style" in name for name in sys.modules)


def test_knobs_exclusive_with_geometry() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_mega_kernel(_config(knobs={}, swap_ab=True))


def test_runtime_requirements(monkeypatch) -> None:
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.core.runtime import NVSHMEM, TORCH_DIST

    monkeypatch.delenv("MEGA_NO_DIST", raising=False)
    bootstrap = BootstrapConfig(rank=0, world_size=1)
    assert create_mega_kernel(_config()).runtime_requirements(bootstrap) == frozenset(
        {TORCH_DIST, NVSHMEM}
    )


def test_arch_gate_names_the_backend(monkeypatch) -> None:
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.core.validation import common as vcommon

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vcommon, "_device_capability", lambda: (10, 0))
    backend = create_mega_kernel(_config())
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=64, token_hidden_size=1024)
    with pytest.raises(vcommon.MoEEpArchError, match=f"{_NAME}.*sm_90"):
        backend.validate_init(BootstrapConfig(rank=0, world_size=1), fleet)


def test_forward_validation() -> None:
    backend = create_mega_kernel(_config(top_k=2))
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=16, token_hidden_size=256)
    ids = torch.zeros(4, 2, dtype=torch.int64)
    weights = torch.full((4, 2), 0.5)
    ok = MoEEpTensors(
        hidden_states=torch.zeros(4, 256, dtype=torch.bfloat16),
        topk_ids=ids,
        topk_weights=weights,
    )
    # quantize_input is irrelevant for BF16 activations (copied, never quantized).
    backend.validate_forward(ok, fleet, quantize_input=True)
    backend.validate_forward(ok, fleet, quantize_input=False)
    for bad, match in (
        (dataclasses.replace(ok, hidden_states=ok.hidden_states.float()), "bf16"),
        (dataclasses.replace(ok, scales=torch.ones(4, 8)), "scales"),
        (
            dataclasses.replace(
                ok, hidden_states=torch.zeros(17, 256, dtype=torch.bfloat16)
            ),
            "max_tokens_per_rank",
        ),
    ):
        with pytest.raises(MoEEpConfigError, match=match):
            backend.validate_forward(bad, fleet, quantize_input=True)
