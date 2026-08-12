"""Host-only unit tests for the sm90_pull_fp8 mega-kernel backend wiring.

No GPU / no kernel compile: config dataclass defaults, registry resolution via
``create_mega_kernel``, public re-exports, and runtime-requirement plumbing.

Deliberately does NOT import ``flashinfer.moe_ep.kernel_src.sm90.*``: the SM90
and SM100 kernel trees share top-level module names and are mutually exclusive
per process, and the ``unit`` run_tests.sh target collects this file in the
same pytest process as SM100-importing tests.  Shim-level validation coverage
lives in ``test_sm90_pull_fp8_kernel_vs_reference.py`` (own process).
"""

from __future__ import annotations

import dataclasses

import pytest

from flashinfer.moe_ep import Sm90PullFp8MegaMoeConfig
from flashinfer.moe_ep.backends.mega.kernel.sm90_pull_fp8 import (
    Sm90PullFp8MegaKernelBackend,
)
from flashinfer.moe_ep.core.kernel.registry import (
    create_mega_kernel,
    is_mega_kernel_config,
)


def _config(**overrides) -> Sm90PullFp8MegaMoeConfig:
    return Sm90PullFp8MegaMoeConfig(intermediate_size=1024, top_k=4, **overrides)


class TestSm90PullFp8Config:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.kernel_name == "sm90_pull_fp8"
        assert cfg.kind == "fp8_e4m3"
        assert cfg.fp8_scale_mode == "per_tensor"
        assert cfg.fp8_accum_mode == "1xacc"
        assert cfg.swap_ab is False
        assert cfg.mma_tiler_mnk is None
        assert cfg.load_balance_mode == "static"
        assert cfg.gate_up_clamp is None
        assert cfg.fast_math is True
        assert cfg.in_kernel_fc2_reduce is False
        assert cfg.token_back_by_dispatch is False
        assert cfg.fc1_activation_dequant_scale == 1.0
        assert cfg.fc2_activation_dequant_scale == 1.0
        # PORT NOTE contract: no knobs field until the sm90 tree grows a tuner.
        assert "knobs" not in {f.name for f in dataclasses.fields(cfg)}

    def test_is_mega_kernel_config(self) -> None:
        assert is_mega_kernel_config(_config())

    def test_registry_resolves_backend(self) -> None:
        backend = create_mega_kernel(_config())
        assert isinstance(backend, Sm90PullFp8MegaKernelBackend)
        assert backend.kernel_name() == "sm90_pull_fp8"
        assert Sm90PullFp8MegaKernelBackend.kernel_name() == "sm90_pull_fp8"

    def test_registry_lists_kernel_in_unknown_error(self) -> None:
        bogus = dataclasses.replace(_config(), kernel_name="definitely_not_a_kernel")
        with pytest.raises(KeyError, match="sm90_pull_fp8"):
            create_mega_kernel(bogus)

    def test_public_reexports(self) -> None:
        import flashinfer.moe_ep as moe_ep

        assert moe_ep.Sm90PullFp8MegaMoeConfig is Sm90PullFp8MegaMoeConfig
        assert callable(moe_ep.preprocess_sm90_pull_fp8_mega_weights)
        assert "Sm90PullFp8MegaMoeConfig" in moe_ep.__all__
        assert "preprocess_sm90_pull_fp8_mega_weights" in moe_ep.__all__


class TestSm90PullFp8RuntimeRequirements:
    def test_requires_nvshmem_and_torch_dist(self, monkeypatch) -> None:
        monkeypatch.delenv("MEGA_NO_DIST", raising=False)
        from flashinfer.moe_ep.config import BootstrapConfig
        from flashinfer.moe_ep.core.runtime import (
            NVSHMEM,
            TORCH_DIST,
            sm90_pull_fp8_runtime_requirements,
        )

        bootstrap = BootstrapConfig(rank=0, world_size=1)
        backend = create_mega_kernel(_config())
        expected = frozenset({TORCH_DIST, NVSHMEM})
        assert sm90_pull_fp8_runtime_requirements(bootstrap) == expected
        assert backend.runtime_requirements(bootstrap) == expected

    def test_mega_no_dist_needs_nothing(self, monkeypatch) -> None:
        monkeypatch.setenv("MEGA_NO_DIST", "1")
        from flashinfer.moe_ep.config import BootstrapConfig
        from flashinfer.moe_ep.core.runtime import sm90_pull_fp8_runtime_requirements

        bootstrap = BootstrapConfig(rank=0, world_size=1)
        assert sm90_pull_fp8_runtime_requirements(bootstrap) == frozenset()


class TestSm90ArchGate:
    def test_validate_mega_arch_sm90_no_cuda_is_noop(self, monkeypatch) -> None:
        import torch

        from flashinfer.moe_ep.core.validation.common import validate_mega_arch_sm90

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        validate_mega_arch_sm90()  # must not raise on CPU-only hosts

    def test_validate_mega_arch_sm90_rejects_non_hopper(self, monkeypatch) -> None:
        import torch

        from flashinfer.moe_ep.core.validation import common as vcommon
        from flashinfer.moe_ep.core.validation.common import MoEEpArchError

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(vcommon, "_device_capability", lambda: (10, 0))
        with pytest.raises(MoEEpArchError, match="sm_90"):
            vcommon.validate_mega_arch_sm90()
        monkeypatch.setattr(vcommon, "_device_capability", lambda: (8, 0))
        with pytest.raises(MoEEpArchError, match="sm_90"):
            vcommon.validate_mega_arch_sm90()
        monkeypatch.setattr(vcommon, "_device_capability", lambda: (9, 0))
        vcommon.validate_mega_arch_sm90()  # exactly Hopper passes
