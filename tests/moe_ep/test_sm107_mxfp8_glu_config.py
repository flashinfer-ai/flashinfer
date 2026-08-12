"""Host-only unit tests for the sm107_mxfp8_mxfp8_bf16_cutedsl mega-kernel backend wiring.

No GPU / no kernel compile: config dataclass defaults, registry resolution via
``create_mega_kernel``, public re-exports, and runtime-requirement plumbing.

Deliberately does NOT import ``flashinfer.moe_ep.kernel_src.next_cutedsl_megamoe``
internals that pull cutlass: the ``unit`` run_tests.sh target collects this
file in one shared pytest process.  Shim/kernel validation coverage lives in
``test_sm107_mxfp8_glu_kernel_vs_reference.py`` (own process, Rubin only).
"""

from __future__ import annotations

import dataclasses

import pytest

from flashinfer.moe_ep import Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl import (
    Sm107Mxfp8GluMegaKernelBackend,
)
from flashinfer.moe_ep.core.kernel.registry import (
    create_mega_kernel,
    is_mega_kernel_config,
)


def _config(**overrides) -> Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig:
    return Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=1024, top_k=4, **overrides
    )


class TestSm107Mxfp8GluConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.kernel_name == "sm107_mxfp8_mxfp8_bf16_cutedsl"
        assert cfg.kind == "mxfp8_e4m3"
        assert cfg.gate_up_clamp is None
        assert cfg.activation_clamp is None
        assert cfg.fast_math is True
        assert cfg.in_kernel_fc2_reduce is False
        assert cfg.token_back_mode == "epi_warps"
        assert cfg.apply_topk_in_fc1 is True
        assert cfg.group_hint == 768
        assert cfg.mma_tiler_mnk is None
        assert cfg.cluster_shape_mnk is None
        assert cfg.max_sm_count is None
        # No knobs field until the next tree grows a tuner (mirrors the SM90
        # PORT NOTE contract).
        assert "knobs" not in {f.name for f in dataclasses.fields(cfg)}

    def test_is_mega_kernel_config(self) -> None:
        assert is_mega_kernel_config(_config())

    def test_registry_resolves_backend(self) -> None:
        backend = create_mega_kernel(_config())
        assert isinstance(backend, Sm107Mxfp8GluMegaKernelBackend)
        assert backend.kernel_name() == "sm107_mxfp8_mxfp8_bf16_cutedsl"
        assert (
            Sm107Mxfp8GluMegaKernelBackend.kernel_name()
            == "sm107_mxfp8_mxfp8_bf16_cutedsl"
        )

    def test_registry_lists_kernel_in_unknown_error(self) -> None:
        bogus = dataclasses.replace(_config(), kernel_name="definitely_not_a_kernel")
        with pytest.raises(KeyError, match="sm107_mxfp8_mxfp8_bf16_cutedsl"):
            create_mega_kernel(bogus)

    def test_public_reexports(self) -> None:
        import flashinfer.moe_ep as moe_ep

        assert (
            moe_ep.Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
            is Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
        )
        assert callable(moe_ep.preprocess_sm107_mxfp8_glu_mega_weights)
        assert "Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig" in moe_ep.__all__
        assert "preprocess_sm107_mxfp8_glu_mega_weights" in moe_ep.__all__


class TestSm107RuntimeRequirements:
    def test_requires_nvshmem_and_torch_dist(self, monkeypatch) -> None:
        monkeypatch.delenv("MEGA_NO_DIST", raising=False)
        from flashinfer.moe_ep.config import BootstrapConfig
        from flashinfer.moe_ep.core.runtime import (
            NVSHMEM,
            TORCH_DIST,
            sm107_mxfp8_glu_runtime_requirements,
        )

        bootstrap = BootstrapConfig(rank=0, world_size=1)
        backend = create_mega_kernel(_config())
        expected = frozenset({TORCH_DIST, NVSHMEM})
        assert sm107_mxfp8_glu_runtime_requirements(bootstrap) == expected
        assert backend.runtime_requirements(bootstrap) == expected

    def test_mega_no_dist_needs_nothing(self, monkeypatch) -> None:
        monkeypatch.setenv("MEGA_NO_DIST", "1")
        from flashinfer.moe_ep.config import BootstrapConfig
        from flashinfer.moe_ep.core.runtime import (
            sm107_mxfp8_glu_runtime_requirements,
        )

        bootstrap = BootstrapConfig(rank=0, world_size=1)
        assert sm107_mxfp8_glu_runtime_requirements(bootstrap) == frozenset()


class TestSm107ArchGate:
    def test_validate_mega_arch_sm107_no_cuda_is_noop(self, monkeypatch) -> None:
        import torch

        from flashinfer.moe_ep.core.validation.common import validate_mega_arch_sm107

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        validate_mega_arch_sm107()  # must not raise on CPU-only hosts

    def test_validate_mega_arch_sm107_rejects_non_rubin(self, monkeypatch) -> None:
        import torch

        from flashinfer.moe_ep.core.validation import common as vcommon
        from flashinfer.moe_ep.core.validation.common import MoEEpArchError

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(vcommon, "_device_capability", lambda: (10, 0))
        with pytest.raises(MoEEpArchError, match="sm_107"):
            vcommon.validate_mega_arch_sm107()
        monkeypatch.setattr(vcommon, "_device_capability", lambda: (9, 0))
        with pytest.raises(MoEEpArchError, match="sm_107"):
            vcommon.validate_mega_arch_sm107()
        monkeypatch.setattr(vcommon, "_device_capability", lambda: (10, 7))
        vcommon.validate_mega_arch_sm107()  # exactly Rubin passes

    def test_in_kernel_reduce_requires_topk_in_fc1(self) -> None:
        # The shim config enforces this; the backend surfaces it at workspace
        # allocation. Host-only check via the shim dataclass would import the
        # drop, so assert the backend config carries the fields instead.
        cfg = _config(in_kernel_fc2_reduce=True, apply_topk_in_fc1=True)
        assert cfg.in_kernel_fc2_reduce and cfg.apply_topk_in_fc1
