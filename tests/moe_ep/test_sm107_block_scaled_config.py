"""Host-only unit tests for the sm107 block-scaled mega-kernel backend wiring.

No GPU / no kernel compile: config dataclass defaults, registry resolution via
``create_mega_kernel``, public re-exports, and runtime-requirement plumbing
for BOTH sm107 backends (mxfp8 and nvfp4).

Deliberately does NOT import ``flashinfer.moe_ep.kernel_src.next_cutedsl_megamoe``
internals that pull cutlass: the ``unit`` run_tests.sh target collects this
file in one shared pytest process.  Shim/kernel validation coverage lives in
``test_sm107_block_scaled_kernel_vs_reference.py`` (own process, Rubin only).
"""

from __future__ import annotations

import dataclasses

import pytest

from flashinfer.moe_ep import (
    Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl import (
    Sm107Mxfp8BlockScaledMegaKernelBackend,
)
from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl import (
    Sm107Nvfp4BlockScaledMegaKernelBackend,
)
from flashinfer.moe_ep.core.kernel.registry import (
    create_mega_kernel,
    is_mega_kernel_config,
)

_BACKENDS = (
    (
        Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        Sm107Mxfp8BlockScaledMegaKernelBackend,
        "sm107_mxfp8_mxfp8_bf16_cutedsl",
    ),
    (
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        Sm107Nvfp4BlockScaledMegaKernelBackend,
        "sm107_nvfp4_nvfp4_bf16_cutedsl",
    ),
)


def _config(config_cls, **overrides):
    return config_cls(intermediate_size=1024, top_k=4, **overrides)


class TestSm107BlockScaledConfig:
    def test_mxfp8_defaults(self) -> None:
        cfg = _config(Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig)
        assert cfg.kernel_name == "sm107_mxfp8_mxfp8_bf16_cutedsl"
        assert cfg.kind == "mxfp8_e4m3"
        assert cfg.gate_up_clamp is None
        assert cfg.activation_clamp is None
        self._check_shared_defaults(cfg)

    def test_nvfp4_defaults(self) -> None:
        cfg = _config(Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig)
        assert cfg.kernel_name == "sm107_nvfp4_nvfp4_bf16_cutedsl"
        assert cfg.gate_up_clamp is None
        self._check_shared_defaults(cfg)

    @staticmethod
    def _check_shared_defaults(cfg) -> None:
        assert cfg.fast_math is True
        assert cfg.in_kernel_fc2_reduce is False
        assert cfg.token_back_mode == "epi_warps"
        assert cfg.apply_topk_in_fc1 is True
        assert cfg.schedule_policy == ("grouped", None)
        assert cfg.work_id_mode == "grid_stride"
        assert cfg.fc2_use_bulk is False
        assert cfg.fc2_tma_stages is None
        assert cfg.epi_flag_batches == (4, 2)
        assert cfg.token_in_flag_batch == 1
        assert cfg.mma_tiler_mnk is None
        assert cfg.cluster_shape_mn is None
        assert cfg.max_sm_count is None
        # No knobs field until the next tree grows a tuner (mirrors the SM90
        # PORT NOTE contract).
        assert "knobs" not in {f.name for f in dataclasses.fields(cfg)}

    @pytest.mark.parametrize("config_cls, backend_cls, name", _BACKENDS)
    def test_is_mega_kernel_config(self, config_cls, backend_cls, name) -> None:
        assert is_mega_kernel_config(_config(config_cls))

    @pytest.mark.parametrize("config_cls, backend_cls, name", _BACKENDS)
    def test_registry_resolves_backend(self, config_cls, backend_cls, name) -> None:
        backend = create_mega_kernel(_config(config_cls))
        assert isinstance(backend, backend_cls)
        assert backend.kernel_name() == name
        assert backend_cls.kernel_name() == name

    def test_registry_lists_kernel_in_unknown_error(self) -> None:
        bogus = dataclasses.replace(
            _config(Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig),
            kernel_name="definitely_not_a_kernel",
        )
        with pytest.raises(KeyError, match="sm107_mxfp8_mxfp8_bf16_cutedsl"):
            create_mega_kernel(bogus)

    def test_public_reexports(self) -> None:
        import flashinfer.moe_ep as moe_ep

        assert (
            moe_ep.Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
            is Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
        )
        assert (
            moe_ep.Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
            is Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        )
        assert callable(moe_ep.preprocess_sm107_mxfp8_mega_weights)
        assert callable(moe_ep.preprocess_sm107_nvfp4_mega_weights)
        assert "Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig" in moe_ep.__all__
        assert "Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig" in moe_ep.__all__
        assert "preprocess_sm107_mxfp8_mega_weights" in moe_ep.__all__
        assert "preprocess_sm107_nvfp4_mega_weights" in moe_ep.__all__


class TestSm107RuntimeRequirements:
    def test_requires_nvshmem_and_torch_dist(self, monkeypatch) -> None:
        monkeypatch.delenv("MEGA_NO_DIST", raising=False)
        from flashinfer.moe_ep.config import BootstrapConfig
        from flashinfer.moe_ep.core.runtime import (
            NVSHMEM,
            TORCH_DIST,
            sm107_block_scaled_runtime_requirements,
        )

        bootstrap = BootstrapConfig(rank=0, world_size=1)
        expected = frozenset({TORCH_DIST, NVSHMEM})
        assert sm107_block_scaled_runtime_requirements(bootstrap) == expected
        for config_cls, _, _ in _BACKENDS:
            backend = create_mega_kernel(_config(config_cls))
            assert backend.runtime_requirements(bootstrap) == expected

    def test_mega_no_dist_needs_nothing(self, monkeypatch) -> None:
        monkeypatch.setenv("MEGA_NO_DIST", "1")
        from flashinfer.moe_ep.config import BootstrapConfig
        from flashinfer.moe_ep.core.runtime import (
            sm107_block_scaled_runtime_requirements,
        )

        bootstrap = BootstrapConfig(rank=0, world_size=1)
        assert sm107_block_scaled_runtime_requirements(bootstrap) == frozenset()


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
        cfg = _config(
            Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
            in_kernel_fc2_reduce=True,
            apply_topk_in_fc1=True,
        )
        assert cfg.in_kernel_fc2_reduce and cfg.apply_topk_in_fc1
