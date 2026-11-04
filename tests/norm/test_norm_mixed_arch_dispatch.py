# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for norm dispatch compatibility on mixed-architecture hosts.

Issue: #5355
When the CuTe DSL's compile target is captured from device 0 and differs
from the target device's architecture, norm should fall back to the CUDA JIT
path instead of attempting to run an incompatible kernel.
"""

import pytest
import torch

from flashinfer.norm import _is_arch_compatible, _use_cuda_norm


class TestArchCompatibility:
    """Test the CUDA binary compatibility logic for architecture matching.

    These tests verify that _is_arch_compatible() correctly implements CUDA's
    forward compatibility rules: a cubin compiled for sm_X.Y can run on sm_X.Z
    when Z >= Y (same major architecture, target minor version >= captured).
    """

    def test_same_arch_exact_match(self):
        """Verify exact architecture matches are compatible.

        When the captured architecture exactly matches the target device's
        compute capability, the kernel should be compatible and use the DSL path.

        Test cases:
        - sm_86 captured, sm_86 device: compatible
        - sm_89 captured, sm_89 device: compatible
        - sm_100a captured, sm_100 device: compatible (variant suffix stripped)
        """
        assert _is_arch_compatible("sm_86", (8, 6))
        assert _is_arch_compatible("sm_89", (8, 9))
        assert _is_arch_compatible("sm_100a", (10, 0))

    def test_forward_compatibility_same_major(self):
        """Verify forward compatibility within the same major architecture.

        CUDA allows forward compatibility: a cubin compiled for an older minor
        version can run on a newer minor version within the same major architecture.
        However, backward compatibility (running on older minor versions) is not
        guaranteed.

        Test cases:
        - sm_86 cubin on sm_89 device: compatible (forward, 8.9 >= 8.6)
        - sm_89 cubin on sm_86 device: incompatible (backward, 8.6 < 8.9)
        - sm_100a cubin on sm_103 device: compatible (forward, 10.3 >= 10.0)
        - sm_103 cubin on sm_100 device: incompatible (backward, 10.0 < 10.3)
        """
        # sm_86 cubin can run on sm_89 (same major, higher minor)
        assert _is_arch_compatible("sm_86", (8, 9))
        assert not _is_arch_compatible("sm_89", (8, 6))

        # sm_100a can run on sm_103 (same major, higher minor)
        assert _is_arch_compatible("sm_100a", (10, 3))
        assert not _is_arch_compatible("sm_103", (10, 0))

    def test_cross_major_incompatible(self):
        """Different major architectures are incompatible."""
        # Hopper (sm_90) vs Ampere (sm_86)
        assert not _is_arch_compatible("sm_90", (8, 6))
        assert not _is_arch_compatible("sm_86", (9, 0))

        # Blackwell (sm_100) vs Hopper (sm_90)
        assert not _is_arch_compatible("sm_100a", (9, 0))
        assert not _is_arch_compatible("sm_90", (10, 0))

    def test_arch_variants(self):
        """Architecture variants (a/f suffixes) should be handled correctly."""
        # sm_100a and sm_100f both map to (10, 0)
        assert _is_arch_compatible("sm_100a", (10, 0))
        assert _is_arch_compatible("sm_100f", (10, 0))
        assert _is_arch_compatible("sm_100", (10, 0))

    def test_invalid_arch_string(self):
        """Invalid architecture strings should return False."""
        assert not _is_arch_compatible("", (8, 6))
        assert not _is_arch_compatible("invalid", (8, 6))
        assert not _is_arch_compatible("86", (8, 6))  # Missing "sm_" prefix
        assert not _is_arch_compatible("sm_", (8, 6))  # Empty after prefix


class TestUseCudaNormDispatch:
    """Test _use_cuda_norm dispatch logic with mixed architectures."""

    @pytest.fixture
    def mock_cuda_devices(self, monkeypatch):
        """Mock CUDA device capabilities for testing."""

        def set_device_caps(caps_list):
            """caps_list: [(major, minor), ...] for cuda:0, cuda:1, ..."""

            def mock_get_device_capability(device):
                if isinstance(device, torch.device):
                    idx = device.index or 0
                else:
                    idx = int(device)
                return caps_list[idx]

            monkeypatch.setattr(
                torch.cuda, "get_device_capability", mock_get_device_capability
            )

        return set_device_caps

    def test_explicit_cuda_norm_env(self, monkeypatch):
        """FLASHINFER_USE_CUDA_NORM=1 should always return True."""
        import flashinfer.norm as norm_module

        monkeypatch.setattr(norm_module, "_USE_CUDA_NORM", True)

        device = torch.device("cuda:0")
        assert _use_cuda_norm(device) is True

    def test_single_arch_uses_dsl(self, monkeypatch, mock_cuda_devices):
        """On single-arch host, DSL path should be used when supported."""
        import flashinfer.norm as norm_module

        monkeypatch.setattr(norm_module, "_USE_CUDA_NORM", False)
        monkeypatch.setattr(norm_module, "_cute_dsl_supports_arch", lambda m, n: True)

        # Mock DSL captured arch matching device 0
        monkeypatch.setattr(
            norm_module,
            "_dsl_captured_arch",
            lambda: "sm_89",
        )

        # Single sm_89 device
        mock_cuda_devices([(8, 9)])

        device = torch.device("cuda:0")
        assert _use_cuda_norm(device) is False  # Use DSL path

    def test_mixed_arch_falls_back_to_cuda(self, monkeypatch, mock_cuda_devices):
        """Mixed-arch host should fall back to CUDA JIT when DSL target != device."""
        import flashinfer.norm as norm_module

        monkeypatch.setattr(norm_module, "_USE_CUDA_NORM", False)
        monkeypatch.setattr(norm_module, "_cute_dsl_supports_arch", lambda m, n: True)

        # Mock DSL captured arch from device 0 (sm_89)
        monkeypatch.setattr(
            norm_module,
            "_dsl_captured_arch",
            lambda: "sm_89",
        )

        # Mixed host: device 0 is sm_89, device 1 is sm_86
        mock_cuda_devices([(8, 9), (8, 6)])

        # On cuda:0 (sm_89), DSL path should be used
        device0 = torch.device("cuda:0")
        assert _use_cuda_norm(device0) is False

        # On cuda:1 (sm_86), DSL compiled for sm_89 should NOT be compatible
        # sm_89 cubin cannot run on sm_86 (backward compatibility not guaranteed)
        device1 = torch.device("cuda:1")
        assert _use_cuda_norm(device1) is True  # Fall back to CUDA JIT

    def test_forward_compat_same_major(self, monkeypatch, mock_cuda_devices):
        """Forward compatibility within same major should work."""
        import flashinfer.norm as norm_module

        monkeypatch.setattr(norm_module, "_USE_CUDA_NORM", False)
        monkeypatch.setattr(norm_module, "_cute_dsl_supports_arch", lambda m, n: True)

        # DSL compiled for sm_86
        monkeypatch.setattr(
            norm_module,
            "_dsl_captured_arch",
            lambda: "sm_86",
        )

        # Device is sm_89 (same major, higher minor)
        mock_cuda_devices([(8, 9)])

        device = torch.device("cuda:0")
        # sm_86 cubin can run on sm_89 (forward compatibility)
        assert _use_cuda_norm(device) is False  # DSL path OK

    def test_dsl_arch_none_uses_dsl(self, monkeypatch, mock_cuda_devices):
        """When DSL captured arch is unavailable, use DSL path if arch supported."""
        import flashinfer.norm as norm_module

        monkeypatch.setattr(norm_module, "_USE_CUDA_NORM", False)
        monkeypatch.setattr(norm_module, "_cute_dsl_supports_arch", lambda m, n: True)
        monkeypatch.setattr(norm_module, "_dsl_captured_arch", lambda: None)

        mock_cuda_devices([(8, 9)])

        device = torch.device("cuda:0")
        # Cannot determine captured arch, assume DSL is OK
        assert _use_cuda_norm(device) is False

    def test_dsl_not_installed_falls_back(self, monkeypatch, mock_cuda_devices):
        """When DSL is not installed, _dsl_captured_arch is None."""
        import flashinfer.norm as norm_module

        monkeypatch.setattr(norm_module, "_USE_CUDA_NORM", False)
        monkeypatch.setattr(norm_module, "_cute_dsl_supports_arch", lambda m, n: True)
        monkeypatch.setattr(norm_module, "_dsl_captured_arch", None)

        mock_cuda_devices([(8, 9)])

        device = torch.device("cuda:0")
        assert _use_cuda_norm(device) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
