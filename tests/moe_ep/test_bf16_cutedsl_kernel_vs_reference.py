"""Single-rank BF16 CuTeDSL MegaMoE integration smoke."""

from __future__ import annotations

import pytest


@pytest.mark.arch_blackwell
def test_bf16_megamoe_public_reference_is_lazy():
    """Keep the CPU import boundary free of the CuTeDSL reference dependency."""
    import flashinfer.moe_ep.kernel_src.cutedsl_megamoe as megamoe

    assert callable(megamoe.get_symm_buffer_for_bf16_mega_moe)
    assert callable(megamoe.bf16_mega_moe)
    # Resolving the raw reference happens only on a GPU test host with CuTeDSL.
    assert "compute_megamoe_reference_bf16" in megamoe.__all__
