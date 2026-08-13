import torch

import flashinfer.fused_moe as fused_moe
import flashinfer.fused_moe.core as core


def test_cutlass_fused_moe_valid_profile_ids_public_api(monkeypatch):
    calls = []

    class FakeModule:
        @staticmethod
        def get_valid_profile_ids(*args, **kwargs):
            calls.append((args, kwargs))
            return [3, 5], [101, 103]

    monkeypatch.setattr(core, "get_compute_capability", lambda device: (9, 0))
    monkeypatch.setattr(
        core, "get_cutlass_fused_moe_module", lambda backend: FakeModule()
    )

    x = torch.empty((12, 64), dtype=torch.bfloat16)
    w1 = torch.empty((8, 128, 32), dtype=torch.uint8)
    w2 = torch.empty((8, 64, 64), dtype=torch.uint8)

    gemm1_ids, gemm2_ids = fused_moe.get_cutlass_fused_moe_valid_profile_ids(
        x,
        w1,
        w2,
        torch.bfloat16,
        top_k=2,
        use_w4_group_scaling=True,
        use_packed_weights=True,
    )

    assert gemm1_ids == [3, 5]
    assert gemm2_ids == [101, 103]
    assert len(calls) == 1
    assert calls[0][0][:3] == (x, w1, w2)
    assert calls[0][1]["top_k"] == 2
    assert calls[0][1]["use_w4_group_scaling"] is True
    assert calls[0][1]["use_packed_weights"] is True
