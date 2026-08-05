"""CPU-only policy tests for the long paged direct-decode route."""

import torch

from flashinfer.msa_ops._cake_sm100 import (
    _is_long_paged_gqa16_direct_decode,
    _select_decode_route,
)


HEAD_DIM = 128
BLOCK_SIZE = 128


def _long_decode_tensors():
    q = torch.empty((512, 64, HEAD_DIM), dtype=torch.bfloat16)
    k = torch.empty((32768, 4, BLOCK_SIZE, HEAD_DIM), dtype=torch.bfloat16)
    return q, k


def test_long_decode_uses_direct_selected_block_route() -> None:
    q, k = _long_decode_tensors()
    common = {
        "q": q,
        "k": k,
        "cu_k": torch.empty(65, dtype=torch.int32),
        "kv_lens": torch.empty(64, dtype=torch.int32),
        "group_size": 16,
        "seqlen_q": 8,
        "paged": True,
        "force_fused": True,
        "workspace": None,
        "route_key": ("long-decode-direct",),
        "capturing": False,
    }

    assert _is_long_paged_gqa16_direct_decode(
        q=q,
        k=k,
        group_size=16,
        seqlen_q=8,
        paged=True,
        force_fused=True,
    )
    assert _select_decode_route(**common) == ("decode", False, True)


def test_long_decode_route_rejects_neighboring_geometries() -> None:
    q, k = _long_decode_tensors()
    common = {
        "q": q,
        "k": k,
        "group_size": 16,
        "seqlen_q": 8,
        "paged": True,
        "force_fused": True,
    }

    for changed in (
        {"seqlen_q": 4},
        {"paged": False},
        {"force_fused": False},
        {"q": torch.empty((256, 64, HEAD_DIM), dtype=torch.bfloat16)},
    ):
        assert not _is_long_paged_gqa16_direct_decode(**{**common, **changed})


def test_neighboring_q4_keeps_folded_m128_route() -> None:
    q = torch.empty((256, 64, HEAD_DIM), dtype=torch.bfloat16)
    k = torch.empty((32768, 4, BLOCK_SIZE, HEAD_DIM), dtype=torch.bfloat16)

    assert _select_decode_route(
        q=q,
        k=k,
        cu_k=torch.empty(65, dtype=torch.int32),
        kv_lens=torch.empty(64, dtype=torch.int32),
        group_size=16,
        seqlen_q=4,
        paged=True,
        force_fused=True,
        workspace=None,
        route_key=("long-decode-neighbor",),
        capturing=False,
    ) == ("m128", False, None)
