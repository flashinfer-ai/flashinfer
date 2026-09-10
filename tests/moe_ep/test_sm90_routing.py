"""Pure-host contracts for deterministic SM90 benchmark routing."""

from __future__ import annotations

import numpy as np
import pytest

import flashinfer.moe_ep.sm90_routing as routing


_STANDARD_EXACT_HASHES = {
    8: "1ba40a6fb0ab731b9085979a1968c60aa6b3a5fa3e13b444f2c0a55bcfb8aa00",
    32: "5499b7ae730372fb6ae53f29b852a07ec10aeeaa11d1e4e99424f3edd9be16ce",
    64: "d78a2b4df5bb769238a2528a76ccf80a980074bdb02c88152fbefbbd0d21e90e",
    128: "1209a05edefdc700fb8d45b54c2291b62d410cdc2934e801ca05f0a84a38b06f",
    256: "415b8f862a97ea9cc498bbc150e1e6d5d7b7111c27b387dcba95169386b1d7e2",
    512: "f5306ed4f8d1fd685fedf370c96e942f715b9481367be5932200e36d444379de",
    1024: "5999065601264efc000004684321ef46c4c1996b6531ecdbd985e8a617ec7dd5",
    2048: "85c6311af059960c02445ee051e3950991f34dc1979e537081d00c7f5da40b53",
}


def _assert_exact_balanced(
    routes: np.ndarray,
    *,
    world_size: int,
    tokens: int,
    topk: int,
    total_experts: int,
    seed: int,
) -> None:
    del seed
    assert routes.shape == (world_size, tokens, topk)
    assert routes.dtype == np.int32
    assert np.all(routes >= 0) and np.all(routes < total_experts)
    assert not np.any(np.diff(np.sort(routes, axis=2), axis=2) == 0)

    expert_counts = np.bincount(routes.reshape(-1), minlength=total_experts)
    assert int(expert_counts.max()) - int(expert_counts.min()) <= 1
    local_experts = total_experts // world_size
    expected_owner_rows = tokens * topk // world_size
    for source_rank in range(world_size):
        owner_counts = np.bincount(
            routes[source_rank].reshape(-1) // local_experts,
            minlength=world_size,
        )
        assert np.all(owner_counts == expected_owner_rows)


@pytest.mark.parametrize(
    ("tokens", "expected_hash"), list(_STANDARD_EXACT_HASHES.items())
)
def test_published_exact_standard_hashes_do_not_change(
    monkeypatch: pytest.MonkeyPatch, tokens: int, expected_hash: str
) -> None:
    def unexpected_fallback(**kwargs: object) -> np.ndarray:
        pytest.fail(f"standard shape unexpectedly used fallback: {kwargs!r}")

    monkeypatch.setattr(
        routing,
        "_generate_sm90_published_exact_balanced_routes_cyclic_numpy",
        unexpected_fallback,
    )
    routes = routing.generate_sm90_published_exact_balanced_routes_numpy(
        world_size=4,
        tokens=tokens,
        topk=6,
        total_experts=384,
        seed=1234,
    )
    assert routing.sm90_route_ids_sha256(routes) == expected_hash


def test_published_exact_fallback_only_runs_after_greedy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = routing._generate_sm90_published_exact_balanced_routes_cyclic_numpy
    fallback_seeds: list[int] = []

    def observed_fallback(**kwargs: object) -> np.ndarray:
        fallback_seeds.append(int(kwargs["seed"]))
        return original(**kwargs)

    monkeypatch.setattr(
        routing,
        "_generate_sm90_published_exact_balanced_routes_cyclic_numpy",
        observed_fallback,
    )
    common = dict(world_size=4, tokens=3, topk=4, total_experts=8)
    routing.generate_sm90_published_exact_balanced_routes_numpy(**common, seed=0)
    assert fallback_seeds == []
    routes = routing.generate_sm90_published_exact_balanced_routes_numpy(
        **common, seed=1
    )
    assert fallback_seeds == [1]
    _assert_exact_balanced(routes, **common, seed=1)


def test_published_exact_small_regression_all_seeds() -> None:
    common = dict(world_size=4, tokens=3, topk=4, total_experts=8)
    for seed in range(100):
        first = routing.generate_sm90_published_exact_balanced_routes_numpy(
            **common, seed=seed
        )
        second = routing.generate_sm90_published_exact_balanced_routes_numpy(
            **common, seed=seed
        )
        assert np.array_equal(first, second)
        _assert_exact_balanced(first, **common, seed=seed)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_published_exact_small_legal_domain(seed: int) -> None:
    for world_size in range(1, 5):
        for local_experts in range(1, 4):
            total_experts = world_size * local_experts
            for tokens in range(6):
                for topk in range(1, total_experts + 1):
                    if tokens and (tokens * topk) % world_size:
                        continue
                    kwargs = dict(
                        world_size=world_size,
                        tokens=tokens,
                        topk=topk,
                        total_experts=total_experts,
                        seed=seed,
                    )
                    routes = (
                        routing.generate_sm90_published_exact_balanced_routes_numpy(
                            **kwargs
                        )
                    )
                    _assert_exact_balanced(routes, **kwargs)
