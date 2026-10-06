from __future__ import annotations



def test_mhc_decode_split_n_environment_override(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.setenv("B12X_MHC_DECODE_SPLITS", "4")
    monkeypatch.setenv("B12X_MHC_DECODE_TILE_N", "3")
    assert residual_kernels._selected_post_pre_decode_split_n(
        num_tokens=16,
        hidden_size=4096,
        compute_capability=(12, 1),
    ) == (4, 3)


def test_mhc_sm121_decode_split_n_policy(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.delenv("B12X_MHC_DECODE_SPLITS", raising=False)
    monkeypatch.delenv("B12X_MHC_DECODE_TILE_N", raising=False)
    select = residual_kernels._selected_post_pre_decode_split_n

    assert select(num_tokens=4, hidden_size=4096, compute_capability=(12, 1)) == (0, 0)
    assert select(num_tokens=8, hidden_size=4096, compute_capability=(12, 1)) == (4, 6)
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 1)) == (8, 6)
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 0)) == (0, 0)
    assert select(num_tokens=16, hidden_size=7168, compute_capability=(12, 1)) == (0, 0)


def test_mhc_decode_finalize_threads_environment_override(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.setenv("B12X_MHC_DECODE_FINALIZE_THREADS", "128")
    assert (
        residual_kernels._selected_mhc_decode_finalize_threads(
            num_tokens=16,
            hidden_size=4096,
            compute_capability=(12, 1),
        )
        == 128
    )


def test_mhc_sm121_decode_finalize_policy(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.delenv("B12X_MHC_DECODE_FINALIZE_THREADS", raising=False)
    select = residual_kernels._selected_mhc_decode_finalize_threads

    assert select(num_tokens=4, hidden_size=4096, compute_capability=(12, 1)) == 0
    assert select(num_tokens=8, hidden_size=4096, compute_capability=(12, 1)) == 512
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 1)) == 128
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 0)) == 0
    assert select(num_tokens=16, hidden_size=7168, compute_capability=(12, 1)) == 0


def test_mhc_sm121_decode_partial_group_policy(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.delenv("B12X_MHC_PARTIALS_PER_CTA", raising=False)
    select = residual_kernels._selected_post_pre_partials_per_cta

    assert select(num_tokens=2, hidden_size=4096, compute_capability=(12, 1)) == 4
    assert select(num_tokens=4, hidden_size=4096, compute_capability=(12, 1)) == 9
    assert select(num_tokens=8, hidden_size=4096, compute_capability=(12, 1)) == 25
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 1)) == 25


def test_mhc_decode_partial_group_policy_preserves_sm120(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.delenv("B12X_MHC_PARTIALS_PER_CTA", raising=False)
    select = residual_kernels._selected_post_pre_partials_per_cta

    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 0)) == 4
    assert select(num_tokens=16, hidden_size=7168, compute_capability=(12, 1)) == 4


def test_mhc_decode_partial_group_environment_override(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.setenv("B12X_MHC_PARTIALS_PER_CTA", "7")
    assert (
        residual_kernels._selected_post_pre_partials_per_cta(
            num_tokens=16,
            hidden_size=4096,
            compute_capability=(12, 1),
        )
        == 7
    )
