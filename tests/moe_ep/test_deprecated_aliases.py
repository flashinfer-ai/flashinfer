"""Deprecated pre-taxonomy name coverage (no CUDA, no comms).

The taxonomy rename (kernel/config names of the form
``sm<arch>_<act>_<weight>_<out>_<style>``) kept the old spellings alive for
external callers such as the vLLM integration patch, via two mechanisms:

- registry aliases: ``register_mega_kernel(..., deprecated_aliases=...)``
  resolves the old kernel_name and emits a DeprecationWarning on use;
- config class aliases in ``flashinfer.moe_ep.__init__``.

These tests pin both down so a re-sync or refactor can't silently drop them.
"""

from __future__ import annotations

import pytest

import flashinfer.moe_ep as moe_ep
from flashinfer.moe_ep.core.kernel import registry

# (deprecated kernel_name, canonical kernel_name, deprecated config class,
#  canonical config class) — one row per pre-taxonomy backend spelling.
ALIAS_ROWS = [
    (
        "deep_gemm_mega",
        "sm100_fp8_fp4_bf16_deepgemm",
        "DeepGemmMegaMoeConfig",
        "Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig",
    ),
    (
        "mxfp8_cutedsl",
        "sm100_mxfp8_mxfp8_bf16_cutedsl",
        "Mxfp8CutedslMegaMoeConfig",
        "Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig",
    ),
    (
        "nvfp4_cutedsl",
        "sm100_nvfp4_nvfp4_bf16_cutedsl",
        "Nvfp4CutedslMegaMoeConfig",
        "Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
    ),
    (
        "sm90_pull_fp8",
        "sm90_fp8_fp8_bf16_pull_cutedsl",
        "Sm90PullFp8MegaMoeConfig",
        "Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig",
    ),
]

IDS = [row[0] for row in ALIAS_ROWS]


@pytest.mark.parametrize(("alias", "canonical", "_old", "_new"), ALIAS_ROWS, ids=IDS)
def test_registry_alias_resolves_to_canonical_backend(
    alias: str, canonical: str, _old: str, _new: str
) -> None:
    assert registry._MEGA_KERNEL_REGISTRY[alias] is (
        registry._MEGA_KERNEL_REGISTRY[canonical]
    )
    assert registry._MEGA_KERNEL_DEPRECATED_ALIASES[alias] == canonical


@pytest.mark.parametrize(("alias", "canonical", "_old", "_new"), ALIAS_ROWS, ids=IDS)
def test_registry_alias_warns_deprecation(
    alias: str, canonical: str, _old: str, _new: str
) -> None:
    with pytest.warns(DeprecationWarning, match=canonical):
        registry._warn_if_deprecated_mega_name(alias)


@pytest.mark.parametrize(("alias", "canonical", "_old", "_new"), ALIAS_ROWS, ids=IDS)
def test_canonical_name_does_not_warn(
    alias: str, canonical: str, _old: str, _new: str
) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        registry._warn_if_deprecated_mega_name(canonical)


def test_available_mega_kernels_hides_aliases() -> None:
    available = registry._available_mega_kernels()
    for alias, canonical, _old, _new in ALIAS_ROWS:
        assert canonical in available
        assert alias not in available


@pytest.mark.parametrize(("alias", "_canonical", "old", "new"), ALIAS_ROWS, ids=IDS)
def test_config_class_alias_is_same_class(
    alias: str, _canonical: str, old: str, new: str
) -> None:
    assert getattr(moe_ep, old) is getattr(moe_ep, new)


@pytest.mark.parametrize(("alias", "_canonical", "old", "_new"), ALIAS_ROWS, ids=IDS)
def test_alias_config_is_mega_kernel_config(
    alias: str, _canonical: str, old: str, _new: str
) -> None:
    # An external caller overriding kernel_name with the old spelling must
    # still route to the mega path.
    class _Stub:
        kernel_name = alias

    assert registry.is_mega_kernel_config(_Stub())
