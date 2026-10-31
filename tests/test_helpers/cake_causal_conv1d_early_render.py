"""Test-only shim for the early render of the non-deterministic causal conv1d package.

That render's backward rules predate the required ``slots`` field of an ``accumulators`` workspace rule (the
package refuses such a rule, so a render that drops the field never undersizes ``dw_acc``).  Its backward programs,
listed here by name, keep one accumulator copy each.  :func:`name_early_render_slots` names ``slots: 1`` on exactly
those records when their rule lacks the field; a render that names ``slots`` itself, and every other record, is
left alone.  The package code never applies a default.
"""

from __future__ import annotations

from types import ModuleType

EARLY_RENDER_BACKWARD = frozenset(
    {
        "cake_causal_conv1d_train_012b6319a0cc717911b9",
        "cake_causal_conv1d_train_064063666308a8552401",
        "cake_causal_conv1d_train_0cc1521134686f49dbc9",
        "cake_causal_conv1d_train_17d30c46fe34f0266e26",
        "cake_causal_conv1d_train_32c5625f33586925f154",
        "cake_causal_conv1d_train_383cb5839559d5b0e72b",
        "cake_causal_conv1d_train_461d358d7090642c65de",
        "cake_causal_conv1d_train_5ed71ee5908c07376183",
        "cake_causal_conv1d_train_61b7fb3c4a7d481011eb",
        "cake_causal_conv1d_train_a47e12df7005fb5fb72b",
        "cake_causal_conv1d_train_a79ed3e33712abf525ad",
        "cake_causal_conv1d_train_befc06774f6fee7c36ce",
        "cake_causal_conv1d_train_c3d9c23921f5573d0577",
        "cake_causal_conv1d_train_c798878a383c652c428d",
        "cake_causal_conv1d_train_d542ad2292353c56bd65",
        "cake_causal_conv1d_train_d7f458b69c0249f63c37",
        "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
        "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
        "cake_causal_conv1d_train_584216799fed8caece3a_kcp",
        "cake_causal_conv1d_train_777e24f8771c3f7abdf8_kcp",
        "cake_causal_conv1d_train_d994e1963ad7724c9c64_kcp",
    }
)


def name_early_render_slots(*registries: ModuleType) -> int:
    """Name ``slots: 1`` on the early render's backward records that lack it; returns how many it named."""

    named = 0
    for registry in registries:
        for name, record in registry.MODULES.items():
            rule = record.get("workspace_rule")
            if (
                rule is not None
                and rule.get("kind") == "accumulators"
                and "slots" not in rule
                and name in EARLY_RENDER_BACKWARD
            ):
                rule["slots"] = 1
                named += 1
    return named


def apply_to_package() -> int:
    """:func:`name_early_render_slots` on the package's single-rank and KCP registries (imported through
    :func:`tests.test_helpers.cake_package.cake_package`, so a vendored copy gets the same shim)."""

    from tests.test_helpers.cake_package import cake_package

    key = "cake.cake_causal_conv1d_train"
    return name_early_render_slots(
        cake_package(key, "cake_jit"), cake_package(key, "kcp_jit")
    )
