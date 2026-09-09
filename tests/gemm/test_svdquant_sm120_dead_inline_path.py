"""Source guards for the optional SM120 inline-LoRA-down experiment.

The optional CTA-level inline LoRA-down routine added by 711d72e never executes
under the production entry, but instantiating it changes ptxas allocation for the
whole kernel. Compiling it out is profitable for the persistent tiles and harmful
for the Stream-K and static-scheduler siblings, so the selection is per config.
These tests pin that contract and keep the three places that encode the list in
lockstep.
"""

import re
from pathlib import Path
from typing import Final


_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_GEMM_DIR: Final = _REPO_ROOT / "include" / "flashinfer" / "gemm"
_COLLECTIVE_PATH: Final = _GEMM_DIR / "nvfp4_svdquant_gemm_collective_sm120.h"
_TEMPLATE_PATH: Final = _GEMM_DIR / "nvfp4_svdquant_gemm_template_sm120.h"
_BINDING_PATH: Final = _REPO_ROOT / "csrc" / "nvfp4_svdquant_gemm_cutlass_sm120.cu"
_JINJA_PATH: Final = _REPO_ROOT / "csrc" / "nvfp4_svdquant_gemm_cutlass_sm120.jinja"

# Configs whose per-thread stack frame shrinks when the dead path is compiled
# out, measured with `cuobjdump -res-usage` on the campaign JIT module.
_EXPECTED_NO_INLINE_DOWN: Final = frozenset(
    {
        "Tactic128x128x128Config",
        "Tactic128x128x128SwapConfig",
        "Tactic128x128x128SkConfig",
        "Tactic128x128x128SwapSkConfig",
        "Tactic128x128x256Config",
        "Tactic128x128x256SwapConfig",
        "Tactic128x128x256SkConfig",
        "Tactic128x128x256SwapSkConfig",
        "Tactic256x128x128Config",
        "Tactic256x128x128SwapConfig",
        "Tactic128x256x128Config",
        "Tactic128x256x128SwapConfig",
        "Tactic128x64x128SwapConfig",
        "Tactic128x64x128SwapSkConfig",
        "Tactic128x64x256SwapConfig",
        "Tactic128x64x256SwapSkConfig",
        "Tactic128x32x128SwapConfig",
        "Tactic128x32x256SwapConfig",
    }
)

# Compiling the path out measurably grows these frames, so they must keep it.
_MUST_KEEP_INLINE_DOWN: Final = frozenset(
    {
        "Tactic256x128x128SkConfig",
        "Tactic256x128x128SwapSkConfig",
        "Tactic128x256x128SkConfig",
        "Tactic128x256x128SwapSkConfig",
        "Tactic256x128x128SwapStaticConfig",
    }
)


def _header_list() -> frozenset[str]:
    body = _TEMPLATE_PATH.read_text(encoding="utf-8")
    start = body.index("#define SVDQ_SM120_NO_INLINE_DOWN_CONFIG_LIST(X)")
    end = body.index(
        "template <class Config>\ninline constexpr bool kCompileInlineDownSm120", start
    )
    return frozenset(re.findall(r"X\((Tactic\w+)\)", body[start:end]))


def _jinja_list() -> frozenset[str]:
    body = _JINJA_PATH.read_text(encoding="utf-8")
    start = body.index("{% set NO_INLINE_DOWN_CONFIGS")
    end = body.index("%}", start)
    return frozenset(re.findall(r'"(Tactic\w+)"', body[start:end]))


def test_collective_still_exposes_the_compile_time_toggle() -> None:
    """The per-config toggle must remain a compile-time template parameter."""
    collective_source = _COLLECTIVE_PATH.read_text(encoding="utf-8")
    template_source = _TEMPLATE_PATH.read_text(encoding="utf-8")

    # No trailing ">" in this match: what matters is that the toggle is a
    # defaulted compile-time template parameter, not that it is the last one in
    # the list. Pinning the ">" made this fail the moment a parameter was added
    # after it, which says nothing about the toggle.
    assert "bool CompileInlineDown_ = true" in collective_source
    assert (
        "static constexpr bool CompileInlineDown = CompileInlineDown_;"
        in collective_source
    )
    # The LoRA rank reached the collective the same way, and for the same reason:
    # it has to be known at compile time for the smem overlay to be sized.
    assert "int LoRaK_ = 32" in collective_source
    assert "static constexpr int LoRaK = LoRaK_;" in collective_source
    assert "if constexpr (CollectiveMainloop::CompileInlineDown)" in collective_source
    assert "CollectiveMainloop::inline_lora_down_m537_k5120(" in collective_source
    assert "using CollectiveMainloop = CollectiveMainloopT<true>;" in template_source
    assert (
        "using CollectiveMainloopNoInlineDown = CollectiveMainloopT<false>;"
        in template_source
    )
    assert "run_tactic_no_inline_down" in template_source


def test_production_dispatch_selects_per_config_not_per_tactic_id() -> None:
    """The old `tactic == 9` dispatch covered one runtime id out of 38 affected shapes."""
    binding_source = _BINDING_PATH.read_text(encoding="utf-8")

    assert "if (tactic == sd::kNoInlineDownRuntimeTactic)" not in binding_source
    assert "if constexpr (sd::kCompileInlineDownSm120<C>)" in binding_source
    assert "return sd::run_tactic_no_inline_down<C>(" in binding_source


def test_selection_list_matches_the_measured_configs() -> None:
    """The header list is the single source of truth for the selection."""
    assert _header_list() == _EXPECTED_NO_INLINE_DOWN


def test_jinja_instantiates_exactly_the_selected_configs() -> None:
    """Drift guard: an entry added in one place but not the other fails to link."""
    assert _jinja_list() == _header_list()


def test_regressing_configs_keep_the_inline_down_path() -> None:
    """Configs whose frame grows when the path is removed must not be selected."""
    assert _MUST_KEEP_INLINE_DOWN.isdisjoint(_header_list())


def test_inline_down_entry_still_reaches_its_only_config() -> None:
    """nvfp4_svdquant_gemm_run_inline_down needs the CompileInlineDown=true build."""
    binding_source = _BINDING_PATH.read_text(encoding="utf-8")

    assert "using C = sd::Tactic256x128x128SwapStaticConfig;" in binding_source
    assert "Tactic256x128x128SwapStaticConfig" not in _header_list()
