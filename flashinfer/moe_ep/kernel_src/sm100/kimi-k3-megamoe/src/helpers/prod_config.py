# Exported for FlashInfer; local changes are recorded in VENDOR.md.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production defaults of the MegaMoE kernel's codegen switches (no env needed).

Every switch below is a bit-identical (exact) optimization that was A/B'd on GB300 and adopted for the config it is
ON for (docs/optimizations.md: v9 / v10 / v12 / v14 / v26 / v33, mx1 rounds 2-3). The defaults reproduce the
measured best kernels byte for byte (PTX), except where noted under ``KERNEL_FLAG_DEFAULTS``.

Env vars stay as OPTIONAL overrides for debugging / A/B: unset (or empty) = the default for the config being built;
``1`` forces a boolean switch on, ``0`` forces it off (``MEGAMOE_FC2_TAIL_BLOCKS`` / ``MEGAMOE_DISPATCH_REG_ROWS``
take an integer; ``0`` disables). Any other value raises. A forced-on switch is still subject to the kernel's own
applicability gates (e.g. EPI_WG2 needs N=256 + non-bulk FC2), exactly as before.

Config key (``config_key``): (quant family, combine family)
    quant family   "nv" = nvfp4 (e4m3 / 16 scales), "mx" = mxfp4_mxfp8 / mxfp4 (ue8m0 / 32 scales, paired amax)
    combine family "bf16" = BF16 combine, "lp" = quantized combine (32e4m3xe8m0 FP8, 16e2m1xbf16 FP4)
"""
from ..helpers import codegen_options as _kimi_options
import os

# Env name -> internal key, kind ("bool" | "int"). Diagnostic-only switches (stamps, IKET fine ranges, AB-stage cap,
# FC1_PIPE, ...) are NOT here: they keep their plain env reads with default off.
_ENV_KIND = {
    "MEGAMOE_SITU_FAST": "bool",
    "MEGAMOE_LEAN_SMEM_TRANS": "bool",
    "MEGAMOE_LEAN_SFIDX": "bool",
    "MEGAMOE_EPI_WG2": "bool",
    "MEGAMOE_DISPATCH_REG_ROWS": "int",
    "MEGAMOE_LEAN_FC2_STSM": "bool",
    "MEGAMOE_LEAN_FC2_QUANT": "bool",
    "MEGAMOE_LEAN_FC1_QUANT": "bool",
    "MEGAMOE_LEAN_FC2_ADDR": "bool",
    "MEGAMOE_LEAN_TILE_DRAIN": "bool",
    "MEGAMOE_LEAN_SCHED_FENCE": "bool",
    "MEGAMOE_EPI_ASYNC_FLAG": "bool",
    "MEGAMOE_EPI_FC2_ASYNC_FLAG": "bool",
    "MEGAMOE_MX_FC1_STAGE2": "bool",
    "MEGAMOE_MX_LEAN_T": "bool",
    "MEGAMOE_FC2_TAIL_BLOCKS": "int",
    "MEGAMOE_EPOCH_COUNTERS": "bool",
    "MEGAMOE_ROUTER_CAP": "int",
}

_V14 = {  # v14-epi3 lean set (S Q q A O F)
    "MEGAMOE_LEAN_FC2_STSM": True,
    "MEGAMOE_LEAN_FC2_QUANT": True,
    "MEGAMOE_LEAN_FC1_QUANT": True,
    "MEGAMOE_LEAN_FC2_ADDR": True,
    "MEGAMOE_LEAN_TILE_DRAIN": True,
    "MEGAMOE_LEAN_SCHED_FENCE": True,
}
_COMMON = {
    # v2 3-MUFU SiTU core: ON (user decision 11-01 ~23:45 EST, "keep SITU_FAST"). It is more accurate than the stock core
    # at the op (24 / 24 cases vs fp64) and mixed end to end (<= 4e-6 relative), and 2-7% faster on NVFP4 (epilogue-bound
    # at N240). MEGAMOE_SITU_FAST=0 selects the stock core: NVFP4 outputs then bit-identical to stock TRT-LLM.
    # Inert for SwiGLU.
    "MEGAMOE_SITU_FAST": True,
    "MEGAMOE_LEAN_SMEM_TRANS": True,  # v9 T (MX: only with MX_LEAN_T; inert otherwise)
    "MEGAMOE_LEAN_SFIDX": True,  # v9 k
    "MEGAMOE_DISPATCH_REG_ROWS": 1,  # v12 r1 (auto-off when hidden * act bits % 64 != 0)
    "MEGAMOE_EPOCH_COUNTERS": True,  # v33 (all configs; MX: new with prod1, see below)
    "MEGAMOE_MX_LEAN_T": False,  # mx1 T: not adopted
    "MEGAMOE_EPI_ASYNC_FLAG": False,
    "MEGAMOE_EPI_FC2_ASYNC_FLAG": False,  # s24: on by the epilogue for NVFP4 + quantized combine at N = 240
    "MEGAMOE_MX_FC1_STAGE2": False,
}

# ============================ PROD1 KERNEL FLAG DEFAULTS (per config) ============================
# Source of each default = the measured best env of that config (benchmark configs nv8 / nvbf / mx8 / mxbf):
#   nv/lp   nv8 : v14 set + EPI_WG2 + FC2_TAIL_BLOCKS=3 + EPOCH                    (TAC_C4_PI_E21)
#   nv/bf16 nvbf: no v14 set (measured neutral on BF16, never adopted: docs/optimizations.md "FP8 / BF16 combine status"),
#                 no WG2 / tail blocks (inapplicable to the bulk epi_warps tactic; verified PTX-inert) + EPOCH
#   mx/*    mx+sR: v14 set minus LEAN_FC2_STSM (= 's') + EPI_ASYNC_FLAG ('Y') + MX_FC1_STAGE2 ('R'); no WG2 (measured
#                 slower on MX: 2.876 vs 2.81 ms). NEW with prod1 (merge of the NVFP4 lineage, not yet A/B'd on MX):
#                 EPOCH_COUNTERS and FC2_TAIL_BLOCKS=3 (the latter only acts on the non-bulk FP8 tactic). Both are
#                 exact by construction; MEGAMOE_EPOCH_COUNTERS=0 MEGAMOE_FC2_TAIL_BLOCKS=0 gives the mx1 PTX back.
KERNEL_FLAG_DEFAULTS = {
    ("nv", "lp"): {**_COMMON, **_V14, "MEGAMOE_EPI_WG2": True, "MEGAMOE_FC2_TAIL_BLOCKS": 3},
    ("nv", "bf16"): {
        **_COMMON,
        **{k: False for k in _V14},
        "MEGAMOE_EPI_WG2": True,  # auto-gated: the BF16 default tactic (bulk FC2) never builds WG2
        "MEGAMOE_FC2_TAIL_BLOCKS": 3,  # inert for epi_warps token-back; acts if a non-bulk BF16 tactic is chosen
    },
    ("mx", "lp"): {
        **_COMMON,
        **_V14,
        "MEGAMOE_LEAN_FC2_STSM": False,
        "MEGAMOE_EPI_WG2": False,
        "MEGAMOE_EPI_ASYNC_FLAG": True,
        "MEGAMOE_MX_FC1_STAGE2": True,
        "MEGAMOE_FC2_TAIL_BLOCKS": 3,
    },
}
KERNEL_FLAG_DEFAULTS[("mx", "bf16")] = dict(KERNEL_FLAG_DEFAULTS[("mx", "lp")])
# ==================================================================================================


def config_key(quant_kind, combine_format) -> tuple:
    """``quant_kind``: QuantKind / str; ``combine_format``: CombineFormat / str (bf16 | 32e4m3xe8m0 | 16e2m1xbf16)."""
    qk = str(getattr(quant_kind, "value", quant_kind))
    fam = "nv" if qk == "nvfp4" else "mx"
    quantized = getattr(combine_format, "is_quantized", None)
    if quantized is None:
        quantized = str(combine_format) != "bf16"
    return fam, ("lp" if quantized else "bf16")


def _parse(env: str, raw: str, kind: str):
    if kind == "bool":
        if raw in ("0", "1"):
            return raw == "1"
        raise ValueError(f"{env}={raw!r}: expected 0 or 1 (unset = prod1 default).")
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{env}={raw!r}: expected an integer (unset = prod1 default).") from error


def env_override(env: str):
    """The env value of a prod1 switch, or None when unset / empty."""
    raw = _kimi_options.environ.get(env)
    if raw is None or raw.strip() == "":
        return None
    return _parse(env, raw.strip(), _ENV_KIND.get(env, "bool"))


_DEFAULTS_ACTIVE = None


def defaults_active() -> bool:
    """The adapter admits SM100/SM103 only."""
    return True


def _stock_default(env: str):
    return 0 if _ENV_KIND.get(env) == "int" else False


def flag(env: str, quant_kind, combine_format):
    """Resolved value of one switch for a config: env override if set, else the config default."""
    value = env_override(env)
    if value is not None:
        return value
    if not defaults_active():
        return _stock_default(env)
    return KERNEL_FLAG_DEFAULTS[config_key(quant_kind, combine_format)][env]


# NVFP4 + FP8 combine buckets on N240 + early token-back (2048 / 4096 / 8192 / 16384, i.e. max tokens/rank > 1024 under
# PROD1_DEFAULT_TACTICS): per-block FC2 readiness for the last 24 local experts instead of 3 (docs/optimizations.md "s21-skew", "s24-u4k").
# s21 (N240 buckets, 3 paired reps, cold / sustained): u4k -1.0 / -1.7%, u8k -1.5 / -0.5%, Gaussian 8k -0.8 / -0.65%,
# u16k -0.5 / -0.2%. With the 2048 bucket on N240 + TBX (s24-async2k), 24 also wins there: u2k -0.64 / -0.51%, Fg-u2k
# -2.0 / -1.5% vs prod1 (s24-z5). Smaller buckets (N256, no early token-back) keep 3.
FC2_TAIL_BLOCKS_N240 = 24
FC2_TAIL_BLOCKS_N240_MIN_T = 1025


def fc2_tail_blocks(quant_kind, combine_format, max_tokens_per_rank: int) -> int:
    """MEGAMOE_FC2_TAIL_BLOCKS for a config and bucket: env override if set, else the bucket-aware default."""
    value = env_override("MEGAMOE_FC2_TAIL_BLOCKS")
    if value is not None:
        return value
    default = flag("MEGAMOE_FC2_TAIL_BLOCKS", quant_kind, combine_format)
    if (
        defaults_active()
        and config_key(quant_kind, combine_format) == ("nv", "lp")
        and max_tokens_per_rank >= FC2_TAIL_BLOCKS_N240_MIN_T
    ):
        return FC2_TAIL_BLOCKS_N240
    return default


def epoch_enabled() -> bool:
    """v33 epoch counters: config-independent (ON for every Blackwell config), so trace-time code that has no problem
    description at hand (DeviceWorkspace, router) resolves it without one."""
    value = env_override("MEGAMOE_EPOCH_COUNTERS")
    if value is not None:
        return bool(value)
    return defaults_active()


# Router DATA-CTA capacity cap (s2-rcap, adopted 10-31): 2048 routed elements per CTA for max_tokens_per_rank <= 4096,
# off (stock geometry) above. Config-independent (the router is the same for every quant / combine).
ROUTER_CAP_DEFAULT = 2048
ROUTER_CAP_MAX_T = 4096


def router_cap(max_tokens_per_rank: int) -> int:
    """Router DATA-CTA element cap for this bucket (0 = stock); MEGAMOE_ROUTER_CAP overrides (0 = off)."""
    forced = env_override("MEGAMOE_ROUTER_CAP")
    if forced is not None:
        return forced
    return ROUTER_CAP_DEFAULT if max_tokens_per_rank <= ROUTER_CAP_MAX_T else 0
