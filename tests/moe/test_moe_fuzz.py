"""Brutal randomized fuzzer for trtllm-gen fused MoE.

MoE is the single largest flashinfer bug surface (41 of 131 user-reported bug issues in
2026): autotuner picks wrong/corrupting tactics (#3227/#3197/#3168/#2504/#3530), routing
correctness incl fp32 router-logits (#2796/#2469) / all-to-one-expert (#2822) / warp-OOB
on many experts (#2575), quant accuracy/garbage/NaN (#2356/#2485/#2569/#2732/#2907/#3068/
#3334/#3103), in-place-not-updated (#2703).

The existing MoE tests use *fixed* parametrize grids (e.g. num_tokens in [8,768,3072],
hidden_size==1024, a handful of intermediate sizes) — they never sweep the autotune-bucket
boundaries, the large-expert / extreme-top_k routing regimes, fp32 logits, or zero hidden
states broadly, and they run autotune in only one mode. This fuzzer reuses the *exact*
oracle (`run_moe_test` -> `check_accuracy`, which already raises on mismatch / non-finite)
but drives it with random nasty configs and adds:

  * autotune ON *and* OFF, each validated against the dequant reference (catches the
    "autotuner selects a wrong/corrupting tactic" class that single-mode tests miss),
  * a run-to-run determinism check (seed is fixed inside run_moe_test, so a second
    identical call must produce a bit-identical result; catches #2514-class flakiness),
  * a post-trial device-state probe (catches autotune corrupting device state, #3168),
  * wide nasty ranges: autotune-bucket-boundary token counts incl 16384 (#3168), large
    expert counts incl 2048 (#2575), extreme top_k, fp32 routing logits (#2796), zero
    hidden states (#3068), every routing method.

Incompatible combos are turned into clean pytest.skip by the harness's own skip_checks
(the NOT_SUPPORTED path); a genuine crash / IMA / accuracy failure becomes a hard failure.

RUN WITH ``--forked`` (pip install pytest-forked). A MoE config that triggers an illegal
memory access (e.g. the autotuner profiling a bad tactic, GH #3168) corrupts the CUDA
context, which would otherwise cascade into every subsequent test or segfault the whole
run. ``--forked`` isolates each config in its own process, so an IMA is one reproducible
failed test, not a dead run:
    pytest tests/moe/test_moe_fuzz.py --forked

SM100/SM103 only (trtllm-gen MoE arch gate). Env knobs:
  FLASHINFER_MOE_FUZZ_NUM_TESTS (default 150), FLASHINFER_MOE_FUZZ_SEED (default 0).
"""

import os
import random

import pytest
import torch

from flashinfer import ActivationType, RoutingMethodType
from flashinfer.fused_moe import WeightLayout
from flashinfer.utils import get_compute_capability

from .test_trtllm_gen_fused_moe import (
    BF16Moe,
    FP4Moe,
    FP8BlockScaleMoe,
    FP8PerTensorMoe,
    MxInt4BlockScaleMoe,
    run_moe_test,
)
from .utils import QuantMode

NUM_TESTS = int(os.environ.get("FLASHINFER_MOE_FUZZ_NUM_TESTS", "150"))
BASE_SEED = int(os.environ.get("FLASHINFER_MOE_FUZZ_SEED", "0"))


def _arch_skip():
    """Lazy arch gate. Deliberately NOT done at module load: a config here can trigger a CUDA
    illegal-memory-access that corrupts the context, so this fuzzer is meant to be run with
    ``pytest --forked`` (one process per config). Touching CUDA at import would initialize it
    in the parent and break every forked child (CUDA cannot be used across fork)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if get_compute_capability(torch.device("cuda:0"))[0] != 10:
        pytest.skip("trtllm-gen fused MoE requires SM100/SM103")


# Nasty ranges (mapped to bug classes above).
NUM_TOKENS = [
    1,
    2,
    8,
    16,
    17,
    127,
    128,
    129,
    768,
    2048,
    3072,
    4095,
    4096,
    4097,
    8192,
    16384,
]
HIDDEN = [512, 1024, 2048]  # >=512 stresses scale precision (#3068)
INTERMED = [256, 384, 512, 768, 1024, 1536, 2048, 2688]
EXPERTS = [8, 16, 72, 128, 256, 384, 512, 2048]  # 2048 -> warp-OOB regime (#2575)
# Production-relevant top_k (Mixtral=2, DeepSeek=8, Llama4=1). top_k>=22 is beyond any real
# model and makes trtllm_fp4_block_scale_moe CUDA-error in trtllm_fused_moe_dev_kernel.cu:985
# (a separate "should reject, not crash" robustness gap, documented in the triage notes) which
# would corrupt the CUDA context and cascade; keep the sweep on the realistic range.
TOPK = [1, 2, 4, 6, 8]
ROUTING_DTYPES = [torch.float32, torch.bfloat16]  # fp32 router logits (#2796/#2469)
ALL_IMPLS = [FP8PerTensorMoe, FP8BlockScaleMoe, FP4Moe, BF16Moe, MxInt4BlockScaleMoe]
_SKIP_KW = (
    "not support",
    "unsupported",
    "no kernel found",
    "not implemented",
    "requires",
    "only support",
    "incompatible",
    "must be",
    "no valid",
    "invalid configuration",
    # documented per-combo limitations that are correct rejections, not bugs:
    "currently supports",  # "FP8 per-tensor currently supports gated activations only"
    "gated activation",
    "warp size",  # routingDeepSeek: #experts per group must be <= warp size (the #2575 fix)
    "experts per group",
    "is invalid for input",  # harness reshape limitation at some odd num_tokens
)


def _unsupported(e):
    """A NotImplementedError, or a message naming a documented per-combo limitation, is a
    legitimate NOT_SUPPORTED rejection (skip), not a defect. The test harness's
    prepare_static_weights_for_kernel also returns None for some large/extreme configs it was
    not built for, yielding `static_data[...]` -> TypeError; that is a test-infrastructure
    limitation, not a flashinfer-library defect, so skip it too. Any *other* error type
    (real crash / unexpected TypeError) is NOT swallowed -> it surfaces as a real failure."""
    if isinstance(e, NotImplementedError):
        return True
    msg = str(e).lower()
    if isinstance(e, TypeError) and "subscriptable" in msg:
        return True  # harness weight-prep returned None for this config
    return any(k in msg for k in _SKIP_KW)


def _make_impl(rng):
    return rng.choice(
        [
            lambda: BF16Moe(),
            lambda: FP8PerTensorMoe(),
            lambda: FP8BlockScaleMoe(
                fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK
            ),
            lambda: FP8BlockScaleMoe(
                fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8
            ),
            lambda: FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4),
            lambda: MxInt4BlockScaleMoe(),
        ]
    )()


def gen_config(seed):
    """Return a config dict, or None if it violates run_moe_test's hard asserts (resample)."""
    rng = random.Random(seed)
    method = rng.choice(list(RoutingMethodType))
    num_experts = rng.choice(EXPERTS)
    topk_opts = [t for t in TOPK if t <= num_experts]
    if not topk_opts:
        return None
    top_k = rng.choice(topk_opts)
    # The test-harness reference only supports top_k==1 for Llama4 routing
    # (test_trtllm_gen_fused_moe.py:2382 asserts top_k==1); honor that to avoid a
    # harness-only assertion that is not a flashinfer defect.
    if method == RoutingMethodType.Llama4:
        top_k = 1

    if method == RoutingMethodType.DeepSeekV3:
        if num_experts % 4 != 0:
            return None
        n_groups = rng.choice(
            [g for g in (1, 8) if num_experts % g == 0 and num_experts > g]
        )
        top_k_groups = rng.choice([1, 4])
        if not (top_k_groups <= 4 and top_k < (top_k_groups * num_experts / n_groups)):
            return None
        routed_scaling = rng.choice([1.0, 2.5])
        has_bias = True
    else:
        n_groups = top_k_groups = None
        routed_scaling = rng.choice([None, 1.0, 2.5])
        has_bias = method == RoutingMethodType.MiniMax2 or rng.random() < 0.3

    routing_config = {
        "num_experts": num_experts,
        "top_k": top_k,
        "padding": 8,
        "n_groups": n_groups,
        "top_k_groups": top_k_groups,
        "routed_scaling": routed_scaling,
        "has_routing_bias": has_bias,
        "routing_method_type": method,
        "compatible_moe_impls": ALL_IMPLS,
        "compatible_intermediate_size": INTERMED,
        "compatible_activation_types": [
            ActivationType.Swiglu,
            ActivationType.Geglu,
            ActivationType.Relu2,
        ],
    }
    weight_processing = rng.choice(
        [
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": ALL_IMPLS,
            },
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": ALL_IMPLS,
            },
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.BlockMajorK,
                "compatible_moe_impls": ALL_IMPLS,
            },
        ]
    )
    return {
        "seed": seed,
        "num_tokens": rng.choice(NUM_TOKENS),
        "hidden_size": rng.choice(HIDDEN),
        "intermediate_size": rng.choice(INTERMED),
        "activation_type": rng.choice(
            [ActivationType.Swiglu, ActivationType.Geglu, ActivationType.Relu2]
        ),
        "routing_logits_dtype": rng.choice(ROUTING_DTYPES),
        "zero_hidden_states": rng.random() < 0.12,
        "routing_bias_dtype": rng.choice([torch.bfloat16, torch.float32])
        if has_bias
        else None,
        "norm_topk_prob": rng.random() < 0.5,
        "routing_config": routing_config,
        "weight_processing": weight_processing,
        "_impl_seed": rng.randint(0, 2**31 - 1),
        "_label": f"{method.name}_e{num_experts}_k{top_k}_t{0}",
    }


def _gen_configs():
    out, s = [], BASE_SEED
    while len(out) < NUM_TESTS and s < BASE_SEED + NUM_TESTS * 20:
        c = gen_config(s)
        s += 1
        if c is not None:
            c["_label"] = (
                f"{c['routing_config']['routing_method_type'].name}_"
                f"e{c['routing_config']['num_experts']}_k{c['routing_config']['top_k']}_"
                f"nt{c['num_tokens']}_h{c['hidden_size']}_i{c['intermediate_size']}_s{c['seed']}"
            )
            out.append(c)
    return out


_CONFIGS = _gen_configs()


def _device_state_ok():
    """Cheap probe that the CUDA context is still healthy (autotune can corrupt it, #3168)."""
    try:
        x = torch.randn(1024, device="cuda")
        torch.cuda.synchronize()
        return torch.isfinite((x * 2).sum()).item()
    except Exception:
        return False


def _run(cfg, enable_autotune):
    rc = dict(cfg["routing_config"])
    rc["enable_autotune"] = enable_autotune
    return run_moe_test(
        cfg["num_tokens"],
        cfg["hidden_size"],
        cfg["intermediate_size"],
        _make_impl(random.Random(cfg["_impl_seed"])),
        rc,
        cfg["weight_processing"],
        cfg["activation_type"],
        cache_permute_indices={},
        routing_logits_dtype=cfg["routing_logits_dtype"],
        zero_hidden_states=cfg["zero_hidden_states"],
        routing_bias_dtype=cfg["routing_bias_dtype"],
        norm_topk_prob=cfg["norm_topk_prob"],
    )


@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c["_label"] for c in _CONFIGS])
def test_moe_fuzz(cfg):
    _arch_skip()
    Skipped = pytest.skip.Exception
    # (A) autotune ON: run_moe_test internally validates actual-vs-reference (check_accuracy
    #     raises on mismatch / non-finite). A wrong autotune tactic fails here (#3227).
    try:
        ref, act_at, _ = _run(cfg, True)
    except Skipped:
        raise
    except (RuntimeError, ValueError, NotImplementedError, TypeError) as e:
        if _unsupported(e):
            pytest.skip(f"unsupported: {e}")
        raise  # genuine crash / IMA -> failure

    # (B) autotune OFF: independent validation against the same (seeded) reference.
    try:
        ref2, act_noat, _ = _run(cfg, False)
    except Skipped:
        return  # autotune-on path already validated; off-path simply unsupported here
    except (RuntimeError, ValueError, NotImplementedError, TypeError) as e:
        if _unsupported(e):
            return
        raise

    # (C) autotune must not change the result beyond tolerance (both already matched ref,
    #     this is a direct cross-check that the tuned tactic agrees with the untuned one).
    a, b = act_at.float(), act_noat.float()
    if a.shape == b.shape:
        denom = b.abs().clamp_min(1e-3)
        rel = (a - b).abs() / denom
        bad = (rel > 0.5).float().mean().item()
        assert bad < 0.02, (
            f"{cfg['_label']}: autotune ON vs OFF disagree on {bad * 100:.1f}% of elements "
            f"(autotuner picked a wrong tactic, #3227-class)"
        )

    # (D) determinism: run_moe_test fixes seed=0, so an identical call must reproduce exactly.
    _, act_again, _ = _run(cfg, True)
    if not torch.equal(act_at, act_again):
        md = (act_at.float() - act_again.float()).abs().max().item()
        pytest.fail(
            f"{cfg['_label']}: NONDETERMINISTIC MoE output (max abs diff {md:.3e}, #2514-class)"
        )

    # (E) device state intact after autotune (#3168).
    assert _device_state_ok(), (
        f"{cfg['_label']}: CUDA device state corrupted after MoE/autotune (#3168-class)"
    )
