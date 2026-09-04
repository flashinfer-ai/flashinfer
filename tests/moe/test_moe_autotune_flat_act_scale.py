"""Regression tests for the flat (1-D) MXFP8 ``hidden_states_scale`` layout.

``flashinfer.mxfp8_quantize`` natively returns a FLAT scale buffer of
``num_tokens * hidden_size // 32`` elements (see
``flashinfer/quantization/fp8_quantization.py``), and TensorRT-LLM forwards that
buffer to ``trtllm_fp4_block_scale_moe`` unmodified.  Every in-repo caller
happens to reshape it to ``[num_tokens, -1]`` first, so the MoE autotuner's
assumption that dim 0 is the token count went untested against the flat layout
and aborted with::

    AssertionError: hidden_states_scale shape (96,) does not match expected
    layout (num_tokens=1, ...)

The C++ launcher accepts any rank here (it derives the SF vector size from
``numel()`` alone), so the assertion — not the kernel — was the bug.

The subtle part is the fix, not the acceptance: the autotuner writes a bucket's
token count *verbatim* into a dynamic dim, so simply returning dim 0 for a flat
tensor would profile against a ``(bucket,)`` scale where the kernel indexes
``bucket * sf_per_token`` elements — a silent out-of-bounds read, strictly worse
than the crash.  ``test_flat_act_scale_profiles_scale_with_sf_per_token`` is the
test that guards against that, and it is the important one.

These tests are CPU-only: the JIT module is mocked, and only shape bookkeeping is
exercised, so no GPU or compiled kernel is required.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

import flashinfer.fused_moe.core as core_mod
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.core import MoeRunnerInputs
from flashinfer.tllm_enums import (
    DtypeTrtllmGen,
    Fp8QuantizationType,
    RoutingInputMode,
)

# GPT-OSS-20b shape from the original TensorRT-LLM report: one token, padded
# hidden 3072, MXFP8 block size 32 -> a (96,) flat activation scale.
HIDDEN_SIZE = 3072
SF_BLOCK_SIZE = 32
SF_PER_TOKEN = HIDDEN_SIZE // SF_BLOCK_SIZE  # 96
NUM_EXPERTS = 128
TOP_K = 8


@pytest.fixture(autouse=True)
def _clear_moe_module_cache():
    """Evict the mocked module from the process-wide cache after every test.

    ``_get_trtllm_moe_sm100_module_impl`` is ``functools.cache``d and the cached
    closure captures ``module.build_and_load()``. Without this teardown the
    MagicMock built below would be served to every later caller in the same
    pytest session, silently breaking unrelated MoE tests.
    """
    yield
    core_mod._get_trtllm_moe_sm100_module_impl.cache_clear()


def _make_runner(fp8_quantization_type=Fp8QuantizationType.NoneFp8):
    """Build a MoERunner with the JIT/cubin machinery mocked out (CPU-safe)."""
    fn = core_mod._get_trtllm_moe_sm100_module_impl
    fn.cache_clear()
    mock_module = MagicMock()
    mock_module.get_library_path.return_value = "/tmp/fake.so"
    with (
        patch.object(
            core_mod,
            "gen_trtllm_gen_fused_moe_sm100_module",
            return_value=mock_module,
        ),
        patch.object(core_mod, "setup_cubin_loader"),
    ):
        MoERunner = fn(enable_rubin=False).MoERunner

    return MoERunner(
        top_k=TOP_K,
        num_local_experts=NUM_EXPERTS,
        dtype_act=DtypeTrtllmGen.MxE4m3,
        dtype_weights=DtypeTrtllmGen.MxE2m1,
        fp8_quantization_type=fp8_quantization_type,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=3072,
        num_experts=NUM_EXPERTS,
    )


def _make_inputs(num_tokens: int, hidden_states_scale):
    return MoeRunnerInputs(
        output=torch.empty((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16),
        routing_logits=None,
        topk_ids=torch.zeros((num_tokens, TOP_K), dtype=torch.int32),
        expert_weights=None,
        hidden_states=torch.empty((num_tokens, HIDDEN_SIZE), dtype=torch.float8_e4m3fn),
        hidden_states_scale=hidden_states_scale,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )


def _flat_scale(num_tokens: int) -> torch.Tensor:
    """The buffer mxfp8_quantize returns: flat, uint8, no token dimension."""
    return torch.ones(num_tokens * SF_PER_TOKEN, dtype=torch.uint8)


def _scale_idx() -> int:
    return MoeRunnerInputs.idx("hidden_states_scale")


@pytest.mark.parametrize("num_tokens", [1, 3, 8])
def test_flat_act_scale_is_accepted(num_tokens):
    """A flat mxfp8 scale must not be rejected by the tuning-config builder.

    This is the exact assertion from the bug report.  It fails on unfixed main.
    """
    runner = _make_runner()
    moe_inputs = _make_inputs(num_tokens, _flat_scale(num_tokens))

    config = runner._make_tuning_config(
        moe_inputs, routing_input_mode=RoutingInputMode.PackedPrecomputed
    )

    # The flat scale must be driven by a ConstraintSpec, never by a dynamic dim:
    # a dynamic dim would be assigned the raw bucket token count.
    scale_idx = _scale_idx()
    dyn_spec = config.dynamic_tensor_specs[0]
    assert scale_idx not in dyn_spec.input_idx, (
        "flat hidden_states_scale must be excluded from the DynamicTensorSpec; "
        "the autotuner assigns such a dim the bucket's token count verbatim, "
        "which under-allocates the scale buffer by sf_per_token."
    )
    assert [c.input_idx for c in config.constraint_specs] == [scale_idx]

    # It must stay in the cold-L2 profiling arena even though it left the
    # dynamic spec (profile_arena_input_indices is derived from the full
    # input set, not from the dynamic spec).
    assert scale_idx in config.profile_arena_input_indices


def test_flat_act_scale_profiles_scale_with_sf_per_token():
    """THE decisive test: profiled scale numel must be bucket * sf_per_token.

    If the flat scale were left as a dynamic dim, the autotuner would allocate a
    ``(bucket,)`` tensor while the kernel indexes ``bucket * 96`` elements — an
    out-of-bounds device read during profiling rather than a loud failure.
    """
    runner = _make_runner()
    num_tokens = 1
    moe_inputs = _make_inputs(num_tokens, _flat_scale(num_tokens))
    config = runner._make_tuning_config(
        moe_inputs,
        tune_max_num_tokens=1024,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
    )

    tuner = AutoTuner.get()
    inputs = moe_inputs.to_list()
    profiles = tuner._generate_optimization_profiles(config, inputs)
    assert profiles, "expected at least one optimization profile"

    scale_idx = _scale_idx()
    hidden_idx = MoeRunnerInputs.idx("hidden_states")

    for profile in profiles:
        shapes = profile.get_opt_shapes()
        bucket = shapes[hidden_idx][0]
        assert shapes[scale_idx] == (bucket * SF_PER_TOKEN,), (
            f"profiled scale shape {shapes[scale_idx]} for bucket {bucket}; "
            f"expected ({bucket * SF_PER_TOKEN},). A shape of ({bucket},) means "
            f"the scale is being resized as a token-count dim, which would make "
            f"the kernel read {SF_PER_TOKEN}x past the end of the buffer."
        )

    # And the tensors actually handed to the kernel must match those shapes.
    prepared = tuner._prepare_input_tensors(profiles[-1], inputs)
    bucket = profiles[-1].get_opt_shapes()[hidden_idx][0]
    assert prepared[scale_idx].numel() == bucket * SF_PER_TOKEN
    assert prepared[scale_idx].dtype == torch.uint8
    assert prepared[scale_idx] is not inputs[scale_idx]


def test_two_dim_act_scale_mapping_is_unchanged():
    """The pre-existing 2-D layout must keep using dim 0 and no ConstraintSpec.

    Guards the autotuner cache key that
    tests/moe/test_trtllm_gen_moe_autotune_tactics.py::_moe_profile_shapes
    hardcodes for 2-D scales (constrained dims are written as -1 in that key).
    """
    runner = _make_runner()
    num_tokens = 4
    scale_2d = torch.ones((num_tokens, SF_PER_TOKEN), dtype=torch.uint8)
    config = runner._make_tuning_config(
        _make_inputs(num_tokens, scale_2d),
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
    )

    assert config.constraint_specs == ()
    dyn_spec = config.dynamic_tensor_specs[0]
    scale_idx = _scale_idx()
    assert scale_idx in dyn_spec.input_idx
    assert dyn_spec.dim_idx[dyn_spec.input_idx.index(scale_idx)] == 0


def test_deepseek_fp8_act_scale_mapping_is_unchanged():
    """DeepSeekFp8 keeps its [hidden//128, num_tokens] contract and dim 1."""
    runner = _make_runner(fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8)
    num_tokens = 4
    scale = torch.ones((HIDDEN_SIZE // 128, num_tokens), dtype=torch.float32)
    config = runner._make_tuning_config(
        _make_inputs(num_tokens, scale),
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
    )

    assert config.constraint_specs == ()
    dyn_spec = config.dynamic_tensor_specs[0]
    scale_idx = _scale_idx()
    assert dyn_spec.dim_idx[dyn_spec.input_idx.index(scale_idx)] == 1


def test_malformed_act_scale_is_still_rejected():
    """Relaxing the layout check must not stop catching genuinely bad scales."""
    runner = _make_runner()
    num_tokens = 5

    # Flat, but numel is not a whole number of per-token scale groups.
    with pytest.raises(AssertionError, match="not a multiple of num_tokens"):
        runner._make_tuning_config(
            _make_inputs(num_tokens, torch.ones(97, dtype=torch.uint8)),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )

    # 2-D with the wrong token count on dim 0.
    with pytest.raises(AssertionError, match="does not match expected layout"):
        runner._make_tuning_config(
            _make_inputs(
                num_tokens,
                torch.ones((num_tokens + 1, SF_PER_TOKEN), dtype=torch.uint8),
            ),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )


def test_flat_act_sf_inferrer_is_cached():
    """The ConstraintSpec callback must be identity-stable across calls.

    ConstraintSpec is hashed into AutoTuner._find_nearest_profile's lru_cache
    key, so a fresh closure per inference call would grow that cache without
    bound (the same hazard make_hybrid_bucket_mapper documents).
    """
    runner = _make_runner()
    hidden_idx = MoeRunnerInputs.idx("hidden_states")

    configs = [
        runner._make_tuning_config(
            _make_inputs(1, _flat_scale(1)),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
        for _ in range(2)
    ]
    infer_a, infer_b = (c.constraint_specs[0].infer_shape for c in configs)
    assert infer_a is infer_b
    assert (
        core_mod._make_flat_act_sf_numel_inferrer(hidden_idx, SF_PER_TOKEN) is infer_a
    )
    # Equal configs must therefore also hash equally.
    assert hash(configs[0].constraint_specs) == hash(configs[1].constraint_specs)
