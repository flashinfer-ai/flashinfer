"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Naming-contract tests for the b12x MoE CuTe-DSL disk cache adopter.

Replicates the contract enforced for the earlier adopters in
``tests/jit/test_cute_dsl_cache.py``, as the design doc's rollout note asks
of new adopters. The kernel-name string is the sole per-kernel cache key --
the module ``meta.json`` guards only arch / DSL version / source hashes -- so
a name that ignores a codegen parameter makes two different kernels collide
on one artifact and the cache silently serves the wrong binary.

1. Signature coverage: every parameter of each kernel getter is expressible
   in the corresponding cache-key function.
2. Per-argument perturbation: changing any single argument changes the name.
3. Symbol safety: names are valid filename/symbol components as produced.
"""

import inspect
import re

import pytest

pytest.importorskip("cutlass")

import torch  # noqa: E402

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md  # noqa: E402

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (  # noqa: E402
    _disk_kernel_name,
    _dynamic_kernel_cache_key,
    _get_dynamic_kernel,
    _get_micro_kernel,
    _get_static_kernel,
    _micro_kernel_cache_key,
    _static_kernel_cache_key,
)

# Getter parameters that deliberately do NOT participate in the cache key.
#
# Each of these only *selects* a value that is itself keyed, so the artifact is
# keyed on what reaches codegen rather than on how it was chosen:
#
# * ``mac_override`` selects the max-active-clusters value, and the resulting
#   ``mac`` is part of every key;
# * ``tile_m`` (dynamic kernel) becomes ``mma_tiler_mn[0]``, and
#   ``mma_tiler_mn`` is part of every key;
# * ``intermediate_size`` (dynamic kernel) selects the branch-major operand
#   extent: when the gated kernel can stream the true intermediate extent it
#   becomes the keyed ``branch_major_extent`` (and the compile fakes' N), and
#   when it cannot the kernel indexes the tile-padded views whose extent is the
#   keyed ``n`` - both outcomes reach codegen only through keyed fields.
NON_CODEGEN_PARAMS = {"mac_override", "tile_m", "intermediate_size"}

STATIC_BASELINE = {
    "activation_precision": "fp4",
    "quant_mode": "nvfp4",
    "state_E": 32,
    "weight_E": 32,
    "m": 64,
    "k": 2048,
    "n": 1024,
    "num_topk": 4,
    "max_rows": 256,
    "mac": 48,
    "mma_tiler_mn": (128, 128),
    "topk_ids_dtype": torch.int32,
    "input_scales_are_reciprocal": False,
    "fast_math": True,
    "activation": "silu",
    "swiglu_alpha": 1.702,
    "swiglu_beta": 1.0,
    "swiglu_limit": None,
    # true weight extent, static workspace rows, merged-groups schedule, deferred init
    "weight_n": 1024,
    "route_rows": 256,
    "merged_groups": False,
    "deferred_init": False,
    "source_scales": False,
}
STATIC_PERTURBED = {
    "activation_precision": "w4a16",
    "quant_mode": "mxfp4",
    "state_E": 16,
    "weight_E": 16,
    "m": 128,
    "k": 4096,
    "n": 2048,
    "num_topk": 2,
    "max_rows": 512,
    "mac": 96,
    "mma_tiler_mn": (128, 64),  # not a transpose of the baseline square tile
    "topk_ids_dtype": torch.int64,
    "input_scales_are_reciprocal": True,
    "fast_math": False,
    "activation": "gelu",
    "swiglu_alpha": -1.702,  # sign flip: sanitized text alone would collide
    "swiglu_beta": 2.0,
    "swiglu_limit": 7.0,
    "weight_n": 320,  # padded n 2048 above vs a true extent below it
    "route_rows": 8192,
    "merged_groups": True,
    "deferred_init": True,
    "source_scales": True,
}

# Fields of the static key that the micro kernel does not specialize on (it
# indexes the tile-padded views, sizes no workspace and has one schedule).
STATIC_ONLY_FIELDS = {
    "activation_precision",
    "weight_n",
    "route_rows",
    "merged_groups",
    "deferred_init",
    "source_scales",
}

MICRO_BASELINE = {
    k: v for k, v in STATIC_BASELINE.items() if k not in STATIC_ONLY_FIELDS
}
MICRO_BASELINE.update(
    share_input_across_experts=False, share_expert_scales=False, single_token=False
)
MICRO_PERTURBED = {
    k: v for k, v in STATIC_PERTURBED.items() if k not in STATIC_ONLY_FIELDS
}
MICRO_PERTURBED.update(
    share_input_across_experts=True, share_expert_scales=True, single_token=True
)

DYNAMIC_BASELINE = {
    "activation_precision": "fp4",
    "quant_mode": "nvfp4",
    "E": 32,
    "k": 2048,
    "n": 1024,
    "num_topk": 4,
    "mac": 48,
    "mma_tiler_mn": (128, 128),
    "topk_ids_dtype": torch.int32,
    "input_scales_are_reciprocal": False,
    "fast_math": True,
    "activation": "silu",
    "swiglu_alpha": 1.702,
    "swiglu_beta": 1.0,
    "swiglu_limit": None,
    "share_input_across_experts": False,
}
DYNAMIC_PERTURBED = {
    "activation_precision": "w4a16",
    "quant_mode": "mxfp4",
    "E": 16,
    "k": 4096,
    "n": 2048,
    "num_topk": 2,
    "mac": 96,
    "mma_tiler_mn": (128, 64),
    "topk_ids_dtype": torch.int64,
    "input_scales_are_reciprocal": True,
    "fast_math": False,
    "activation": "gelu",
    "swiglu_alpha": -1.702,
    "swiglu_beta": 2.0,
    "swiglu_limit": 7.0,
    "share_input_across_experts": True,
}

ADOPTERS = [
    (
        "static",
        _get_static_kernel,
        _static_kernel_cache_key,
        STATIC_BASELINE,
        STATIC_PERTURBED,
    ),
    (
        "micro",
        _get_micro_kernel,
        _micro_kernel_cache_key,
        MICRO_BASELINE,
        MICRO_PERTURBED,
    ),
    (
        "dynamic",
        _get_dynamic_kernel,
        _dynamic_kernel_cache_key,
        DYNAMIC_BASELINE,
        DYNAMIC_PERTURBED,
    ),
]

# Getter parameters absent from a key function because that kernel genuinely
# does not specialize on them: the dynamic kernel takes its runtime-shaped
# operands as pointers, so one artifact serves every m / max_rows.
KEY_OMISSIONS = {"dynamic": {"m", "max_rows"}}


@pytest.mark.parametrize("label,getter,key_fn", [(a[0], a[1], a[2]) for a in ADOPTERS])
def test_key_signature_covers_getter_params(label, getter, key_fn):
    """Every kernel-getter parameter must be expressible in the cache key.

    Fails the moment a parameter is added to a getter without threading it
    into the key (and therefore into the on-disk artifact name).
    """
    getter_params = set(inspect.signature(getter).parameters)
    key_params = set(inspect.signature(key_fn).parameters)
    missing = (
        getter_params
        - key_params
        - NON_CODEGEN_PARAMS
        - KEY_OMISSIONS.get(label, set())
    )
    assert not missing, (
        f"{getter.__name__} has codegen parameter(s) {sorted(missing)} that "
        f"{key_fn.__name__} cannot encode. Add them to the key function (or, "
        "if provably non-codegen, to NON_CODEGEN_PARAMS / KEY_OMISSIONS with a "
        "justification)."
    )


@pytest.mark.parametrize(
    "label,key_fn,baseline,perturbed,param",
    [(a[0], a[2], a[3], a[4], p) for a in ADOPTERS for p in sorted(a[3])],
)
def test_disk_name_varies_with_every_argument(
    label, key_fn, baseline, perturbed, param
):
    """Changing any single codegen argument must change the on-disk name."""
    baseline_name = _disk_kernel_name(label, key_fn(**baseline))
    kwargs = dict(baseline)
    kwargs[param] = perturbed[param]
    perturbed_name = _disk_kernel_name(label, key_fn(**kwargs))
    assert perturbed_name != baseline_name, (
        f"the {label} kernel's on-disk name ignores argument {param!r}: two "
        "different kernel specializations would collide on one cache artifact."
    )


@pytest.mark.parametrize(
    "label,key_fn,baseline", [(a[0], a[2], a[3]) for a in ADOPTERS]
)
def test_disk_name_is_symbol_safe(label, key_fn, baseline):
    """Names must already be valid symbol/filename components.

    ``JitSpecCuteDsl`` sanitizes names before use; a name relying on that
    sanitization could collide with a different name that sanitizes to the
    same string, so the raw name must not need it.
    """
    name = _disk_kernel_name(f"{label}_m64_k2048", key_fn(**baseline))
    assert re.fullmatch(r"[0-9A-Za-z_]+", name), name


@pytest.mark.parametrize(
    "label,key_fn,baseline", [(a[0], a[2], a[3]) for a in ADOPTERS]
)
def test_disk_name_is_stable_for_equal_keys(label, key_fn, baseline):
    """The same key must map to the same artifact name within a process.

    Guards against a name derived from anything unstable (object identity,
    iteration order); without this the cache would never hit.
    """
    first = _disk_kernel_name(label, key_fn(**baseline))
    second = _disk_kernel_name(label, key_fn(**dict(baseline)))
    assert first == second


def test_kernel_types_do_not_collide():
    """The three kernel families must never share an artifact name."""
    names = {
        _disk_kernel_name(label, key_fn(**baseline))
        for label, _, key_fn, baseline, _ in ADOPTERS
    }
    assert len(names) == len(ADOPTERS)


Q35 = dict(
    quant_mode="nvfp4",
    num_experts=256,
    intermediate_size=512,
    hidden_size=2048,
    activation="silu",
    num_topk=8,
    capacity_tokens=8192,
)
Q38 = dict(
    quant_mode="nvfp4",
    num_experts=512,
    intermediate_size=320,
    hidden_size=2560,
    activation="silu",
    num_topk=10,
    capacity_tokens=8192,
)
SM = md.get_num_sm(torch.device("cuda")) if torch.cuda.is_available() else 0
measured_device = pytest.mark.skipif(
    SM != 110, reason="the measured entries are for the 110-SM SM120 device"
)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cutover resolution needs a CUDA device"
)
class TestCutoverRegistry:
    @pytest.fixture(autouse=True)
    def _clean_cutover_state(self, monkeypatch):
        for name in (
            "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS",
            "B12X_STATIC_COMPACT_CUTOVER_PAIRS",
            "B12X_DYNAMIC_STATIC_CUTOVER_PAIRS",
            "B12X_LEVEL10_STATIC_CUTOVER_PAIRS",
        ):
            monkeypatch.delenv(name, raising=False)
        md._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
        yield
        md._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()

    def test_registry_is_keyed_by_the_full_exact_key(self):
        for key, pairs in md._STATIC_CUTOVER_REGISTRY.items():
            assert len(key) == 8 and isinstance(pairs, int) and pairs > 0
            quant, act, e, h, i, k, sm, capacity = key
            assert quant in ("nvfp4", "mxfp4") and isinstance(act, str)
            assert all(isinstance(v, int) for v in (e, h, i, k, sm, capacity))
            # every entry was measured at the wrapper capacity of the deployment scope
            assert capacity == 8192

    @measured_device
    def test_exact_key_hits_return_the_measured_entry(self):
        assert md._get_static_compact_cutover_pairs("fp4", **Q35) == 1792
        assert (
            md._get_static_compact_cutover_pairs("fp4", **Q38) == 8704
        )  # merged-three policy, not a model-specific registry entry

    @measured_device
    @pytest.mark.parametrize(
        "capacity_tokens", [None, 0, 1024, 4096, 8191, 8193, 16384]
    )
    def test_unmeasured_or_missing_capacity_falls_back_to_the_density_rule(
        self, capacity_tokens
    ):
        """The registry entries were measured at capacity 8192 only; the lookup is fail-closed
        on the capacity part instead of widening the measurement to a class."""
        assert (
            md._get_static_compact_cutover_pairs(
                "fp4", **{**Q35, "capacity_tokens": capacity_tokens}
            )
            == 2048
        )
        assert (
            md._static_cutover_registry_lookup(
                "nvfp4", "silu", 256, 2048, 512, 8, 110, capacity_tokens
            )
            is None
        )

    @measured_device
    def test_near_misses_fall_back_to_the_density_rule(self):
        # different hidden size (Qwen3.5-122B TP2 shares E / I / top-k with 35B): 8 rows x 256 experts
        assert (
            md._get_static_compact_cutover_pairs("fp4", **{**Q35, "hidden_size": 3072})
            == 2048
        )
        # different activation / top-k
        assert (
            md._get_static_compact_cutover_pairs(
                "fp4", **{**Q35, "activation": "relu2"}
            )
            == 2048
        )
        assert (
            md._get_static_compact_cutover_pairs("fp4", **{**Q35, "num_topk": 4})
            == 2048
        )
        # another SM count is another device
        assert (
            md._static_cutover_registry_lookup(
                "nvfp4", "silu", 256, 2048, 512, 8, 148, 8192
            )
            is None
        )
        # a partial key (no hidden size / activation / capacity) never consults the registry
        assert (
            md._get_static_compact_cutover_pairs(
                "fp4", quant_mode="nvfp4", num_experts=256, intermediate_size=512
            )
            == 2048
        )
        # the generic recipe keeps its flat boundary
        assert (
            md._get_static_compact_cutover_pairs(
                "fp4", **{**Q35, "quant_mode": "mxfp4"}
            )
            == 640
        )

    @measured_device
    def test_wrappers_of_different_capacity_resolve_independently_in_one_process(self):
        """The resolution cache is keyed by the capacity too: a measured-capacity wrapper and
        a smaller one alternate without one result leaking into the other."""
        assert md._get_static_compact_cutover_pairs("fp4", **Q35) == 1792
        assert (
            md._get_static_compact_cutover_pairs(
                "fp4", **{**Q35, "capacity_tokens": 4096}
            )
            == 2048
        )  # density rule (8 padded rows x 256) at the unmeasured capacity
        assert md._get_static_compact_cutover_pairs("fp4", **Q38) == 8704
        assert (
            md._get_static_compact_cutover_pairs(
                "fp4", **{**Q38, "capacity_tokens": 4096}
            )
            == 8704
        )  # the merged-three schedule does not depend on a model/capacity registry hit

    @pytest.mark.parametrize("intermediates", [(512, 256), (256, 512)])
    def test_intermediate_extents_resolve_independently_in_one_process(
        self, intermediates
    ):
        # Same padding/single-slice class, but only I512 has a registry entry.
        expected = {512: 1792, 256: 2048}
        for intermediate in intermediates * 2:
            assert (
                md._get_static_compact_cutover_pairs(
                    "fp4", **{**Q35, "intermediate_size": intermediate}, sm_count=110
                )
                == expected[intermediate]
            )

    def test_environment_override_wins_over_the_registry(self, monkeypatch):
        monkeypatch.setenv("FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS", "4096")
        md._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
        assert md._get_static_compact_cutover_pairs("fp4", **Q35) == 4096

    @measured_device
    def test_backend_selection_uses_the_registry_only_with_the_full_key(self):
        common = dict(
            num_topk=8,
            activation_precision="fp4",
            quant_mode="nvfp4",
            num_experts=256,
            intermediate_size=512,
            hidden_size=2048,
            activation="silu",
            capacity_tokens=8192,
        )
        assert (
            md.select_sm120_moe_backend(num_tokens=224, **common) == "static"
        )  # 1792 pairs = the registry boundary (r=7)
        assert (
            md.select_sm120_moe_backend(num_tokens=225, **common) == "dynamic"
        )  # 1800 pairs: dynamic under r=7
        # the same shape at another capacity or without a capacity keeps the density rule (2048)
        assert (
            md.select_sm120_moe_backend(
                num_tokens=225, **{**common, "capacity_tokens": 4096}
            )
            == "static"
        )
        assert (
            md.select_sm120_moe_backend(
                num_tokens=225, **{**common, "capacity_tokens": None}
            )
            == "static"
        )
        q38 = dict(
            num_topk=10,
            activation_precision="fp4",
            quant_mode="nvfp4",
            num_experts=512,
            intermediate_size=320,
            hidden_size=2560,
            activation="silu",
            capacity_tokens=8192,
        )
        assert md.select_sm120_moe_backend(num_tokens=819, **q38) == "static"
        assert md.select_sm120_moe_backend(num_tokens=820, **q38) == "static"
        assert md.select_sm120_moe_backend(num_tokens=870, **q38) == "static"
        assert md.select_sm120_moe_backend(num_tokens=871, **q38) == "dynamic"
        assert (
            md.select_sm120_moe_backend(
                num_tokens=820, **{**q38, "capacity_tokens": 4096}
            )
            == "static"
        )
        partial = dict(
            num_topk=8,
            activation_precision="fp4",
            quant_mode="nvfp4",
            num_experts=256,
            intermediate_size=512,
        )
        assert (
            md.select_sm120_moe_backend(num_tokens=225, **partial) == "static"
        )  # density rule (2048) without the full key

    @pytest.mark.parametrize("intermediate", [272, 288, 304, 320, 352, 384])
    @pytest.mark.parametrize("experts", [256, 512, 1024])
    def test_merged_three_policy_is_shape_based(self, intermediate, experts):
        # Deliberately not the Q38 registry key; no H or capacity supplied.
        args = dict(
            quant_mode="nvfp4",
            num_experts=experts,
            intermediate_size=intermediate,
            activation="silu",
            num_topk=4,
        )
        assert md._get_static_compact_cutover_pairs(**args) == 17 * experts

    @pytest.mark.parametrize("intermediate", [128, 256, 272, 320, 384, 400, 512])
    @pytest.mark.parametrize("routed_pairs", [1919, 1920])
    def test_merged_groups_uses_slice_count_and_routing_density(
        self, intermediate, routed_pairs
    ):
        """Only three-slice extents at the routing-density floor use merged groups."""
        expected = 256 < intermediate <= 384 and routed_pairs >= 1920
        assert md._static_merged_groups(intermediate, routed_pairs) is expected

    def test_partial_keys_do_not_alias_activation_or_slice_count(self):
        args = dict(
            quant_mode="nvfp4",
            num_experts=512,
            intermediate_size=320,
            activation="silu",
            num_topk=4,
        )
        for _ in range(2):
            assert md._get_static_compact_cutover_pairs(**args) == 8704
            assert (
                md._get_static_compact_cutover_pairs(**{**args, "activation": "relu2"})
                == 8192
            )
            assert (
                md._get_static_compact_cutover_pairs(**{**args, "num_topk": 1}) == 8192
            )
            assert (
                md._get_static_compact_cutover_pairs(
                    **{**args, "intermediate_size": 448}
                )
                == 8192
            )

    def test_workspace_allocation_covers_the_new_boundary(self):
        from flashinfer import B12xMoEWrapper

        args = dict(
            num_experts=512,
            top_k=10,
            hidden_size=256,
            intermediate_size=320,
            use_cuda_graph=True,
            max_num_tokens=8192,
        )
        w = B12xMoEWrapper(**args)
        assert w._static_workspace.max_rows == 8704
        assert w._dynamic_workspace is not None
        assert (
            md.select_sm120_moe_backend(
                num_tokens=870, num_topk=w.top_k, **w._dispatch_kwargs
            )
            == "static"
        )
        assert (
            md.select_sm120_moe_backend(
                num_tokens=871, num_topk=w.top_k, **w._dispatch_kwargs
            )
            == "dynamic"
        )
