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

Naming-contract tests for the GDN CuTe-DSL disk cache adopters.

The specialization name is the sole per-kernel on-disk cache key (the module
meta.json guards arch / DSL version / source hash, NOT per-kernel codegen
parameters), so for every adopter the name must be a function of every
codegen argument. Pattern follows tests/jit/test_cute_dsl_cache.py:

1. Signature coverage: every parameter of the @functools.cache'd kernel
   getter appears in the name function's signature.
2. Per-argument perturbation: changing any single argument changes the name.
3. Symbol safety: names stay within [A-Za-z0-9_] (filename + TVM-FFI symbol).
"""

import inspect
import re

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available():
    pytest.skip(
        "GDN kernel modules read device properties at import time",
        allow_module_level=True,
    )

from flashinfer.gdn_kernels.cute_dsl_cache_naming import (  # noqa: E402
    format_name_part,
    make_kernel_name,
)
from flashinfer.gdn_kernels.gdn_decode_nontranspose import (  # noqa: E402
    _get_compiled_decode_kernel_nontranspose,
    _nontranspose_kernel_name,
)
from flashinfer.gdn_kernels.gdn_decode_pretranspose import (  # noqa: E402
    _get_compiled_decode_kernel,
    _pretranspose_kernel_name,
)
from flashinfer.gdn_kernels.gdn_decode_mtp import (  # noqa: E402
    _get_compiled_mtp_kernel,
    _get_compiled_mtp_kernel_inline,
    _mtp_kernel_name,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_state import (  # noqa: E402
    _bf16_state_kernel_name,
)
from flashinfer.gdn_kernels.blackwell.gdn_prefill import (  # noqa: E402
    _get_compiled_cache,
    _prefill_kernel_name,
)

_SYMBOL_SAFE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Shared name formatter
# ---------------------------------------------------------------------------


def test_format_name_part_symbol_safe_and_distinct():
    # One group per value kind: a cache-key slot keeps its type across calls,
    # so distinctness is only required within a kind. (Deliberate aliases
    # exist across kinds: bool 1/0 vs int, torch.dtype vs its str() form.)
    groups = [
        [True, False],
        [0, 1, -1, 128],
        [1.0, 0.5, -0.5, 1e-6, 0.08838834764831845],
        [torch.bfloat16, torch.float16, torch.float32, torch.int32],
        ["v3_mtp_bf16_tiled_dynB", "mtp_bf16_dynB", "torch.bfloat16"],
        [
            (8192, 16384, 128, 1),
            (16384, 128, 1),
            (torch.bfloat16, torch.float32, torch.int32),
            (),
            None,
        ],
    ]
    for group in groups:
        parts = [format_name_part(v) for v in group]
        for v, p in zip(group, parts, strict=False):
            assert re.fullmatch(r"[A-Za-z0-9_]*", p), f"{v!r} formatted to unsafe {p!r}"
        assert len(set(parts)) == len(parts), (group, parts)


def test_make_kernel_name_caps_length_without_collision():
    long_a = make_kernel_name("v", *(range(200)))
    long_b = make_kernel_name("v", *(list(range(199)) + [999]))
    assert len(long_a) <= 210
    assert len(long_b) <= 210
    assert long_a != long_b


# ---------------------------------------------------------------------------
# Baselines (realistic decode shapes: 16 q/k heads, 32 v heads, head size 128,
# matching the shapes exercised by tests/gdn/test_decode_delta_rule.py)
# ---------------------------------------------------------------------------

NONTRANSPOSE_BASELINE = {
    "use_small_batch": True,
    "T": 1,
    "H": 16,
    "HV": 32,
    "K": 128,
    "V": 128,
    "dtype": torch.bfloat16,
    "scale": 0.08838834764831845,
    "use_qk_l2norm": True,
}

PRETRANSPOSE_BASELINE = {
    "T": 1,
    "H": 16,
    "HV": 32,
    "K": 128,
    "V": 128,
    "dtype": torch.bfloat16,
    "scale": 0.08838834764831845,
    "use_qk_l2norm": True,
    "use_pool_indexing": False,
    "stride1": 0,
    "stride2": 0,
    "stride3": 0,
}

MTP_BASELINE = {
    "variant": "warp",
    "T": 2,
    "H": 16,
    "HV": 32,
    "K": 128,
    "V": 128,
    "cache_steps": 0,
    "disable_state_update": False,
    "use_pool_indexing": False,
    "pool_strides_key": None,
    "scale": 0.08838834764831845,
    "use_qk_l2norm": True,
    "tile_v": 64,
    "vec_size": 8,
    "dtype_key": (torch.float32, torch.float32, torch.int32),
    "ilp_rows": 4,
    "use_smem_v": False,
    "use_packed_fma": True,
    "per_token_pool_scatter": False,
}

PREFILL_BASELINE = {
    "io_dtype_str": "torch.bfloat16",
    "state_dtype_str": "torch.float32",
    "HQ": 32,
    "HV": 16,
    "is_GQA": True,
    "use_initial_state": True,
    "store_final_state": True,
    "enable_checkpoints": False,
    "use_state_indices": False,
    "cu_seqlens_dtype_str": "torch.int32",
    "state_indices_dtype_str": "none",
    "cu_checkpoints_dtype_str": "none",
    "initial_state_inner_strides": None,
    "output_state_inner_strides": None,
    "num_sm": 148,
}

# In-process cache tuples as built at each gdn_decode_bf16_state call site.
BF16_STATE_BASELINES = {
    "wide_vec": (
        "v3_mtp_bf16_tiled_dynB",
        2,  # T
        16,  # H
        32,  # HV
        128,  # K
        128,  # V
        -1,  # pool_size_key
        (-1,),  # pool_slot_stride
        64,  # tile_v
        False,  # effective_disable_final
        False,  # cache_intermediate_states
        True,  # use_qk_l2norm_in_kernel
        0.08838834764831845,  # scale
        1.0,  # softplus_beta
        20.0,  # softplus_threshold
        True,  # use_packed_fma
        True,  # same_pool
        False,  # disable_output
        0,  # recovery_steps
        False,  # per_request_accepted_steps
        False,  # per_token_pool_scatter
        False,  # per_token_pool_scatter_flat
        (torch.float32, torch.float32, torch.int32),  # _dtype_key
    ),
    "wide_vec_t1": (
        "v3_mtp_bf16_tiled_dynB",
        1,
        16,
        32,
        128,
        128,
        -1,
        (-1,),
        64,
        False,
        False,
        True,
        0.08838834764831845,
        1.0,
        20.0,
        True,
        True,
        (torch.float32, torch.float32, torch.int32),
    ),
    "mtp_ilp4": (
        "mtp_bf16_dynB",
        2,
        16,
        32,
        128,
        128,
        -1,
        (-1,),
        16,  # tile_v
        4,  # ilp_rows
        False,  # disable_state_update
        False,  # cache_intermediate_states
        True,  # use_qk_l2norm_in_kernel
        0.08838834764831845,
        1.0,
        20.0,
        True,
        True,
        False,  # disable_output
        False,  # per_request_accepted_steps
        False,  # per_token_pool_scatter
        False,  # per_token_pool_scatter_flat
        (torch.float32, torch.float32, torch.int32),
    ),
}


# ---------------------------------------------------------------------------
# 1. Signature coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "getter,name_fn",
    [
        pytest.param(
            _get_compiled_decode_kernel_nontranspose,
            _nontranspose_kernel_name,
            id="nontranspose",
        ),
        pytest.param(
            _get_compiled_decode_kernel, _pretranspose_kernel_name, id="pretranspose"
        ),
        pytest.param(_get_compiled_mtp_kernel, _mtp_kernel_name, id="mtp_warp"),
        pytest.param(
            _get_compiled_mtp_kernel_inline, _mtp_kernel_name, id="mtp_inline"
        ),
        pytest.param(_get_compiled_cache, _prefill_kernel_name, id="prefill"),
    ],
)
def test_kernel_name_signature_covers_getter_params(getter, name_fn):
    getter_params = set(inspect.signature(getter).parameters)
    name_params = set(inspect.signature(name_fn).parameters)
    missing = getter_params - name_params
    assert not missing, (
        f"{getter.__name__} has codegen parameter(s) {sorted(missing)} that "
        f"{name_fn.__name__} cannot encode; add them to the name function."
    )


# ---------------------------------------------------------------------------
# 2. Per-argument perturbation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param,alternate",
    [
        ("use_small_batch", False),
        ("T", 2),
        ("H", 32),
        ("HV", 64),
        ("K", 64),
        ("V", 64),
        ("dtype", torch.float16),
        ("scale", 0.0625),
        ("use_qk_l2norm", False),
    ],
)
def test_nontranspose_name_varies_with_every_argument(param, alternate):
    baseline = _nontranspose_kernel_name(**NONTRANSPOSE_BASELINE)
    kwargs = dict(NONTRANSPOSE_BASELINE)
    kwargs[param] = alternate
    assert _nontranspose_kernel_name(**kwargs) != baseline, (
        f"_nontranspose_kernel_name ignores {param!r}"
    )


@pytest.mark.parametrize(
    "param,alternate",
    [
        ("T", 2),
        ("H", 32),
        ("HV", 64),
        ("K", 64),
        ("V", 64),
        ("dtype", torch.float16),
        ("scale", 0.0625),
        ("use_qk_l2norm", False),
        ("use_pool_indexing", True),
        ("stride1", 16384),
        ("stride2", 128),
        ("stride3", 1),
    ],
)
def test_pretranspose_name_varies_with_every_argument(param, alternate):
    baseline = _pretranspose_kernel_name(**PRETRANSPOSE_BASELINE)
    kwargs = dict(PRETRANSPOSE_BASELINE)
    kwargs[param] = alternate
    assert _pretranspose_kernel_name(**kwargs) != baseline, (
        f"_pretranspose_kernel_name ignores {param!r}"
    )


@pytest.mark.parametrize(
    "param,alternate",
    [
        ("variant", "inline"),
        ("T", 3),
        ("H", 32),
        ("HV", 64),
        ("K", 64),
        ("V", 64),
        ("cache_steps", 2),
        ("disable_state_update", True),
        ("use_pool_indexing", True),
        ("pool_strides_key", (262144, 16384, 128, 1)),
        ("scale", 0.0625),
        ("use_qk_l2norm", False),
        ("tile_v", 128),
        ("vec_size", 4),
        ("dtype_key", (torch.bfloat16, torch.float32, torch.int32)),
        ("ilp_rows", 8),
        ("use_smem_v", True),
        ("use_packed_fma", False),
        ("per_token_pool_scatter", True),
    ],
)
def test_mtp_name_varies_with_every_argument(param, alternate):
    baseline = _mtp_kernel_name(**MTP_BASELINE)
    kwargs = dict(MTP_BASELINE)
    kwargs[param] = alternate
    assert _mtp_kernel_name(**kwargs) != baseline, f"_mtp_kernel_name ignores {param!r}"


@pytest.mark.parametrize(
    "param,alternate",
    [
        ("io_dtype_str", "torch.float16"),
        ("state_dtype_str", "torch.bfloat16"),
        ("HQ", 64),
        ("HV", 32),
        ("is_GQA", False),
        ("use_initial_state", False),
        ("store_final_state", False),
        ("enable_checkpoints", True),
        ("use_state_indices", True),
        ("cu_seqlens_dtype_str", "torch.int64"),
        ("state_indices_dtype_str", "torch.int32"),
        ("cu_checkpoints_dtype_str", "torch.int32"),
        ("initial_state_inner_strides", (16384, 128, 1)),
        ("output_state_inner_strides", (16384, 128, 1)),
        ("num_sm", 132),
    ],
)
def test_prefill_name_varies_with_every_argument(param, alternate):
    baseline = _prefill_kernel_name(**PREFILL_BASELINE)
    kwargs = dict(PREFILL_BASELINE)
    kwargs[param] = alternate
    assert _prefill_kernel_name(**kwargs) != baseline, (
        f"_prefill_kernel_name ignores {param!r}"
    )


def _perturb(value):
    """Return a same-type value that must map to a different name fragment."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 2.0 + 1.0
    if isinstance(value, str):
        return value + "_alt"
    if isinstance(value, tuple):
        if not value:
            return (1,)
        if isinstance(value[0], torch.dtype):
            swapped = torch.float16 if value[0] != torch.float16 else torch.bfloat16
            return (swapped,) + value[1:]
        return (_perturb(value[0]),) + value[1:]
    raise TypeError(f"unhandled baseline component {value!r}")


@pytest.mark.parametrize("variant", sorted(BF16_STATE_BASELINES))
def test_bf16_state_name_varies_with_every_key_component(variant):
    key = BF16_STATE_BASELINES[variant]
    baseline = _bf16_state_kernel_name(variant, key)
    for i in range(len(key)):
        perturbed = key[:i] + (_perturb(key[i]),) + key[i + 1 :]
        assert _bf16_state_kernel_name(variant, perturbed) != baseline, (
            f"_bf16_state_kernel_name ignores cache_key[{i}] = {key[i]!r} "
            f"for variant {variant!r}"
        )


def test_bf16_state_name_distinguishes_variants():
    names = {
        variant: _bf16_state_kernel_name(variant, key)
        for variant, key in BF16_STATE_BASELINES.items()
    }
    assert len(set(names.values())) == len(names), names


# ---------------------------------------------------------------------------
# 3. Symbol safety
# ---------------------------------------------------------------------------


def test_all_baseline_names_are_symbol_safe():
    names = [
        _nontranspose_kernel_name(**NONTRANSPOSE_BASELINE),
        _pretranspose_kernel_name(**PRETRANSPOSE_BASELINE),
        _mtp_kernel_name(**MTP_BASELINE),
        _prefill_kernel_name(**PREFILL_BASELINE),
        *(
            _bf16_state_kernel_name(variant, key)
            for variant, key in BF16_STATE_BASELINES.items()
        ),
        _mtp_kernel_name(
            **{**MTP_BASELINE, "pool_strides_key": (262144, 16384, 128, 1)}
        ),
        _prefill_kernel_name(
            **{**PREFILL_BASELINE, "initial_state_inner_strides": (16384, 128, 1)}
        ),
    ]
    for name in names:
        assert _SYMBOL_SAFE.fullmatch(name), f"unsafe specialization name {name!r}"
        assert len(name) <= 210, f"specialization name too long: {name!r}"


# ---------------------------------------------------------------------------
# 4. Disk-cache round trip
# ---------------------------------------------------------------------------


def test_nontranspose_disk_cache_round_trip(monkeypatch, tmp_path):
    """A second process must reload the exported artifact instead of recompiling.

    Simulated in-process: clear the in-process cache, then forbid cute.compile
    and re-run the same specialization against the populated disk cache.
    """
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("nontranspose decode kernel requires SM90 or later")

    import flashinfer.jit.env as jit_env
    import flashinfer.gdn_kernels.gdn_decode_nontranspose as nt_mod
    from flashinfer.gdn_decode import gated_delta_rule_decode

    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    nt_mod._get_compiled_decode_kernel_nontranspose.cache_clear()

    torch.manual_seed(0)
    B, H, HV, D = 2, 16, 32, 128
    dev = torch.device("cuda")
    q = torch.randn(B, 1, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.nn.functional.normalize(
        torch.randn(B, 1, H, D, dtype=torch.bfloat16, device=dev), p=2.0, dim=-1
    )
    v = torch.randn(B, 1, HV, D, dtype=torch.bfloat16, device=dev)
    a = torch.randn(B, 1, HV, dtype=torch.bfloat16, device=dev) * 0.1
    b = torch.randn(B, 1, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    dt_bias = torch.rand(HV, dtype=torch.float32, device=dev)
    state = torch.randn(B, HV, D, D, dtype=torch.float32, device=dev)

    out1, state1 = gated_delta_rule_decode(q, k, v, state.clone(), A_log, a, dt_bias, b)
    artifacts = list(tmp_path.glob("gdn_decode_nontranspose_*_cute_dsl/*.o"))
    assert len(artifacts) == 1, f"expected one exported artifact, got {artifacts}"

    nt_mod._get_compiled_decode_kernel_nontranspose.cache_clear()

    def _no_recompile(*args, **kwargs):
        raise AssertionError("cute.compile ran despite a valid disk artifact")

    monkeypatch.setattr(nt_mod.cute, "compile", _no_recompile)
    out2, state2 = gated_delta_rule_decode(q, k, v, state.clone(), A_log, a, dt_bias, b)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)
    torch.testing.assert_close(state1, state2, atol=0, rtol=0)
