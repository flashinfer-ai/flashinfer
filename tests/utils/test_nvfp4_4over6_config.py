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

Pure-Python contract tests for the public NVFP4 4over6 setting.

No GPU, no JIT, no kernel launch: this file exercises only
``flashinfer.quantization.nvfp4_quantization_utils``, so it runs in every CI
lane and is the fast feedback loop for the ``nvfp4_4over6`` parameter
introduced by issue #5141: type ``Optional[NVFP44Over6Config]``, with the
*omitted* argument (a private ``_UNSET`` default) as the third case.

What is pinned here:

1. **Precedence.** An omitted argument reads the environment on every call
   (and warns when the environment turns 4over6 on); ``None`` turns 4over6
   off and ignores the environment; an explicit ``NVFP44Over6Config`` is used
   verbatim with **no per-field merge** against the environment.
2. **Serialization.** ``eval(repr(x)) == x``, and ``pickle`` / ``deepcopy``
   preserve the identity of the unset sentinel — a framework-level
   quantization config has to survive both.
3. **The wire format.** The exact int64 packings that
   ``csrc/`` decodes; a silent change here miscompiles every 4over6 kernel.
4. **The on-disk cache key.** ``nvfp4_4over6_cache_key`` string literals are
   part of every user's CuTe-DSL artifact directory, so they are pinned
   character-for-character (see ``tests/jit/test_cute_dsl_cache.py`` for the
   cross-check against the kernel-name function itself).
5. **Canonicalization.** A subclass of ``NVFP44Over6Config`` must collapse to
   the base class before it can reach a ``@functools.cache`` key.

The environment half of every truth-table case uses pytest's ``monkeypatch``
rather than raw ``os.environ`` mutation: the headline acceptance criterion of
#5141 is that multiple 4over6 recipes can be exercised in one process without
hand-rolled save/restore blocks, and this file should model that.
"""

import copy
import dataclasses
import pickle
import warnings

import pytest
import torch

from flashinfer.quantization.nvfp4_quantization_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    NVFP4_4OVER6_CODE_FROM_ENV,
    NVFP4_4OVER6_CODE_STANDARD,
    _UNSET,
    NVFP44Over6Config,
    NVFP44Over6ErrMode,
    make_nvfp4_global_scale,
    nvfp4_4over6_cache_key,
    nvfp4_4over6_code,
    nvfp4_4over6_from_code,
    nvfp4_e4m3_max,
    resolve_nvfp4_4over6,
)

NVFP4_4OVER6_ENV_VARS = (
    "FLASHINFER_NVFP4_4OVER6",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
)


def _apply_env(monkeypatch, env: dict) -> None:
    """Put the 4over6 environment in exactly the state ``env`` describes.

    Every variable not named in ``env`` is deleted, so an ambient value in the
    developer's shell cannot make a case pass (or fail) for the wrong reason.
    """
    for name in NVFP4_4OVER6_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


# The four environment states the truth table crosses with each setting.
ENV_STATES = {
    "unset": ({}, None),
    "on": (
        {"FLASHINFER_NVFP4_4OVER6": "1"},
        NVFP44Over6Config(),
    ),
    "on_mse": (
        {
            "FLASHINFER_NVFP4_4OVER6": "1",
            "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        },
        NVFP44Over6Config(err_mode=NVFP44Over6ErrMode.MSE),
    ),
    "on_e4m3_256": (
        {
            "FLASHINFER_NVFP4_4OVER6": "1",
            "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
        },
        NVFP44Over6Config(e4m3_max=256),
    ),
}

# A recipe that differs from every ENV_STATES recipe in every field, so a
# per-field merge with the environment could not possibly reproduce it.
PINNED_RECIPE = NVFP44Over6Config(
    e4m3_max=256,
    err_mode=NVFP44Over6ErrMode.MSE,
    err_use_fast_math=True,
)

ALL_RECIPES = [
    NVFP44Over6Config(),
    NVFP44Over6Config(e4m3_max=256),
    NVFP44Over6Config(err_mode=NVFP44Over6ErrMode.MSE),
    NVFP44Over6Config(err_use_fast_math=True),
    NVFP44Over6Config(e4m3_max=256, err_mode=NVFP44Over6ErrMode.MSE),
    NVFP44Over6Config(e4m3_max=256, err_use_fast_math=True),
    NVFP44Over6Config(err_mode=NVFP44Over6ErrMode.MSE, err_use_fast_math=True),
    PINNED_RECIPE,
]


# ---------------------------------------------------------------------------
# resolve_nvfp4_4over6 truth table
# ---------------------------------------------------------------------------


class TestResolveTruthTable:
    """{env unset, env=1, env=1+MSE, env=1+256} x {omitted, None, config}."""

    @pytest.mark.parametrize("env_id", sorted(ENV_STATES))
    def test_omitted_argument_reads_the_environment(self, monkeypatch, env_id):
        env, expected = ENV_STATES[env_id]
        _apply_env(monkeypatch, env)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            assert resolve_nvfp4_4over6() == expected

    @pytest.mark.parametrize("env_id", sorted(ENV_STATES))
    def test_environment_enabling_4over6_is_deprecated(self, monkeypatch, env_id):
        """The env vars are a compatibility shim: warn (visibly) only when they act."""
        env, expected = ENV_STATES[env_id]
        _apply_env(monkeypatch, env)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_nvfp4_4over6()
        deprecations = [w for w in caught if issubclass(w.category, FutureWarning)]
        if expected is None:
            assert not deprecations
        else:
            assert len(deprecations) == 1
            assert repr(expected) in str(deprecations[0].message)
            assert "nvfp4_4over6=" in str(deprecations[0].message)

    @pytest.mark.parametrize("env_id", sorted(ENV_STATES))
    def test_none_ignores_the_environment(self, monkeypatch, env_id):
        """``FLASHINFER_NVFP4_4OVER6=1`` cannot turn an explicit ``None`` back on."""
        env, _ = ENV_STATES[env_id]
        _apply_env(monkeypatch, env)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert resolve_nvfp4_4over6(None) is None

    @pytest.mark.parametrize("env_id", sorted(ENV_STATES))
    @pytest.mark.parametrize("recipe", ALL_RECIPES, ids=repr)
    def test_explicit_config_does_not_merge_with_environment(
        self, monkeypatch, env_id, recipe
    ):
        """An explicit recipe is used verbatim — field by field.

        The dangerous failure mode is a *partial* merge: e.g. taking
        ``err_mode`` from the config but ``e4m3_max`` from
        ``FLASHINFER_NVFP4_4OVER6_E4M3_USE_256``.  That silently rescales the
        tensor, so assert each field individually rather than only ``==``.
        """
        env, _ = ENV_STATES[env_id]
        _apply_env(monkeypatch, env)
        resolved = resolve_nvfp4_4over6(recipe)
        assert resolved == recipe
        assert resolved.e4m3_max == recipe.e4m3_max
        assert resolved.err_mode == recipe.err_mode
        assert resolved.err_use_fast_math == recipe.err_use_fast_math

    def test_default_config_is_not_topped_up_from_environment(self, monkeypatch):
        """``NVFP44Over6Config()`` under a fully non-default environment.

        Every field of the explicit recipe is at its default, so a merge
        implemented as "config overrides env where the field was set" would
        pass the parametrized test above but fail here.
        """
        _apply_env(
            monkeypatch,
            {
                "FLASHINFER_NVFP4_4OVER6": "1",
                "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
                "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
                "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
            },
        )
        assert resolve_nvfp4_4over6(NVFP44Over6Config()) == NVFP44Over6Config()

    def test_environment_is_read_on_every_call(self, monkeypatch):
        """An omitted argument must not latch: the docstring promises a per-call read."""
        _apply_env(monkeypatch, {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            assert resolve_nvfp4_4over6() is None
            monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "1")
            assert resolve_nvfp4_4over6() == NVFP44Over6Config()
            monkeypatch.delenv("FLASHINFER_NVFP4_4OVER6")
            assert resolve_nvfp4_4over6() is None

    @pytest.mark.parametrize("value", ["0", "true", "TRUE", "yes", ""])
    def test_only_the_literal_1_enables_4over6(self, monkeypatch, value):
        """Legacy contract: ``env_flag_enabled`` compares against ``"1"``."""
        _apply_env(monkeypatch, {"FLASHINFER_NVFP4_4OVER6": value})
        assert resolve_nvfp4_4over6() is None

    def test_unset_sentinel_is_the_default_and_private(self):
        """The default is a sentinel, not a public type: callers omit the
        argument rather than spelling it, and its repr names the env vars."""
        assert resolve_nvfp4_4over6.__defaults__ == (_UNSET,)
        assert "FLASHINFER_NVFP4_4OVER6" in repr(_UNSET)
        assert _UNSET is not None and not isinstance(_UNSET, NVFP44Over6Config)

    @pytest.mark.parametrize("bogus", ["auto", 3, 0, 1.5, True, ["MAE"], object()])
    def test_bogus_setting_raises_type_error(self, bogus):
        with pytest.raises(TypeError, match="nvfp4_4over6"):
            resolve_nvfp4_4over6(bogus)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    @pytest.mark.parametrize("e4m3_max", [0, 1, 127, 255, 257, 448.0, 512])
    def test_bad_e4m3_max_rejected(self, e4m3_max):
        if e4m3_max == 448.0:
            # 448.0 == 448 so it is accepted; keep the case documented.
            assert NVFP44Over6Config(e4m3_max=e4m3_max).e4m3_max == 448.0
            return
        with pytest.raises(ValueError, match="256 or 448"):
            NVFP44Over6Config(e4m3_max=e4m3_max)

    @pytest.mark.parametrize("err_mode", ["mad", "", "MAE2", 2, -1])
    def test_bad_err_mode_rejected(self, err_mode):
        with pytest.raises(ValueError, match="MAE or MSE"):
            NVFP44Over6Config(err_mode=err_mode)

    @pytest.mark.parametrize("text,expected", [("mae", "MAE"), ("mse", "MSE")])
    def test_err_mode_string_is_normalized(self, text, expected):
        cfg = NVFP44Over6Config(err_mode=text)
        assert cfg.err_mode is NVFP44Over6ErrMode[expected]
        assert cfg.err_mode_name == expected
        # A string and the enum member must be the same config, or they would
        # split the kernel-compilation cache.
        assert cfg == NVFP44Over6Config(err_mode=NVFP44Over6ErrMode[expected])
        assert hash(cfg) == hash(
            NVFP44Over6Config(err_mode=NVFP44Over6ErrMode[expected])
        )

    def test_frozen(self):
        cfg = NVFP44Over6Config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.e4m3_max = 256


# ---------------------------------------------------------------------------
# Serialization: repr / pickle / deepcopy
# ---------------------------------------------------------------------------


_EVAL_NAMESPACE = {
    "NVFP44Over6Config": NVFP44Over6Config,
    "NVFP44Over6ErrMode": NVFP44Over6ErrMode,
}


def _eval_repr(obj):
    return eval(repr(obj), dict(_EVAL_NAMESPACE))


class TestReprRoundTrip:
    @pytest.mark.parametrize("recipe", ALL_RECIPES, ids=repr)
    def test_config_round_trip(self, recipe):
        assert _eval_repr(recipe) == recipe
        assert hash(_eval_repr(recipe)) == hash(recipe)

    def test_config_repr_emits_non_default_fields_only(self):
        assert repr(NVFP44Over6Config()) == "NVFP44Over6Config()"
        assert repr(NVFP44Over6Config(e4m3_max=256)) == (
            "NVFP44Over6Config(e4m3_max=256)"
        )
        assert repr(NVFP44Over6Config(err_mode="MSE")) == (
            "NVFP44Over6Config(err_mode=NVFP44Over6ErrMode.MSE)"
        )
        assert repr(NVFP44Over6Config(err_use_fast_math=True)) == (
            "NVFP44Over6Config(err_use_fast_math=True)"
        )
        assert repr(PINNED_RECIPE) == (
            "NVFP44Over6Config(e4m3_max=256, "
            "err_mode=NVFP44Over6ErrMode.MSE, err_use_fast_math=True)"
        )

    @pytest.mark.parametrize("member", list(NVFP44Over6ErrMode), ids=lambda m: m.name)
    def test_err_mode_round_trip(self, member):
        assert repr(member) == f"NVFP44Over6ErrMode.{member.name}"
        assert _eval_repr(member) is member


class TestPickleAndDeepcopy:
    """The unset sentinel must stay a singleton: ``is _UNSET`` is load-bearing.

    ``resolve_nvfp4_4over6`` dispatches with ``is _UNSET``, so a config that
    came back from ``pickle`` (a torch DataLoader worker, a cached framework
    config) or ``deepcopy`` with a mere equal-but-distinct sentinel would
    silently resolve as an unknown setting and raise ``TypeError``.
    """

    def test_pickle_preserves_identity(self):
        assert pickle.loads(pickle.dumps(_UNSET)) is _UNSET

    def test_deepcopy_preserves_identity(self):
        assert copy.deepcopy(_UNSET) is _UNSET
        assert copy.copy(_UNSET) is _UNSET

    def test_round_tripped_sentinel_still_resolves(self, monkeypatch):
        _apply_env(monkeypatch, {})
        revived = pickle.loads(pickle.dumps(_UNSET))
        assert resolve_nvfp4_4over6(revived) is None

    @pytest.mark.parametrize("member", list(NVFP44Over6ErrMode), ids=lambda m: m.name)
    def test_err_mode_pickle_preserves_identity(self, member):
        assert pickle.loads(pickle.dumps(member)) is member

    @pytest.mark.parametrize("recipe", ALL_RECIPES, ids=repr)
    def test_config_pickle_and_deepcopy(self, recipe):
        assert pickle.loads(pickle.dumps(recipe)) == recipe
        assert copy.deepcopy(recipe) == recipe
        assert hash(copy.deepcopy(recipe)) == hash(recipe)


# ---------------------------------------------------------------------------
# Wire format: nvfp4_4over6_code / nvfp4_4over6_from_code
# ---------------------------------------------------------------------------


class TestWireFormat:
    """The int64 crossing the torch-custom-op / TVM-FFI boundary.

    These literals are the Python half of a two-sided contract (the C++ half
    decodes the same bits).  They are spelled out rather than recomputed from
    the packing expression, so a change to the packing shows up here as a
    failing assertion instead of being silently mirrored.
    """

    def test_sentinel_values(self):
        assert NVFP4_4OVER6_CODE_FROM_ENV == -1
        assert NVFP4_4OVER6_CODE_STANDARD == 0

    @pytest.mark.parametrize(
        "config,code",
        [
            (None, 0),
            (NVFP44Over6Config(), 1),
            (NVFP44Over6Config(e4m3_max=256), 3),
            (NVFP44Over6Config(err_mode="MSE"), 5),
            (NVFP44Over6Config(e4m3_max=256, err_mode="MSE"), 7),
            (NVFP44Over6Config(err_use_fast_math=True), 17),
            (NVFP44Over6Config(e4m3_max=256, err_use_fast_math=True), 19),
            (
                NVFP44Over6Config(e4m3_max=256, err_mode="MSE", err_use_fast_math=True),
                23,
            ),
        ],
        ids=lambda v: repr(v) if not isinstance(v, int) else str(v),
    )
    def test_documented_packings(self, config, code):
        assert nvfp4_4over6_code(config) == code

    def test_standard_code_is_even_and_enabled_codes_are_odd(self):
        """bit0 is the enable bit; C++ dispatches on it."""
        assert nvfp4_4over6_code(None) % 2 == 0
        for recipe in ALL_RECIPES:
            assert nvfp4_4over6_code(recipe) % 2 == 1

    @pytest.mark.parametrize("recipe", ALL_RECIPES, ids=repr)
    def test_round_trip(self, recipe):
        assert nvfp4_4over6_from_code(nvfp4_4over6_code(recipe)) == recipe

    def test_round_trip_off(self):
        assert nvfp4_4over6_from_code(nvfp4_4over6_code(None)) is None

    def test_from_env_code_decodes_to_unset(self):
        assert nvfp4_4over6_from_code(-1) is _UNSET
        assert nvfp4_4over6_from_code(NVFP4_4OVER6_CODE_FROM_ENV) is _UNSET

    def test_codes_are_unique_across_all_recipes(self):
        codes = {nvfp4_4over6_code(r) for r in ALL_RECIPES}
        assert len(codes) == len(ALL_RECIPES)
        assert NVFP4_4OVER6_CODE_STANDARD not in codes
        assert NVFP4_4OVER6_CODE_FROM_ENV not in codes

    def test_code_rejects_an_unresolved_setting(self):
        """``nvfp4_4over6_code`` is typed on the *resolved* two-state value.

        Handing it the unset sentinel is the bug the type signature exists to
        prevent; it must not quietly return a valid-looking code.
        """
        with pytest.raises(AttributeError):
            nvfp4_4over6_code(_UNSET)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestCacheKey:
    """Backwards compatibility of every user's on-disk CuTe-DSL artifact name.

    ``_nvfp4_kernel_name`` has always appended ``_4over6_<e4m3>_<mode>_<fast>``;
    ``nvfp4_4over6_cache_key`` must keep producing exactly that suffix or the
    whole cache silently invalidates (and recompiles) on upgrade.  The
    cross-check against the kernel-name function itself lives in
    ``tests/jit/test_cute_dsl_cache.py``, which can import the CuTe-DSL module.
    """

    def test_off(self):
        assert nvfp4_4over6_cache_key(None) == "off"

    @pytest.mark.parametrize(
        "config,key",
        [
            (NVFP44Over6Config(), "4over6_448_MAE_0"),
            (NVFP44Over6Config(err_use_fast_math=True), "4over6_448_MAE_1"),
            (NVFP44Over6Config(err_mode="MSE"), "4over6_448_MSE_0"),
            (
                NVFP44Over6Config(err_mode="MSE", err_use_fast_math=True),
                "4over6_448_MSE_1",
            ),
            (NVFP44Over6Config(e4m3_max=256), "4over6_256_MAE_0"),
            (
                NVFP44Over6Config(e4m3_max=256, err_use_fast_math=True),
                "4over6_256_MAE_1",
            ),
            (
                NVFP44Over6Config(e4m3_max=256, err_mode="MSE"),
                "4over6_256_MSE_0",
            ),
            (PINNED_RECIPE, "4over6_256_MSE_1"),
        ],
        ids=lambda v: v if isinstance(v, str) else repr(v),
    )
    def test_legacy_suffix_strings(self, config, key):
        assert nvfp4_4over6_cache_key(config) == key

    def test_keys_are_unique(self):
        keys = {nvfp4_4over6_cache_key(r) for r in ALL_RECIPES}
        assert len(keys) == len(ALL_RECIPES)
        assert nvfp4_4over6_cache_key(None) not in keys

    @pytest.mark.parametrize("recipe", ALL_RECIPES + [None], ids=repr)
    def test_keys_are_symbol_safe(self, recipe):
        """The key is embedded in a kernel symbol and a directory name."""
        import re

        assert re.fullmatch(r"[0-9A-Za-z_]+", nvfp4_4over6_cache_key(recipe))

    def test_string_and_enum_err_mode_share_a_key(self):
        assert nvfp4_4over6_cache_key(
            NVFP44Over6Config(err_mode="MSE")
        ) == nvfp4_4over6_cache_key(NVFP44Over6Config(err_mode=NVFP44Over6ErrMode.MSE))


# ---------------------------------------------------------------------------
# Subclass canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe", ALL_RECIPES, ids=repr)
def test_resolve_canonicalizes_subclass_to_base(recipe):
    """A subclass must not reach a ``@functools.cache`` key.

    ``tests/utils/test_fp4_quantize.py`` subclasses ``NVFP44Over6Config`` purely
    to attach a readable pytest id.  A dataclass ``__eq__`` requires
    ``other.__class__ is self.__class__``, and the extra field changes the
    hash, so an un-canonicalized subclass would compile (and cache to disk) a
    second copy of an identical kernel — and worse, one per pytest id.
    """

    import dataclasses

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _IdConfig(NVFP44Over6Config):
        id: str = "pytest-id"

    subclassed = _IdConfig(
        e4m3_max=recipe.e4m3_max,
        err_mode=recipe.err_mode,
        err_use_fast_math=recipe.err_use_fast_math,
    )
    # Precondition: the subclass really is a different key before resolution.
    assert subclassed != recipe
    assert hash(subclassed) != hash(recipe)

    resolved = resolve_nvfp4_4over6(subclassed)
    assert type(resolved) is NVFP44Over6Config
    assert resolved == recipe
    assert hash(resolved) == hash(recipe)
    assert nvfp4_4over6_cache_key(resolved) == nvfp4_4over6_cache_key(recipe)
    assert nvfp4_4over6_code(resolved) == nvfp4_4over6_code(recipe)


def test_resolve_returns_a_plain_config_unchanged():
    """No needless copying: an already-canonical config is returned as-is."""
    recipe = NVFP44Over6Config(err_mode="MSE")
    assert resolve_nvfp4_4over6(recipe) is recipe


# ---------------------------------------------------------------------------
# Scale helpers: legacy vs three-state parameter
# ---------------------------------------------------------------------------


class TestScaleHelpers:
    """``nvfp4_e4m3_max`` / ``make_nvfp4_global_scale``.

    These take the **resolved** recipe (``nvfp4_4over6_config=``) and default
    to ``None``, never reading the environment: their pre-existing contract is
    that no recipe means the 448 clamp, and promoting that to an environment
    read would rescale every existing caller's tensor.
    """

    def test_default_is_full_e4m3_range(self, monkeypatch):
        _apply_env(monkeypatch, {"FLASHINFER_NVFP4_4OVER6": "1"})
        # Legacy default must NOT consult the environment.
        assert nvfp4_e4m3_max() == FLOAT8_E4M3_MAX == 448.0
        assert nvfp4_e4m3_max(None) == 448.0

    @pytest.mark.parametrize("e4m3_max", [448, 256])
    def test_resolved_config_selects_the_clamp(self, e4m3_max):
        cfg = NVFP44Over6Config(e4m3_max=e4m3_max)
        assert nvfp4_e4m3_max(cfg) == float(e4m3_max)
        assert nvfp4_e4m3_max(nvfp4_4over6_config=cfg) == float(e4m3_max)

    def test_no_three_state_spelling_on_scale_helpers(self):
        """One recipe type everywhere: the helpers take ``nvfp4_4over6_config``
        only, so there is no second keyword to disagree with it."""
        x = torch.zeros(4, 16, dtype=torch.float32)
        unresolved_keyword = {"nvfp4_4over6": NVFP44Over6Config()}
        with pytest.raises(TypeError):
            nvfp4_e4m3_max(**unresolved_keyword)
        with pytest.raises(TypeError):
            make_nvfp4_global_scale(x, True, **unresolved_keyword)

    @pytest.mark.parametrize("e4m3_max", [448, 256])
    def test_per_token_scale_inverts_e4m3_max_times_six(self, e4m3_max):
        """The invariant the kernel relies on: ``e4m3_max * 6 * scale == 1``.

        ``1 / (e4m3_max * 6)`` is not exactly representable in fp32 for either
        clamp, so the product comes back one ulp off 1.0 — hence ``approx``
        rather than an exact compare.  What matters is that the scale and the
        clamp come from the *same* recipe.
        """
        cfg = NVFP44Over6Config(e4m3_max=e4m3_max)
        x = torch.zeros(4, 16, dtype=torch.float32)
        scale = make_nvfp4_global_scale(x, True, nvfp4_4over6_config=cfg)

        assert scale.shape == (1,)
        assert scale.dtype == torch.float32
        assert scale.device == x.device  # CPU: the helper must not require CUDA
        product = float(scale.item()) * nvfp4_e4m3_max(cfg) * FLOAT4_E2M1_MAX
        assert product == pytest.approx(1.0, rel=1e-6)

    def test_per_token_scale_default_matches_standard_nvfp4(self):
        """Legacy default (no recipe) is the 448 scale, unchanged."""
        x = torch.zeros(4, 16, dtype=torch.float32)
        default = make_nvfp4_global_scale(x, True)
        standard = make_nvfp4_global_scale(x, True, nvfp4_4over6_config=None)
        assert float(default.item()) == float(standard.item())
        assert float(default.item()) == pytest.approx(
            1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX), rel=1e-6
        )

    def test_per_token_scale_differs_between_clamps(self):
        """Guards the whole point: the two clamps must not collapse."""
        x = torch.zeros(4, 16, dtype=torch.float32)
        s448 = make_nvfp4_global_scale(x, True, nvfp4_4over6_config=NVFP44Over6Config())
        s256 = make_nvfp4_global_scale(
            x, True, nvfp4_4over6_config=NVFP44Over6Config(e4m3_max=256)
        )
        assert float(s448.item()) != float(s256.item())

    @pytest.mark.parametrize("e4m3_max", [448, 256])
    def test_per_tensor_amax_scale_uses_the_clamp(self, e4m3_max):
        cfg = NVFP44Over6Config(e4m3_max=e4m3_max)
        x = torch.full((4, 16), 2.0, dtype=torch.float32)
        scale = make_nvfp4_global_scale(x, False, nvfp4_4over6_config=cfg)
        expected = e4m3_max * FLOAT4_E2M1_MAX / 2.0
        assert float(scale.item()) == pytest.approx(expected, rel=1e-6)

    def test_per_tensor_all_zero_input_saturates(self):
        """Pre-existing edge case; the new parameter must not disturb it."""
        x = torch.zeros(4, 16, dtype=torch.float32)
        scale = make_nvfp4_global_scale(
            x, False, nvfp4_4over6_config=NVFP44Over6Config()
        )
        assert float(scale.item()) == torch.finfo(torch.float32).max

    @pytest.mark.parametrize("e4m3_max", [448, 256])
    def test_explicit_global_scale_wins_over_amax(self, e4m3_max):
        """``global_scale=`` bypasses the amax path in per-tensor mode."""
        x = torch.full((4, 16), 2.0, dtype=torch.float32)
        scale = make_nvfp4_global_scale(
            x,
            False,
            global_scale=0.125,
            nvfp4_4over6_config=NVFP44Over6Config(e4m3_max=e4m3_max),
        )
        assert float(scale.item()) == 0.125
