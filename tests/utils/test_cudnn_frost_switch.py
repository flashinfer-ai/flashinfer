"""CPU tests for FlashInfer's cuDNN FROST-engine switch and the decode
wrapper's ``backend="auto"`` -> cudnn rule. No GPU, no cudnn import needed."""

import os
import sys
import types
import warnings

import pytest
import torch

import flashinfer.cudnn_frost as frost
from flashinfer.decode import _DECODE_AUTO_CUDNN_ENV, _auto_decode_prefers_cudnn

FI = frost.FI_FROST_ENV
FE = frost.FE_FROST_ENV


@pytest.fixture
def clean_env(monkeypatch):
    for name in (FI, FE, _DECODE_AUTO_CUDNN_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "cudnn", raising=False)
    frost._warn_frontend_too_old.cache_clear()
    yield monkeypatch
    frost._warn_frontend_too_old.cache_clear()


def _fake_cudnn(monkeypatch, version):
    mod = types.ModuleType("cudnn")
    if version is not None:
        mod.__version__ = version
    monkeypatch.setitem(sys.modules, "cudnn", mod)
    return mod


# --------------------------------------------------------------------------- switch


@pytest.mark.parametrize(
    "fi,fe,expected",
    [
        (None, None, False),
        (None, "1", True),
        ("1", None, True),
        ("true", None, True),
        (" On ", None, True),
        ("0", "1", False),  # FlashInfer's variable wins over the frontend's
        ("1", "0", True),
        ("no", None, False),
        ("garbage", None, False),
    ],
)
def test_frost_engines_requested_precedence(clean_env, fi, fe, expected):
    if fi is not None:
        clean_env.setenv(FI, fi)
    if fe is not None:
        clean_env.setenv(FE, fe)
    assert frost.frost_engines_requested() is expected


def test_configure_forwards_to_the_frontend_variable(clean_env):
    clean_env.setenv(FI, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert frost.configure_cudnn_frost_engines() is True
        # idempotent
        assert frost.configure_cudnn_frost_engines() is True
    assert os.environ[FE] == "1"

    clean_env.setenv(FI, "0")
    clean_env.setenv(FE, "1")
    assert frost.configure_cudnn_frost_engines() is False
    assert os.environ[FE] == "0"


def test_configure_leaves_frontend_variable_alone_when_unset(clean_env):
    assert frost.configure_cudnn_frost_engines() is False
    assert FE not in os.environ
    clean_env.setenv(FE, "1")
    assert frost.configure_cudnn_frost_engines() is True
    assert os.environ[FE] == "1"


def test_configure_warns_when_cudnn_was_imported_first(clean_env):
    _fake_cudnn(clean_env, "1.30.0")
    clean_env.setenv(FI, "1")
    with pytest.warns(RuntimeWarning, match="already imported"):
        assert frost.configure_cudnn_frost_engines() is False
    # The late FI override was declined: availability must use FE's setting.
    assert FE not in os.environ
    assert frost.frost_decode_engines_available((10, 0)) is False

    # same setting as what the frontend read: nothing to warn about
    clean_env.setenv(FE, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert frost.configure_cudnn_frost_engines() is True
    assert frost.frost_decode_engines_available((10, 0)) is True


def test_availability_uses_forwarded_setting_not_a_late_fi_override(clean_env):
    clean_env.setenv(FI, "1")
    assert frost.configure_cudnn_frost_engines() is True
    _fake_cudnn(clean_env, "1.30.0")
    clean_env.setenv(FI, "0")
    # Editing FI's alias after configuration does not change FE's setting.
    assert frost.frost_decode_engines_available((10, 0)) is True
    clean_env.setenv(FE, "0")
    assert frost.frost_decode_engines_available((10, 0)) is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.30.0", (1, 30, 0)),
        ("1.31.0.dev123", (1, 31, 0)),
        ("1.29.1+cu13", (1, 29, 1)),
        ("1.30", (1, 30)),
        ("", None),
        (None, None),
        ("nightly", None),
    ],
)
def test_cudnn_frontend_version_parsing(clean_env, raw, expected):
    _fake_cudnn(clean_env, raw)
    assert frost.cudnn_frontend_version() == expected


def test_cudnn_frontend_version_without_package(clean_env):
    clean_env.setitem(sys.modules, "cudnn", None)  # makes `import cudnn` fail
    assert frost.cudnn_frontend_version() is None


@pytest.mark.parametrize(
    "requested,version,cc,expected",
    [
        (True, "1.30.0", (10, 0), True),
        (True, "1.30.0", (10, 3), True),
        (True, "1.31.0.dev5", (10, 0), True),
        (True, "1.30.0", (9, 0), False),  # Hopper has no FROST decode row
        (True, "1.30.0", (10, 7), False),  # Rubin rows have no decode tile yet
        (True, "1.30.0", (12, 0), False),
        (False, "1.30.0", (10, 0), False),
        (True, None, (10, 0), False),  # cudnn-frontend not installed
    ],
)
def test_frost_decode_engines_available(clean_env, requested, version, cc, expected):
    if version is None:
        clean_env.setitem(sys.modules, "cudnn", None)
    else:
        _fake_cudnn(clean_env, version)
    if requested:
        clean_env.setenv(FE, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert frost.frost_decode_engines_available(cc) is expected


def test_frost_decode_engines_warn_once_on_old_frontend(clean_env):
    _fake_cudnn(clean_env, "1.29.0")
    clean_env.setenv(FE, "1")
    with pytest.warns(RuntimeWarning, match="1.29.0"):
        assert frost.frost_decode_engines_available((10, 0)) is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # second call: no repeat
        assert frost.frost_decode_engines_available((10, 0)) is False


# ----------------------------------------------------------------------------- auto


def _base():
    return dict(
        compute_capability=(10, 0),
        frost_available=True,
        cudnn_available=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        head_dim=128,
        num_qo_heads=64,
        num_kv_heads=4,
        batch_size=32,  # 128 (batch, KV head) units: one B200 wave
        page_size=16,
        pos_encoding_mode="NONE",
        window_left=-1,
        logits_soft_cap=0.0,
        q_len_per_req=1,
        sm_count=148,
        override="",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"compute_capability": (10, 3)},
        {
            "q_data_type": torch.float16,
            "kv_data_type": torch.float16,
            "o_data_type": torch.float16,
        },
        {"num_qo_heads": 64, "num_kv_heads": 8, "batch_size": 16},  # 128 units
        {"num_qo_heads": 16, "num_kv_heads": 16, "batch_size": 8},  # MHA: 128 units
        {
            "num_qo_heads": 128,
            "num_kv_heads": 1,
            "batch_size": 128,
        },  # MQA, whole tile, 128 units
        {"batch_size": 8},
        {"batch_size": 37},  # 148 units: exactly one wave
        {"page_size": 8},
        {"page_size": 64},
        {"page_size": 128},
        {"page_size": 256},
        {"q_len_per_req": 2},
        {"q_len_per_req": 4},  # 64/4: 64 rows in the d128 tile
        {"num_qo_heads": 64, "num_kv_heads": 1, "q_len_per_req": 2},  # 128 rows
        {"logits_soft_cap": None},
        {"frost_available": False, "override": "1"},  # forced without FROST
        {"override": "true"},
    ],
)
def test_auto_prefers_cudnn_inside_the_measured_envelope(changes):
    kwargs = {**_base(), **changes}
    assert _auto_decode_prefers_cudnn(**kwargs) is True


@pytest.mark.parametrize(
    "changes",
    [
        {"frost_available": False},  # backend engine: slow MTP, rejects sinks
        {"cudnn_available": False},
        {"compute_capability": (9, 0)},
        {"compute_capability": (10, 7)},
        {"compute_capability": (12, 0)},
        {"q_data_type": torch.float8_e4m3fn, "kv_data_type": torch.float8_e4m3fn},
        {"kv_data_type": torch.float8_e4m3fn},  # bf16 q over fp8 KV
        {"o_data_type": torch.float16},  # mixed output dtype
        {"head_dim": 64},
        {"head_dim": 192},
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 2,
        },  # d256 tile: 89 vs 70 us at b=32
        {"head_dim": 512},
        {"num_qo_heads": 96, "num_kv_heads": 8},  # GLM-4.5: group 12
        {"num_qo_heads": 40, "num_kv_heads": 8},  # group 5
        {"num_qo_heads": 48, "num_kv_heads": 8},  # group 6
        {"num_qo_heads": 30, "num_kv_heads": 8},  # not a multiple
        {"num_kv_heads": 0},
        {"batch_size": 7},
        {"batch_size": 1},
        {"batch_size": 38},  # 152 units: past one wave at d128
        {"batch_size": 128},  # 512 units: 479 vs 373 us
        {
            "num_qo_heads": 64,
            "num_kv_heads": 8,
            "batch_size": 32,
        },  # 256 units: 163 vs 111 us
        {"num_qo_heads": 64, "num_kv_heads": 64},  # MHA at b=32: 2048 units
        {"sm_count": 0},
        {"page_size": 24},
        {"page_size": 4},
        {"page_size": 96},
        {"window_left": 128},
        {"window_left": 0},
        {"logits_soft_cap": 30.0},
        {"pos_encoding_mode": "ROPE_LLAMA"},
        {"pos_encoding_mode": "ALIBI"},
        {"q_len_per_req": 5},
        {"q_len_per_req": 8},
        {"q_len_per_req": 0},
        {"num_qo_heads": 64, "num_kv_heads": 1, "q_len_per_req": 4},  # 256 > 128
        {"override": "0"},
        {"override": "off", "frost_available": True},
        {"override": "0", "frost_available": False},
    ],
)
def test_auto_keeps_fa2_outside_the_envelope(changes):
    kwargs = {**_base(), **changes}
    assert _auto_decode_prefers_cudnn(**kwargs) is False


def test_auto_reads_the_override_from_the_environment(clean_env):
    kwargs = {**_base(), "override": None}
    assert _auto_decode_prefers_cudnn(**kwargs) is True
    clean_env.setenv(_DECODE_AUTO_CUDNN_ENV, "0")
    assert _auto_decode_prefers_cudnn(**kwargs) is False
    clean_env.setenv(_DECODE_AUTO_CUDNN_ENV, "1")
    assert _auto_decode_prefers_cudnn(**{**kwargs, "frost_available": False}) is True
    clean_env.delenv(_DECODE_AUTO_CUDNN_ENV)
    assert _auto_decode_prefers_cudnn(**{**kwargs, "frost_available": False}) is False
