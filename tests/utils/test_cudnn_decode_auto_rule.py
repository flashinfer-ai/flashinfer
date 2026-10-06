"""CPU tests for the decode wrapper's ``backend="auto"`` -> cudnn rule and its
cudnn-frontend version gate. No GPU, no real cudnn import needed."""

import sys
import types

import pytest
import torch

import flashinfer.cudnn.utils as cudnn_utils
from flashinfer.decode import _DECODE_AUTO_CUDNN_ENV, _auto_decode_prefers_cudnn


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(_DECODE_AUTO_CUDNN_ENV, raising=False)
    monkeypatch.delitem(sys.modules, "cudnn", raising=False)
    return monkeypatch


def _fake_cudnn(monkeypatch, version):
    mod = types.ModuleType("cudnn")
    if version is not None:
        mod.__version__ = version
    monkeypatch.setitem(sys.modules, "cudnn", mod)
    return mod


# -------------------------------------------------------------- version gate


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
    assert cudnn_utils.cudnn_frontend_version() == expected


def test_cudnn_frontend_version_without_package(clean_env):
    clean_env.setitem(sys.modules, "cudnn", None)  # makes `import cudnn` fail
    assert cudnn_utils.cudnn_frontend_version() is None


@pytest.mark.parametrize(
    "version,cc,expected",
    [
        ("1.30.0", (10, 0), True),
        ("1.30.0", (10, 3), True),
        ("1.31.0.dev5", (10, 0), True),
        ("1.30.0", (9, 0), False),  # Hopper has no FROST decode row
        ("1.30.0", (10, 7), False),  # Rubin's row has no decode tile yet
        ("1.30.0", (12, 0), False),
        ("1.29.0", (10, 0), False),  # FROST declines paged KV before 1.30
        ("1.30", (10, 0), True),
        (None, (10, 0), False),  # cudnn-frontend not installed
    ],
)
def test_cudnn_frontend_serves_frost_decode(clean_env, version, cc, expected):
    if version is None:
        clean_env.setitem(sys.modules, "cudnn", None)
    else:
        _fake_cudnn(clean_env, version)
    assert cudnn_utils.cudnn_frontend_serves_frost_decode(cc) is expected


# ------------------------------------------------------------------- the rule


def _base():
    # 64/4 at b=32 with two rows: 32 rows per CTA, 128 CTAs.
    return dict(
        compute_capability=(10, 0),
        frontend_serves_frost_decode=True,
        cudnn_available=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        head_dim=128,
        num_qo_heads=64,
        num_kv_heads=4,
        batch_size=32,
        page_size=16,
        pos_encoding_mode="NONE",
        window_left=-1,
        logits_soft_cap=0.0,
        q_len_per_req=2,
        fa2_available=True,
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
        {"q_len_per_req": 3},
        {"q_len_per_req": 4},  # 64 rows per CTA
        {
            "num_qo_heads": 64,
            "num_kv_heads": 8,
            "q_len_per_req": 4,
        },  # 32 rows, 256 CTAs
        {"num_qo_heads": 64, "num_kv_heads": 8, "q_len_per_req": 4, "batch_size": 8},
        {"num_qo_heads": 64, "num_kv_heads": 1, "batch_size": 64},  # 128 rows, 64 CTAs
        {"batch_size": 16},  # 64 CTAs
        {"batch_size": 128},
        {"page_size": 8},
        {"page_size": 64},
        {"page_size": 128},
        {"page_size": 256},
        {"logits_soft_cap": None},
        {"frontend_serves_frost_decode": False, "override": "1"},  # forced
        {"override": "true"},
        {"replay_hint": True},  # the hint changes nothing at d128
        # single-token d256: the decode tile's unit band
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
        },  # 128 CTAs
        {
            "head_dim": 256,
            "num_qo_heads": 64,
            "num_kv_heads": 8,
            "q_len_per_req": 1,
            "batch_size": 12,
        },  # 96
        {
            "head_dim": 256,
            "num_qo_heads": 16,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "batch_size": 64,
        },  # 256
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 2,
            "q_len_per_req": 1,
            "batch_size": 64,
        },  # group 16
        {
            "head_dim": 256,
            "num_qo_heads": 8,
            "num_kv_heads": 1,
            "q_len_per_req": 1,
            "batch_size": 128,
        },
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "batch_size": 16,
            "replay_hint": True,
        },  # 64 CTAs, split
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "page_size": 64,
        },
    ],
)
def test_auto_prefers_cudnn_inside_the_envelope(changes):
    assert _auto_decode_prefers_cudnn(**{**_base(), **changes}) is True


@pytest.mark.parametrize(
    "changes",
    [
        {"frontend_serves_frost_decode": False},  # older frontend: backend engine
        {"cudnn_available": False},
        {"compute_capability": (9, 0)},
        {"compute_capability": (10, 7)},
        {"compute_capability": (12, 0)},
        {"q_len_per_req": 1},  # single-token decode stays on fa2
        {"q_len_per_req": 5},
        {"q_len_per_req": 8},
        {"q_len_per_req": 0},
        {"num_qo_heads": 64, "num_kv_heads": 8},  # 16 rows per CTA: the tile loses
        {"num_qo_heads": 16, "num_kv_heads": 16, "q_len_per_req": 4},  # MHA: 4 rows
        {
            "num_qo_heads": 64,
            "num_kv_heads": 1,
            "batch_size": 64,
            "q_len_per_req": 4,
        },  # 256 rows
        {"batch_size": 8},  # 32 CTAs
        {"batch_size": 1},
        {"batch_size": 0},
        {"num_qo_heads": 64, "num_kv_heads": 1},  # 128 rows but 32 CTAs
        {"q_data_type": torch.float8_e4m3fn, "kv_data_type": torch.float8_e4m3fn},
        {"kv_data_type": torch.float8_e4m3fn},  # bf16 q over fp8 KV
        {"o_data_type": torch.float16},  # mixed output dtype
        {"head_dim": 64},
        {"head_dim": 192},
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 2,
        },  # multi-token d256: prefill tile
        {"head_dim": 512},
        # single-token d256 outside the band
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "batch_size": 16,
        },  # 64 CTAs unsplit
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "batch_size": 8,
        },  # 32 CTAs
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "batch_size": 8,
            "replay_hint": True,
        },
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "batch_size": 128,
        },  # 512 CTAs
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "batch_size": 128,
            "replay_hint": True,
        },
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 16,
            "q_len_per_req": 1,
        },  # group 2
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 32,
            "q_len_per_req": 1,
        },  # MHA
        {
            "head_dim": 256,
            "num_qo_heads": 64,
            "num_kv_heads": 2,
            "q_len_per_req": 1,
            "batch_size": 64,
        },  # group 32: prefill tile
        {
            "head_dim": 256,
            "num_qo_heads": 96,
            "num_kv_heads": 8,
            "q_len_per_req": 1,
            "batch_size": 16,
        },  # group 12
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 2,
        },  # two rows: prefill tile
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "window_left": 128,
        },
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "page_size": 24,
        },
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "kv_data_type": torch.float8_e4m3fn,
        },
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "compute_capability": (9, 0),
        },
        {
            "head_dim": 256,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "q_len_per_req": 1,
            "frontend_serves_frost_decode": False,
        },
        {
            "replay_hint": True,
            "batch_size": 8,
        },  # the hint does not widen the d128 envelope
        {"num_qo_heads": 96, "num_kv_heads": 8},  # GLM-4.5: group 12
        {"num_qo_heads": 40, "num_kv_heads": 8},  # group 5
        {"num_qo_heads": 48, "num_kv_heads": 8},  # group 6
        {"num_qo_heads": 30, "num_kv_heads": 8},  # not a multiple
        {"num_kv_heads": 0},
        {"page_size": 24},
        {"page_size": 4},
        {"page_size": 96},
        {"window_left": 128},
        {"window_left": 0},
        {"logits_soft_cap": 30.0},
        {"pos_encoding_mode": "ROPE_LLAMA"},
        {"pos_encoding_mode": "ALIBI"},
        {"override": "0"},
        {"override": "off", "frontend_serves_frost_decode": True},
        {"override": "0", "frontend_serves_frost_decode": False},
    ],
)
def test_auto_keeps_fa2_outside_the_envelope(changes):
    assert _auto_decode_prefers_cudnn(**{**_base(), **changes}) is False


@pytest.mark.parametrize(
    "changes",
    [
        {
            "num_qo_heads": 64,
            "num_kv_heads": 8,
        },  # 16 rows per CTA: no fa2 kernel, cudnn
        {"batch_size": 1},
        {"q_len_per_req": 8},
        {"num_qo_heads": 96, "num_kv_heads": 8},  # group 12
        {"head_dim": 256, "num_qo_heads": 32, "num_kv_heads": 2},
        {"window_left": 128},
        {"page_size": 24},
    ],
)
def test_auto_takes_cudnn_when_fa2_cannot_serve_the_rows(changes):
    """Multi-token rows without tensor cores have no fa2 kernel: cudnn takes
    every plan its decode path can run, envelope or not."""
    kwargs = {**_base(), "fa2_available": False, **changes}
    assert _auto_decode_prefers_cudnn(**kwargs) is True


@pytest.mark.parametrize(
    "changes",
    [
        {"q_len_per_req": 1},  # the CUDA-core kernel serves one row
        {"pos_encoding_mode": "ROPE_LLAMA"},  # the cudnn path does not apply RoPE
        {"logits_soft_cap": 30.0},
        {"head_dim": 64},
        {"kv_data_type": torch.float8_e4m3fn},
        {"frontend_serves_frost_decode": False},
        {"compute_capability": (9, 0)},
        {"override": "0"},
    ],
)
def test_auto_still_declines_what_cudnn_cannot_run_without_fa2(changes):
    kwargs = {**_base(), "fa2_available": False, **changes}
    assert _auto_decode_prefers_cudnn(**kwargs) is False


def test_auto_reads_the_override_from_the_environment(clean_env):
    kwargs = {**_base(), "override": None}
    assert _auto_decode_prefers_cudnn(**kwargs) is True
    clean_env.setenv(_DECODE_AUTO_CUDNN_ENV, "0")
    assert _auto_decode_prefers_cudnn(**kwargs) is False
    clean_env.setenv(_DECODE_AUTO_CUDNN_ENV, "1")
    old = {**kwargs, "frontend_serves_frost_decode": False}
    assert _auto_decode_prefers_cudnn(**old) is True
    clean_env.delenv(_DECODE_AUTO_CUDNN_ENV)
    assert _auto_decode_prefers_cudnn(**old) is False


# -------------------------------------------------------------- replay hint probe


class _PygraphWithHint:
    def __init__(
        self, io_data_type=None, *, is_cuda_graph_replay_expected=False, **kwargs
    ):
        pass


class _PygraphWithoutHint:
    def __init__(self, io_data_type=None, **kwargs):
        pass


def _graph_forwarding(handle, name="g", **graph_kwargs):
    pass


def _graph_fixed(handle, name="g"):
    pass


@pytest.mark.parametrize(
    "pygraph,graph,expected",
    [
        (_PygraphWithHint, _graph_forwarding, True),
        (_PygraphWithoutHint, _graph_forwarding, False),  # 1.30: no such keyword
        (_PygraphWithHint, _graph_fixed, False),  # the helper would reject the keyword
        (None, _graph_forwarding, False),  # no pygraph class at all
    ],
)
def test_cudnn_frontend_accepts_cuda_graph_replay_hint(
    clean_env, pygraph, graph, expected
):
    """The probe reads the signatures: the pygraph constructor must take the
    keyword and the cudnn.graph helper must forward graph keywords."""
    mod = _fake_cudnn(clean_env, "1.31.0")
    if pygraph is not None:
        mod.pygraph = pygraph
    mod.graph = graph
    cudnn_utils.cudnn_frontend_accepts_cuda_graph_replay_hint.cache_clear()
    try:
        assert cudnn_utils.cudnn_frontend_accepts_cuda_graph_replay_hint() is expected
    finally:
        cudnn_utils.cudnn_frontend_accepts_cuda_graph_replay_hint.cache_clear()


def test_cudnn_frontend_accepts_cuda_graph_replay_hint_without_package(clean_env):
    clean_env.setitem(sys.modules, "cudnn", None)  # makes `import cudnn` fail
    cudnn_utils.cudnn_frontend_accepts_cuda_graph_replay_hint.cache_clear()
    try:
        assert cudnn_utils.cudnn_frontend_accepts_cuda_graph_replay_hint() is False
    finally:
        cudnn_utils.cudnn_frontend_accepts_cuda_graph_replay_hint.cache_clear()
