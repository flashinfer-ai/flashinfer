"""B4 — NcclEpFleet / NcclEpHandle unit tests (mocked NCCL library)."""

from __future__ import annotations

import ctypes
from unittest import mock

import pytest


@pytest.fixture
def fake_nccl_ep_module():
    """Inject a minimal fake ``nccl_ep`` module into ``sys.modules``."""
    import sys

    if "nccl_ep" in sys.modules:
        yield sys.modules["nccl_ep"]
        return

    fake_mod = mock.Mock()
    fake_mod.ncclNDTensor_t = ctypes.c_void_p
    fake_mod.NCCL_EP_ALGO_LOW_LATENCY = 0
    fake_mod.NCCL_EP_ALGO_HIGH_THROUGHPUT = 1
    fake_mod.HAVE_NCCL_EP = True
    fake_mod.HAVE_TORCH = True

    # Stub get_nccl_comm_from_group → returns a fake pointer.
    fake_mod.get_nccl_comm_from_group = lambda group=None: ctypes.c_void_p(0xC0FFEE)

    class _GroupCfg(ctypes.Structure):
        _fields_ = [
            ("version", ctypes.c_uint),
            ("algorithm", ctypes.c_int),
            ("num_experts", ctypes.c_uint),
            ("max_tokens_per_rank", ctypes.c_uint),
            ("token_size_bytes", ctypes.c_uint),
            ("rdma_buffer_size", ctypes.c_ulong),
            ("num_qp_per_rank", ctypes.c_uint),
            ("num_channels", ctypes.c_uint),
        ]

    fake_mod.ncclEpGroupConfig_t = _GroupCfg

    class _DispatchCfg(ctypes.Structure):
        _fields_ = [("round_scales", ctypes.c_uint)]

    fake_mod.ncclEpDispatchConfig_t = _DispatchCfg

    class _Tags:
        NCCL_EP_TENSOR_TAG_TOKENS = 1
        NCCL_EP_TENSOR_TAG_TOPK_IDX = 2
        NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS = 3
        NCCL_EP_TENSOR_TAG_SCALES = 4
        NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE = 5
        NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST = 6
        NCCL_EP_TENSOR_TAG_TOKENS_PER_EXPERTS = 7

    fake_mod.ncclEpTensorTag_t = _Tags

    class _Dtypes:
        ncclBfloat16 = 9
        ncclInt64 = 4

        @classmethod
        def from_torch(cls, dtype):
            import torch

            return {
                torch.bfloat16: cls.ncclBfloat16,
                torch.int64: cls.ncclInt64,
                torch.int32: 2,  # ncclInt32
                torch.float32: 7,  # ncclFloat32
            }[dtype]

    fake_mod.ncclDataTypeEnum = _Dtypes

    sys.modules["nccl_ep"] = fake_mod
    yield fake_mod
    del sys.modules["nccl_ep"]


@pytest.fixture
def fake_nccl_lib():
    lib = mock.Mock()
    lib.ncclEpCreateGroup = mock.Mock(return_value=ctypes.c_void_p(0xDEADBEEF))
    lib.ncclEpGroupDestroy = mock.Mock()
    lib.ncclEpCreateHandle = mock.Mock(return_value=ctypes.c_void_p(0xBADF00D))
    lib.ncclEpHandleDestroy = mock.Mock()
    lib.ncclEpDispatch = mock.Mock()
    lib.ncclEpCombine = mock.Mock()
    lib.ncclEpHandleGetNumRecvTokens = mock.Mock(return_value=64)
    lib.ncclEpComplete = mock.Mock()
    lib._funcs = {
        "ncclEpTensorCreate": mock.Mock(return_value=0),
        "ncclEpTensorDestroy": mock.Mock(return_value=0),
        "ncclEpTensorGetData": mock.Mock(return_value=0),
        "ncclEpTensorGetSizes": mock.Mock(return_value=0),
    }
    lib.NCCL_CHECK = lambda rc: None
    return lib


@pytest.fixture
def patched_lib(fake_nccl_lib):
    """Patch get_nccl_lib in BOTH ndtensor and the modules that call it."""
    from flashinfer.moe_ep.nccl_ep import fleet, handle, ndtensor

    with (
        mock.patch.object(ndtensor, "get_nccl_lib", return_value=fake_nccl_lib),
        mock.patch.object(fleet, "get_nccl_lib", return_value=fake_nccl_lib),
        mock.patch.object(handle, "get_nccl_lib", return_value=fake_nccl_lib),
    ):
        yield fake_nccl_lib


def test_fleet_init_populates_group_config(fake_nccl_ep_module, patched_lib):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetAlgoKnobNumChannelsPerRank,
        FleetAlgoKnobQuantization,
        FleetParams,
        QuantType,
        create_fleet,
    )

    bootstrap = BootstrapConfig(world_size=8, rank=0, stream=0, nccl_comm=0x1234)
    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=128,
        token_hidden_size=4096,
        dtype_bytes=2,
        algorithm=EpAlgorithm.LOW_LATENCY,
    )
    knobs = [
        FleetAlgoKnobQuantization(quants=frozenset({QuantType.FP8E4M3})),
        FleetAlgoKnobNumChannelsPerRank(n=12),
    ]

    fleet = create_fleet(bootstrap, params, knobs, backend="nccl_ep")

    # ncclEpCreateGroup was called once with the right config struct fields.
    assert patched_lib.ncclEpCreateGroup.call_count == 1
    cfg = patched_lib.ncclEpCreateGroup.call_args.args[1]
    assert cfg.algorithm == 0  # LOW_LATENCY
    assert cfg.num_experts == 8
    assert cfg.max_tokens_per_rank == 128
    assert cfg.token_size_bytes == 4096 * 2
    assert cfg.num_channels == 12
    # FP8 quant flag flows into the handle, not the group config; verified below.
    assert fleet.use_fp8 is True


def test_handle_create_passes_use_fp8(fake_nccl_ep_module, patched_lib):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetAlgoKnobQuantization,
        FleetParams,
        HandleParams,
        QuantType,
        create_fleet,
    )

    bootstrap = BootstrapConfig(world_size=8, rank=0, nccl_comm=0x1)
    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=4096,
        algorithm=EpAlgorithm.LOW_LATENCY,
    )
    fleet = create_fleet(
        bootstrap,
        params,
        [FleetAlgoKnobQuantization(quants=frozenset({QuantType.FP8E4M3}))],
        backend="nccl_ep",
    )

    topk = torch.zeros(64, 4, dtype=torch.int64, device="cuda")
    _ = fleet.create_handle(HandleParams(topk_ids=topk))
    assert patched_lib.ncclEpCreateHandle.call_count == 1
    assert patched_lib.ncclEpCreateHandle.call_args.kwargs["use_fp8"] is True


def test_dispatch_round_scales_from_ue8m0(fake_nccl_ep_module, patched_lib):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DispatchInputParams,
        EpAlgorithm,
        FleetAlgoKnobQuantization,
        FleetParams,
        HandleParams,
        QuantType,
        create_fleet,
    )

    bootstrap = BootstrapConfig(world_size=8, rank=0, nccl_comm=0x1)
    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=4096,
        algorithm=EpAlgorithm.LOW_LATENCY,
    )
    fleet = create_fleet(
        bootstrap,
        params,
        [
            FleetAlgoKnobQuantization(
                quants=frozenset({QuantType.FP8E4M3, QuantType.UE8M0})
            )
        ],
        backend="nccl_ep",
    )

    topk = torch.zeros(64, 4, dtype=torch.int64, device="cuda")
    h = fleet.create_handle(HandleParams(topk_ids=topk))
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    _ = h.dispatch(DispatchInputParams(x=[x]))
    assert patched_lib.ncclEpDispatch.call_count == 1
    cfg = patched_lib.ncclEpDispatch.call_args.args[8]
    assert cfg.round_scales == 1


def test_complete_called_internally_after_dispatch(fake_nccl_ep_module, patched_lib):
    """LL mode requires ncclEpComplete after dispatch; we issue it from
    inside dispatch() rather than waiting for caller.complete(). Handle.
    complete() is now a no-op (kept for HandleAlgoKnobSplitOperation API
    parity, deferred to a future HT-mode pipelined commit)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DispatchInputParams,
        EpAlgorithm,
        FleetParams,
        HandleParams,
        create_fleet,
    )

    bootstrap = BootstrapConfig(world_size=8, rank=0, nccl_comm=0x1)
    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=4096,
        algorithm=EpAlgorithm.LOW_LATENCY,
    )
    fleet = create_fleet(bootstrap, params, [], backend="nccl_ep")
    topk = torch.zeros(64, 4, dtype=torch.int64, device="cuda")
    h = fleet.create_handle(HandleParams(topk_ids=topk))
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    _ = h.dispatch(DispatchInputParams(x=[x]))
    # dispatch issues exactly one ncclEpComplete (post-dispatch).
    assert patched_lib.ncclEpComplete.call_count == 1
    # Handle.complete() is a no-op now.
    h.complete()
    assert patched_lib.ncclEpComplete.call_count == 1
