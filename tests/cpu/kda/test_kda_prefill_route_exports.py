"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Every prepared BF16 KDA prefill call over the serving shape domain resolves
to an exported module.

CPU-only: the device queries are stubbed to SM100a / SM103a with 148 or 160
SMs, every tensor lives on the meta device (the planner reads shapes, strides,
dtypes and alignment only) and module builds are recorded instead of compiled,
with the registry lookup of ``flashinfer.jit.cake_kda_tf32._factory``.  The
domain is the one the route fix was derived from: B in {1, 2, 4, 8, 16, 128},
H in {4, ..., 96}, 64 to 8192 tokens, both gates, with and without state
indices and 64-token BF16 checkpoint rows, on an FP32 in-place state pool, with
a dense beta and with the strided beta slice serving allocators hand over.
"""

import contextlib
import itertools
from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_kda_tf32_runtime as runtime
import flashinfer.jit.cake_kda_tf32 as registry
from flashinfer import prepare_bf16_kda_prefill

HEAD_DIM = 128
CAPABILITIES = ((10, 0), (10, 3))
SM_COUNTS = (148, 160)
BATCHES = (1, 2, 4, 8, 16, 128)
HEADS = (4, 6, 8, 12, 16, 24, 32, 64, 96)
TOKENS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
LOWER_BOUNDS = (-5.0, None)
INDEXED = (True, False)
CHECKPOINTS = (0, 64)


class _Planner:
    """Stubbed device plus the module-request log of one planner call."""

    def __init__(self):
        self.capability = (10, 0)
        self.sm_count = 148
        self.requests = []
        self.work_bins = []

    def configure(self, capability, sm_count):
        self.capability = tuple(capability)
        self.sm_count = sm_count
        self.requests = []
        self.work_bins = []


class _StubModule:
    schedule = None

    def __init__(self, name, module):
        self.name = name
        self.module = module

    def prepare(self, *, grid, **bindings):
        return None

    def launch(self, *, grid, **bindings):
        return None


class _NoDevice(contextlib.AbstractContextManager):
    def __init__(self, *args, **kwargs):
        pass

    def __exit__(self, *exc):
        return False


@pytest.fixture
def planner(monkeypatch):
    rec = _Planner()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: rec.capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=rec.sm_count),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", _NoDevice)

    def data_ptr(tensor):
        # meta storage has no address: alignment checks see an aligned base
        try:
            base = tensor.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001
            base = 0
        return base + tensor.storage_offset() * tensor.element_size()

    monkeypatch.setattr(torch.Tensor, "data_ptr", data_ptr)

    def require_tensor(tensor, *, name, dtype, ndim, contiguous=True):
        # the runtime check minus ``is_cuda`` (meta tensors)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}")
        if contiguous and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def upload_int_batch(device, host_lists, dtype):
        uploads = {"_host_pinned": None}
        for name, values in host_lists.items():
            uploads[name] = torch.empty(len(values), dtype=dtype, device=device)
        return uploads

    def build_kda_module(factory, *args, **kwargs):
        name = factory.args[0]
        arch = registry.device_arch()
        key = registry._key(args, kwargs)
        module = registry.FACTORIES[arch].get(name, {}).get(key)
        rec.requests.append((name, module))
        rec.work_bins.append(bool(factory.keywords.get("route_work_bins", False)))
        if module is None:
            raise NotImplementedError(
                f"Unexported KDA schedule specialization: {(arch, name, *key)}"
            )
        return _StubModule(name, module)

    monkeypatch.setattr(runtime, "_require_tensor", require_tensor)
    monkeypatch.setattr(runtime, "_upload_int_batch", upload_int_batch)
    monkeypatch.setattr(runtime, "_device_sm_count", lambda device: rec.sm_count)
    monkeypatch.setattr(runtime, "_fused_affine_epilogue_enabled", lambda: False)
    monkeypatch.setattr(runtime, "_build_kda_module", build_kda_module)
    monkeypatch.setattr(registry, "prepare_descriptors", lambda prepared: None)
    return rec


def _prepare(
    B, H, T, *, lower_bound, indexed, checkpoint_every_n_tokens, strided=False
):
    dev = "meta"
    shape = (B, T, H, HEAD_DIM)
    q, k, v, g = (
        torch.empty(shape, dtype=torch.bfloat16, device=dev) for _ in range(4)
    )
    pool = torch.empty(
        (B + 2 if indexed else B, H, HEAD_DIM, HEAD_DIM),
        dtype=torch.float32,
        device=dev,
    )
    kwargs = dict(
        A_log=torch.empty((H,), dtype=torch.float32, device=dev),
        dt_bias=torch.empty((H, HEAD_DIM), dtype=torch.float32, device=dev),
        out=torch.empty(shape, dtype=torch.bfloat16, device=dev),
        initial_state=pool,
        final_state=pool,
        lower_bound=lower_bound,
    )
    if indexed:
        kwargs["state_indices"] = torch.empty((B,), dtype=torch.int32, device=dev)
    if checkpoint_every_n_tokens:
        rows = B * ((T + checkpoint_every_n_tokens - 1) // checkpoint_every_n_tokens)
        kwargs.update(
            state_checkpoints=torch.empty(
                (rows, H, HEAD_DIM, HEAD_DIM), dtype=torch.bfloat16, device=dev
            ),
            checkpoint_cu_starts=torch.empty((B + 1,), dtype=torch.int64, device=dev),
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )
    if strided:
        # serving allocators hand beta over as a slice of a wider row: the
        # token pitch differs from the head count
        beta = torch.empty((B, T, H + 24), dtype=torch.bfloat16, device=dev)
        beta = beta[:, :, 8 : 8 + H]
    else:
        beta = torch.empty((B, T, H), dtype=torch.bfloat16, device=dev)
    return prepare_bf16_kda_prefill(q, k, v, g, beta, **kwargs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("beta_layout", ["dense", "strided"])
@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("capability", CAPABILITIES, ids=["sm100a", "sm103a"])
def test_every_serving_shape_requests_an_exported_module(
    planner, capability, sm_count, beta_layout
):
    planner.configure(capability, sm_count)
    unexported = []
    for B, H, T, lower_bound, indexed, cp in itertools.product(
        BATCHES, HEADS, TOKENS, LOWER_BOUNDS, INDEXED, CHECKPOINTS
    ):
        planner.requests = []
        try:
            _prepare(
                B,
                H,
                T,
                lower_bound=lower_bound,
                indexed=indexed,
                checkpoint_every_n_tokens=cp,
                strided=beta_layout == "strided",
            )
        except NotImplementedError as exc:
            unexported.append((B, H, T, lower_bound, indexed, cp, str(exc)[:120]))
            continue
        assert planner.requests, (B, H, T, lower_bound, indexed, cp)
        assert all(module is not None for _name, module in planner.requests)
    assert not unexported, (
        f"{len(unexported)} shapes request unexported bodies: {unexported[:8]}"
    )


M64 = ("fused_m64_independent_dvsplit_fp32_state", "cake_kda_bf16_5535e79c76a18330f6f1")
DIRECT = ("fused_direct_m128", "cake_kda_bf16_2064d4aca94aef841cf8")
PREDICTION_FIRST = (
    "fused_prediction_first_direct_m128",
    "cake_kda_bf16_c410d3932af8bc84666e",
)
UNBOUNDED_CARRIER = (
    "fused_unbounded_softplus_direct_m128_fp32_carrier",
    "cake_kda_bf16_893740628d53688ac269",
)
UNBOUNDED_ROWS = (
    "fused_unbounded_softplus_direct_m128",
    "cake_kda_bf16_9a336afc613e4f9ee1c7",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("capability", "B", "H", "T", "lower_bound", "indexed", "cp", "expected"),
    [
        # grids whose BF16 family is not exported take the exported sibling
        ((10, 0), 1, 8, 2048, -5.0, True, 0, M64),  # small-BH owner/helper
        ((10, 0), 2, 16, 4096, -5.0, False, 0, M64),  # BT16 prepare/chain
        ((10, 0), 2, 16, 4096, -5.0, True, 0, M64),  # BT16 prepare/chain, indexed
        ((10, 0), 1, 12, 512, -5.0, True, 0, M64),  # H12 BT16 prepare/chain
        ((10, 0), 4, 64, 1024, -5.0, True, 0, DIRECT),  # head-grouped persistent
        ((10, 3), 1, 96, 4096, -5.0, True, 0, PREDICTION_FIRST),  # vtile
        # measured flags without an exported body keep the sibling body
        ((10, 3), 2, 64, 1024, -5.0, True, 0, PREDICTION_FIRST),  # tensor-state decay
        ((10, 0), 1, 12, 512, None, True, 64, UNBOUNDED_ROWS),  # pair-packed beta
        # plain calls (no state indices) take the serving ABI
        ((10, 0), 8, 16, 1024, -5.0, False, 0, DIRECT),
        ((10, 0), 1, 16, 1024, None, False, 0, UNBOUNDED_CARRIER),
        # unbounded composites exist for pair-packed H12 windows without rows
        ((10, 0), 1, 16, 4096, None, True, 0, UNBOUNDED_CARRIER),
        ((10, 0), 1, 16, 4096, None, True, 64, UNBOUNDED_ROWS),
        (
            (10, 0),
            1,
            12,
            8192,
            None,
            True,
            0,
            (
                "split_seq_affine_prefix_direct_m128_fp32_state_apply_leanmap",
                "cake_kda_bf16_4e7e4609dadc8f6bcdfe",
            ),
        ),
        # the bounded composite keeps its windows; prediction-first has no
        # window body and stays off there
        (
            (10, 3),
            1,
            24,
            8192,
            -5.0,
            True,
            0,
            (
                "split_seq_affine_prefix_direct_m128_fp32_state",
                "cake_kda_bf16_b38d251875e94cb4aaa6",
            ),
        ),
        # short H12 grids keep the bounded N32 body without the early state pack
        (
            (10, 0),
            1,
            12,
            64,
            -5.0,
            True,
            0,
            ("fused_direct_m128", "cake_kda_bf16_5be344cbddc29563f1ee"),
        ),
        (
            (10, 3),
            16,
            12,
            128,
            -5.0,
            True,
            0,
            ("fused_direct_m128", "cake_kda_bf16_5be344cbddc29563f1ee"),
        ),
        # H12 short rows with checkpoint rows: the bounded N32 body without the
        # early state pack, one task per CTA where every task has an SM of its own
        # and work-binned where a bin runs tasks back to back
        (
            (10, 0),
            4,
            12,
            128,
            -5.0,
            True,
            64,
            ("fused_checkpoint_direct_m128_n32", "cake_kda_bf16_5be344cbddc29563f1ee"),
        ),
        (
            (10, 0),
            16,
            12,
            128,
            -5.0,
            True,
            64,
            ("fused_checkpoint_direct_m128_n32", "cake_kda_bf16_5be344cbddc29563f1ee"),
        ),
        # unchanged selections
        ((10, 3), 8, 32, 1024, -5.0, True, 0, PREDICTION_FIRST),
        (
            (10, 0),
            8,
            12,
            1024,
            -5.0,
            False,
            64,
            (
                "fused_checkpoint_tma_direct_m128_n16",
                "cake_kda_bf16_51e813006f17147a02a5",
            ),
        ),
    ],
)
def test_route_selection_pins(
    planner, capability, B, H, T, lower_bound, indexed, cp, expected
):
    planner.configure(capability, 148)
    prepared = _prepare(
        B, H, T, lower_bound=lower_bound, indexed=indexed, checkpoint_every_n_tokens=cp
    )
    schedule, module = expected
    assert str(prepared.schedule) == schedule
    assert planner.requests[-1][1] == module


@pytest.mark.cpu_only
@pytest.mark.parametrize("capability", CAPABILITIES, ids=["sm100a", "sm103a"])
@pytest.mark.parametrize(("B", "binned"), [(4, False), (16, True), (64, True)])
def test_n16_checkpoint_launch_asks_for_work_bins_where_a_bin_chains_tasks(
    planner, capability, B, binned
):
    # The H12 serving launch of 64-token sequences with 64-token checkpoint rows
    # runs the N16 checkpoint-TMA body; with more (sequence, head) tasks than
    # SMs a bin runs tasks back to back and the launch asks for work bins.
    planner.configure(capability, 148)
    prepared = _prepare(
        B, 12, 64, lower_bound=-5.0, indexed=True, checkpoint_every_n_tokens=64
    )
    assert str(prepared.schedule) == "fused_checkpoint_tma_direct_m128_n16"
    assert planner.requests[-1][1] == "cake_kda_bf16_51e813006f17147a02a5"
    assert planner.work_bins[-1] is binned


UNBOUNDED_COMPOSITE_APPLY = (
    "split_seq_affine_prefix_direct_m128_fp32_state_apply_leanmap",
    "cake_kda_bf16_4e7e4609dadc8f6bcdfe",
)
UNBOUNDED_COMPOSITE_CHAIN = (
    "split_seq_affine_prefix_direct_m128_fp32_state_checkpoint64",
    "cake_kda_bf16_b38d251875e94cb4aaa6",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("capability", "T", "cp", "expected"),
    [
        ((10, 0), 4096, 0, UNBOUNDED_COMPOSITE_APPLY),
        ((10, 0), 8192, 64, UNBOUNDED_COMPOSITE_CHAIN),
        ((10, 3), 4096, 0, UNBOUNDED_COMPOSITE_APPLY),
        ((10, 3), 8192, 64, UNBOUNDED_COMPOSITE_CHAIN),
    ],
)
def test_strided_beta_keeps_the_unbounded_composite(
    planner, capability, T, cp, expected
):
    # With a strided beta every window body runs under the serving ABI, whose
    # unbounded flavours are exported for every head count (with and without
    # BF16 checkpoint rows); the same shapes with a dense beta take the
    # sequential body (pinned above).
    planner.configure(capability, 148)
    prepared = _prepare(
        1,
        16,
        T,
        lower_bound=None,
        indexed=True,
        checkpoint_every_n_tokens=cp,
        strided=True,
    )
    schedule, module = expected
    assert str(prepared.schedule) == schedule
    assert planner.requests[-1][1] == module
