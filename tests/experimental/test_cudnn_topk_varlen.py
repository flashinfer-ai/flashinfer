# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch/ownership contracts and semantic GPU checks for optional cuDNN Top-K."""

from contextlib import nullcontext
from dataclasses import dataclass
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

import flashinfer
from flashinfer.experimental.cudnn_topk_varlen import backend as cb
from flashinfer.experimental.cudnn_topk_varlen import support as cs
from flashinfer.topk_varlen import topk_varlen as tv


class _TensorMetadata:
    """CPU-only stand-in: never allocates GPU storage or emulates computation."""

    def __init__(self, shape, dtype=torch.bfloat16):
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device("cuda", 0)
        self.is_cuda = True
        self.contiguous = True
        self.negative = self.conjugate = False
        self.pointer = 4096

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugate

    def data_ptr(self):
        return self.pointer


@pytest.fixture
def metadata(monkeypatch):
    capability = [10, 3]
    monkeypatch.setattr(
        cs,
        "torch",
        SimpleNamespace(
            Tensor=_TensorMetadata,
            bfloat16=torch.bfloat16,
            int32=torch.int32,
            cuda=SimpleNamespace(get_device_capability=lambda _: tuple(capability)),
        ),
    )
    probes = []

    def probe(cc):
        probes.append(cc)
        return object()

    monkeypatch.setattr(cb, "frontend_api", probe)
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    scores = _TensorMetadata((512, 32768))
    lengths = _TensorMetadata((512,), torch.int32)
    return scores, lengths, capability, probes


def test_gate_off_does_not_probe_frontend(metadata, monkeypatch):
    scores, lengths, _, probes = metadata
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS")
    assert not cs.check_cudnn_top_k_varlen(scores, lengths, 512)
    assert probes == []
    assert cs.check_cudnn_top_k_varlen(scores, lengths, 512, backend="cudnn")
    assert probes == [103]


@pytest.mark.parametrize(
    "cc,shape,admitted",
    [
        (103, (512, 16384), True),
        (103, (512, 32768), True),
        (103, (512, 131072), True),
        (107, (512, 16384), True),
        (107, (512, 32768), True),
        (107, (256, 131072), True),
        (107, (256, 32768), False),
        (103, (256, 131072), False),
        (107, (3, 16384), False),
        (100, (512, 32768), False),
    ],
)
def test_exact_auto_allowlist(metadata, cc, shape, admitted):
    scores, lengths, capability, probes = metadata
    capability[:] = divmod(cc, 10)
    scores.shape, lengths.shape = shape, (shape[0],)
    assert cs.check_cudnn_top_k_varlen(scores, lengths, 512) == admitted
    assert probes == ([cc] if admitted else [])


@pytest.mark.parametrize(
    "mutation",
    [
        "fp32",
        "strided",
        "shifted",
        "negative",
        "conjugate",
        "length_dtype",
        "length_stride",
        "length_device",
        "length_negative",
        "length_shifted",
        "hint",
        "values",
        "k",
        "next_n",
        "cr",
        "boolean_k",
        "bad_output",
    ],
)
def test_ineligible_calls_do_not_probe_frontend(metadata, mutation):
    scores, lengths, _, probes = metadata
    kwargs = {"top_k": 512}
    if mutation == "fp32":
        scores.dtype = torch.float32
    elif mutation == "strided":
        scores.contiguous = False
    elif mutation == "shifted":
        scores.pointer += 2
    elif mutation in ("negative", "conjugate"):
        setattr(scores, mutation, True)
    elif mutation == "length_dtype":
        lengths.dtype = torch.int64
    elif mutation == "length_stride":
        lengths.contiguous = False
    elif mutation == "length_device":
        lengths.device = torch.device("cuda", 1)
    elif mutation == "length_negative":
        lengths.negative = True
    elif mutation == "length_shifted":
        lengths.pointer += 4
    elif mutation == "hint":
        kwargs["pre_idx"] = object()
    elif mutation == "values":
        kwargs["return_values"] = True
    elif mutation in ("k", "boolean_k"):
        kwargs["top_k"] = 1024 if mutation == "k" else True
    elif mutation == "next_n":
        kwargs["next_n"] = 2
        lengths.shape = (256,)
    elif mutation == "cr":
        kwargs["compress_ratio"] = 2
    elif mutation == "bad_output":
        kwargs["out_indices"] = object()
    assert not cs.check_cudnn_top_k_varlen(scores, lengths, **kwargs)
    assert probes == []


def test_explicit_contract_and_flat_output(metadata):
    scores, lengths, _, probes = metadata
    scores.shape, lengths.shape = (8, 2053), (4,)
    out = _TensorMetadata((8 * 2048,), torch.int32)
    assert cs.check_cudnn_top_k_varlen(
        scores,
        lengths,
        2048,
        next_n=2,
        compress_ratio=3,
        out_indices=out,
        backend="cudnn",
    )
    assert probes == [103]


def test_registry_and_existing_heuristic_order():
    assert "cudnn" in flashinfer.top_k_varlen.experimental_backends
    assert flashinfer.top_k_varlen.is_backend_supported("cudnn", 103)
    assert not flashinfer.top_k_varlen.is_backend_supported("cudnn", 100)
    scores = torch.empty((512, 32768), device="meta", dtype=torch.bfloat16)
    lengths = torch.empty((512,), device="meta", dtype=torch.int32)
    old = ["radix", "radix_filter", "radix_cutlass"]
    old_order = tv._top_k_varlen_heuristic(old, scores, lengths, 512)
    new_order = tv._top_k_varlen_heuristic(old + ["cudnn"], scores, lengths, 512)
    assert new_order == ["cudnn"] + old_order


def test_missing_old_frontend_and_dependency_errors(monkeypatch):
    cb.frontend_api.cache_clear()
    monkeypatch.setattr(cb, "version", lambda _: "4.8.0")
    monkeypatch.setattr(cb.importlib, "import_module", lambda _: SimpleNamespace())
    assert cb.frontend_api(103) is None  # installed frontend predates public API
    cb.frontend_api.cache_clear()
    monkeypatch.setattr(cb, "version", lambda _: "4.7.0")
    assert cb.frontend_api(107) is None
    cb.frontend_api.cache_clear()
    monkeypatch.setattr(cb, "version", lambda _: "4.8.0")

    def broken(_):
        raise ModuleNotFoundError("internal dependency broken", name="unrelated")

    monkeypatch.setattr(cb.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="internal dependency"):
        cb.frontend_api(103)
    cb.frontend_api.cache_clear()


def test_metadata_plan_cache_fresh_bindings_and_capture_miss(monkeypatch):
    cb._prepare.cache_clear()
    capture = [False]
    made, calls = [], []

    @dataclass
    class Desc:
        dtype: object
        shape: tuple
        stride: tuple
        stride_order: tuple
        device: object

    class Plan:
        def __init__(self, scores, lengths, *config):
            assert isinstance(scores, Desc) and isinstance(lengths, Desc)
            made.append((scores, lengths, config))

        def compile(self):
            pass

        def execute(self, *args):
            calls.append(args)

    monkeypatch.setattr(cb, "frontend_api", lambda _: Plan)
    monkeypatch.setitem(sys.modules, "cudnn.api_base", SimpleNamespace(TensorDesc=Desc))
    monkeypatch.setattr(
        cb,
        "torch",
        SimpleNamespace(
            device=torch.device,
            bfloat16=torch.bfloat16,
            int32=torch.int32,
            cuda=SimpleNamespace(
                device=lambda _: nullcontext(),
                is_current_stream_capturing=lambda: capture[0],
                get_device_capability=lambda _: (10, 3),
            ),
        ),
    )
    for _ in range(2):
        scores = _TensorMetadata((8, 2053))
        lengths, output = object(), object()
        assert cb.run(scores, lengths, 512, 1, 1, output) == (output, None)
    assert len(made) == 1
    assert calls[0][0] is not calls[1][0]
    assert calls[0][2] is not calls[1][2]
    capture[0] = True
    cb.run(scores, lengths, 512, 1, 1, output)  # warmed metadata hits
    scores.shape = (16, 2053)
    with pytest.raises(RuntimeError, match="eagerly before CUDA graph"):
        cb.run(scores, lengths, 512, 1, 1, output)
    assert len(made) == 1
    capture[0] = False
    previous = weakref.ref(cb._prepare(0, 8, 2053, 512, 1, 1))
    for cols in range(3000, 3129):
        cb._prepare(0, 8, cols, 512, 1, 1)
    assert previous() is not None  # compiled owner survives newer geometries
    assert cb._prepare(0, 8, 2053, 512, 1, 1) is previous()
    assert len(made) == 130
    cb._prepare.cache_clear()

    def compile_error(self):
        raise RuntimeError("ordinary compiler error")

    monkeypatch.setattr(Plan, "compile", compile_error)
    with pytest.raises(RuntimeError, match="ordinary compiler error"):
        cb._prepare(0, 8, 2053, 512, 1, 1)
    assert cb._prepare.cache_info().currsize == 0


def _gpu_ready():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    if cc not in (103, 107) or cb.frontend_api(cc) is None:
        pytest.skip("SM103/SM107 and frontend IndexerTopKVarlen required")
    return cc


def _assert_semantics(scores, lengths, indices, k, next_n=1, compress_ratio=1):
    host_lengths = lengths.cpu().tolist()
    for row in range(scores.shape[0]):
        n = min(
            scores.shape[1],
            max(
                0,
                (host_lengths[row // next_n] - next_n + 1 + row % next_n)
                // compress_ratio,
            ),
        )
        count = min(n, k)
        picked = indices[row, :count].long()
        assert bool((indices[row, count:] == -1).all())
        if count:
            assert bool(((picked >= 0) & (picked < n)).all())
            assert picked.unique().numel() == count
            actual = scores[row].gather(0, picked).sort().values
            expected = scores[row, :n].topk(count).values.sort().values
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "k,cols,next_n,cr", [(512, 8192, 1, 1), (1024, 2053, 2, 2), (2048, 32768, 1, 1)]
)
def test_gpu_explicit_fresh_outputs_and_changed_graph(k, cols, next_n, cr, monkeypatch):
    _gpu_ready()
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    scores = torch.randn((8, cols), dtype=torch.bfloat16, device="cuda")
    lengths = torch.full((8 // next_n,), cols * cr, dtype=torch.int32, device="cuda")
    flat = torch.empty(8 * k, dtype=torch.int32, device="cuda")

    def invoke():
        return flashinfer.top_k_varlen(
            scores,
            lengths,
            k,
            next_n=next_n,
            compress_ratio=cr,
            out_indices=flat,
            backend="cudnn",
        )

    output, values = invoke()
    assert (
        values is None
        and output.shape == (8, k)
        and output.data_ptr() == flat.data_ptr()
    )
    _assert_semantics(scores, lengths, output, k, next_n, cr)
    replacement = torch.randn_like(scores)
    fresh, values = flashinfer.top_k_varlen(
        replacement, lengths, k, next_n=next_n, compress_ratio=cr, backend="cudnn"
    )
    assert values is None and fresh.data_ptr() != flat.data_ptr()
    _assert_semantics(replacement, lengths, fresh, k, next_n, cr)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for length, equal in [
        (cols // 2, False),
        (0, False),
        (k - 1, True),
        (2**31 - 1, True),
        (-(2**31), False),
        (cols, False),
    ]:
        scores.fill_(0) if equal else scores.normal_()
        lengths.fill_(length)
        graph.replay()
        _assert_semantics(scores, lengths, output, k, next_n, cr)


@pytest.mark.parametrize("cols", [16384, 32768, 131072])
def test_gpu_opt_in_auto_and_gate_off_control(cols, monkeypatch):
    cc = _gpu_ready()
    rows = 256 if (cc, cols) == (107, 131072) else 512
    scores = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda")
    lengths = torch.randint(513, cols + 1, (rows,), dtype=torch.int32, device="cuda")
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    old, _ = flashinfer.top_k_varlen(scores, lengths, 512)
    assert "cudnn" not in flashinfer.top_k_varlen.suitable_auto_backends
    _assert_semantics(scores, lengths, old, 512)
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    selected, values = flashinfer.top_k_varlen(scores, lengths, 512)
    assert flashinfer.top_k_varlen.suitable_auto_backends[0] == "cudnn"
    assert values is None
    _assert_semantics(scores, lengths, selected, 512)


def test_gpu_two_graphs_share_code_with_independent_bindings():
    _gpu_ready()
    work = []
    for _ in range(2):
        stream = torch.cuda.Stream()
        scores = torch.randn((8, 8192), dtype=torch.bfloat16, device="cuda")
        lengths = torch.full((8,), 8192, dtype=torch.int32, device="cuda")
        output = torch.empty((8, 512), dtype=torch.int32, device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            flashinfer.top_k_varlen(
                scores, lengths, 512, out_indices=output, backend="cudnn"
            )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            flashinfer.top_k_varlen(
                scores, lengths, 512, out_indices=output, backend="cudnn"
            )
        work.append((stream, graph, scores, lengths, output))
    assert work[0][-1].data_ptr() != work[1][-1].data_ptr()
    for phase in range(3):
        for stream, graph, scores, lengths, _ in work:
            with torch.cuda.stream(stream):
                scores.normal_()
                lengths.fill_((8192, 0, 4096)[phase])
                graph.replay()
        # Both graph launches are queued before either stream is waited upon.
        for stream, _, scores, lengths, output in work:
            stream.synchronize()
            _assert_semantics(scores, lengths, output, 512)


def test_gpu_reject_alias_and_lazy_views():
    _gpu_ready()
    scores = torch.randn((8, 8192), dtype=torch.bfloat16, device="cuda")
    lengths = torch.full((8,), 8192, dtype=torch.int32, device="cuda")
    aliased = scores.view(torch.int32).flatten()[: 8 * 512].view(8, 512)
    with pytest.raises(ValueError, match="overlap"):
        flashinfer.top_k_varlen(
            scores, lengths, 512, out_indices=aliased, backend="cudnn"
        )
    with pytest.raises(ValueError):
        flashinfer.top_k_varlen(torch._neg_view(scores), lengths, 512, backend="cudnn")
