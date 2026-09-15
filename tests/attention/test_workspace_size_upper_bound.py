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
"""

import pathlib

import pytest
import torch

from flashinfer import decode as decode_module
from flashinfer import prefill as prefill_module

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _bare_prefill_wrapper(backend="auto", is_cuda_graph_enabled=False):
    cls = prefill_module.BatchPrefillWithPagedKVCacheWrapper
    wrapper = cls.__new__(cls)
    wrapper._backend = backend
    wrapper._jit_module = None
    wrapper.device = torch.device("cpu")
    wrapper._use_cuda_graph = is_cuda_graph_enabled
    wrapper._max_total_num_rows = None
    wrapper._float_workspace_buffer = torch.empty(1, dtype=torch.uint8)
    return wrapper


def _bare_decode_wrapper(backend="auto", use_tensor_cores=True):
    cls = decode_module.BatchDecodeWithPagedKVCacheWrapper
    wrapper = cls.__new__(cls)
    wrapper._backend = backend
    wrapper._jit_module = None
    wrapper.device = torch.device("cpu")
    wrapper._use_tensor_cores = use_tensor_cores
    wrapper._use_cuda_graph = False
    wrapper._float_workspace_buffer = torch.empty(1, dtype=torch.uint8)
    return wrapper


class _ModuleWithoutBound:
    """A backend module that has no upper-bound entry point, like fa3."""


class _ModuleWithBound:
    def __init__(self):
        self.args = None
        self.size_args = None

    def workspace_size_upper_bound(self, *args):
        self.args = args
        return (1024, 64)

    def workspace_size(self, *args):
        self.size_args = args
        return (512, 32)


def test_prefill_bound_resolves_auto_with_the_head_dims_plan_uses(monkeypatch):
    """`auto` has to land on the same backend the plan will get.

    The selector takes the head dimensions into account, so a bound that
    resolved without them could describe a different scheduler than the one
    that ends up planning.
    """
    seen = {}

    def fake_determine(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "fa2"

    monkeypatch.setattr(prefill_module, "determine_attention_backend", fake_determine)
    bound_module = _ModuleWithBound()
    monkeypatch.setattr(
        prefill_module, "get_batch_prefill_module", lambda *a, **k: bound_module
    )

    wrapper = _bare_prefill_wrapper()
    wrapper.workspace_size_upper_bound(
        max_batch_size=4,
        max_total_num_rows=64,
        max_num_pages_per_request=8,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=192,
        page_size=16,
        head_dim_vo=128,
    )

    assert seen["kwargs"]["head_dim_qk"] == 192
    assert seen["kwargs"]["head_dim_vo"] == 128


def test_prefill_bound_and_workspace_size_resolve_auto_the_same_way(monkeypatch):
    """Sizing and bounding must not drift onto different backends."""
    calls = []

    def fake_determine(*args, **kwargs):
        calls.append(kwargs)
        return "fa2"

    monkeypatch.setattr(prefill_module, "determine_attention_backend", fake_determine)

    resolved = prefill_module._resolve_prefill_backend(
        "auto",
        torch.device("cpu"),
        "NONE",
        False,
        False,
        torch.float16,
        torch.float16,
        192,
        128,
    )
    assert resolved == "fa2"
    assert calls == [{"head_dim_qk": 192, "head_dim_vo": 128}]

    assert (
        prefill_module._resolve_prefill_backend(
            "fa3",
            torch.device("cpu"),
            "NONE",
            False,
            False,
            torch.float16,
            torch.float16,
            128,
            128,
        )
        == "fa3"
    )
    # An explicit backend is never re-resolved.
    assert len(calls) == 1


def test_prefill_bound_refuses_a_backend_without_an_entry_point(monkeypatch):
    """A backend with no bound falls closed rather than borrowing another's."""
    monkeypatch.setattr(
        prefill_module, "determine_attention_backend", lambda *a, **k: "fa3"
    )
    monkeypatch.setattr(
        prefill_module,
        "get_batch_prefill_module",
        lambda *a, **k: _ModuleWithoutBound(),
    )

    wrapper = _bare_prefill_wrapper()
    with pytest.raises(NotImplementedError):
        wrapper.workspace_size_upper_bound(
            max_batch_size=4,
            max_total_num_rows=64,
            max_num_pages_per_request=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=128,
            page_size=16,
        )


@pytest.mark.parametrize(
    ("q_data_type", "kv_data_type", "expected"),
    [
        pytest.param(torch.float16, torch.float16, "fa2", id="fp16-stays-fa2"),
        pytest.param(
            torch.float8_e4m3fn, torch.float8_e4m3fn, "fa3", id="fp8-asks-the-selector"
        ),
    ],
)
def test_decode_tensor_core_bound_resolves_auto_like_plan(
    monkeypatch, q_data_type, kv_data_type, expected
):
    """Tensor-core decode is planned as a prefill, and which one depends on dtype."""
    monkeypatch.setattr(
        decode_module, "determine_attention_backend", lambda *a, **k: "fa3"
    )
    assert (
        decode_module._resolve_decode_tensor_core_backend(
            "auto",
            torch.device("cpu"),
            "NONE",
            q_data_type,
            kv_data_type,
            128,
            1,
        )
        == expected
    )


def test_decode_tensor_core_bound_refuses_fa3_without_an_entry_point(monkeypatch):
    """An FP8 decode that resolves to fa3 must not be given the fa2 bound."""
    monkeypatch.setattr(
        decode_module, "determine_attention_backend", lambda *a, **k: "fa3"
    )
    monkeypatch.setattr(
        decode_module, "get_batch_prefill_module", lambda *a, **k: _ModuleWithoutBound()
    )

    wrapper = _bare_decode_wrapper()
    with pytest.raises(NotImplementedError):
        wrapper.workspace_size_upper_bound(
            max_batch_size=4,
            max_num_pages_per_request=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
        )


@pytest.mark.parametrize(
    ("fixed_split_size", "disable_split_kv", "expected_fixed"),
    [
        pytest.param(8, False, 8, id="fixed-split"),
        pytest.param(None, True, -1, id="split-disabled"),
    ],
)
def test_decode_bound_forwards_the_split_settings(
    monkeypatch, fixed_split_size, disable_split_kv, expected_fixed
):
    """A fixed split bypasses the scheduler ceiling, so the bound must see it.

    The two settings are exercised apart: disabling the split makes a fixed
    split size moot, so passing both would not prove the fixed-split branch
    reaches the module.
    """
    monkeypatch.setattr(
        decode_module, "determine_attention_backend", lambda *a, **k: "fa2"
    )
    bound_module = _ModuleWithBound()
    monkeypatch.setattr(
        decode_module, "get_batch_prefill_module", lambda *a, **k: bound_module
    )

    wrapper = _bare_decode_wrapper()
    wrapper.workspace_size_upper_bound(
        max_batch_size=4,
        max_num_pages_per_request=8,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=16,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )

    # (buffer, batch, rows, pages, qo, kv, page_size, graph, qk, vo,
    #  fixed_split_size, disable_split_kv, colocated)
    assert bound_module.args[10] == expected_fixed
    assert bound_module.args[11] is disable_split_kv


def test_decode_bound_rejects_a_zero_q_len_per_req():
    wrapper = _bare_decode_wrapper()
    with pytest.raises(ValueError):
        wrapper.workspace_size_upper_bound(
            max_batch_size=4,
            max_num_pages_per_request=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            q_len_per_req=0,
        )


def test_decode_cuda_core_bound_refuses_the_split_settings():
    """Only the tensor-core path plans through a scheduler that takes them."""
    wrapper = _bare_decode_wrapper(use_tensor_cores=False)
    with pytest.raises(NotImplementedError):
        wrapper.workspace_size_upper_bound(
            max_batch_size=4,
            max_num_pages_per_request=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            fixed_split_size=8,
        )


def test_cta_tile_q_candidates_have_a_single_source():
    """The selector and the bound must iterate the same list.

    The bound covers a tile it cannot predict by evaluating every candidate,
    so a second copy of the list is a silent way to weaken it.
    """
    utils = (_REPO_ROOT / "include/flashinfer/utils.cuh").read_text()
    scheduler = (_REPO_ROOT / "include/flashinfer/attention/scheduler.cuh").read_text()

    assert utils.count("constexpr uint32_t kFA2CtaTileQCandidates[]") == 1
    assert "kFA2CtaTileQCandidates[]" not in scheduler
    assert "kFA2CtaTileQCandidates" in scheduler
    # The selector's result is checked against the list rather than assumed.
    assert "FA2CtaTileQIsCandidate(cta_tile_q)" in utils


def test_prefill_sizing_and_bounding_ask_the_resolver_the_same_question(monkeypatch):
    """The two sizing paths must hand the resolver identical arguments.

    Counting call sites only shows they go through the same function; this
    shows they go through it with the same question, which is what makes the
    answers comparable.
    """
    calls = []

    def recording_resolver(*args):
        calls.append(args[1:])
        return "fa2"

    monkeypatch.setattr(prefill_module, "_resolve_prefill_backend", recording_resolver)
    bound_module = _ModuleWithBound()
    monkeypatch.setattr(
        prefill_module, "get_batch_prefill_module", lambda *a, **k: bound_module
    )

    wrapper = _bare_prefill_wrapper()
    shape = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=192,
        page_size=16,
        head_dim_vo=128,
        pos_encoding_mode="NONE",
        window_left=-1,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    wrapper.workspace_size_upper_bound(
        max_batch_size=4,
        max_total_num_rows=64,
        max_num_pages_per_request=8,
        **shape,
    )

    qo_indptr = torch.tensor([0, 3], dtype=torch.int32)
    wrapper.workspace_size(
        qo_indptr=qo_indptr,
        paged_kv_indptr=torch.tensor([0, 2], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([8], dtype=torch.int32),
        **shape,
    )

    assert len(calls) == 2
    # device, pos_encoding_mode, fp16 reduction, custom mask, dtypes, head dims
    assert calls[0] == calls[1]


def test_decode_sizing_and_bounding_ask_the_resolver_the_same_question(monkeypatch):
    """Tensor-core decode resolves the same way for both sizing paths."""
    calls = []

    def recording_resolver(*args):
        calls.append(args[1:])
        return "fa2"

    monkeypatch.setattr(
        decode_module, "_resolve_decode_tensor_core_backend", recording_resolver
    )
    bound_module = _ModuleWithBound()
    monkeypatch.setattr(
        decode_module, "get_batch_prefill_module", lambda *a, **k: bound_module
    )

    wrapper = _bare_decode_wrapper()
    shape = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=16,
        pos_encoding_mode="NONE",
        window_left=-1,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    wrapper.workspace_size_upper_bound(
        max_batch_size=4,
        max_num_pages_per_request=8,
        **shape,
    )
    wrapper.workspace_size(
        indptr=torch.tensor([0, 2], dtype=torch.int32),
        indices=torch.tensor([0, 1], dtype=torch.int32),
        last_page_len=torch.tensor([8], dtype=torch.int32),
        **shape,
    )

    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_plan_and_sizing_share_the_backend_resolvers():
    """A structural guard: only the shared helpers decide `auto`.

    Sizing that resolved `auto` separately from the plan it sizes for is the
    defect these helpers exist to prevent, so no wrapper path may keep its own
    copy of the selection.
    """
    prefill_source = pathlib.Path(prefill_module.__file__).read_text()
    decode_source = pathlib.Path(decode_module.__file__).read_text()

    # The only direct calls left are inside the helpers and the module-level
    # functional entry points, which take no wrapper state.
    assert prefill_source.count("self._backend = determine_attention_backend(") == 0
    assert decode_source.count("self._backend = determine_attention_backend(") == 0
    # plan(), workspace_size() and workspace_size_upper_bound() all go through
    # the helper, whose definition is the third occurrence in each file.
    assert prefill_source.count("_resolve_prefill_backend(") >= 4
    assert decode_source.count("_resolve_decode_tensor_core_backend(") >= 4


# ---------------------------------------------------------------------------
# GPU tests. These build the real wrappers and call the real entry points, so
# nothing here may be reached through a monkeypatched module: the point is to
# find out whether the bound the C++ computes actually covers what the planner
# asks for.
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="workspace upper-bound GPU tests require CUDA",
)


def _distributions(max_batch_size: int, max_total_num_rows: int, max_pages: int):
    """Request shapes inside the bounds, biased rather than uniform.

    The bound's claim is that it covers any distribution of rows and KV
    lengths over the batch, which a uniform grid cannot test.
    """
    shapes = []
    for batch_size in range(1, max_batch_size + 1):
        rows = max_total_num_rows
        even = max(rows // batch_size, 1)
        # uniform
        shapes.append(([even] * batch_size, [max_pages] * batch_size))
        if batch_size == 1:
            continue
        # one request carries almost everything
        skewed = [rows - (batch_size - 1)] + [1] * (batch_size - 1)
        shapes.append((skewed, [max_pages] * batch_size))
        # short and long alternating
        alternating = [
            1 if index % 2 else max(even * 2, 1) for index in range(batch_size)
        ]
        shapes.append((alternating, [max_pages] * batch_size))
        # the long-q request is not the long-kv one
        pages = [1] * batch_size
        pages[-1] = max_pages
        shapes.append((skewed, pages))
        # a request with no query rows, where the planner allows it
        zero_query = [0] + [even] * (batch_size - 1)
        shapes.append((zero_query, [max_pages] * batch_size))
    return [
        (q, kv)
        for q, kv in shapes
        if sum(q) <= max_total_num_rows and len(q) <= max_batch_size
    ]


def _boundary_query_lens():
    return [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]


def _paged_inputs(q_lens, kv_pages, page_size, device="cuda"):
    qo_indptr = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    kv_indptr = torch.tensor(
        [0, *torch.tensor(kv_pages).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    kv_indices = torch.arange(int(kv_indptr[-1]), dtype=torch.int32, device=device)
    last_page_len = torch.full(
        (len(q_lens),), page_size, dtype=torch.int32, device=device
    )
    return qo_indptr, kv_indptr, kv_indices, last_page_len


@requires_cuda
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize(
    ("fixed_split_size", "disable_split_kv"),
    [
        pytest.param(None, False, id="default-split"),
        pytest.param(4, False, id="fixed-split"),
        pytest.param(None, True, id="split-disabled"),
    ],
)
def test_prefill_upper_bound_covers_every_reachable_shape(
    use_cuda_graph, fixed_split_size, disable_split_kv
):
    """No plan inside the bounds may need more than the bound reports."""
    import flashinfer

    page_size = 16
    max_batch_size, max_total_num_rows, max_pages = 4, 64, 8
    num_qo_heads, num_kv_heads, head_dim = 8, 2, 128

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        use_cuda_graph=use_cuda_graph,
    )
    bound_float, bound_int = wrapper.workspace_size_upper_bound(
        max_batch_size=max_batch_size,
        max_total_num_rows=max_total_num_rows,
        max_num_pages_per_request=max_pages,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )

    for q_lens, kv_pages in _distributions(
        max_batch_size, max_total_num_rows, max_pages
    ):
        qo_indptr, kv_indptr, kv_indices, last_page_len = _paged_inputs(
            q_lens, kv_pages, page_size
        )
        try:
            shape_float, shape_int = wrapper.workspace_size(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
            )
        except ValueError:
            # A shape the planner itself refuses is out of scope.
            continue
        assert shape_float <= bound_float, (q_lens, kv_pages, shape_float, bound_float)
        assert shape_int <= bound_int, (q_lens, kv_pages, shape_int, bound_int)


@requires_cuda
@pytest.mark.parametrize("query_len", _boundary_query_lens())
def test_prefill_upper_bound_covers_the_tile_boundaries(query_len):
    """The tile the planner picks changes around these lengths."""
    import flashinfer

    page_size = 16
    max_batch_size, max_pages = 4, 8
    max_total_num_rows = max_batch_size * max(_boundary_query_lens())
    num_qo_heads, num_kv_heads, head_dim = 8, 2, 128

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    bound_float, bound_int = wrapper.workspace_size_upper_bound(
        max_batch_size=max_batch_size,
        max_total_num_rows=max_total_num_rows,
        max_num_pages_per_request=max_pages,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
    )

    for batch_size in range(1, max_batch_size + 1):
        q_lens = [query_len] * batch_size
        qo_indptr, kv_indptr, kv_indices, last_page_len = _paged_inputs(
            q_lens, [max_pages] * batch_size, page_size
        )
        shape_float, shape_int = wrapper.workspace_size(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=True,
        )
        assert shape_float <= bound_float, (query_len, batch_size, shape_float)
        assert shape_int <= bound_int, (query_len, batch_size, shape_int)


@requires_cuda
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_decode_upper_bound_covers_every_reachable_shape(use_tensor_cores):
    """Both decode planners have to stay inside their own bound."""
    import flashinfer

    page_size = 16
    max_batch_size, max_pages = 8, 8
    num_qo_heads, num_kv_heads, head_dim = 8, 2, 128

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        use_tensor_cores=use_tensor_cores,
    )
    bound_float, bound_int = wrapper.workspace_size_upper_bound(
        max_batch_size=max_batch_size,
        max_num_pages_per_request=max_pages,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )

    for batch_size in range(1, max_batch_size + 1):
        for pages in ({1}, {max_pages}, {1, max_pages}):
            kv_pages = [
                sorted(pages)[index % len(pages)] for index in range(batch_size)
            ]
            _, kv_indptr, kv_indices, last_page_len = _paged_inputs(
                [1] * batch_size, kv_pages, page_size
            )
            shape_float, shape_int = wrapper.workspace_size(
                indptr=kv_indptr,
                indices=kv_indices,
                last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
            )
            assert shape_float <= bound_float, (batch_size, kv_pages, shape_float)
            assert shape_int <= bound_int, (batch_size, kv_pages, shape_int)


@requires_cuda
def test_prefill_plan_succeeds_with_buffers_sized_from_the_bound():
    """A bound is only useful if a plan actually fits in it."""
    import flashinfer

    page_size = 16
    max_batch_size, max_total_num_rows, max_pages = 4, 64, 8
    num_qo_heads, num_kv_heads, head_dim = 8, 2, 128

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    bound_float, bound_int = wrapper.workspace_size_upper_bound(
        max_batch_size=max_batch_size,
        max_total_num_rows=max_total_num_rows,
        max_num_pages_per_request=max_pages,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
    )
    wrapper.reset_workspace_buffer(
        torch.empty(max(bound_float, 1), dtype=torch.uint8, device="cuda"),
        torch.empty(max(bound_int, 1), dtype=torch.uint8, device="cuda"),
    )

    for q_lens, kv_pages in _distributions(
        max_batch_size, max_total_num_rows, max_pages
    ):
        if 0 in q_lens:
            continue
        qo_indptr, kv_indptr, kv_indices, last_page_len = _paged_inputs(
            q_lens, kv_pages, page_size
        )
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=True,
        )
