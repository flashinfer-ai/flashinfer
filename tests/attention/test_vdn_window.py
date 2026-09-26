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

import gc
import random
import warnings
import weakref

import pytest
import torch

from flashinfer.vdn import VDNWindowAttentionWrapper


@pytest.fixture(scope="module", autouse=True)
def ieee_float32_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture(scope="module")
def workspace():
    if not torch.cuda.is_available():
        pytest.skip("VDN requires CUDA")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("VDN requires SM120")
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _oracle(query, key, value, start, frames, spatial, bounds, anchors, scale):
    """Independent token-pair mask and FP32 math; no production planner helpers."""
    length = query.shape[0]
    frame = [
        (token - start) // spatial if start <= token < start + frames * spatial else -1
        for token in range(length)
    ]
    mask = torch.tensor(
        [
            [
                qf == -1
                or kf == -1
                or (anchors in ("rows", "both") and qf in (0, frames - 1))
                or (anchors in ("columns", "both") and kf in (0, frames - 1))
                or (bounds[qf][0] <= kf <= bounds[qf][1])
                for kf in frame
            ]
            for qf in frame
        ],
        device=query.device,
        dtype=torch.bool,
    )
    logits = query.float().transpose(0, 1) @ key.float().permute(1, 2, 0)
    probabilities = (logits * scale).masked_fill(~mask, -torch.inf).softmax(-1)
    return (probabilities @ value.float().transpose(0, 1)).transpose(0, 1)


def _visible_indices(length, start, frames, spatial, bounds, anchors, row, device):
    """Enumerate keys independently for sampled-row and analytic references."""
    end = start + frames * spatial
    qf = (row - start) // spatial if start <= row < end else -1
    visible = []
    for token in range(length):
        kf = (token - start) // spatial if start <= token < end else -1
        if (
            qf == -1
            or kf == -1
            or (anchors in ("rows", "both") and qf in (0, frames - 1))
            or (anchors in ("columns", "both") and kf in (0, frames - 1))
            or bounds[qf][0] <= kf <= bounds[qf][1]
        ):
            visible.append(token)
    return torch.tensor(visible, dtype=torch.long, device=device)


def _sampled_oracle(
    query, key, value, start, frames, spatial, bounds, anchors, scale, rows
):
    outputs = []
    for row in rows:
        indices = _visible_indices(
            query.shape[0], start, frames, spatial, bounds, anchors, row, query.device
        )
        # Head chunks bound temporary FP32 K/V storage independently of the
        # total head count. No S-by-S attention matrix is materialized.
        heads = []
        for head in range(0, query.shape[1], 7):
            q = query[row, head : head + 7].float().unsqueeze(1)
            k = (
                key[:, head : head + 7]
                .index_select(0, indices)
                .float()
                .permute(1, 2, 0)
            )
            v = (
                value[:, head : head + 7]
                .index_select(0, indices)
                .float()
                .transpose(0, 1)
            )
            heads.append(((q @ k * scale).softmax(-1) @ v).squeeze(1))
        outputs.append(torch.cat(heads))
    return torch.stack(outputs)


def _assert_output(actual, expected):
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.015)
    relative_error = (actual.float() - expected).norm() / expected.norm().clamp_min(
        1e-8
    )
    assert relative_error.item() < 0.006


@pytest.mark.parametrize("anchors", ["none", "rows", "columns", "both"])
@pytest.mark.parametrize("kind", ["frame", "chunk"])
@pytest.mark.parametrize(
    "frames,prefix,suffix",
    [(0, 3, 4), (1, 0, 0), (2, 2, 3), (7, 3, 0), (13, 0, 4), (13, 0, 0)],
)
def test_window_mask(workspace, anchors, kind, frames, prefix, suffix):
    torch.manual_seed(11)
    spatial, heads = 3, 2
    length = prefix + frames * spatial + suffix
    if kind == "frame":
        bounds = [(frame - 1, frame + 1) for frame in range(frames)]
    else:
        bounds = [(frame // 3 * 3 - 3, frame // 3 * 3 + 5) for frame in range(frames)]
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(length, heads, prefix, frames, spatial, bounds, anchor_frames=anchors)
    query, key, value = [
        torch.randn(length, heads, 128, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]
    _assert_output(
        wrapper.run(query, key, value),
        _oracle(query, key, value, prefix, frames, spatial, bounds, anchors, 128**-0.5),
    )


def _strided_inputs(length, heads, layout):
    if layout in (386, 514):
        data = torch.randn(length, heads, layout, dtype=torch.bfloat16, device="cuda")
        return data[..., :128], data[..., 128:256], data[..., 256:384]
    if layout == "last_dim":
        return [
            torch.randn(length, heads, 256, dtype=torch.bfloat16, device="cuda")[
                ..., ::2
            ]
            for _ in range(3)
        ]
    if layout == "misaligned":
        return [
            torch.randn(length * heads * 128 + 1, dtype=torch.bfloat16, device="cuda")[
                1:
            ].view(length, heads, 128)
            for _ in range(3)
        ]
    if layout == "broadcast":
        return [
            torch.randn(1, 1, 128, dtype=torch.bfloat16, device="cuda").expand(
                length, heads, 128
            )
            for _ in range(3)
        ]
    if layout == "transposed":
        return [
            torch.randn(
                heads, 128, length, dtype=torch.bfloat16, device="cuda"
            ).permute(2, 0, 1)
            for _ in range(3)
        ]
    return [
        torch.randn(length, heads, 128, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]


@pytest.mark.parametrize(
    "layout", [128, 386, 514, "last_dim", "misaligned", "broadcast", "transposed"]
)
@pytest.mark.parametrize("scale", [0.01, 0.3, 0.125])
def test_strides_and_reuse(workspace, layout, scale):
    torch.manual_seed(43)
    length, heads, start, frames, spatial = 41, 7, 3, 7, 5
    bounds = [(frame - 1, frame + 2) for frame in range(frames)]
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(length, heads, start, frames, spatial, bounds, sm_scale=scale)
    previous = None
    for _ in range(2):
        query, key, value = _strided_inputs(length, heads, layout)
        actual = wrapper.run(query, key, value)
        _assert_output(
            actual,
            _oracle(query, key, value, start, frames, spatial, bounds, "both", scale),
        )
        if previous is not None:
            assert not torch.equal(actual, previous)
        previous = actual


@pytest.mark.parametrize(
    "out_kind", ["separate", "query", "key", "value", "misaligned"]
)
def test_out(workspace, out_kind):
    length, heads, start, frames, spatial = 25, 2, 2, 7, 3
    bounds = [(frame, frame) for frame in range(frames)]
    query, key, value = _strided_inputs(length, heads, 128)
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(length, heads, start, frames, spatial, bounds)
    expected = _oracle(
        query, key, value, start, frames, spatial, bounds, "both", 128**-0.5
    )
    if out_kind == "separate":
        output = torch.empty_like(query)
    elif out_kind == "misaligned":
        storage = torch.full(
            (query.numel() + 2,), 17, dtype=query.dtype, device=query.device
        )
        output = storage[1:-1].view(query.shape)
    else:
        output = {"query": query, "key": key, "value": value}[out_kind]
    actual = wrapper.run(query, key, value, out=output)
    assert actual is output
    _assert_output(actual, expected)
    if out_kind == "misaligned":
        assert storage[0].item() == 17 and storage[-1].item() == 17


def test_replan(workspace):
    wrapper = VDNWindowAttentionWrapper(workspace)
    for length, heads, start, frames, spatial, anchors in [
        (29, 7, 3, 7, 3, "both"),
        (18, 2, 2, 2, 7, "none"),
        (5, 1, 3, 0, 2, "rows"),
    ]:
        bounds = [(frame, frame) for frame in range(frames)]
        wrapper.plan(
            length, heads, start, frames, spatial, bounds, anchor_frames=anchors
        )
        query, key, value = _strided_inputs(length, heads, 128)
        expected = _oracle(
            query, key, value, start, frames, spatial, bounds, anchors, 128**-0.5
        )
        _assert_output(wrapper.run(query, key, value), expected)
        with pytest.raises(ValueError):
            wrapper.plan(0, heads, start, frames, spatial, bounds)
        _assert_output(wrapper.run(query, key, value), expected)


def test_failed_device_replan_invalidates_plan(workspace, monkeypatch):
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(5, 2, 0, 1, 5, [(0, 0)])

    def fail_plan(*args, **kwargs):
        raise RuntimeError("injected planning failure")

    monkeypatch.setattr(
        "flashinfer.vdn.BatchPrefillWithPagedKVCacheWrapper.plan", fail_plan
    )
    with pytest.raises(RuntimeError, match="injected"):
        wrapper.plan(5, 2, 0, 1, 5, [(0, 0)])
    tensor = torch.empty(5, 2, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="plan"):
        wrapper.run(tensor, tensor, tensor)


@pytest.mark.parametrize("anchors", ["none", "rows", "columns", "both"])
def test_outside_clip_windows(workspace, anchors):
    bounds = [(9, 10)] * 7
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(26, 2, 3, 7, 3, bounds, anchor_frames=anchors)
    query, key, value = _strided_inputs(26, 2, 128)
    _assert_output(
        wrapper.run(query, key, value),
        _oracle(query, key, value, 3, 7, 3, bounds, anchors, 128**-0.5),
    )


@pytest.mark.parametrize(
    "bad_workspace",
    [None, torch.empty(1), torch.empty(1, dtype=torch.uint8)],
)
def test_workspace_cpu_validation(bad_workspace):
    with pytest.raises(ValueError, match="workspace"):
        VDNWindowAttentionWrapper(bad_workspace)


@pytest.mark.parametrize("kind", ["small", "dtype", "rank", "stride", "alignment"])
def test_workspace_validation(workspace, kind):
    if kind == "small":
        bad = workspace[:-1]
    elif kind == "dtype":
        bad = workspace.view(torch.bfloat16)
    elif kind == "rank":
        bad = workspace.view(2, -1)
    elif kind == "stride":
        bad = torch.empty(
            workspace.numel() * 2, dtype=torch.uint8, device=workspace.device
        )[::2]
    else:
        bad = torch.empty(
            workspace.numel() + 1, dtype=torch.uint8, device=workspace.device
        )[1:]
    with pytest.raises(ValueError):
        VDNWindowAttentionWrapper(bad)


def test_wrong_architecture(workspace, monkeypatch):
    monkeypatch.setattr("flashinfer.vdn.get_compute_capability", lambda device: (9, 0))
    with pytest.raises(RuntimeError, match="SM120"):
        VDNWindowAttentionWrapper(workspace)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seq_len": 0},
        {"seq_len": 2**31},
        {"seq_len": True},
        {"seq_len": 4.5},
        {"num_heads": 0},
        {"num_heads": 2**24},
        {"video_start": -1},
        {"video_start": 6},
        {"num_frames": -1},
        {"num_frames": 3},
        {"tokens_per_frame": 0},
        {"tokens_per_frame": 6},
        {"window_bounds": []},
        {"window_bounds": [(0, 1)]},
        {"window_bounds": [(0, 1, 2), (0, 1)]},
        {"window_bounds": [(True, 1), (0, 1)]},
        {"window_bounds": [(0.5, 1), (0, 1)]},
        {"window_bounds": [(1, 0), (0, 1)]},
        {"sm_scale": float("inf")},
        {"sm_scale": float("nan")},
        {"sm_scale": 0.0},
        {"sm_scale": -0.1},
        {"sm_scale": 1e-100},
        {"sm_scale": 1e100},
        {"sm_scale": True},
        {"sm_scale": "0.1"},
        {"anchor_frames": "all"},
    ],
)
def test_plan_validation(workspace, overrides):
    params = dict(
        seq_len=10,
        num_heads=2,
        video_start=1,
        num_frames=2,
        tokens_per_frame=3,
        window_bounds=[(0, 1), (0, 1)],
    )
    params.update(overrides)
    with pytest.raises(ValueError):
        VDNWindowAttentionWrapper(workspace).plan(**params)


def test_plan_rejects_empty_rows_and_index_overflow(workspace):
    wrapper = VDNWindowAttentionWrapper(workspace)
    with pytest.raises(ValueError, match="visible key"):
        wrapper.plan(6, 2, 0, 2, 3, [(3, 4), (3, 4)], anchor_frames="none")
    # Few host-side groups but >2**31 paged indices: reject before GPU allocation.
    with pytest.raises(ValueError, match="paged KV index"):
        wrapper.plan(
            2**31 - 1, 1, 2**31 - 3, 2, 1, [(0, 0), (1, 1)], anchor_frames="none"
        )


def test_run_before_plan(workspace):
    tensor = torch.empty(5, 2, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="plan"):
        VDNWindowAttentionWrapper(workspace).run(tensor, tensor, tensor)


@pytest.mark.parametrize("name", ["query", "key", "value", "out"])
@pytest.mark.parametrize("kind", ["dtype", "shape", "device", "grad", "workspace"])
def test_run_validation(workspace, name, kind):
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(5, 2, 0, 1, 5, [(0, 0)])
    args = {
        name: torch.empty(5, 2, 128, dtype=torch.bfloat16, device="cuda")
        for name in ("query", "key", "value", "out")
    }
    if kind == "dtype":
        bad = args[name].float()
    elif kind == "shape":
        bad = args[name][:-1]
    elif kind == "device":
        bad = args[name].cpu()
    elif kind == "grad":
        bad = args[name].requires_grad_()
    else:
        bad = workspace.view(torch.bfloat16)[: 5 * 2 * 128].view(5, 2, 128)
    args[name] = bad
    with pytest.raises(ValueError):
        wrapper.run(**args)


@pytest.mark.parametrize("name", ["query", "key", "value", "out"])
@pytest.mark.parametrize(
    "kind",
    [
        "fp16",
        "fp8_e4m3",
        "fp8_e5m2",
        "d64",
        "d256",
        "one_head",
        "four_heads",
        "rank2",
        "rank4",
        "non_tensor",
    ],
)
def test_unsupported_input_contracts(workspace, name, kind):
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(5, 2, 0, 1, 5, [(0, 0)])
    args = {
        key: torch.empty(5, 2, 128, dtype=torch.bfloat16, device=workspace.device)
        for key in ("query", "key", "value", "out")
    }
    dtypes = {
        "fp16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }
    shapes = {
        "d64": (5, 2, 64),
        "d256": (5, 2, 256),
        "one_head": (5, 1, 128),
        "four_heads": (5, 4, 128),
        "rank2": (5, 256),
        "rank4": (5, 2, 1, 128),
    }
    if kind in dtypes:
        args[name] = args[name].new_empty(args[name].shape, dtype=dtypes[kind])
    elif kind in shapes:
        args[name] = args[name].new_empty(shapes[kind])
    else:
        args[name] = object()
    with pytest.raises(ValueError, match=name):
        wrapper.run(**args)


def test_out_must_be_contiguous(workspace):
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(5, 2, 0, 1, 5, [(0, 0)])
    tensor = torch.randn(5, 2, 128, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(5, 2, 256, dtype=torch.bfloat16, device="cuda")[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        wrapper.run(tensor, tensor, tensor, out=output)


def test_execution_contract(workspace, monkeypatch):
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(5, 2, 0, 1, 5, [(0, 0)])
    tensor = torch.randn(5, 2, 128, dtype=torch.bfloat16, device="cuda")
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match="stream"):
            wrapper.plan(5, 2, 0, 1, 5, [(0, 0)])
        with pytest.raises(RuntimeError, match="stream"):
            wrapper.run(tensor, tensor, tensor)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eager"):
        wrapper.plan(5, 2, 0, 1, 5, [(0, 0)])
    with pytest.raises(RuntimeError, match="eager"):
        wrapper.run(tensor, tensor, tensor)


@pytest.mark.parametrize(
    "length,heads",
    [
        (127, 1),
        (128, 7),
        (129, 56),
        (255, 7),
        (256, 56),
        (257, 1),
        (1023, 56),
        (1024, 1),
        (1025, 7),
    ],
)
@pytest.mark.parametrize("kind", ["dense", "self", "irregular"])
def test_tile_boundaries(workspace, length, heads, kind):
    torch.manual_seed(length)
    frames, prefix = 7, 5
    spatial = (length - prefix) // frames
    if kind == "dense":
        bounds = [(-3, frames + 4)] * frames
    elif kind == "self":
        bounds = [(frame, frame) for frame in range(frames)]
    else:
        rng = random.Random(length)
        bounds = [
            tuple(sorted(rng.sample(range(-3, frames + 4), 2))) for _ in range(frames)
        ]
    query, key, value = _strided_inputs(length, heads, 514)
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(length, heads, prefix, frames, spatial, bounds, anchor_frames="none")
    _assert_output(
        wrapper.run(query, key, value),
        _oracle(query, key, value, prefix, frames, spatial, bounds, "none", 128**-0.5),
    )


@pytest.mark.parametrize("anchors", ["none", "rows", "columns", "both"])
def test_nonconsecutive_equal_windows(workspace, anchors):
    bounds = [(-2, 2), (3, 3), (-2, 2), (5, 9), (-2, 2), (5, 9), (2, 4)]
    query, key, value = _strided_inputs(249, 7, 386)
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(249, 7, 5, 7, 34, bounds, anchor_frames=anchors)
    _assert_output(
        wrapper.run(query, key, value),
        _oracle(query, key, value, 5, 7, 34, bounds, anchors, 128**-0.5),
    )


@pytest.mark.parametrize("anchors", ["none", "rows", "columns", "both"])
@pytest.mark.parametrize("kind", ["uniform", "constant_value", "peaked"])
def test_analytic_inputs(workspace, anchors, kind):
    length, heads, start, frames, spatial = 137, 7, 4, 7, 19
    bounds = [(frame - 1, frame + 1) for frame in range(frames)]
    query, key, value = _strided_inputs(length, heads, 386)
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(length, heads, start, frames, spatial, bounds, anchor_frames=anchors)
    rows = [0, 3, 4, 22, 23, 61, 99, 117, 118, 136]
    if kind == "uniform":
        query.zero_()
        expected = torch.stack(
            [
                value.index_select(
                    0,
                    _visible_indices(
                        length,
                        start,
                        frames,
                        spatial,
                        bounds,
                        anchors,
                        row,
                        value.device,
                    ),
                )
                .float()
                .mean(0)
                for row in rows
            ]
        )
    elif kind == "constant_value":
        # Per-head/channel constants are binary fractions, exactly BF16 representable.
        constants = (
            torch.randint(-8, 9, (heads, 128), device="cuda").to(torch.bfloat16) / 8
        )
        value.copy_(constants)
        expected = constants.float().expand(len(rows), -1, -1)
    else:
        # Every row has global key 0 available; its logit exceeds all others by
        # >1400, so BF16 output must equal that key's value exactly.
        query.fill_(4)
        key.fill_(-4)
        key[0].fill_(4)
        expected = value[0].float().expand(len(rows), -1, -1)
    actual = wrapper.run(query, key, value)[rows]
    if kind == "uniform":
        _assert_output(actual, expected)
    else:
        torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("anchors", ["none", "rows", "columns", "both"])
def test_masked_value_sentinel(workspace, anchors):
    start, frames, spatial = 3, 7, 19
    length = start + frames * spatial + 5
    bounds = [(frame, frame) for frame in range(frames)]
    query, key, value = _strided_inputs(length, 7, 514)
    query.zero_()
    value.zero_()
    # Frame 5 is neither an anchor nor in frame 2's self-only window.
    value[start + 5 * spatial : start + 6 * spatial].fill_(64)
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(length, 7, start, frames, spatial, bounds, anchor_frames=anchors)
    output = wrapper.run(query, key, value)
    invisible_rows = list(range(start + 2 * spatial, start + 3 * spatial))
    torch.testing.assert_close(
        output[invisible_rows], torch.zeros_like(output[invisible_rows]), rtol=0, atol=0
    )
    # Global rows and frame 5 must observe the nonzero sentinel.
    visible_rows = [0, start + 5 * spatial, length - 1]
    assert bool((output[visible_rows] > 0).all())
    expected = _sampled_oracle(
        query,
        key,
        value,
        start,
        frames,
        spatial,
        bounds,
        anchors,
        128**-0.5,
        visible_rows,
    )
    _assert_output(output[visible_rows], expected)


@pytest.mark.parametrize("frames,heads", [(37, 56), (107, 7)])
@pytest.mark.parametrize("layout", [386, 514])
def test_long_sequence_sampled_fp32(workspace, request, frames, heads, layout):
    if not request.config.getoption("--full"):
        pytest.skip("long-sequence sampled FP32 oracle requires --full")
    torch.manual_seed(frames + layout)
    start, spatial = 3623, 510
    length = start + frames * spatial
    bounds = [
        ((frame // 5 - 1) * 5, (frame // 5 + 2) * 5 - 1) for frame in range(frames)
    ]
    query, key, value = _strided_inputs(length, heads, layout)
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(length, heads, start, frames, spatial, bounds)
    rows = sorted(
        {
            0,
            start - 1,
            start,
            start + spatial - 1,
            start + spatial,
            start + 2 * spatial - 1,
            start + (frames // 2) * spatial,
            start + (frames // 2 + 1) * spatial - 1,
            start + (frames - 2) * spatial,
            start + (frames - 1) * spatial - 1,
            start + (frames - 1) * spatial,
            length - 1,
        }
    )
    actual = wrapper.run(query, key, value)
    assert bool(torch.isfinite(actual).all())
    expected = _sampled_oracle(
        query, key, value, start, frames, spatial, bounds, "both", 128**-0.5, rows
    )
    _assert_output(actual[rows], expected)
    difference = actual[rows].float() - expected
    request.node.user_properties.extend(
        [
            ("sampled_query_rows", rows),
            ("relative_l2_error", (difference.norm() / expected.norm()).item()),
            ("max_abs_error", difference.abs().max().item()),
        ]
    )


def test_nondefault_stream(workspace):
    stream = torch.cuda.Stream(device=workspace.device)
    stream.wait_stream(torch.cuda.current_stream(workspace.device))
    with torch.cuda.stream(stream):
        wrapper = VDNWindowAttentionWrapper(workspace)
        wrapper.plan(129, 7, 3, 7, 18, [(frame, frame + 1) for frame in range(7)])
        query, key, value = _strided_inputs(129, 7, 386)
        actual = wrapper.run(query, key, value)
    stream.synchronize()
    _assert_output(
        actual,
        _oracle(
            query,
            key,
            value,
            3,
            7,
            18,
            [(frame, frame + 1) for frame in range(7)],
            "both",
            128**-0.5,
        ),
    )


@pytest.mark.parametrize("operation", ["plan", "run"])
def test_real_cuda_graph_rejection(workspace, operation):
    stream = torch.cuda.Stream(device=workspace.device)
    stream.wait_stream(torch.cuda.current_stream(workspace.device))
    with torch.cuda.stream(stream):
        wrapper = VDNWindowAttentionWrapper(workspace)
        wrapper.plan(129, 7, 3, 7, 18, [(frame, frame) for frame in range(7)])
        query, key, value = _strided_inputs(129, 7, 514)
        wrapper.run(query, key, value)
        marker = torch.zeros((), device="cuda")
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with torch.cuda.graph(graph, stream=stream):
            marker.add_(1)
            with pytest.raises(RuntimeError, match="eager"):
                if operation == "plan":
                    wrapper.plan(
                        129, 7, 3, 7, 18, [(frame, frame) for frame in range(7)]
                    )
                else:
                    wrapper.run(query, key, value)
    graph.replay()
    torch.cuda.synchronize()
    assert marker.item() == 1
    with torch.cuda.stream(stream):
        actual = wrapper.run(query, key, value)
    stream.synchronize()
    _assert_output(
        actual,
        _oracle(
            query,
            key,
            value,
            3,
            7,
            18,
            [(frame, frame) for frame in range(7)],
            "both",
            128**-0.5,
        ),
    )


@pytest.mark.parametrize("name", ["query", "key", "value", "out"])
def test_mismatched_cuda_device(workspace, name):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(5, 2, 0, 1, 5, [(0, 0)])
    args = {
        key: torch.empty(5, 2, 128, dtype=torch.bfloat16, device=workspace.device)
        for key in ("query", "key", "value", "out")
    }
    other = (workspace.device.index + 1) % torch.cuda.device_count()
    args[name] = torch.empty(5, 2, 128, dtype=torch.bfloat16, device=f"cuda:{other}")
    with pytest.raises(ValueError, match=name):
        wrapper.run(**args)


def test_noncurrent_cuda_device_guard(workspace):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    query, key, value = _strided_inputs(129, 7, 514)
    other = (workspace.device.index + 1) % torch.cuda.device_count()
    with torch.cuda.device(other):
        wrapper = VDNWindowAttentionWrapper(workspace)
        wrapper.plan(129, 7, 3, 7, 18, [(frame, frame) for frame in range(7)])
        actual = wrapper.run(query, key, value)
        assert torch.cuda.current_device() == other
        assert actual.device == workspace.device
    _assert_output(
        actual,
        _oracle(
            query,
            key,
            value,
            3,
            7,
            18,
            [(frame, frame) for frame in range(7)],
            "both",
            128**-0.5,
        ),
    )


def test_workspace_and_plan_are_released(workspace):
    query, key, value = _strided_inputs(17, 2, 128)
    warmup = VDNWindowAttentionWrapper(workspace)
    warmup.plan(17, 2, 2, 3, 5, [(frame, frame) for frame in range(3)])
    warmup.run(query, key, value)
    del warmup
    torch.cuda.synchronize()
    gc.collect()
    baseline = torch.cuda.memory_allocated(workspace.device)
    owned = torch.empty_like(workspace)
    wrapper = VDNWindowAttentionWrapper(owned)
    wrapper.plan(17, 2, 2, 3, 5, [(frame, frame) for frame in range(3)])
    wrapper.run(query, key, value)
    references = [
        weakref.ref(obj)
        for obj in (owned, wrapper, wrapper._wrapper, wrapper._query_order)
    ]
    del wrapper, owned
    torch.cuda.synchronize()
    gc.collect()
    assert all(ref() is None for ref in references)
    assert torch.cuda.memory_allocated(workspace.device) <= baseline


def test_copy_kernels_large_index(workspace, request):
    """Opt-in 16 GiB copy test, without quadratic attention or huge metadata."""
    if not request.config.getoption("--full"):
        pytest.skip("large-index copy regression requires --full and 18 GiB free VRAM")
    if torch.cuda.mem_get_info()[0] < 18 * 1024**3:
        pytest.skip("large-index copy regression needs 18 GiB free VRAM")
    from flashinfer.triton.vdn import _pack_query_value, _scatter_output

    elements = 2**31 + 1024
    rows = elements // 128
    query = torch.full((rows, 1, 128), 0.5, device="cuda", dtype=torch.bfloat16)
    query[:32] = (
        torch.arange(4096, device="cuda", dtype=torch.float32)
        .view(32, 1, 128)
        .to(query.dtype)
    )
    query[-32:] = -query[:32]
    order = torch.arange(rows - 1, -1, -1, device="cuda", dtype=torch.int32)
    packed_query, packed_value, output = [torch.empty_like(query) for _ in range(3)]
    grid = ((elements + 1023) // 1024,)
    _pack_query_value[grid](
        query,
        query,
        order,
        packed_query,
        packed_value,
        elements,
        128,
        128,
        128,
        128,
        128,
        True,
        1024,
    )
    _scatter_output[grid](packed_query, order, output, elements, 128, 1024)
    torch.cuda.synchronize()
    for section in (slice(0, 32), slice(-32, None)):
        torch.testing.assert_close(output[section], query[section], rtol=0, atol=0)
        torch.testing.assert_close(
            packed_value[section], query[section], rtol=0, atol=0
        )
    torch.testing.assert_close(packed_query[:32], query[-32:].flip(0), rtol=0, atol=0)
    torch.testing.assert_close(packed_query[-32:], query[:32].flip(0), rtol=0, atol=0)
