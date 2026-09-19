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

import pytest
import torch

import flashinfer
from flashinfer.topk import TopKTieBreak


def _reference(scores, lengths, k, tie_break):
    """What the selection means, in plain torch.

    Only the first ``lengths[i]`` columns of a row are candidates; a row with
    fewer than ``k`` of them fills the rest of its output with ``-1``. Ties at
    the boundary are broken by index, in the direction asked for.
    """
    rows, _ = scores.shape
    out = torch.full((rows, k), -1, dtype=torch.int32, device=scores.device)
    for row in range(rows):
        length = int(lengths[row])
        if length <= 0:
            continue
        candidates = scores[row, :length].to(torch.float64)
        order = torch.arange(length, device=scores.device, dtype=torch.float64)
        if tie_break == TopKTieBreak.SMALL:
            key = torch.stack([candidates, -order], dim=1)
        elif tie_break == TopKTieBreak.LARGE:
            key = torch.stack([candidates, order], dim=1)
        else:
            key = torch.stack([candidates, -order], dim=1)
        # Sort by score, then by the index direction the tie-break names.
        ranked = sorted(
            range(length), key=lambda i: (-key[i, 0].item(), -key[i, 1].item())
        )
        taken = ranked[: min(k, length)]
        out[row, : len(taken)] = torch.tensor(
            taken, dtype=torch.int32, device=scores.device
        )
    return out


def _prepared(rows, width, k, device, **kwargs):
    return flashinfer.PreparedTopKRaggedTransform(
        num_rows=rows,
        max_len=width,
        k=k,
        dtype=torch.float32,
        device=device,
        **kwargs,
    )


def _run(prepared, scores, lengths, offsets=None):
    device = scores.device
    rows = scores.size(0)
    if offsets is None:
        offsets = torch.zeros(rows, dtype=torch.int32, device=device)
    workspace = torch.zeros(prepared.workspace_size(), dtype=torch.uint8, device=device)
    out = torch.empty(rows, prepared.k, dtype=torch.int32, device=device)
    return prepared.run(scores, offsets, lengths, out=out, workspace=workspace)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 64])
def test_a_row_shorter_than_k_fills_the_rest_with_minus_one(device, length):
    """Below, at, and above ``k`` candidates, including a row with none."""
    rows, width, k = 4, 128, 4
    generator = torch.Generator(device=device).manual_seed(length)
    scores = torch.randn(rows, width, device=device, generator=generator)
    lengths = torch.full((rows,), length, dtype=torch.int32, device=device)

    prepared = _prepared(rows, width, k, device, deterministic=True)
    out = _run(prepared, scores, lengths)

    expected_live = min(k, length)
    for row in range(rows):
        live = out[row][out[row] >= 0]
        assert live.numel() == expected_live, f"row {row}"
        padding = out[row][expected_live:]
        assert bool((padding == -1).all()), "the tail past the row's length"
        assert bool((live < length).all()), "an index past the row's length"
        assert live.unique().numel() == live.numel(), "a repeated index"


def test_rows_of_different_lengths_in_one_batch(device):
    """Every row is cut by its own length, not by the widest one."""
    width, k = 256, 8
    lengths = torch.tensor([0, 1, 7, 8, 9, 200], dtype=torch.int32, device=device)
    rows = lengths.numel()
    generator = torch.Generator(device=device).manual_seed(11)
    scores = torch.randn(rows, width, device=device, generator=generator)

    prepared = _prepared(rows, width, k, device, deterministic=True)
    out = _run(prepared, scores, lengths)

    for row in range(rows):
        live = out[row][out[row] >= 0]
        assert live.numel() == min(k, int(lengths[row]))
        assert bool((live < int(lengths[row])).all())


@pytest.mark.parametrize("tie_break", [TopKTieBreak.SMALL, TopKTieBreak.LARGE])
def test_the_tie_break_decides_which_boundary_index_wins(device, tie_break):
    """Every score equal, so the whole selection is the tie-break's answer."""
    rows, width, k = 3, 64, 8
    scores = torch.zeros(rows, width, device=device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)

    prepared = _prepared(
        rows, width, k, device, deterministic=True, tie_break=tie_break
    )
    out = _run(prepared, scores, lengths)

    chosen = out.sort(dim=1).values
    if tie_break == TopKTieBreak.SMALL:
        expected = torch.arange(k, dtype=torch.int32, device=device)
    else:
        expected = torch.arange(width - k, width, dtype=torch.int32, device=device)
    for row in range(rows):
        torch.testing.assert_close(chosen[row], expected, rtol=0, atol=0)


def test_the_selection_matches_a_plain_torch_reference(device):
    """Distinct scores, so the answer is the answer whatever the backend."""
    rows, width, k = 6, 512, 16
    generator = torch.Generator(device=device).manual_seed(5)
    # A permutation per row: no ties, so the top-k set is unambiguous.
    scores = torch.stack(
        [
            torch.randperm(width, device=device, generator=generator).float()
            for _ in range(rows)
        ]
    )
    lengths = torch.tensor(
        [width, width // 2, k, k - 1, 1, 0], dtype=torch.int32, device=device
    )

    prepared = _prepared(rows, width, k, device, deterministic=True)
    out = _run(prepared, scores, lengths)
    expected = _reference(scores, lengths, k, TopKTieBreak.NONE)

    for row in range(rows):
        got_live = out[row][out[row] >= 0].sort().values
        want_live = expected[row][expected[row] >= 0].sort().values
        torch.testing.assert_close(got_live, want_live, rtol=0, atol=0)


def test_the_offsets_are_added_to_every_selected_index(device):
    """The transform half: a row's indices come back in its own KV range."""
    rows, width, k = 4, 128, 8
    generator = torch.Generator(device=device).manual_seed(7)
    scores = torch.randn(rows, width, device=device, generator=generator)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    offsets = torch.arange(0, rows * width, width, dtype=torch.int32, device=device)

    prepared = _prepared(rows, width, k, device, deterministic=True)
    with_offsets = _run(prepared, scores, lengths, offsets=offsets)
    without = _run(prepared, scores, lengths)

    torch.testing.assert_close(
        with_offsets, without + offsets.reshape(-1, 1), rtol=0, atol=0
    )


def test_running_allocates_nothing(device):
    """The reason this exists: nothing lands in a graph's private pool."""
    rows, width, k = 8, 4096, 256
    generator = torch.Generator(device=device).manual_seed(13)
    scores = torch.randn(rows, width, device=device, generator=generator)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    offsets = torch.zeros(rows, dtype=torch.int32, device=device)

    prepared = _prepared(
        rows, width, k, device, deterministic=True, dsa_graph_safe=True
    )
    workspace = torch.zeros(prepared.workspace_size(), dtype=torch.uint8, device=device)
    out = torch.empty(rows, k, dtype=torch.int32, device=device)
    prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    torch.cuda.synchronize()

    before = torch.cuda.memory_allocated()
    prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


def test_it_replays_under_a_graph(device):
    """Captured once, replayed against new scores."""
    rows, width, k = 8, 1024, 32
    generator = torch.Generator(device=device).manual_seed(17)
    scores = torch.randn(rows, width, device=device, generator=generator)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    offsets = torch.zeros(rows, dtype=torch.int32, device=device)

    prepared = _prepared(
        rows, width, k, device, deterministic=True, dsa_graph_safe=True
    )
    workspace = torch.zeros(prepared.workspace_size(), dtype=torch.uint8, device=device)
    out = torch.empty(rows, k, dtype=torch.int32, device=device)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.run(scores, offsets, lengths, out=out, workspace=workspace)

    scores.copy_(torch.randn(rows, width, device=device, generator=generator))
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()

    prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        replayed.sort(dim=1).values, out.sort(dim=1).values, rtol=0, atol=0
    )


def test_a_workspace_too_small_is_refused(device):
    """Told before the kernel runs, not by a fault inside it."""
    rows, width, k = 4, 256, 8
    scores = torch.randn(rows, width, device=device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    offsets = torch.zeros(rows, dtype=torch.int32, device=device)
    prepared = _prepared(rows, width, k, device, deterministic=True)
    needed = prepared.workspace_size()
    out = torch.empty(rows, k, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="workspace needs"):
        prepared.run(
            scores,
            offsets,
            lengths,
            out=out,
            workspace=torch.zeros(needed - 1, dtype=torch.uint8, device=device),
        )


def test_a_mismatched_output_is_refused(device):
    """The output is the caller's, so its shape and dtype are checked."""
    rows, width, k = 4, 256, 8
    scores = torch.randn(rows, width, device=device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    offsets = torch.zeros(rows, dtype=torch.int32, device=device)
    prepared = _prepared(rows, width, k, device, deterministic=True)
    workspace = torch.zeros(prepared.workspace_size(), dtype=torch.uint8, device=device)

    with pytest.raises(ValueError, match="shape"):
        prepared.run(
            scores,
            offsets,
            lengths,
            out=torch.empty(rows, k + 1, dtype=torch.int32, device=device),
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="int32"):
        prepared.run(
            scores,
            offsets,
            lengths,
            out=torch.empty(rows, k, dtype=torch.int64, device=device),
            workspace=workspace,
        )


def test_the_backend_is_fixed_when_the_transform_is_prepared(device):
    """Chosen once, so a run cannot quietly land on a different kernel."""
    prepared = _prepared(8, 4096, 256, device, deterministic=True)
    assert prepared.backend in ("cub", "radix")
    first = prepared.backend
    scores = torch.randn(8, 4096, device=device)
    lengths = torch.full((8,), 4096, dtype=torch.int32, device=device)
    _run(prepared, scores, lengths)
    assert prepared.backend == first


@pytest.fixture
def forced_cub(monkeypatch):
    """Make the resolver pick CUB, whatever the shape heuristics prefer.

    Without this the deterministic settings QSA uses always land on radix, and
    the CUB half of the prepared path would never run.
    """
    monkeypatch.setenv("FLASHINFER_TOPK_ALGO", "cub")


def _cub_prepared(rows, width, k, device, **kwargs):
    prepared = _prepared(rows, width, k, device, deterministic=False, **kwargs)
    if prepared.backend != "cub":
        pytest.skip(f"this device resolves to {prepared.backend}, not cub")
    return prepared


def test_the_cub_backend_sizes_its_own_workspace(device, forced_cub):
    """CUB answers from the shape it was prepared for, with no tensors."""
    rows, width, k = 8, 4096, 256
    prepared = _cub_prepared(rows, width, k, device)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    assert prepared.workspace_size() > 0
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "the query allocated"


def test_the_cub_backend_runs_into_the_callers_buffers(device, forced_cub):
    """Same answer as the per-call API, into an output the caller owns."""
    rows, width, k = 8, 4096, 256
    generator = torch.Generator(device=device).manual_seed(19)
    scores = torch.stack(
        [
            torch.randperm(width, device=device, generator=generator).float()
            for _ in range(rows)
        ]
    )
    lengths = torch.tensor(
        [width, width // 2, k, k - 1, 1, 0, width, 3], dtype=torch.int32, device=device
    )
    offsets = torch.zeros(rows, dtype=torch.int32, device=device)
    prepared = _cub_prepared(rows, width, k, device)
    out = _run(prepared, scores, lengths)

    expected = _reference(scores, lengths, k, TopKTieBreak.NONE)
    for row in range(rows):
        got_live = out[row][out[row] >= 0].sort().values
        want_live = expected[row][expected[row] >= 0].sort().values
        torch.testing.assert_close(got_live, want_live, rtol=0, atol=0)

    reference = flashinfer.top_k_ragged_transform(
        scores, offsets, lengths, k, deterministic=False
    )
    torch.testing.assert_close(
        out.sort(dim=1).values, reference.sort(dim=1).values, rtol=0, atol=0
    )


def test_the_cub_backend_allocates_nothing_and_replays(device, forced_cub):
    """The same two properties the radix path has to hold, on CUB."""
    rows, width, k = 8, 4096, 256
    generator = torch.Generator(device=device).manual_seed(21)
    scores = torch.randn(rows, width, device=device, generator=generator)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    offsets = torch.zeros(rows, dtype=torch.int32, device=device)
    prepared = _cub_prepared(rows, width, k, device)
    workspace = torch.zeros(prepared.workspace_size(), dtype=torch.uint8, device=device)
    out = torch.empty(rows, k, dtype=torch.int32, device=device)
    prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    torch.cuda.synchronize()

    before = torch.cuda.memory_allocated()
    prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    expected = out.clone()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.sort(dim=1).values, expected.sort(dim=1).values, rtol=0, atol=0
    )


def test_the_cub_backend_refuses_a_workspace_one_byte_short(device, forced_cub):
    """The exact size works; a byte less is refused before the kernel."""
    rows, width, k = 8, 4096, 256
    scores = torch.randn(rows, width, device=device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    offsets = torch.zeros(rows, dtype=torch.int32, device=device)
    prepared = _cub_prepared(rows, width, k, device)
    needed = prepared.workspace_size()
    out = torch.empty(rows, k, dtype=torch.int32, device=device)

    prepared.run(
        scores,
        offsets,
        lengths,
        out=out,
        workspace=torch.zeros(needed, dtype=torch.uint8, device=device),
    )
    with pytest.raises(ValueError, match="workspace needs"):
        prepared.run(
            scores,
            offsets,
            lengths,
            out=out,
            workspace=torch.zeros(needed - 1, dtype=torch.uint8, device=device),
        )


def test_the_clusters_backend_is_refused_when_it_would_be_chosen(device, monkeypatch):
    """A contract test, not an SM100 run.

    The clusters backend allocates its own output and overflow buffer, so it
    cannot serve a prepared transform. This device may not be one it would ever
    be chosen on, so the choice is forced and the refusal checked.
    """
    monkeypatch.setattr(
        flashinfer.topk, "resolve_ragged_transform_backend", lambda **_: "clusters"
    )
    with pytest.raises(NotImplementedError, match="clusters"):
        _prepared(8, 4096, 256, device)


def test_a_deterministic_run_repeats_its_output_exactly(device):
    """Not the same set -- the same order, which is what deterministic means."""
    rows, width, k = 8, 2048, 64
    generator = torch.Generator(device=device).manual_seed(23)
    scores = torch.randn(rows, width, device=device, generator=generator)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    offsets = torch.zeros(rows, dtype=torch.int32, device=device)
    prepared = _prepared(
        rows, width, k, device, deterministic=True, dsa_graph_safe=True
    )
    assert prepared.backend == "radix"
    workspace = torch.zeros(prepared.workspace_size(), dtype=torch.uint8, device=device)
    out = torch.empty(rows, k, dtype=torch.int32, device=device)

    prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    torch.cuda.synchronize()
    first = out.clone()
    for _ in range(4):
        out.zero_()
        prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, first, rtol=0, atol=0)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.run(scores, offsets, lengths, out=out, workspace=workspace)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, first, rtol=0, atol=0)


def test_offsets_reach_the_live_indices_and_leave_the_padding_alone(device):
    """A short row keeps its ``-1`` tail however far its range is shifted."""
    width, k = 256, 8
    lengths = torch.tensor([0, 3, 8, 9], dtype=torch.int32, device=device)
    rows = lengths.numel()
    offsets = torch.tensor([0, 1000, 2000, 3000], dtype=torch.int32, device=device)
    generator = torch.Generator(device=device).manual_seed(29)
    scores = torch.randn(rows, width, device=device, generator=generator)

    prepared = _prepared(rows, width, k, device, deterministic=True)
    out = _run(prepared, scores, lengths, offsets=offsets)

    for row in range(rows):
        length = int(lengths[row])
        live_count = min(k, length)
        live = out[row][:live_count]
        # Shifted once into the row's own range, and never past its length.
        assert bool((live >= int(offsets[row])).all())
        assert bool((live < int(offsets[row]) + length).all())
        padding = out[row][live_count:]
        assert bool((padding == -1).all()), "the padding took an offset"


def test_a_call_that_differs_from_the_prepared_shape_is_refused(device):
    """The backend was chosen for one shape, dtype and device."""
    rows, width, k = 4, 256, 8
    prepared = _prepared(rows, width, k, device, deterministic=True)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    offsets = torch.zeros(rows, dtype=torch.int32, device=device)
    out = torch.empty(rows, k, dtype=torch.int32, device=device)
    scores = torch.randn(rows, width, device=device)
    workspace = torch.zeros(prepared.workspace_size(), dtype=torch.uint8, device=device)

    with pytest.raises(ValueError, match="shape"):
        prepared.run(
            torch.randn(rows, width * 2, device=device),
            offsets,
            lengths,
            out=out,
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="dtype|must be torch"):
        prepared.run(
            scores.to(torch.float16), offsets, lengths, out=out, workspace=workspace
        )
    with pytest.raises(ValueError, match="contiguous"):
        wide = torch.randn(rows, width * 2, device=device)
        prepared.run(wide[:, ::2], offsets, lengths, out=out, workspace=workspace)
    with pytest.raises(ValueError, match="aligned"):
        prepared.run(
            scores,
            offsets,
            lengths,
            out=out,
            workspace=torch.zeros(
                workspace.numel() + 16, dtype=torch.uint8, device=device
            )[1:],
        )
