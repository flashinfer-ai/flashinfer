# SPDX-License-Identifier: Apache-2.0
"""CPU arithmetic and SM120 native-XQA bucket behavior tests."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch


def _check_aligned_boundaries(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise unittest.SkipTest(
            "A C++17 host compiler is needed for the shared arithmetic test"
        )
    header = (
        Path(__file__).resolve().parents[2] / "flashinfer/experimental/"
        "xqa_decode_buckets/csrc/task_geometry.h"
    )
    source = tmp_path / "geometry.cpp"
    source.write_text(f'''#include "{header.as_posix()}"
#include <cassert>
#include <cstdint>
#include <initializer_list>
int main() {{
  using flashinfer::xqa_buckets::aligned_boundary;
  using flashinfer::xqa_buckets::direct_boundary;
  using flashinfer::xqa_buckets::direct_split_count;
  using flashinfer::xqa_buckets::direct_tile_count;
  assert(direct_split_count(0, 2) == 0);
  assert(direct_split_count(512, 170) == 2);
  assert(direct_boundary(13, 512, 512) == 13);
  assert(direct_boundary(512, 512, 512) == 512);
  for (int length : {{0, 1, 255, 256, 257, 512, 8193, 2147483647}}) {{
    for (int cap : {{1, 2, 32, 170, 1024}}) {{
      int tiles = direct_tile_count(length), count = direct_split_count(length, cap);
      assert(tiles == (int64_t(length) + 255) / 256);
      assert(count >= 0 && count <= cap && count <= tiles);
      int previous = 0;
      for (int i = 0; i < count; ++i) {{
        assert(aligned_boundary(i, tiles, count) == previous);
        int end = aligned_boundary(i + 1, tiles, count);
        assert(end > previous && end <= tiles);
        previous = end;
      }}
      assert(previous == tiles);
    }}
  }}
  for (int n : {{1, 2, 33, 512, 8388607}}) {{
    for (int count : {{1, 2, 32, 170, 340, 1024}}) {{
      if (count > n) continue;
      int previous = 0;
      for (int i = 0; i <= count; ++i) {{
        int boundary = aligned_boundary(i, n, count);
        assert(direct_boundary(i, n, count) == boundary);
        assert(boundary == int(int64_t(i) * n / count));
        assert(boundary >= previous && boundary <= n);
        if (i) assert(boundary > previous);
        previous = boundary;
      }}
      assert(previous == n);
    }}
  }}
}}
''')
    binary = tmp_path / "geometry.exe"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)


def _require_sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise unittest.SkipTest("Requires SM120")


def _check_host_planner(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise unittest.SkipTest("A C++17 compiler is needed for the host planner test")
    header = (
        Path(__file__).resolve().parents[2]
        / "flashinfer/experimental/xqa_decode_buckets/csrc/host_plan.h"
    )
    source = tmp_path / "host_plan.cpp"
    source.write_text(f'''#include "{header.as_posix()}"
#include <cassert>
#include <vector>
int main() {{
  using namespace flashinfer::xqa_buckets;
  int lengths[] = {{0, 768, 512}};
  std::vector<int> storage(descriptor_words(3, 4));
  plan_host(lengths, storage.data(), 3, 4, 24, false);
  std::vector<int> expected = {{1,0,2,1, 1,2,3,1, 2,0,1,1, 2,1,2,1,
      0,0,0,0, 0,0,0,0, 0,0, 0,2, 2,2, 0,1, 1,3, 3,4, 0,0, 2,4,3,2}};
  assert(storage == expected);
  plan_host(lengths, storage.data(), 3, 4, 24, true);
  expected = {{1,0,1,1, 1,1,3,1, 2,0,2,1, 0,0,0,0,
      0,0,0,0, 0,0,0,0, 0,0, 0,2, 2,1, 0,1, 1,2, 2,3, 0,0, 2,3,3,2}};
  assert(storage == expected);
  for (bool aligned : {{false, true}}) {{
    for (int page_size : {{16, 32, 64}}) {{
    for (int batch : {{1, 3, 8, 32}}) {{
      for (int ctas : {{32, 170, 340}}) {{
        for (int generation = 0; generation < 5; ++generation) {{
          std::vector<int> lens(batch);
          for (int r = 0; r < batch; ++r)
            lens[r] = generation == 0 ? 0 :
                (generation == 1 ? 4096 : (generation == 2 ? (r ? 4096 : 131072) :
                  (generation == 3 ? (r * 257) % 8193 : max_table_width(page_size) * page_size)));
          int capacity = ctas + batch - 1;
          std::vector<int> data(descriptor_words(batch, ctas) + 2, -71);
          plan_host(lens.data(), data.data() + 1, batch, ctas,
                    max_table_width(page_size), aligned, page_size);
          assert(data.front() == -71 && data.back() == -71);
          auto* tasks = reinterpret_cast<XQABucketTask*>(data.data() + 1);
          auto* requests = reinterpret_cast<XQABucketRange*>(data.data() + 1 + 4 * capacity);
          auto* spans = reinterpret_cast<XQABucketSpan*>(data.data() + 1 + 4 * capacity + 2 * batch);
          auto* h = data.data() + data.size() - 5;
          assert(h[1] <= capacity && h[2] <= ctas);
          int next = 0;
          for (int r = 0; r < batch; ++r) {{
            assert(requests[r].begin == next);
            int tile = 0;
            for (int i = 0; i < requests[r].count; ++i) {{
              auto t = tasks[next++];
              assert(t.request == r && t.begin == tile && t.end > tile && t.stride == 1);
              tile = t.end;
            }}
            assert(tile == direct_tile_count(lens[r]));
          }}
          assert(next == h[1]);
          next = 0;
          for (int c = 0; c < ctas; ++c) {{
            auto s = spans[c];
            if (s.end == 0) {{ assert(s.begin == 0); continue; }}
            assert(s.begin == next && s.end > next && s.end <= h[1]);
            next = s.end;
          }}
          assert(next == h[1]);
        }}
      }}
    }}
    }}
  }}
  for (int page_size : {{16, 32, 64}}) {{
    lengths[0] = 0; lengths[1] = page_size; lengths[2] = 1;
    plan_host(lengths, storage.data(), 3, 4, 1, false, page_size);
    lengths[1] = page_size + 1;
    bool rejected = false;
    try {{ plan_host(lengths, storage.data(), 3, 4, 1, false, page_size); }}
    catch (const std::invalid_argument&) {{ rejected = true; }}
    assert(rejected);
  }}
  for (int invalid : {{-1, 769}}) {{
    lengths[1] = invalid;
    bool rejected = false;
    try {{ plan_host(lengths, storage.data(), 3, 4, 24, false); }}
    catch (const std::invalid_argument&) {{ rejected = true; }}
    assert(rejected);
  }}
  lengths[0] = 0; lengths[1] = 5; lengths[2] = 6;
  plan_host(lengths, storage.data(), 3, 4, 24, false, 32, 5);
  for (int invalid : {{1, 2, 3, 4}}) {{
    lengths[1] = invalid;
    bool rejected = false;
    try {{ plan_host(lengths, storage.data(), 3, 4, 24, false, 32, 5); }}
    catch (const std::invalid_argument&) {{ rejected = true; }}
    assert(rejected);
  }}
}}
''')
    binary = tmp_path / "host_plan.exe"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)


def _case(
    lengths,
    *,
    interleaved=False,
    capacity_length=None,
    page_size=32,
    num_kv_heads=1,
    head_group_ratio=6,
    q_len_per_req=1,
):
    device = torch.device("cuda", torch.cuda.current_device())
    # Reserve the same per-request capacity for changed row lengths/page mappings.
    capacity_length = max(lengths) if capacity_length is None else capacity_length
    assert capacity_length >= max(lengths)
    width = max(1, (capacity_length + page_size - 1) // page_size)
    batch, pages = len(lengths), width * len(lengths)
    if interleaved:
        padding = 2 if num_kv_heads > 1 else 1
        half = page_size * num_kv_heads * 144 * padding
        backing = torch.empty((pages, 2 * half), dtype=torch.uint8, device=device)
        data = [
            backing.as_strided(
                (pages, page_size, num_kv_heads, 128),
                (2 * half, num_kv_heads * 128 * padding, 128 * padding, 1),
                side * half,
            )
            for side in range(2)
        ]
        sf_bytes = backing.view(torch.float8_e4m3fn)
        scales = [
            sf_bytes.as_strided(
                (pages, page_size, num_kv_heads, 16),
                (2 * half, num_kv_heads * 16 * padding, 16 * padding, 1),
                side * half + page_size * num_kv_heads * 128 * padding,
            )
            for side in range(2)
        ]
    else:
        data = [
            torch.empty(
                (pages, page_size, num_kv_heads, 128), dtype=torch.uint8, device=device
            )
            for _ in range(2)
        ]
        scales = [
            torch.empty(
                (pages, page_size, num_kv_heads, 16),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            for _ in range(2)
        ]
    for value in data:
        value.copy_(
            torch.randint(0, 256, value.shape, dtype=torch.uint8, device=device)
        )
    for value in scales:
        value.copy_(
            (
                (torch.rand(value.shape, device=device) * 0.5 + 0.25)
                * (
                    torch.arange(1, num_kv_heads + 1, device=device) / num_kv_heads
                ).view(1, 1, -1, 1)
            ).to(value.dtype)
        )
    table = torch.arange(pages, dtype=torch.int32, device=device).view(batch, width)
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    queries = [
        torch.randn(
            (batch, num_kv_heads * head_group_ratio, 256)
            if q_len_per_req == 1
            else (batch, q_len_per_req, num_kv_heads * head_group_ratio, 256),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.25
        for _ in range(2)
    ]
    return queries, data, scales, table, lens


def _reference(query, data, scales, table, lengths, bmm1, bmm2):
    # All FP4 codes, nontrivial E4M3 scales and the native BF16 operand rounding.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=query.device,
    )
    output = torch.zeros_like(query)
    page_size = data[0].shape[1]
    heads = data[0].shape[2]
    query_rows = query.unsqueeze(1) if query.ndim == 3 else query
    output_rows = output.unsqueeze(1) if output.ndim == 3 else output
    tokens = query_rows.shape[1]
    ratio = query_rows.shape[2] // heads
    for row, length in enumerate(lengths):
        if not length:
            continue
        assert length >= tokens
        ids = table[row, : (length + page_size - 1) // page_size].long()
        operands = []
        for packed, sf in zip(data, scales, strict=True):
            values = packed[ids].reshape(-1, heads, 128)[:length]
            codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2).long()
            block_scales = (
                sf.float()[ids]
                .reshape(-1, heads, 16)[:length]
                .repeat_interleave(16, -1)
            )
            operands.append((lut[codes] * block_scales).to(torch.bfloat16).float())
        # Independent prefix reference: no task boundaries or packed-mask math.
        for token in range(tokens):
            visible = length - tokens + token + 1
            for head in range(heads):
                qh = slice(head * ratio, (head + 1) * ratio)
                scores = (
                    query_rows[row, token, qh].float() @ operands[0][:visible, head].T
                ) * bmm1
                output_rows[row, token, qh] = (
                    scores.softmax(-1) @ operands[1][:visible, head] * bmm2
                ).to(output.dtype)
    return output


def _assert_output(actual, expected):
    assert torch.isfinite(actual).all()
    relative_rms = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-8)
    assert relative_rms.item() < 0.03


def _assert_plan(plan, lengths):
    if plan.direct:
        assert plan.ctas % len(lengths) == 0
        assert all(
            value.numel() == 0
            for value in (plan.tasks, plan.requests, plan.spans, plan.header)
        )
        return
    counts = [(n + 255) // 256 for n in lengths]
    limit = max(1, (sum(counts) + plan.ctas - 1) // plan.ctas)
    if plan.mode == "aligned":
        lo, hi = 1, max(1, max(counts))
        while lo < hi:
            mid = (lo + hi) // 2
            if sum((n + mid - 1) // mid for n in counts) <= plan.ctas:
                hi = mid
            else:
                lo = mid + 1
        limit = lo
    tasks, requests, prefix = [], [], 0
    spans = [[0, 0] for _ in range(plan.ctas)]
    for row, count in enumerate(counts):
        first = len(tasks)
        if plan.mode == "aligned":
            parts = (count + limit - 1) // limit
            tasks.extend(
                [row, i * count // parts, (i + 1) * count // parts, 1]
                for i in range(parts)
            )
        elif count:
            for bucket in range(prefix // limit, (prefix + count - 1) // limit + 1):
                if spans[bucket][0] == spans[bucket][1]:
                    spans[bucket][0] = len(tasks)
                tasks.append(
                    [
                        row,
                        max(prefix, bucket * limit) - prefix,
                        min(prefix + count, (bucket + 1) * limit) - prefix,
                        1,
                    ]
                )
                spans[bucket][1] = len(tasks)
        requests.append([first, len(tasks) - first])
        prefix += count
    used = (prefix + limit - 1) // limit
    if plan.mode == "aligned":
        spans[: len(tasks)] = [[i, i + 1] for i in range(len(tasks))]
        used = len(tasks)
    assert plan.header.cpu().tolist() == [
        limit,
        len(tasks),
        used,
        max(n for _, n in requests),
    ]
    assert plan.tasks[: len(tasks)].cpu().tolist() == tasks
    assert plan.requests.cpu().tolist() == requests
    assert plan.spans.cpu().tolist() == spans


def _check_changed_graph(
    mode,
    interleaved,
    *,
    direct=False,
    capacity_length=None,
    direct_batch=1,
    planner="gpu",
    page_size=32,
    num_kv_heads=1,
    head_group_ratio=6,
    q_len_per_req=1,
    query_tile=16,
):
    _require_sm120()
    from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets
    from flashinfer.experimental.xqa_decode_buckets import backend

    initial = [
        0,
        1,
        page_size - 1,
        page_size,
        page_size + 1,
        255,
        256,
        257,
        511,
        512,
        513,
        0,
        8193,
    ]
    changed = [
        513,
        0,
        257,
        1,
        0,
        512,
        page_size - 1,
        256,
        page_size + 1,
        255,
        page_size,
        8193,
        0,
    ]
    replay_lengths = (initial, changed, [0] * len(initial), initial)
    if direct and direct_batch == 1:
        initial = [512]
        replay_lengths = [
            [n]
            for n in (0, 1, 255, 256, 257, 512, 8193, 65537, capacity_length, 0, 512)
            if n <= capacity_length
        ]
    elif direct:
        if direct_batch == 4:
            initial = [512, 257, 1, 0]
            replay_lengths = (
                initial,
                [0, 512, 256, 1],
                [255, 0, 257, 512],
                [512, 1, 0, 256],
                [0] * 4,
                initial,
            )
        elif direct_batch == 3:
            initial = [0, 257, 513]
            replay_lengths = (initial, [513, 0, 1], [1, 256, 0], [0] * 3, initial)
        else:
            assert direct_batch == 32
            initial = [0, 1, 255, 256] * 8
            replay_lengths = (initial, [256, 0, 1, 255] * 8, [0] * 32, initial)
    if q_len_per_req > 1:
        initial = [max(n, q_len_per_req) if n else 0 for n in initial]
        replay_lengths = [
            [max(n, q_len_per_req) if n else 0 for n in row] for row in replay_lengths
        ]
    capacity = max(initial) if capacity_length is None else capacity_length
    width = max(1, (capacity + page_size - 1) // page_size)
    sms = backend.get_device_sm_count(torch.device("cuda", torch.cuda.current_device()))
    expected_ctas, eligible_direct = backend.plan_geometry(
        len(initial),
        width,
        sms,
        page_size,
        num_kv_heads,
        head_group_ratio,
        q_len_per_req,
        query_tile,
    )
    if direct and not eligible_direct:
        raise unittest.SkipTest(
            f"Direct fixture exceeds SM budget: B={len(initial)}, width={width}, "
            f"page_size={page_size}, heads={num_kv_heads}, ratio={head_group_ratio}, "
            f"T={q_len_per_req}, M={query_tile}, SMs={sms}"
        )
    queries, data, scales, table, lens = _case(
        initial,
        interleaved=interleaved,
        capacity_length=capacity_length,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_group_ratio=head_group_ratio,
        q_len_per_req=q_len_per_req,
    )
    plan = create_xqa_decode_bucket_plan(
        lens,
        page_table_width=table.shape[1],
        mode=mode,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_group_ratio=head_group_ratio,
        q_len_per_req=q_len_per_req,
        planner=planner,
        query_tile=query_tile,
        host_seq_lens=torch.tensor(initial, dtype=torch.int32)
        if planner == "host"
        else None,
    )
    assert plan.direct == direct
    assert plan.ctas == expected_ctas
    assert plan.page_size == page_size
    assert plan.num_kv_heads == num_kv_heads
    assert plan.head_group_ratio == head_group_ratio
    assert plan.q_len_per_req == q_len_per_req
    assert plan.query_tile == query_tile
    assert plan.num_groups == num_kv_heads * (
        (q_len_per_req * head_group_ratio + query_tile - 1) // query_tile
    )
    print(
        dict(
            mode=mode,
            planner=planner,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            head_group_ratio=head_group_ratio,
            q_len_per_req=q_len_per_req,
            query_tile=query_tile,
            resources=plan.resources,
            ctas=plan.ctas,
        ),
        flush=True,
    )
    workspaces = [plan.create_workspace() for _ in queries]
    guards = []
    if plan.num_groups > 1:
        for index in range(len(workspaces)):
            scratch_guard = torch.full(
                (plan.scratch_bytes + 32,), 167, dtype=torch.uint8, device=lens.device
            )
            counter_guard = torch.full(
                (len(initial) * plan.num_groups + 8,),
                -73,
                dtype=torch.int32,
                device=lens.device,
            )
            counter_guard[4:-4].zero_()
            workspaces[index] = backend.BucketWorkspace(
                scratch_guard[16:-16], counter_guard[4:-4]
            )
            guards.append((scratch_guard, counter_guard))
    assert all(
        w.semaphores.numel() == len(initial) * plan.num_groups for w in workspaces
    )
    if not direct:
        assert plan.tasks.shape[0] == plan.ctas + len(initial) - 1
    assert len({workspace.scratch.data_ptr() for workspace in workspaces}) == len(
        workspaces
    )
    assert len({workspace.semaphores.data_ptr() for workspace in workspaces}) == len(
        workspaces
    )
    backings = [
        torch.full(
            (len(initial) + 1, num_kv_heads * head_group_ratio, 256)
            if q_len_per_req == 1
            else (
                len(initial) + 1,
                q_len_per_req,
                num_kv_heads * head_group_ratio,
                256,
            ),
            19,
            dtype=torch.bfloat16,
            device=lens.device,
        )
        for _ in queries
    ]
    outputs = [value[:-1] for value in backings]
    bmm1, bmm2 = 0.7 / 16, 1.3

    def compute_layers():
        for query, workspace, out in zip(queries, workspaces, outputs, strict=True):
            xqa_decode_with_buckets(
                query,
                *data,
                *scales,
                table,
                plan,
                workspace,
                bmm1_scale=bmm1,
                bmm2_scale=bmm2,
                out=out,
            )

    # Compute must consume the plan, even when assignment is unavailable.
    with patch.object(
        backend, "_prepare", side_effect=AssertionError("implicit assignment")
    ):
        compute_layers()
    graph = torch.cuda.CUDAGraph()
    no_assignment = (
        patch.object(
            backend, "_prepare", side_effect=AssertionError("direct assignment")
        )
        if direct
        else nullcontext()
    )
    with no_assignment, torch.cuda.graph(graph):
        if planner == "gpu":
            plan.prepare()
        compute_layers()
    pointers = [
        value.data_ptr()
        for value in (plan.tasks, plan.requests, plan.spans, plan.header)
    ]
    for lengths in replay_lengths:
        lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=lens.device))
        if planner == "host":
            plan.host_seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
            plan.prepare()
        table.copy_(table.roll(1, dims=1))
        if direct and len(initial) > 1:
            table.copy_(table.roll(1, dims=0))
        for query in queries:
            query.normal_(0, 0.25)
        inputs = [
            *queries,
            *data,
            *scales,
            table,
            lens,
            plan.causal_mask.view(torch.int16),
        ]
        snapshots = [value.clone() for value in inputs]
        graph.replay()
        torch.cuda.synchronize()
        _assert_plan(plan, lengths)
        for scratch_guard, counter_guard in guards:
            assert (scratch_guard[:16] == 167).all() and (
                scratch_guard[-16:] == 167
            ).all()
            assert (counter_guard[:4] == -73).all() and (
                counter_guard[-4:] == -73
            ).all()
        assert pointers == [
            value.data_ptr()
            for value in (plan.tasks, plan.requests, plan.spans, plan.header)
        ]
        for query, out, backing, workspace in zip(
            queries, outputs, backings, workspaces, strict=True
        ):
            expected = _reference(query, data, scales, table, lengths, bmm1, bmm2)
            _assert_output(out, expected)
            for head in range(num_kv_heads):
                hs = slice(head * head_group_ratio, (head + 1) * head_group_ratio)
                _assert_output(out[..., hs, :], expected[..., hs, :])
            assert (backing[-1] == 19).all()
            assert (workspace.semaphores == 0).all()
            for row, length in enumerate(lengths):
                if not length:
                    assert (out[row] == 0).all()
        assert all(
            torch.equal(value, saved)
            for value, saved in zip(inputs, snapshots, strict=True)
        )


def _check_module_case(lengths, mode):
    _require_sm120()
    from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets

    queries, data, scales, table, lens = _case(lengths)
    plan = create_xqa_decode_bucket_plan(
        lens, page_table_width=table.shape[1], mode=mode
    )
    workspace = plan.create_workspace()
    actual = xqa_decode_with_buckets(
        queries[0],
        *data,
        *scales,
        table,
        plan,
        workspace,
        bmm1_scale=0.7 / 16,
        bmm2_scale=1.3,
    )
    _assert_plan(plan, lengths)
    _assert_output(
        actual, _reference(queries[0], data, scales, table, lengths, 0.7 / 16, 1.3)
    )
    assert (workspace.semaphores == 0).all()


class GeometryTests(unittest.TestCase):
    def test_query_tiles(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "flashinfer/experimental/xqa_decode_buckets/geometry.py"
        )
        spec = importlib.util.spec_from_file_location("xqa_tile_geometry", path)
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
        for tokens in range(2, 6):
            for ratio in (4, 6, 8):
                for heads in (1, 2, 8):
                    for tile in (16, 32):
                        self.assertEqual(
                            geometry.query_group_count(heads, ratio, tokens, tile),
                            heads * ((tokens * ratio + tile - 1) // tile),
                        )
        self.assertEqual(
            geometry.plan_geometry(32, 16, 170, 32, 4, 6, 3, 32), (43, False)
        )
        self.assertEqual(geometry.plan_geometry(4, 16, 170, 32, 4, 6, 3, 32), (8, True))
        self.assertEqual(geometry.query_group_count(8, 8, 5, 32), 16)
        # M32 can remove row groups and expand static direct eligibility.
        self.assertEqual(
            geometry.plan_geometry(3, 257, 170, 32, 1, 6, 5, 32), (99, True)
        )
        for sms in (132, 170, 200):
            self.assertFalse(
                geometry.plan_geometry(3, (sms + 1) * 8, sms, 32, 1, 6, 5, 32)[1]
            )
        for tile in (True, 0, 8, 64, 16.0):
            with self.assertRaises(ValueError):
                geometry.query_group_count(1, 6, 3, tile)
        with self.assertRaisesRegex(ValueError, "requires q_len_per_req > 1"):
            geometry.query_group_count(1, 6, 1, 32)

    def test_multi_query_groups_and_mask(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "flashinfer/experimental/xqa_decode_buckets/geometry.py"
        )
        spec = importlib.util.spec_from_file_location("xqa_multi_geometry", path)
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
        for tokens in range(1, 6):
            words = geometry.causal_mask_words(tokens)
            for row, word in enumerate(words):
                self.assertEqual(
                    [bool(word & (1 << col)) for col in range(tokens)],
                    [col <= row for col in range(tokens)],
                )
                self.assertEqual(word >> tokens, 0)
            for heads in (1, 2, 8):
                for ratio in (4, 6, 8):
                    self.assertEqual(
                        geometry.query_group_count(heads, ratio, tokens),
                        heads * ((tokens * ratio + 15) // 16),
                    )
        self.assertEqual(geometry.plan_geometry(4, 16, 170, 32, 8, 8, 5), (8, True))
        self.assertEqual(geometry.plan_geometry(4, 17, 170, 32, 8, 8, 5), (8, False))
        self.assertEqual(geometry.plan_geometry(13, 257, 170, 32, 8, 8, 5), (13, False))
        for tokens in (True, 0, 6, 2.0):
            with self.assertRaises(ValueError):
                geometry.query_group_count(1, 6, tokens)
        with self.assertRaises(ValueError):
            geometry.query_group_count((1 << 31) - 1, 8, 5)

    def test_causal_reference(self):
        # Zero logits make each row the known mean of its visible V prefix.
        query = torch.zeros((1, 5, 4, 256), dtype=torch.bfloat16)
        key = torch.zeros((1, 16, 1, 128), dtype=torch.uint8)
        value = torch.zeros_like(key)
        for token, code in enumerate((0x00, 0x22, 0x44, 0x66, 0x77)):
            value[0, token].fill_(code)
        sf = torch.ones((1, 16, 1, 16)).to(torch.float8_e4m3fn)
        table = torch.zeros((1, 1), dtype=torch.int32)
        actual = _reference(query, [key, value], [sf, sf], table, [5], 1.0, 1.0)
        expected = (
            torch.tensor([0, 0.5, 1, 1.75, 2.6], dtype=torch.bfloat16)
            .view(1, 5, 1, 1)
            .expand_as(query)
        )
        _assert_output(actual, expected)
        value[0, 4].fill_(0xFF)
        changed = _reference(query, [key, value], [sf, sf], table, [5], 1.0, 1.0)
        _assert_output(changed[:, :4], expected[:, :4])
        self.assertGreater(
            (changed[:, 4].float() - actual[:, 4].float()).abs().max().item(), 1
        )

    def test_static_capacity_and_validation(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "flashinfer/experimental/xqa_decode_buckets/geometry.py"
        )
        spec = importlib.util.spec_from_file_location("xqa_bucket_geometry", path)
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
        for batch, width, sms, expected in (
            (1, 1, 132, 1),
            (1, 16, 132, 2),
            (4, 16, 132, 8),
            (32, 4096, 132, 132),
            (32, 1, 4, 32),
        ):
            self.assertEqual(geometry.plan_geometry(batch, width, sms)[0], expected)
        for batch, width, sms, expected in (
            (1, 4096, 132, (132, True)),
            (4, 16, 8, (8, True)),
            (4, 16, 7, (7, False)),
            (32, 1, 4, (32, False)),
            (4, 264, 132, (132, True)),
            (4, 265, 132, (132, False)),
            (3, 17, 9, (9, True)),
            (3, 17, 8, (8, False)),
        ):
            self.assertEqual(geometry.plan_geometry(batch, width, sms), expected)
        for values in (
            (True, 16, 20),
            (0, 16, 20),
            (33, 16, 20),
            (1, True, 20),
            (1, 0, 20),
            (1, -1, 20),
            (1, geometry.MAX_TABLE_WIDTH + 1, 20),
            (1, 16, True),
            (1, 16, 0),
            (1, 16, 1.5),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                geometry.plan_geometry(*values)
        for page_size in (16, 32, 64):
            self.assertEqual(
                geometry.plan_geometry(4, 512 // page_size, 132, page_size), (8, True)
            )
            self.assertEqual(
                geometry.plan_geometry(4, 8192 // page_size, 132, page_size),
                (128, True),
            )
            self.assertEqual(
                geometry.plan_geometry(4, 8448 // page_size, 132, page_size),
                (132, True),
            )
            self.assertEqual(
                geometry.plan_geometry(4, 8448 // page_size + 1, 132, page_size),
                (132, False),
            )
            self.assertEqual(
                geometry.max_table_width(page_size), ((1 << 31) - 256) // page_size
            )
            with self.assertRaises(ValueError):
                geometry.plan_capacity(
                    4, 132, geometry.max_table_width(page_size) + 1, page_size
                )
        for page_size in (True, 0, 8, 24, 128, 32.0):
            with self.assertRaises(ValueError):
                geometry.plan_geometry(4, 16, 132, page_size)

    def test_aligned_boundaries_use_wide_intermediates(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_aligned_boundaries(Path(directory))

    def test_head_group_geometry_and_bounds(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "flashinfer/experimental/xqa_decode_buckets/geometry.py"
        )
        spec = importlib.util.spec_from_file_location("xqa_head_geometry", path)
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
        for heads in (1, 2, 4, 8, 100):
            for ratio in (4, 6, 8):
                geometry.check_head_config(heads, ratio)
            budget = (170 + heads - 1) // heads
            self.assertEqual(
                geometry.plan_geometry(8, 4096, 170, 32, heads), (max(8, budget), False)
            )
            self.assertEqual(
                geometry.plan_geometry(1, 4096, 170, 32, heads), (budget, True)
            )
            self.assertEqual(
                geometry.workspace_capacity(8, max(8, budget), 4096, 32, heads),
                (max(8, budget) + 7) * heads,
            )
        self.assertEqual(geometry.plan_geometry(4, 16, 170, 32, 4), (8, True))
        self.assertEqual(geometry.plan_geometry(4, 16, 170, 32, 32), (6, False))
        self.assertEqual(geometry.plan_geometry(4, 16, 170, 32, 100), (4, False))
        for heads, ratio in (
            (0, 6),
            (-1, 6),
            (True, 6),
            (1.0, 6),
            (1, True),
            (1, 3),
            (1, 16),
        ):
            with self.assertRaises(ValueError):
                geometry.check_head_config(heads, ratio)
        with self.assertRaises(ValueError):
            geometry.workspace_capacity(32, 170, 4096, 32, 1 << 30)

    def test_compiled_host_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_host_planner(Path(directory))


class BucketGraphTests(unittest.TestCase):
    def test_host_replay(self):
        _require_sm120()
        for mode in ("persistent", "aligned"):
            for interleaved in (False, True):
                with self.subTest(mode=mode, interleaved=interleaved):
                    _check_changed_graph(mode, interleaved, planner="host")
            for batch, capacity in ((1, 131072), (4, 512)):
                with self.subTest(mode=mode, direct_batch=batch, planner="host"):
                    _check_changed_graph(
                        mode,
                        False,
                        direct=True,
                        capacity_length=capacity,
                        direct_batch=batch,
                        planner="host",
                    )

    def test_direct_replay(self):
        _require_sm120()
        for mode in ("persistent", "aligned"):
            for interleaved in (False, True):
                for capacity in (512, 131072):
                    with self.subTest(
                        mode=mode, interleaved=interleaved, capacity=capacity
                    ):
                        _check_changed_graph(
                            mode, interleaved, direct=True, capacity_length=capacity
                        )
                for batch, capacity in ((4, 512), (3, 513), (32, 256)):
                    with self.subTest(
                        mode=mode, interleaved=interleaved, direct_batch=batch
                    ):
                        _check_changed_graph(
                            mode,
                            interleaved,
                            direct=True,
                            capacity_length=capacity,
                            direct_batch=batch,
                        )

    def test_persistent_interleaved(self):
        _check_changed_graph("persistent", True)

    def test_persistent_separate(self):
        _check_changed_graph("persistent", False)

    def test_aligned_interleaved(self):
        _check_changed_graph("aligned", True)

    def test_aligned_separate(self):
        _check_changed_graph("aligned", False)


class BucketModuleTests(unittest.TestCase):
    def test_small_uniform_and_skew(self):
        _require_sm120()
        for mode in ("persistent", "aligned"):
            for lengths in ([512], [8064] * 32, [131072] + [4096] * 31):
                with self.subTest(mode=mode, lengths=lengths):
                    _check_module_case(lengths, mode)


class BucketPageTests(unittest.TestCase):
    def test_page_replay(self):
        _require_sm120()
        for page_size in (16, 64):
            for mode in ("persistent", "aligned"):
                for planner in ("gpu", "host"):
                    for interleaved in (False, True):
                        with self.subTest(
                            page_size=page_size,
                            mode=mode,
                            planner=planner,
                            interleaved=interleaved,
                        ):
                            _check_changed_graph(
                                mode, interleaved, planner=planner, page_size=page_size
                            )
                    with self.subTest(
                        page_size=page_size, mode=mode, planner=planner, direct_batch=4
                    ):
                        _check_changed_graph(
                            mode,
                            False,
                            planner=planner,
                            page_size=page_size,
                            direct=True,
                            direct_batch=4,
                            capacity_length=512,
                        )
            _check_changed_graph(
                "persistent",
                True,
                page_size=page_size,
                direct=True,
                capacity_length=131072,
            )


class BucketHeadTests(unittest.TestCase):
    def test_grouped_replay(self):
        _require_sm120()
        for heads in (1, 2, 4, 8):
            for ratio in (4, 6, 8):
                for mode in ("persistent", "aligned"):
                    for planner in ("gpu", "host"):
                        for interleaved in (False, True):
                            with self.subTest(
                                heads=heads,
                                ratio=ratio,
                                mode=mode,
                                planner=planner,
                                interleaved=interleaved,
                            ):
                                _check_changed_graph(
                                    mode,
                                    interleaved,
                                    planner=planner,
                                    num_kv_heads=heads,
                                    head_group_ratio=ratio,
                                )

    def test_grouped_direct_and_pages(self):
        _require_sm120()
        for heads, ratio in ((2, 4), (4, 6), (8, 8)):
            for mode in ("persistent", "aligned"):
                for planner in ("gpu", "host"):
                    for batch, capacity in ((1, 131072), (4, 512)):
                        with self.subTest(
                            heads=heads,
                            ratio=ratio,
                            mode=mode,
                            planner=planner,
                            batch=batch,
                        ):
                            _check_changed_graph(
                                mode,
                                True,
                                planner=planner,
                                num_kv_heads=heads,
                                head_group_ratio=ratio,
                                direct=True,
                                direct_batch=batch,
                                capacity_length=capacity,
                            )
        for page_size, ratio in ((16, 4), (64, 8)):
            for mode in ("persistent", "aligned"):
                for planner in ("gpu", "host"):
                    _check_changed_graph(
                        mode,
                        True,
                        planner=planner,
                        page_size=page_size,
                        num_kv_heads=4,
                        head_group_ratio=ratio,
                    )


def _check_future_kv(
    mode,
    interleaved,
    planner="gpu",
    page_size=32,
    direct=True,
    num_kv_heads=1,
    head_group_ratio=6,
    query_tile=16,
):
    _require_sm120()
    from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets

    tokens, length = 5, 257
    lengths = [length] if direct else [length, 0, tokens]
    # Reserve more tiles than the whole SM budget, including the M32/G1 case.
    # This is cold fixture construction; no capture or decode path reads it.
    capacity = (
        512
        if direct
        else (
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            + 1
        )
        * 256
    )
    queries, data, scales, table, lens = _case(
        lengths,
        interleaved=interleaved,
        capacity_length=capacity,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_group_ratio=head_group_ratio,
        q_len_per_req=tokens,
    )
    query = queries[0]
    query.fill_(0.25)
    data[0].zero_()
    data[1].fill_(0x11)
    for sf in scales:
        sf.fill_(0.5)
    page, offset = (length - 1) // page_size, (length - 1) % page_size
    # A strong final key exposes an accidental future-V contribution immediately.
    data[0][page, offset].fill_(0x77)
    plan = create_xqa_decode_bucket_plan(
        lens,
        page_table_width=table.shape[1],
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_group_ratio=head_group_ratio,
        q_len_per_req=tokens,
        mode=mode,
        query_tile=query_tile,
        planner=planner,
        host_seq_lens=torch.tensor(lengths, dtype=torch.int32)
        if planner == "host"
        else None,
    )
    assert plan.direct == direct
    workspace, output = plan.create_workspace(), torch.empty_like(query)

    def compute():
        xqa_decode_with_buckets(
            query,
            *data,
            *scales,
            table,
            plan,
            workspace,
            bmm1_scale=1 / 16,
            bmm2_scale=1.3,
            out=output,
        )

    compute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if planner == "gpu":
            plan.prepare()
        compute()
    if planner == "host":
        plan.prepare()
    graph.replay()
    torch.cuda.synchronize()
    before = output.clone()
    _assert_output(before, _reference(query, data, scales, table, lengths, 1 / 16, 1.3))
    data[1][page, offset].fill_(0xFF)
    if planner == "host":
        plan.prepare()
    graph.replay()
    torch.cuda.synchronize()
    _assert_output(output, _reference(query, data, scales, table, lengths, 1 / 16, 1.3))
    _assert_output(output[:, :-1], before[:, :-1])
    assert (output[0, -1].float() - before[0, -1].float()).abs().max().item() > 1
    assert (workspace.semaphores == 0).all()
    print(
        dict(
            causal_future_v=True,
            length=length,
            q_len_per_req=tokens,
            query_tile=query_tile,
            mode=mode,
            planner=planner,
            page_size=page_size,
            direct=direct,
            num_kv_heads=num_kv_heads,
            head_group_ratio=head_group_ratio,
            resources=plan.resources,
        ),
        flush=True,
    )


class BucketQueryTileTests(unittest.TestCase):
    def test_causal_partition_smoke(self):
        _check_future_kv("persistent", False, head_group_ratio=8, query_tile=32)

    def test_prepared_m32_replay(self):
        for tokens, heads, ratio, page in ((3, 4, 6, 32), (5, 2, 8, 64), (2, 2, 4, 16)):
            for mode in ("persistent", "aligned"):
                for planner in ("gpu", "host"):
                    for interleaved in (False, True):
                        _check_changed_graph(
                            mode,
                            interleaved,
                            planner=planner,
                            page_size=page,
                            num_kv_heads=heads,
                            head_group_ratio=ratio,
                            q_len_per_req=tokens,
                            query_tile=32,
                        )

    def test_direct_m32_replay(self):
        for tokens, heads, ratio, page in ((3, 4, 6, 32), (5, 2, 8, 64)):
            for mode in ("persistent", "aligned"):
                for planner in ("gpu", "host"):
                    for batch, capacity in ((1, 8193), (4, 512)):
                        with self.subTest(
                            tokens=tokens,
                            heads=heads,
                            ratio=ratio,
                            page=page,
                            mode=mode,
                            planner=planner,
                            batch=batch,
                        ):
                            _check_changed_graph(
                                mode,
                                batch == 4,
                                planner=planner,
                                page_size=page,
                                num_kv_heads=heads,
                                head_group_ratio=ratio,
                                q_len_per_req=tokens,
                                query_tile=32,
                                direct=True,
                                direct_batch=batch,
                                capacity_length=capacity,
                            )

    def test_future_kv_m32(self):
        for heads, ratio, page in ((1, 6, 32), (2, 8, 64)):
            for direct in (False, True):
                for mode in ("persistent", "aligned"):
                    for planner in ("gpu", "host"):
                        _check_future_kv(
                            mode,
                            page == 64,
                            planner,
                            page,
                            direct,
                            heads,
                            ratio,
                            query_tile=32,
                        )

    def test_default_m16_anchor(self):
        _check_changed_graph(
            "persistent", False, num_kv_heads=4, head_group_ratio=6, q_len_per_req=3
        )
        _check_changed_graph("persistent", False, direct=True, capacity_length=512)


class BucketMultiQueryTests(unittest.TestCase):
    def test_causal_partition_smoke(self):
        _check_future_kv("persistent", False)

    def test_uniform_query_replay(self):
        _require_sm120()
        configs = (
            (2, 2, 4, 16),
            (3, 4, 6, 32),
            (4, 4, 4, 32),
            (5, 8, 8, 64),
            (3, 1, 6, 16),
            (5, 1, 4, 32),
        )
        for tokens, heads, ratio, page in configs:
            for mode in ("persistent", "aligned"):
                for planner in ("gpu", "host"):
                    for interleaved in (False, True):
                        with self.subTest(
                            tokens=tokens,
                            heads=heads,
                            ratio=ratio,
                            page=page,
                            mode=mode,
                            planner=planner,
                            interleaved=interleaved,
                        ):
                            _check_changed_graph(
                                mode,
                                interleaved,
                                planner=planner,
                                page_size=page,
                                num_kv_heads=heads,
                                head_group_ratio=ratio,
                                q_len_per_req=tokens,
                            )

    def test_direct_query_replay(self):
        _require_sm120()
        for tokens, heads, ratio, page in ((2, 2, 4, 16), (3, 4, 6, 32), (5, 8, 8, 64)):
            for mode in ("persistent", "aligned"):
                for planner in ("gpu", "host"):
                    for batch, capacity in ((1, 131072), (4, 512)):
                        with self.subTest(
                            tokens=tokens,
                            heads=heads,
                            ratio=ratio,
                            page=page,
                            mode=mode,
                            planner=planner,
                            batch=batch,
                        ):
                            _check_changed_graph(
                                mode,
                                True,
                                planner=planner,
                                page_size=page,
                                num_kv_heads=heads,
                                head_group_ratio=ratio,
                                q_len_per_req=tokens,
                                direct=True,
                                direct_batch=batch,
                                capacity_length=capacity,
                            )

    def test_future_kv_perturbation(self):
        for page in (16, 32, 64):
            for direct in (False, True):
                for mode in ("persistent", "aligned"):
                    for planner in ("gpu", "host"):
                        _check_future_kv(mode, page != 32, planner, page, direct)
        for mode in ("persistent", "aligned"):
            for planner in ("gpu", "host"):
                for interleaved in (False, True):
                    _check_future_kv(
                        mode,
                        interleaved,
                        planner,
                        direct=False,
                        num_kv_heads=2,
                        head_group_ratio=4,
                    )

    def test_q1_compatibility_anchor(self):
        _check_changed_graph("persistent", False, direct=True, capacity_length=512)
        _check_changed_graph("persistent", True, num_kv_heads=4, head_group_ratio=6)
        with self.subTest(direct_batch=4):
            _check_changed_graph(
                "aligned",
                False,
                num_kv_heads=2,
                head_group_ratio=4,
                direct=True,
                direct_batch=4,
                capacity_length=512,
            )


class HostPlannerTests(unittest.TestCase):
    def test_descriptor_abi_and_validation(self):
        _require_sm120()
        from flashinfer.xqa import create_xqa_decode_bucket_plan
        from flashinfer.experimental.xqa_decode_buckets.jit import get_module

        host_lengths = torch.zeros(32, dtype=torch.int32)
        lens = host_lengths.cuda()
        for mode in ("persistent", "aligned"):
            host = create_xqa_decode_bucket_plan(
                lens,
                page_table_width=4096,
                mode=mode,
                planner="host",
                host_seq_lens=host_lengths,
            )
            gpu = create_xqa_decode_bucket_plan(lens, page_table_width=4096, mode=mode)
            assert host._host_descriptors.is_pinned()
            assert host._device_descriptors.numel() == 6 * (host.ctas + 32)
            assert (
                len(
                    {
                        t.untyped_storage().data_ptr()
                        for t in (
                            host.tasks,
                            host.requests,
                            host.spans,
                            host.header,
                        )
                    }
                )
                == 1
            )
            pointers = [
                t.data_ptr()
                for t in (host.tasks, host.requests, host.spans, host.header)
            ]
            for lengths in (
                [4096] * 32,
                [131072] + [4096] * 31,
                [0, 1, 255, 256, 257, 512, 8193, 0] * 4,
                [0] * 32,
                [4096] * 32,
            ):
                host_lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
                lens.copy_(host_lengths)
                # Back-to-back refreshes exercise bounded staging reuse before
                # the first copy is necessarily complete.
                host.prepare()
                host.prepare()
                gpu.prepare()
                torch.cuda.synchronize()
                _assert_plan(host, lengths)
                slots = host.header[1].item()
                for a, b in (
                    (host.header, gpu.header),
                    (host.requests, gpu.requests),
                    (host.spans, gpu.spans),
                    (host.tasks[:slots], gpu.tasks[:slots]),
                ):
                    assert torch.equal(a, b)
                assert pointers == [
                    t.data_ptr()
                    for t in (
                        host.tasks,
                        host.requests,
                        host.spans,
                        host.header,
                    )
                ]
            module, _ = get_module(mode == "persistent", lens.device)
            for invalid in (-1, 131073):
                host_lengths[0] = invalid
                with self.assertRaises((ValueError, RuntimeError)):
                    module.plan_host(
                        host_lengths,
                        host._host_descriptors,
                        host.ctas,
                        4096,
                        mode == "aligned",
                        1,
                    )
            host_lengths.zero_()
            graph = torch.cuda.CUDAGraph()
            with (
                self.assertRaisesRegex(RuntimeError, "outside capture"),
                torch.cuda.graph(graph),
            ):
                host.prepare()
        for metadata in (
            None,
            torch.zeros(32),
            torch.zeros(64, dtype=torch.int32)[::2],
            lens,
            torch.zeros(31, dtype=torch.int32),
        ):
            with self.assertRaises((TypeError, ValueError)):
                create_xqa_decode_bucket_plan(
                    lens,
                    page_table_width=4096,
                    planner="host",
                    host_seq_lens=metadata,
                )
        with self.assertRaises(ValueError):
            create_xqa_decode_bucket_plan(
                lens, page_table_width=4096, planner="invalid"
            )
        for page_size in (True, 8, 24, 128):
            with self.assertRaises(ValueError):
                create_xqa_decode_bucket_plan(
                    lens, page_table_width=4096, page_size=page_size
                )
        for heads, ratio in ((0, 6), (True, 6), (1, 3), (1, 16)):
            with self.assertRaises(ValueError):
                create_xqa_decode_bucket_plan(
                    lens,
                    page_table_width=4096,
                    num_kv_heads=heads,
                    head_group_ratio=ratio,
                )
        for tokens in (True, 0, 6, 2.0):
            with self.assertRaises(ValueError):
                create_xqa_decode_bucket_plan(
                    lens, page_table_width=4096, q_len_per_req=tokens
                )
        with self.assertRaisesRegex(ValueError, "grid-Y"):
            create_xqa_decode_bucket_plan(
                lens, page_table_width=4096, num_kv_heads=host.grid_y_limit + 1
            )


if __name__ == "__main__":
    unittest.main()
