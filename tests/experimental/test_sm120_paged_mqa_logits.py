"""SM120a paged FP8 MQA lightning-indexer logits (DeepGEMM signatures).

Covers the exported ``(heads, page_kv, next_n)`` programs against a dequantized
PyTorch reference and an independent Python mirror of the schedule, the
``clean_logits=False`` write extent, CUDA Graph replay with changed contents,
one-shot/plan equivalence, the padded block-stride cache view, and the
host-only route/target helpers.

The GPU tests skip while the installed catalog carries no route for the shape
or the device is not an exported architecture; the host-only helpers and the
architecture-target gate run everywhere.
"""

import pytest
import torch
from flashinfer.experimental.deepgemm_sm120_paged_mqa_logits import (
    sm120_paged_mqa as _runtime,
)
from flashinfer.sm120_paged_mqa_logits import (
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    prepare_sm120_paged_mqa_logits,
)

HEAD_DIM = 128
FUSED_ROW_BYTES = HEAD_DIM + 4
ATOL, RTOL = 0.1, 0.1  # FP8 e4m3
# (heads, page_kv, next_n, batch, average context)
CASES = [
    (32, 128, 1, 4, 1024),
    (32, 128, 4, 3, 1024),
    (32, 64, 2, 8, 2048),
    (64, 64, 1, 8, 2048),
    (64, 64, 4, 2, 512),
]


def _skip_unless_route(heads, page_kv, next_n):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    if not _runtime.route_available(heads, page_kv, next_n):
        pytest.skip(
            f"catalog has no route for H={heads}, page {page_kv}, next_n {next_n}"
        )


def _inputs(heads, page_kv, next_n, batch, avg_ctx, *, pad=0, seed=11):
    """Decode operands in the DeepGEMM/vLLM fused-page contract."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    lo = max(page_kv, int(0.7 * avg_ctx))
    hi = max(page_kv + 1, int(1.3 * avg_ctx))
    ctx_last = torch.randint(
        lo, hi, (batch,), device=device, dtype=torch.int32, generator=generator
    )
    offsets = (next_n - 1 - torch.arange(next_n, device=device, dtype=torch.int32))[
        None, :
    ]
    context_lens = (ctx_last[:, None] - offsets).clamp_min(1).contiguous()
    max_context_len = int(ctx_last.max())
    blocks = (ctx_last + page_kv - 1) // page_kv
    width = int(blocks.max())
    pages = int(blocks.sum()) + batch
    perm = torch.randperm(pages, device=device, generator=generator).to(torch.int32)
    block_table = torch.zeros(batch, width, device=device, dtype=torch.int32)
    offset = 0
    for b in range(batch):
        n = int(blocks[b])
        block_table[b, :n] = perm[offset : offset + n]
        offset += n
    q = torch.randn(
        batch, next_n, heads, HEAD_DIM, device=device, generator=generator
    ).to(torch.float8_e4m3fn)
    kv = torch.randn(pages, page_kv, HEAD_DIM, device=device, generator=generator)
    scale = (kv.abs().amax(dim=-1).clamp_min(1e-4) / 448.0).contiguous()
    kv_fp8 = (kv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    row_bytes = page_kv * FUSED_ROW_BYTES + pad
    fused = torch.zeros(pages, row_bytes, device=device, dtype=torch.uint8)
    fused[:, : page_kv * HEAD_DIM] = kv_fp8.reshape(pages, -1).view(torch.uint8)
    fused[:, page_kv * HEAD_DIM : page_kv * FUSED_ROW_BYTES] = scale.view(torch.uint8)
    kv_cache = fused if pad else fused.view(pages, page_kv, 1, FUSED_ROW_BYTES)
    weights = (
        torch.rand(batch * next_n, heads, device=device, generator=generator) + 0.1
    )
    return dict(
        q=q,
        kv_cache=kv_cache,
        kv_fp8=kv_fp8,
        scale=scale,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        max_context_len=max_context_len,
        page_kv=page_kv,
    )


def _reference(data):
    """logits[b * next_n + t, k] = sum_h relu(q . kv[k]) * w[h] * scale[k]."""
    q, kv_fp8, scale = data["q"], data["kv_fp8"], data["scale"]
    weights, context_lens = data["weights"], data["context_lens"]
    block_table, page_kv = data["block_table"], data["page_kv"]
    max_context_len = data["max_context_len"]
    batch, next_n, _heads, _dim = q.shape
    out = torch.full(
        (batch * next_n, max_context_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for b in range(batch):
        ctx_last = int(context_lens[b, -1])
        n_blocks = (ctx_last + page_kv - 1) // page_kv
        phys = block_table[b, :n_blocks].long()
        keys = kv_fp8[phys].float().reshape(-1, HEAD_DIM)[:ctx_last]
        key_scale = scale[phys].reshape(-1)[:ctx_last]
        for t in range(next_n):
            scores = torch.relu(q[b, t].float() @ keys.T)
            row = (scores * weights[b * next_n + t][:, None]).sum(0) * key_scale
            ctx_t = int(context_lens[b, t])
            out[b * next_n + t, :ctx_t] = row[:ctx_t]
    return out


def _schedule_reference(context_lens, split_kv, atoms, num_sms):
    """Independent mirror of the SM120 scheduler (non-varlen branch)."""
    rows = context_lens.cpu().tolist()
    batch = len(rows)
    prefix, total_segs = [], 0
    for row in rows:
        total_segs += (row[-1] + split_kv - 1) // split_kv
        prefix.append(total_segs)
    total = total_segs * atoms
    if total == 0:
        return [[batch * atoms, 0] for _ in range(num_sms + 1)]
    per_sm, remainder = divmod(total, num_sms)
    pivot = num_sms - remainder
    meta = []
    for sm in range(num_sms):
        start = sm * per_sm + (sm - pivot if sm > pivot else 0)
        low, high = 0, batch
        while low < high:
            mid = (low + high) // 2
            if prefix[mid] * atoms <= start:
                low = mid + 1
            else:
                high = mid
        index = min(low, batch - 1)
        before = prefix[index - 1] if index > 0 else 0
        offset = start - before * atoms
        segments = prefix[index] - before
        atom = offset // segments if segments > 0 else 0
        split = offset % segments if segments > 0 else 0
        meta.append([index * atoms + atom, split])
    meta.append([batch * atoms, 0])
    return meta


def _written_extent(context_lens, row, columns):
    """Columns the logits program touches for one output row.

    The schedule issues ``ceil(context_lens[b, -1] / split_kv)`` segments of
    ``split_kv`` columns per request and nothing writes past them
    (``clean_logits=False``).
    """
    split_kv = _runtime.split_kv()
    next_n = int(context_lens.shape[1])
    ctx_last = int(context_lens[row // next_n, -1])
    return min((ctx_last + split_kv - 1) // split_kv * split_kv, columns)


def _check(logical, reference, data, physical):
    lengths = data["context_lens"].reshape(-1)
    position = torch.arange(data["max_context_len"], device=logical.device)[None, :]
    inside = position < lengths[:, None]
    assert torch.isfinite(logical[inside]).all()
    # FP8 e4m3 tolerance, the same rule the producer's validator applies.
    error = (logical - reference).abs().masked_fill(~inside, 0.0)
    tolerance = ATOL + RTOL * reference.abs().masked_fill(~inside, 0.0)
    assert bool((error <= tolerance).all())
    # Nothing is written past the schedule's per-request extent.
    for row in range(physical.shape[0]):
        extent = _written_extent(data["context_lens"], row, physical.shape[1])
        tail = physical[row, extent:]
        assert bool(torch.isnan(tail).all()), row


@pytest.mark.parametrize("heads,page_kv,next_n,batch,avg_ctx", CASES)
def test_sm120_paged_mqa_logits(heads, page_kv, next_n, batch, avg_ctx):
    _skip_unless_route(heads, page_kv, next_n)
    data = _inputs(heads, page_kv, next_n, batch, avg_ctx)
    plan = prepare_sm120_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
    )
    assert plan.route_name == _runtime.route_name(heads, page_kv, next_n)
    assert plan.launch_count == 1 and plan.kernel_launches == 2
    assert plan.program_names == [plan.route["sequence"]]
    plan.output.fill_(float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        data["weights"].mul_(0.5)  # a device-side dependency must reach the launch
        plan.run()
    stream.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    # The sequence rebuilt the schedule; it matches the independent mirror.
    atoms = _runtime.next_n_atoms(next_n)
    expected = _schedule_reference(
        data["context_lens"], _runtime.split_kv(), atoms, plan.num_sms
    )
    assert plan.schedule_meta.cpu().tolist() == expected
    # The standalone metadata entry produces the same schedule.
    meta = get_paged_mqa_logits_metadata(data["context_lens"], page_kv, plan.num_sms)
    assert meta.dtype == torch.int32
    assert tuple(meta.shape) == _runtime.metadata_shape(plan.num_sms)
    assert torch.equal(meta, plan.schedule_meta)
    # One-shot entry on the same operands: identical bits where both wrote.
    one_shot = fp8_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        meta,
        data["max_context_len"],
    )
    written = ~torch.isnan(plan.logical_output)
    assert torch.equal(one_shot[written], plan.logical_output[written])


@pytest.mark.parametrize("heads,page_kv,next_n,batch,avg_ctx", CASES[:1])
def test_sm120_paged_graph_replay(heads, page_kv, next_n, batch, avg_ctx):
    """Replay reproduces the eager bits and follows changed tensor contents."""
    _skip_unless_route(heads, page_kv, next_n)
    data = _inputs(heads, page_kv, next_n, batch, avg_ctx, seed=23)
    plan = prepare_sm120_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
    )
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    eager = plan.logical_output.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    with torch.cuda.stream(stream):
        plan.output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    inside = (
        torch.arange(data["max_context_len"], device="cuda")[None, :]
        < data["context_lens"].reshape(-1)[:, None]
    )
    assert torch.equal(plan.logical_output[inside], eager[inside])
    # Changed inputs, including the context lengths the scheduler reads.
    with torch.cuda.stream(stream):
        data["weights"].neg_()
        data["context_lens"].copy_((data["context_lens"] - 3).clamp_min(1))
        plan.output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    assert not torch.equal(plan.logical_output[inside], eager[inside])


def test_sm120_paged_padded_block_stride():
    """A page row padded past ``page_kv * 132`` bytes is accepted as a 2-D view."""
    heads, page_kv, next_n = 32, 128, 1
    _skip_unless_route(heads, page_kv, next_n)
    data = _inputs(heads, page_kv, next_n, 4, 1024, pad=512, seed=31)
    assert data["kv_cache"].ndim == 2
    plan = prepare_sm120_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
        page_kv=page_kv,
    )
    assert plan.block_stride_bytes == page_kv * FUSED_ROW_BYTES + 512
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    with pytest.raises(ValueError):
        # The padded view cannot be routed without an explicit page size.
        prepare_sm120_paged_mqa_logits(
            data["q"],
            data["kv_cache"],
            data["weights"],
            data["context_lens"],
            data["block_table"],
            data["max_context_len"],
        )


def test_sm120_paged_strided_per_layer_view():
    """The per-layer view of a block-outermost engine layout (every layer's page
    in one block: ``stride(0) > page_kv * 132``, not contiguous) is accepted as
    the 4-D cache and scores bitwise what the dense layout scores."""
    heads, page_kv, next_n = 32, 128, 1
    _skip_unless_route(heads, page_kv, next_n)
    data = _inputs(heads, page_kv, next_n, 4, 1024, seed=37)
    dense = data["kv_cache"]
    assert dense.ndim == 4 and dense.is_contiguous()
    pages = int(dense.shape[0])
    layer_bytes = page_kv * FUSED_ROW_BYTES
    # Two layers per block; the indexer's pages sit in the second layer slot.
    block = torch.zeros(pages, 2 * layer_bytes, device=dense.device, dtype=torch.uint8)
    block[:, layer_bytes:] = dense.reshape(pages, layer_bytes)
    strided = torch.as_strided(
        block,
        (pages, page_kv, 1, FUSED_ROW_BYTES),
        (2 * layer_bytes, FUSED_ROW_BYTES, FUSED_ROW_BYTES, 1),
        storage_offset=layer_bytes,
    )
    assert not strided.is_contiguous() and strided.stride(0) == 2 * layer_bytes
    args = (
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
    )
    plan_dense = prepare_sm120_paged_mqa_logits(data["q"], dense, *args)
    plan_dense.output.fill_(float("nan"))
    plan_dense.run()
    plan = prepare_sm120_paged_mqa_logits(data["q"], strided, *args)
    assert plan.page_kv == page_kv and plan.block_stride_bytes == 2 * layer_bytes
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    lengths = data["context_lens"].reshape(-1)
    position = torch.arange(data["max_context_len"], device=dense.device)[None, :]
    inside = position < lengths[:, None]
    assert torch.equal(plan.logical_output[inside], plan_dense.logical_output[inside])
    with pytest.raises(ValueError):
        # Token rows must stay dense: a view that skips every other row is refused.
        prepare_sm120_paged_mqa_logits(
            data["q"],
            torch.as_strided(
                block,
                (pages, page_kv // 2, 1, FUSED_ROW_BYTES),
                (2 * layer_bytes, 2 * FUSED_ROW_BYTES, FUSED_ROW_BYTES, 1),
                storage_offset=layer_bytes,
            ),
            *args,
        )


def test_sm120_paged_host_helpers_and_rejections():
    catalog = _runtime._catalog()
    assert catalog["schema"] == _runtime.CATALOG_SCHEMA
    assert _runtime.route_name(64, 64, 2) == "sm120:fp8:h64:p64:n2"
    assert _runtime.metadata_shape(84) == (85, 2)
    assert _runtime.split_kv() == 128
    stride = _runtime.paged_logits_stride(1000)
    assert stride % 256 == 0 and stride >= 1000
    assert not _runtime.route_available(16, 64, 1)
    assert not _runtime.route_available(64, 128, 1)
    assert not _runtime.route_available(32, 64, 3)
    with pytest.raises(ValueError):
        _runtime.next_n_atoms(3)
    # Every logits route is the ordered (metadata, logits) pair behind one
    # prepared sequence; the metadata route is the standalone scheduler.
    metadata_route = _runtime.metadata_route_name()
    for name, record in catalog["routes"].items():
        stages = [stage for stage, _program in record["stages"]]
        if name == metadata_route:
            assert stages == ["metadata"] and not record["sequence"]
        else:
            assert stages == ["metadata", "logits"] and record["sequence"]
            assert record["kernel_launches"] == 2
    # One Q atom per request for every shipped next_n: 1 and 2 pair the tokens as
    # DeepGEMM does; 4 scores the whole request from one atom (DeepGEMM: two
    # 2-token atoms), so its schedule has half as many items as DeepGEMM's.
    for next_n in _runtime.exported_next_n():
        assert _runtime.next_n_atoms(next_n) == 1


def test_sm120_arch_targets_follow_the_compilation_context(monkeypatch):
    """Targets come from FLASHINFER_CUDA_ARCH_LIST, never a toolkit probe."""
    from flashinfer.jit.core import sm120a_nvcc_flags

    assert "-gencode=arch=compute_120a,code=sm_120a" in sm120a_nvcc_flags
    assert _runtime._nvcc_flags("sm_120a") == sm120a_nvcc_flags
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
    assert _runtime.supported_capabilities() == ((12, 0),)
    assert _runtime.require_supported_capability((12, 0)) == (12, 0)
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "12.0a")
    assert _runtime.supported_capabilities() == ((12, 0),)
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.0a 10.3a")
    assert _runtime.supported_capabilities() == ()
    with pytest.raises(RuntimeError, match="not a build target"):
        _runtime.require_supported_capability((12, 0))
    with pytest.raises(RuntimeError, match="requires compute capability"):
        _runtime.require_supported_capability((10, 0))


def test_sm120_bindings_cover_every_program_argument():
    """Every argument of every program's launcher contract is bound by name.

    Contract-level check that fails before a JIT build or a launch would: the
    sequence binding addresses its stages as ``"<stage>.<name>"`` and the
    standalone scheduler by bare argument name.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    catalog = _runtime._catalog()
    device = torch.device("cuda")
    num_sms, batch, pages, max_context_len = 4, 2, 4, 128
    checked = 0
    for name, record in catalog["routes"].items():
        if name == _runtime.metadata_route_name():
            heads, page_kv, next_n = 32, 64, 1
        else:
            heads = int(record["num_heads"])
            page_kv = int(record["page_kv"])
            next_n = int(record["next_n"])
        context_lens = torch.full(
            (batch, next_n), max_context_len, dtype=torch.int32, device=device
        )
        schedule_meta = torch.zeros(
            _runtime.metadata_shape(num_sms), dtype=torch.int32, device=device
        )
        bindings = {
            "metadata": _runtime.metadata_bindings(
                context_lens, schedule_meta, num_sms=num_sms
            )
        }
        if name != _runtime.metadata_route_name():
            q = torch.zeros(
                (batch, next_n, heads, HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            fused = torch.zeros(
                (pages, page_kv * FUSED_ROW_BYTES), dtype=torch.uint8, device=device
            )
            weights = torch.zeros(
                (batch * next_n, heads), dtype=torch.float32, device=device
            )
            block_table = torch.zeros((batch, pages), dtype=torch.int32, device=device)
            output = torch.zeros(
                (batch * next_n, _runtime.paged_logits_stride(max_context_len)),
                dtype=torch.float32,
                device=device,
            )
            bindings["logits"] = _runtime.logits_bindings(
                q,
                fused,
                weights,
                context_lens,
                block_table,
                schedule_meta,
                output,
                num_sms=num_sms,
            )
        sequence = record.get("sequence")
        if sequence:
            plan = catalog["programs"][sequence]["arg_plan"]
            missing = [
                key
                for _kind, key in plan
                if key.split(".", 1)[1] not in bindings[key.split(".", 1)[0]]
            ]
        else:
            stage, program = record["stages"][0]
            plan = catalog["programs"][program]["arg_plan"]
            missing = [key for _kind, key in plan if key not in bindings[stage]]
        assert not missing, f"{name}: unbound arguments {missing}"
        checked += 1
    assert checked == len(catalog["routes"])
