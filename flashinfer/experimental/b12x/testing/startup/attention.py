"""Concrete Qwen QSA and GLM pooled sparse-attention preparation requests."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace

import torch

from b12x.preparation import FrozenMapping, PreparedCall, PreparationRequest


@dataclass
class _Expected:
    kind: str
    data: dict


def _norm_producer(destination, eps):
    from vllm import _custom_ops as ops

    source = torch.randn_like(destination)
    weight = torch.ones(destination.shape[-1], dtype=destination.dtype,
                        device=destination.device)

    def produce():
        ops.rms_norm(destination.view(-1, destination.shape[-1]),
                     source.view(-1, source.shape[-1]), weight, eps)

    return produce, (source, weight)


def _intervals(rows, max_seqs, cache_tokens, device):
    batch = min(rows, max_seqs)
    counts = [rows // batch + (r < rows % batch) for r in range(batch)]
    if max(counts) > cache_tokens:
        raise ValueError("startup query interval exceeds _cache_tokens")
    starts, positions, requests = [0], [], []
    for request, count in enumerate(counts):
        starts.append(starts[-1] + count)
        positions.extend(range(cache_tokens - count, cache_tokens))
        requests.extend([request] * count)
    return (
        counts,
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(positions, dtype=torch.int64, device=device),
        torch.tensor(requests, dtype=torch.int32, device=device),
    )


def _glm_pool_views(storage, page):
    """Native packed MLA and virtual C4 selector views over one pool."""
    pool_pages = int(storage.shape[0])
    kv = storage[:, : page * 528].view(pool_pages, page, 528)
    parent_stride = storage.stride(0) // (64 * 132)
    virtual_pages = (pool_pages - 1) * parent_stride + page // 256
    index_cache = storage.as_strided(
        (virtual_pages, 64 * 132), (64 * 132, 1), storage_offset=page * 528)
    return kv, index_cache, parent_stride


def _qsa_requests(meta, device, rows):
    from b12x.attention import qsa

    tp = int(meta["_tp"])
    batch = int(meta.get("_max_seqs", 1))
    context = int(meta.get("_cache_tokens", 4096))
    spec = int(meta.get("_spec_tokens", 3))
    ratio = int(meta["indexer_compress_ratio"])
    page = int(meta.get("_page_size", 16))
    if page % ratio or context < 1:
        raise ValueError("QSA page size must be compression-aligned")
    rope = meta.get("rope_parameters", {})
    sections = tuple(rope.get("mrope_section", ())) or None
    axes = 3 if sections is not None else 1
    rotary = int(int(meta["head_dim"]) * float(rope.get("partial_rotary_factor", 1.0)))
    pages_per_request = math.ceil(context / page)
    capacity, live_pages = pages_per_request * page, batch * pages_per_request
    main_stride = 2 * page * max(1, int(meta["num_key_value_heads"]) // tp) * int(meta["head_dim"])
    compressed_stride = (page // ratio) * int(meta["indexer_head_dim"])
    main_first_page = (1 << 31) // main_stride + 1
    compressed_first_page = (1 << 31) // compressed_stride + 1
    common = dict(
        device=device, max_batch=batch, max_raw_state_slots=batch,
        max_seq_len=capacity, num_main_cache_pages=main_first_page + live_pages,
        num_compressed_cache_pages=compressed_first_page + live_pages,
        main_page_size=page, compressed_page_size=page // ratio,
        max_speculative_tokens=spec, q_heads=int(meta["num_attention_heads"]) // tp,
        kv_heads=max(1, int(meta["num_key_value_heads"]) // tp),
        head_dim=int(meta["head_dim"]), index_heads=int(meta["indexer_n_heads"]),
        index_kv_heads=int(meta.get("indexer_kv_heads", 1)),
        index_head_dim=int(meta["indexer_head_dim"]), index_rotary_dim=rotary,
        compress_ratio=ratio, budget=int(meta["indexer_budget"]), position_axes=axes,
        mrope_sections=sections, mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
        rms_norm_eps=float(meta.get("rms_norm_eps", 1e-6)),
    )
    shared = {}

    def fixture():
        if shared:
            return shared
        main = torch.empty(
            (common["num_main_cache_pages"], 2, page, common["kv_heads"], common["head_dim"]),
            dtype=torch.bfloat16, device=device)
        main[main_first_page:].normal_()
        compressed = torch.empty(
            (common["num_compressed_cache_pages"], page // ratio, common["index_head_dim"]),
            dtype=torch.bfloat16, device=device)
        compressed_initial = torch.randn(
            (live_pages, page // ratio, common["index_head_dim"]),
            dtype=torch.bfloat16, device=device)
        compressed[compressed_first_page:].copy_(compressed_initial)
        local_table = torch.arange(live_pages, dtype=torch.int32, device=device).view(
            batch, pages_per_request)
        table, compressed_table = local_table + main_first_page, local_table + compressed_first_page
        touched = []
        for request in range(batch):
            max_interval = max(
                m // min(m, batch) + (request < m % min(m, batch))
                if request < min(m, batch) else 0 for m in rows)
            touched.extend(request * (capacity // ratio) + group
                           for group in range((context - max_interval) // ratio, context // ratio))
        source_indices = torch.tensor(touched, dtype=torch.int64, device=device)
        frequencies = 1.0 / float(rope.get("rope_theta", 10000000)) ** (
            torch.arange(0, rotary, 2, dtype=torch.float32, device=device) / rotary)
        angle = torch.arange(capacity, dtype=torch.float32, device=device)[:, None] * frequencies
        shared.update(
            main=main, compressed=compressed, local_table=local_table, table=table,
            compressed_table=compressed_table, compressed_initial=compressed_initial,
            reset_indices=source_indices + compressed_first_page * (page // ratio),
            reset_values=compressed_initial.flatten(0, 1)[source_indices].clone(),
            rope_cos=angle.cos().to(torch.bfloat16), rope_sin=angle.sin().to(torch.bfloat16),
            q_weight=torch.zeros(common["index_head_dim"], dtype=torch.float32, device=device),
            k_weight=torch.zeros(common["index_head_dim"], dtype=torch.float32, device=device),
        )
        return shared
    def invocation(caps):
        main_page_stride = 2 * page * caps.kv_heads * caps.head_dim
        return qsa.invocation_from_descriptors(caps, operands={
            "request_ids": {"dtype": "int32", "strides": (1,)},
            "rope_positions": {"dtype": "int64", "strides": (axes, 1)},
            "index_query": {
                "dtype": "bfloat16",
                "strides": (caps.index_heads * caps.index_head_dim, caps.index_head_dim, 1),
            },
            "raw_index_key": {
                "dtype": "bfloat16", "strides": (caps.index_head_dim, 1),
            },
            "main_k_cache": {
                "dtype": "bfloat16",
                "strides": (main_page_stride, caps.kv_heads * caps.head_dim, caps.head_dim, 1),
            },
            "main_v_cache": {
                "dtype": "bfloat16",
                "strides": (main_page_stride, caps.kv_heads * caps.head_dim, caps.head_dim, 1),
            },
            "main_block_table": {
                "dtype": "int32", "strides": (caps.main_table_width, 1),
            },
            "compressed_k_cache": {
                "dtype": "bfloat16",
                "strides": (caps.compressed_page_size * caps.index_head_dim, caps.index_head_dim, 1),
            },
            "compressed_block_table": {
                "dtype": "int32", "strides": (caps.compressed_table_width, 1),
            },
            "raw_k_ring": {
                "dtype": "bfloat16",
                "strides": (caps.raw_ring_capacity * caps.index_head_dim, caps.index_head_dim, 1),
            },
            "raw_logical_positions": {"dtype": "int64", "strides": (caps.raw_ring_capacity, 1)},
            "raw_rope_positions": {
                "dtype": "int64", "strides": (caps.raw_ring_capacity * axes, axes, 1),
            },
            "raw_interval_start_positions": {"dtype": "int64", "strides": (1,)},
            "raw_state_slot_ids": {"dtype": "int64", "strides": (1,)},
            "index_q_norm_weight": {"dtype": "float32", "strides": (1,)},
            "index_k_norm_weight": {"dtype": "float32", "strides": (1,)},
            "rope_cos": {"dtype": "bfloat16", "strides": (rotary // 2, 1)},
            "rope_sin": {"dtype": "bfloat16", "strides": (rotary // 2, 1)},
        })


    requests = []
    for m in rows:
        caps = qsa.Caps(**common, max_q_rows=max(m, batch))
        declaration = qsa.plan(caps, invocation=invocation(caps))

        def call(state, *, m=m, caps=caps):
            base = fixture()
            counts, cu, positions, ids = _intervals(m, batch, context, device)
            q = torch.empty((m, caps.q_heads, caps.head_dim), dtype=torch.bfloat16, device=device)
            iq = torch.empty((m, caps.index_heads, caps.index_head_dim), dtype=torch.bfloat16, device=device)
            raw = torch.empty((m, caps.index_head_dim), dtype=torch.bfloat16, device=device)
            producers = [_norm_producer(value, caps.rms_norm_eps) for value in (q, iq, raw)]
            boundaries = torch.full((batch + 1,), m, dtype=torch.int32, device=device)
            boundaries[:cu.numel()].copy_(cu)
            lengths = torch.zeros(batch, dtype=torch.int32, device=device)
            lengths[:len(counts)] = context
            accepted = torch.zeros_like(lengths)
            accepted[:len(counts)] = 1
            prefilling = torch.tensor([count > 1 + spec for count in counts],
                                      dtype=torch.bool, device=device)
            if len(counts) < batch:
                prefilling = torch.cat((prefilling, torch.zeros(batch - len(counts),
                                                               dtype=torch.bool, device=device)))
            ring = torch.randn((batch, caps.raw_ring_capacity, caps.index_head_dim),
                               dtype=torch.bfloat16, device=device)
            tags = torch.full((batch, caps.raw_ring_capacity), -1, dtype=torch.int64, device=device)
            rope_tags = torch.full((*tags.shape, axes), -1, dtype=torch.int64, device=device)
            anchors = torch.full((batch,), -1, dtype=torch.int64, device=device)
            for request, count in enumerate(counts):
                first = context - count
                anchors[request] = first - 1
                for prior in range(max(0, first - caps.raw_ring_capacity), first):
                    tags[request, prior % caps.raw_ring_capacity] = prior
                    rope_tags[request, prior % caps.raw_ring_capacity] = prior
            output = torch.empty(
                (caps.max_q_rows, caps.q_heads, caps.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            selected = torch.empty(
                (caps.max_q_rows, caps.selection_width),
                dtype=torch.int32,
                device=device,
            )
            scratch = torch.empty(
                (state._layout.total_nbytes,),
                dtype=torch.uint8,
                device=device,
            )
            initial = dict(
                base=base,
                ring=ring.clone(),
                tags=tags.clone(),
                rope_tags=rope_tags.clone(),
                anchors=anchors.clone(),
            )
            binding = state.bind_for_preparation(
                scratch=scratch,
                main_k_cache=base["main"][:, 0],
                main_v_cache=base["main"][:, 1],
                main_block_table=base["table"],
                compressed_k_cache=base["compressed"],
                compressed_block_table=base["compressed_table"],
                raw_k_ring=ring,
                raw_logical_positions=tags,
                raw_rope_positions=rope_tags,
                raw_interval_start_positions=anchors,
                raw_state_slot_ids=torch.arange(batch, dtype=torch.int64, device=device),
                index_q_norm_weight=base["q_weight"],
                index_k_norm_weight=base["k_weight"],
                rope_cos=base["rope_cos"],
                rope_sin=base["rope_sin"],
                output=output,
                selected_positions=selected,
            )
            dynamic = dict(query=q, index_query=iq, raw_index_key=raw, request_ids=ids,
                           query_positions=positions,
                           rope_positions=positions[:, None].expand(m, axes).contiguous(),
                           sequence_lengths=lengths, query_start_loc=boundaries,
                           num_accepted_tokens=accepted, is_prefilling=prefilling)

            def reset():
                base["compressed"].flatten(0, 1).index_copy_(
                    0, base["reset_indices"], base["reset_values"]
                )
                ring.copy_(initial["ring"])
                tags.copy_(initial["tags"])
                rope_tags.copy_(initial["rope_tags"])
                anchors.copy_(initial["anchors"])
                scratch.zero_()
                output.zero_()
                selected.fill_(-1)

            def produce():
                reset()
                for producer, _ in producers:
                    producer()

            return PreparedCall(
                run=lambda: state.run_for_preparation(binding, **dynamic),
                produce=produce,
                reset=reset,
                restore=reset,
                owners=(binding, producers, initial,
                _Expected("qsa", dict(binding=binding, dynamic=dynamic, initial=initial))))

        name = f"attention.qsa.m{m}"
        requests.append(declaration.request(
            name=name,
            prepare_call=call, benchmark_call=call, retain_benchmark_call=True))
    return requests


def _glm_requests(meta, device, rows):
    from b12x.attention import dsa_indexer, sparse_mla
    from vllm.models.glm5next.nvidia.ops.glm_kpool import fwht128_quant_fp8

    tp, max_seqs, context, spec = (int(meta["_tp"]), int(meta.get("_max_seqs", 1)),
                                   int(meta.get("_cache_tokens", 4096)),
                                   int(meta.get("_spec_tokens", 3)))
    pool, topk, heads = (int(meta["index_kpool"]), int(meta["index_topk"]),
                         int(meta["index_n_heads"]))
    if (pool, topk, heads, int(meta["index_head_dim"])) != (4, 2048, 32, 128):
        raise ValueError("GLM_NEXT requires C4 pooled512 indexing with 32 replicated heads")
    if not meta.get("index_kpool_always_select_tail", False):
        raise ValueError("GLM_NEXT startup requires always-selected causal tail")
    page = int(meta.get("_page_size", 256))
    if page % 256:
        raise ValueError("GLM pooled cache pages must be divisible by 256")
    pages_per_request, pool_page_width = math.ceil(context / page), math.ceil(context / 256)
    pool_topk, width = topk // pool, topk + pool - 1
    qheads, latent = int(meta["num_attention_heads"]) // tp, int(meta["kv_lora_rank"])
    scale = float(int(meta["qk_nope_head_dim"]) + int(meta.get("qk_rope_head_dim", 0))) ** -0.5
    first_page = (1 << 31) // (page * 561) + 1
    live_pages, pool_pages = max_seqs * pages_per_request, first_page + max_seqs * pages_per_request
    shared = {}

    def fixture():
        if shared:
            return shared
        storage = torch.empty((pool_pages, page * 561), dtype=torch.uint8, device=device)
        kv, index_cache, parent_stride = _glm_pool_views(storage, page)
        history = torch.randn((live_pages * page, latent), dtype=torch.bfloat16, device=device)
        slots = torch.arange(first_page * page, pool_pages * page, dtype=torch.int64, device=device)
        sparse_mla.concat_and_cache_glm_next_mla(history, kv, slots)
        subpages = page // 256
        live_index = storage[first_page:, page * 528:].view(live_pages, subpages, 64 * 132)
        values = torch.randn((live_pages, subpages, 64, 128), dtype=torch.bfloat16, device=device)
        scales = values.float().abs().amax(-1).clamp_min(1e-4).div(448).log2().ceil().exp2()
        live_index[..., :64 * 128].copy_((values.float() / scales[..., None]).to(
            torch.float8_e4m3fn).view(torch.uint8).flatten(2))
        live_index[..., 64 * 128:].copy_(scales.contiguous().view(torch.uint8).flatten(2))
        table = torch.arange(first_page, pool_pages, dtype=torch.int32, device=device).view(
            max_seqs, pages_per_request)
        shared.update(storage=storage, kv=kv, index=index_cache, table=table,
                      pool_table=(table[:, :, None] * parent_stride +
                                  torch.arange(subpages, dtype=torch.int32, device=device)).flatten(1))
        return shared

    requests = []
    for m in rows:
        active_batch = min(m, max_seqs)
        mode = "decode" if m <= active_batch * (1 + spec) else "prefill"
        counts = [m // active_batch + (r < m % active_batch) for r in range(active_batch)]
        request_inputs, predecessors = {}, {}

        def inputs(*, m=m):
            if request_inputs:
                return request_inputs["value"]
            base = fixture()
            _, _, positions, ids = _intervals(m, max_seqs, context, device)
            q = torch.randn((m, heads, 128), dtype=torch.bfloat16, device=device)
            qfp8, qscale = torch.empty_like(q, dtype=torch.float8_e4m3fn), torch.empty((m, heads), dtype=torch.float32, device=device)
            base_weights, weights = (torch.randn((m, heads), dtype=torch.float32, device=device),
                                     torch.empty((m, heads), dtype=torch.float32, device=device))
            qmla = torch.empty((m, qheads, latent), dtype=torch.bfloat16, device=device)
            norm, norm_owners = _norm_producer(qmla, float(meta.get("rms_norm_eps", 1e-5)))
            result = dict(base=base, positions=positions, ids=ids, q=q, qfp8=qfp8,
                          qscale=qscale, base_weights=base_weights, weights=weights,
                          lengths=((positions + 1) // pool).to(torch.int32),
                          active_width=torch.tensor([pool_page_width * 64], dtype=torch.int32, device=device),
                          pool_ids=torch.empty((m, pool_topk), dtype=torch.int32, device=device),
                          selected=torch.empty((m, width), dtype=torch.int32, device=device),
                          selected_lengths=torch.empty(m, dtype=torch.int32, device=device),
                          qmla=qmla, norm=norm, norm_owners=norm_owners)
            def produce():
                fwht128_quant_fp8(q.view(-1, 128), qfp8.view(-1, 128), qscale.view(-1))
                torch.mul(base_weights, qscale, out=weights)
                weights.mul_((128 * heads) ** -0.5)
            result["produce"] = produce
            request_inputs["value"] = result
            return result

        slices = [(0, m)] if mode == "decode" else []
        if mode == "prefill":
            offset = 0
            for count in counts:
                slices.append((offset, offset + count))
                offset += count
        index_names = []
        for start, end in slices:
            n = end - start
            caps = dsa_indexer.Caps(device=device, num_q_heads=heads, max_q_rows=n,
                max_page_table_width=pool_page_width, topk=pool_topk, mode=mode, max_batch=n)
            # The declaration carries exactly the physical high-pid ABI used by this slice.
            def descriptor(shape, dtype, strides=None):
                return FrozenMapping({
                    "shape": tuple(shape),
                    "strides": tuple(strides or tuple(
                        math.prod(shape[index + 1:]) for index in range(len(shape)))),
                    "dtype": dtype,
                    "alignment": 16,
                })
            table_shape = (n, pool_page_width)
            invocation = dsa_indexer.invocation_from_descriptors(
                caps, operands={
                    "q_fp8": descriptor((n, heads, 128), "float8_e4m3fn"),
                    "query_weights": descriptor((n, heads), "float32"),
                    "index_k_cache": descriptor(
                        ((pool_pages - 1) * (page // 256) + 1, 64 * 132),
                        "uint8", (64 * 132, 1)),
                    "page_table": descriptor(table_shape, "int32"),
                    "cache_lengths": descriptor((n,), "int32"),
                    "active_width": descriptor((1,), "int32"),
                    "output_indices": descriptor((n, pool_topk), "int32"),
                    "output_scores": None,
                })
            name = f"attention.glm_indexer.m{m}.r{start}_{end}"
            index_names.append(name)

            def index_call(state, *, start=start, end=end, n=n, mode=mode):
                s = inputs()
                table = s["base"]["pool_table"][s["ids"][start:end].long()].contiguous()
                if mode == "prefill":
                    table = table[:1].expand(n, -1)
                binding = state.bind(
                    scratch=torch.empty(state.layout.scratch_specs()[0].shape,
                                        dtype=torch.uint8, device=device),
                    real_page_table=table,
                    cache_seqlens_int32=s["lengths"][start:end],
                    active_width=s["active_width"],
                    expected_num_q_heads=heads,
                    shared_page_table=mode == "prefill",
                    output_physical_slots=False,
                )
                expected_binding = SimpleNamespace(
                    q_fp8=s["qfp8"][start:end],
                    query_weights=s["weights"][start:end],
                    index_k_cache=s["base"]["index"],
                    output_indices=s["pool_ids"][start:end],
                )
                pool_ids_initial = s["pool_ids"][start:end].clone()

                def restore_index():
                    s["pool_ids"][start:end].copy_(pool_ids_initial)

                def run_index():
                    return state.run(
                        binding, q_fp8=s["qfp8"][start:end],
                        query_weights=s["weights"][start:end],
                        index_k_cache=s["base"]["index"],
                        output_indices=s["pool_ids"][start:end],
                    )
                predecessors[start] = run_index
                return PreparedCall(
                    run=run_index,
                    produce=s["produce"],
                    reset=restore_index,
                    restore=restore_index,
                    owners=(
                        binding,
                        s,
                        _Expected(
                            "indexer",
                            dict(binding=expected_binding, table=table,
                                 lengths=s["lengths"][start:end]),
                        ),
                    ),
                )

            requests.append(declaration.request(
                name=name,
                prepare_call=index_call, benchmark_call=index_call,
                retain_benchmark_call=True))

        mla_mode = "decode" if mode == "decode" else "extend"
        caps = sparse_mla.Caps(device=device, num_q_heads=qheads, max_q_rows=m, max_width=width,
            softmax_scale=scale, dtype=torch.bfloat16, kv_dtype=torch.uint8, head_dim=latent,
            v_head_dim=latent, model_type=int(sparse_mla.ModelType.GLM_NEXT), mode=mla_mode,
            max_batch=m, max_chunks_per_row=math.ceil(width / 64), page_size=page)
        declaration = sparse_mla.plan(caps)
        name = f"attention.glm_sparse_mla.m{m}"

        def mla_call(state, *, m=m):
            s = inputs()
            binding = state.bind(scratch=torch.empty(state.scratch_specs()[0].shape,
                dtype=torch.uint8, device=device), q=s["qmla"], kv_cache=s["base"]["kv"],
                selected_indices=s["selected"], cache_lengths=(s["positions"] + 1).to(torch.int32),
                selected_lengths=s["selected_lengths"])
            state.prime(binding, kv_cache=s["base"]["kv"])
            selected_initial = s["selected"].clone()
            selected_lengths_initial = s["selected_lengths"].clone()

            def restore_mla():
                s["selected"].copy_(selected_initial)
                s["selected_lengths"].copy_(selected_lengths_initial)

            def produce():
                s["produce"]()
                for predecessor in predecessors.values():
                    predecessor()
                s["norm"]()
                sparse_mla.expand_pooled_topk_to_physical_slots(
                    s["pool_ids"], s["positions"], s["ids"], s["base"]["table"],
                    s["selected"], s["selected_lengths"], pool_size=pool, block_size=page,
                    block_stride_rows=page, num_cache_blocks=s["base"]["kv"].shape[0])
            return PreparedCall(
                run=lambda: state.run(binding, kv_cache=s["base"]["kv"]),
                produce=produce,
                reset=restore_mla,
                restore=restore_mla,
                owners=(binding, s, _Expected("mla", dict(
                    q=s["qmla"], kv=s["base"]["kv"], first_page=first_page, page=page,
                    indices=s["selected"], counts=s["selected_lengths"], scale=scale,
                    latent=latent))),
            )

        requests.append(declaration.request(name=name,
            prepare_call=mla_call, benchmark_call=mla_call, dependencies=tuple(index_names),
            retain_benchmark_call=True))
    return requests


def make_benchmark_requests(
    metadata: Mapping, *, device: torch.device, rows: tuple[int, ...]
) -> list[PreparationRequest]:
    meta = dict(metadata.get("text_config", metadata))
    meta.update({k: v for k, v in metadata.items() if k.startswith("_")})
    if not rows or any(int(m) <= 0 for m in rows):
        raise ValueError("attention startup rows must be positive")
    if int(meta.get("_max_seqs", 1)) < 1 or int(meta.get("_spec_tokens", 3)) < 0:
        raise ValueError("invalid attention startup capacities")
    model_type = str(meta.get("model_type", ""))
    if "qwen3_8" in model_type:
        return _qsa_requests(meta, device, rows)
    if "glm5_next" in model_type:
        return _glm_requests(meta, device, rows)
    return []

def _index_membership(indices, columns):
    counts = torch.zeros(
        (indices.shape[0], columns), device=indices.device, dtype=torch.int32
    )
    counts.scatter_add_(1, indices.clamp_min(0).long(), (indices >= 0).to(torch.int32))
    return counts != 0


def test_actual(call: PreparedCall):
    """Selectors promise a set, not score-sorted integer output positions."""
    context = next(owner for owner in call.owners if isinstance(owner, _Expected))
    if context.kind == "indexer":
        return _index_membership(call.output, context.data["table"].shape[1] * 64)
    return call.output


def test_expected(call: PreparedCall):
    """Numerical references, invoked only by the post-startup correctness test."""
    context = next(owner for owner in call.owners if isinstance(owner, _Expected))
    data = context.data
    if context.kind == "qsa":
        from b12x.attention.qsa.reference import (
            gemma_rmsnorm_reference,
            packed_stream_compress_reference,
            paged_store_compressed_reference,
            score_select_reference,
            sparse_paged_gqa_reference,
        )

        binding, dynamic, initial = data["binding"], data["dynamic"], data["initial"]
        caps = binding.plan.caps
        half = caps.index_rotary_dim // 2

        def rope(value, positions):
            # The declared text workload uses identical positions on all M-RoPE
            # axes, retaining the actual three-axis tensor/compiled contract.
            position = positions[:, 0] if positions.ndim == 2 else positions
            if value.ndim == 1 and position.ndim == 1:
                position = position[0]
            cosine = binding.rope_cos[position.long()]
            sine = binding.rope_sin[position.long()]
            if value.ndim == 3:
                cosine, sine = cosine[:, None], sine[:, None]
            result = value.clone()
            left, right = value[..., :half], value[..., half : 2 * half]
            result[..., :half] = (left * cosine) - (right * sine)
            result[..., half : 2 * half] = (right * cosine) + (left * sine)
            return result

        compressed = initial["base"]["compressed_initial"].clone()
        compressed_table = initial["base"]["local_table"]
        selected = torch.full_like(
            binding.selected_positions[: dynamic["query"].shape[0]], -1
        )
        index_query = rope(
            gemma_rmsnorm_reference(
                dynamic["index_query"], binding.index_q_norm_weight, caps.rms_norm_eps
            ),
            dynamic["rope_positions"],
        )
        boundaries = dynamic["query_start_loc"].tolist()
        for request, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if start == end:
                continue
            group_ids, representatives, _ = packed_stream_compress_reference(
                dynamic["raw_index_key"][start:end],
                dynamic["query_positions"][start:end],
                dynamic["rope_positions"][start:end],
                initial["ring"][request].clone(),
                initial["tags"][request].clone(),
                initial["rope_tags"][request].clone(),
                prior_interval_start_position=int(initial["anchors"][request]),
                num_accepted_tokens=int(dynamic["num_accepted_tokens"][request]),
                is_prefilling=bool(dynamic["is_prefilling"][request]),
                compress_ratio=caps.compress_ratio,
                key_norm_weight=binding.index_k_norm_weight,
                eps=caps.rms_norm_eps,
                rope=rope,
            )
            paged_store_compressed_reference(
                compressed, compressed_table, request, group_ids, representatives
            )
            keys = compressed[compressed_table[request].long()].flatten(0, 1)
            _, selected[start:end] = score_select_reference(
                index_query[start:end],
                keys,
                dynamic["query_positions"][start:end],
                int(dynamic["sequence_lengths"][request]),
                caps.compress_ratio,
                caps.budget,
            )
        return sparse_paged_gqa_reference(
            dynamic["query"],
            binding.main_k_cache,
            binding.main_v_cache,
            binding.main_block_table,
            dynamic["request_ids"],
            selected,
            dynamic["query_positions"],
        )
    if context.kind == "indexer":
        from b12x.attention.dsa_indexer.reference import paged_decode_logits_reference

        binding = data["binding"]
        pages, inverse = torch.unique(
            data["table"].flatten(), sorted=True, return_inverse=True
        )
        live_cache = binding.index_k_cache[pages.long()].contiguous()
        compact_table = inverse.view_as(data["table"]).to(torch.int32)
        logits = paged_decode_logits_reference(
            q_fp8=binding.q_fp8,
            weights=binding.query_weights,
            index_k_cache=live_cache,
            real_page_table=compact_table,
            query_row_to_batch=torch.arange(
                binding.q_fp8.shape[0], dtype=torch.int32, device=binding.q_fp8.device
            ),
            seqlens_per_query=data["lengths"],
        )
        count = min(binding.output_indices.shape[1], logits.shape[1])
        values, indices = torch.topk(logits, count, dim=-1)
        result = torch.full_like(binding.output_indices, -1)
        result[:, :count] = torch.where(
            torch.isfinite(values), indices.to(torch.int32), -1
        )
        return _index_membership(result, logits.shape[1])
    if context.kind == "mla":
        from b12x.attention._shared.mla.reference import sparse_mla_reference

        live_cache = data["kv"][data["first_page"] :].contiguous().view(-1, 1, 528)
        indices = torch.where(
            data["indices"] >= 0,
            data["indices"] - data["first_page"] * data["page"],
            -1,
        )
        return sparse_mla_reference(
            q_all=data["q"],
            kv_cache=live_cache,
            page_table_1=indices,
            active_token_counts=data["counts"],
            sm_scale=data["scale"],
            v_head_dim=data["latent"],
        )
    raise ValueError(f"unknown attention reference {context.kind!r}")
