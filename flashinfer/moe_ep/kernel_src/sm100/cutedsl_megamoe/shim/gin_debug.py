# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostic snapshots for qualifying the legacy GIN communication protocol.

These helpers run outside timing/capture. They consume the actual BF16 FC2
terms and saved routing metadata, so they isolate communication rounding from
the independent GEMM/activation oracle. No raw kernel module is imported here.
"""

from __future__ import annotations

import math


def snapshot_nvfp4_gin_terms(buffer, *, include_dispatch=False) -> dict:
    """Copy live FC2 terms and provenance to CPU before the next launch.

    The returned rows are independent of buffer lifetime. Each key identifies
    (source global rank, source token, top-k slot, global expert). Counts and
    segment bases come from device snapshots that survive tail counter clear.
    """
    import torch

    frontend = buffer._frontend
    session = frontend._gin_session
    mega = frontend._mega
    if session is None or session._closed or mega is None or mega.compiled is None:
        raise ValueError("A completed live GIN launch is required for a snapshot.")
    torch.cuda.synchronize(session.device)
    kernel = mega.kernel

    def region(name, dtype):
        if name in kernel._local_offsets:
            root = mega.local_workspace
            offset = kernel._local_offsets[name]
            spec = kernel._local_region_by_name[name]
        else:
            root = mega.shared_workspace
            offset = kernel._shared_offsets[name]
            spec = kernel._shared_region_by_name[name]
        return root.narrow(0, offset, spec.nbytes).view(dtype).reshape(spec.shape)

    intra = region("intra_token_count_snapshot", torch.int32).cpu()
    base = region("expert_pool_base_snapshot", torch.int32).cpu()
    relay = region("internode_recv_count_per_source_snapshot", torch.int32).cpu()
    relay_base = region("relay_pool_base_snapshot", torch.int32).cpu()
    metadata = region("token_src_metadata", torch.uint8).cpu()
    terms = region("fc2_output_workspace", torch.bfloat16)
    keys, rows, sf_rows = [], [], []
    if include_dispatch:
        sf_base = region("expert_sf_pool_base_snapshot", torch.int32).cpu()
        relay_sf_base = region("relay_sf_pool_base_snapshot", torch.int32).cpu()
    config = frontend.config
    for expert in range(config.num_experts_per_rank):
        segments = [(int(base[expert]), int(intra[expert]))]
        segments += [
            (int(relay_base[expert, slot]), int(relay[slot, expert]))
            for slot in range(relay.shape[0])
        ]
        for segment, (start, count) in enumerate(segments):
            for row in range(start, start + count):
                packed = int.from_bytes(bytes(metadata[row].tolist()), "little")
                token = packed & 0xFFFFFFFF
                topk = (packed >> 32) & 0xFFFF
                rank = (packed >> 48) & 0xFFFF
                if not (
                    rank < config.world_size
                    and token < config.num_tokens_per_rank
                    and topk < config.num_topk
                ):
                    raise AssertionError(
                        "Invalid GIN source metadata in a live pool row."
                    )
                keys.append(
                    (
                        rank,
                        token,
                        topk,
                        config.rank * config.num_experts_per_rank + expert,
                    )
                )
                rows.append(row)
                if include_dispatch:
                    sf_start = (
                        sf_base[expert]
                        if segment == 0
                        else relay_sf_base[expert, segment - 1]
                    )
                    sf_rows.append(int(sf_start) + row - start)
    selected = terms[
        torch.tensor(rows, dtype=torch.int64, device=terms.device), 0
    ].cpu()
    result = dict(keys=keys, terms=selected)
    if include_dispatch:
        row_ids = torch.tensor(rows, dtype=torch.int64)
        sf_axis = torch.tensor(sf_rows, dtype=torch.int64)[:, None]
        sf_words = kernel.sf_uint32_per_token
        word_ids = torch.arange(sf_words)[None, :]
        # Replica of the physical 128-token SF atom, independent of DSL code.
        positions = (
            ((sf_axis // 128) * sf_words + word_ids) * 128
            + (sf_axis % 32) * 4
            + (sf_axis % 128) // 32
        )
        wire = frontend._gin_wire_layout
        messages = (
            session.root.narrow(
                0,
                wire.relay_recv_buffer_offset,
                math.prod(wire.relay_recv_buffer_shape),
            )
            .reshape(wire.relay_recv_buffer_shape)
            .cpu()
        )
        result.update(
            l1_data=region("l1_token_buffer", torch.uint8).cpu()[row_ids],
            l1_sf=region("l1_sf_buffer", torch.int32)
            .cpu()
            .flatten()[positions]
            .contiguous()
            .view(torch.uint8),
            l1_weights=region("l1_topk_weights_buffer", torch.float32)
            .cpu()
            .flatten()[row_ids],
            input_data=buffer.x.view(torch.uint8).cpu(),
            input_sf=buffer.x_sf.view(torch.uint8).cpu(),
            input_weights=buffer.topk_weights.cpu(),
            wire_messages=messages,
            wire_splits=session.views["internode_recv_splits"].cpu(),
            wire_layout=wire.relay_message,
        )
    return result


def ordered_nvfp4_gin_reference(shards: list[dict], routes, *, ranks_per_node: int):
    """Apply the protocol's exact FP32/BF16 tree to fixed BF16 FC2 terms.

    ``routes`` is CPU int64 (world, capacity, topk). Invalid negative IDs have
    no term. Every valid route must occur exactly once in the expert shards.
    Local BF16 terms are added in top-k order. Each remote node sums its hits
    in FP32, rounds once to BF16, and contributes once in node-index order.
    """
    import torch

    world, capacity, topk = routes.shape
    experts_per_rank = 12  # First qualification case, not a general limit.
    hidden = shards[0]["terms"].shape[-1]
    terms = {}
    for shard in shards:
        for key, value in zip(shard["keys"], shard["terms"], strict=True):
            rank, token, slot, expert = key
            if int(routes[rank, token, slot]) != expert:
                raise AssertionError(f"GIN route provenance mismatch for {key}.")
            if key in terms:
                raise AssertionError(f"Duplicate GIN expert term for {key}.")
            terms[key] = value
    output = torch.zeros((world, capacity, hidden), dtype=torch.bfloat16)
    expected_terms = 0
    for rank in range(world):
        origin = rank // ranks_per_node
        for token in range(capacity):
            nodes: list[list[torch.Tensor]] = [
                [] for _ in range(world // ranks_per_node)
            ]
            for slot in range(topk):
                expert = int(routes[rank, token, slot])
                if expert < 0:
                    continue
                key = (rank, token, slot, expert)
                if key not in terms:
                    raise AssertionError(f"Missing GIN expert term for {key}.")
                nodes[expert // experts_per_rank // ranks_per_node].append(terms[key])
                expected_terms += 1
            acc = torch.zeros(hidden, dtype=torch.float32)
            for term in nodes[origin]:
                acc += term.float()
            for node, group in enumerate(nodes):
                if node == origin or not group:
                    continue
                partial = torch.zeros(hidden, dtype=torch.float32)
                for term in group:
                    partial += term.float()
                acc += partial.to(torch.bfloat16).float()
            output[rank, token] = acc.to(torch.bfloat16)
    if len(terms) != expected_terms:
        raise AssertionError("GIN snapshots contain unexpected live terms.")
    return output
