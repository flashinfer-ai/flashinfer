"""Fixed-term protocol and independent Torch oracles for real N2R2 GIN."""

import struct

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
    ordered_nvfp4_gin_reference,
    snapshot_nvfp4_gin_terms,
    to_blocked,
)

from .test_nvfp4_cutedsl_gin_multinode import _bootstrap, _inputs, _layer
from .test_moe_ep_nvfp4_cutedsl_mega_multirank import _all_gather_stack
from .test_nvfp4_cutedsl_kernel_vs_reference import (
    _plain_nvfp4_from_bf16,
    _torch_nvfp4_mega_reference,
)

pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_blackwell, pytest.mark.node_exclusive]


def _gather(tensor):
    return _all_gather_stack(tensor).flatten(0, 1)


def test_ordered_gin_reference_rounds_remote_node_once():
    routes = torch.full((4, 1, 3), -1, dtype=torch.int64)
    routes[0, 0] = torch.tensor([0, 24, 25])
    shards = [
        dict(keys=[], terms=torch.empty((0, 2), dtype=torch.bfloat16)) for _ in range(4)
    ]
    shards[0] = dict(
        keys=[(0, 0, 0, 0)], terms=torch.tensor([[-1.0, -1.0]], dtype=torch.bfloat16)
    )
    shards[2] = dict(
        keys=[(0, 0, 1, 24), (0, 0, 2, 25)],
        terms=torch.tensor([[1.0, 1.0], [1 / 256, 1 / 256]], dtype=torch.bfloat16),
    )
    result = ordered_nvfp4_gin_reference(shards, routes, ranks_per_node=2)
    assert torch.equal(result, torch.zeros_like(result))
    # A flat FP32 sum gives 1/256, proving that the oracle models the node cast.
    assert (-1.0 + 1.0 + 1 / 256) != 0
    shards[2] = dict(keys=[(0, 0, 1, 24)], terms=shards[2]["terms"][:1])
    with pytest.raises(AssertionError, match="Missing GIN expert term"):
        ordered_nvfp4_gin_reference(shards, routes, ranks_per_node=2)


@pytest.mark.parametrize("case", ["matrix", "balanced"])
def test_nvfp4_gin_protocol_and_torch_oracles(case):
    bootstrap = _bootstrap()
    layer = _layer(bootstrap, retain_weights=True)
    failed = False
    try:
        # First forward performs the formal backend's lazy initialization.
        first = _inputs(bootstrap.rank, 0, case)
        layer.forward(first)
        torch.cuda.synchronize()
        weights = layer._gin_oracle_weights
        problem = dict(w13=weights.w13, w2=weights.w2, intermediate=1536)
        fc1, sf1, fc2, sf2 = _plain_nvfp4_from_bf16(problem)
        transformed = layer._transformed
        # Confirm the reference quantized exactly the weights that FI launched.
        assert torch.equal(transformed[0][0].transpose(1, 2).view(torch.uint8), fc1)
        assert torch.equal(transformed[1][0].transpose(1, 2).view(torch.uint8), fc2)
        for actual, plain in ((transformed[0][1], sf1), (transformed[1][1], sf2)):
            expected = torch.stack(
                [to_blocked(plain[e]).view(torch.uint8).reshape(-1) for e in range(12)]
            )
            assert torch.equal(actual.view(torch.uint8), expected)
        global_fc1, global_sf1, global_fc2, global_sf2 = (
            _gather(value) for value in (fc1, sf1, fc2, sf2)
        )
        del problem, weights, fc1, sf1, fc2, sf2
        layer._gin_oracle_weights = None

        for iteration in range(12):
            inputs = _inputs(bootstrap.rank, iteration, case)
            output = layer.forward(inputs).clone()
            torch.cuda.synchronize()
            workspace = layer._workspace
            shard = snapshot_nvfp4_gin_terms(workspace, include_dispatch=True)
            gathered = [None] * 4
            dist.all_gather_object(gathered, (shard, workspace.topk_idx.cpu()))
            routes = torch.stack([item[1] for item in gathered])
            _validate_dispatch(gathered, bootstrap.rank)
            reference = ordered_nvfp4_gin_reference(
                [item[0] for item in gathered], routes, ranks_per_node=2
            )
            n = inputs.hidden_states.shape[0]
            actual = output.cpu()
            expected = reference[bootstrap.rank, :n]
            mismatch = (
                (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
            )
            maximum = (
                (actual.float() - expected.float()).abs().max().item() if n else 0.0
            )
            print(
                f"[rank {bootstrap.rank}] protocol iteration={iteration} max_diff={maximum} mismatches={mismatch}",
                flush=True,
            )
            assert mismatch == 0

            # Use the unchanged FI independent-GEMM tolerance, separately from
            # the exact protocol assertion above. No widening for GIN.
            if n:
                math_reference = _torch_nvfp4_mega_reference(
                    act_packed=workspace.x[:n],
                    act_sf=workspace.x_sf[:n],
                    topk_idx=workspace.topk_idx[:n],
                    topk_weights=workspace.topk_weights[:n],
                    fc1_weight=global_fc1,
                    fc1_sf=global_sf1,
                    fc2_weight=global_fc2,
                    fc2_sf=global_sf2,
                    hidden=7168,
                    intermediate=1536,
                    gate_up_clamp=None,
                )
                actual_float, expected_float = output.float(), math_reference.float()
                atol = 2e-3 * expected_float.abs().max().item()
                torch.testing.assert_close(
                    actual_float, expected_float, atol=atol, rtol=0.05
                )
                relative = (
                    actual_float - expected_float
                ).norm() / expected_float.norm().clamp_min(1e-6)
                assert relative.item() < 0.02
                print(
                    f"[rank {bootstrap.rank}] torch iteration={iteration} rel_l2={relative.item()}",
                    flush=True,
                )
            dist.barrier()
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            layer.destroy()


def _validate_dispatch(gathered, rank):
    """Independent byte oracle for transport messages and physical L1 pools."""
    shards = [item[0] for item in gathered]
    routes = [item[1] for item in gathered]
    shard = shards[rank]
    for row, (source, token, topk, expert) in enumerate(shard["keys"]):
        assert int(routes[source][token, topk]) == expert
        assert torch.equal(
            shard["l1_data"][row].flatten(),
            shards[source]["input_data"][token].flatten(),
        )
        assert torch.equal(
            shard["l1_sf"][row].flatten(), shards[source]["input_sf"][token].flatten()
        )
        assert struct.pack("<f", shard["l1_weights"][row].item()) == struct.pack(
            "<f", shards[source]["input_weights"][token, topk].item()
        )

    # N2R2 has one remote slot and one rail peer of equal local rank.
    node = rank // 2
    source = (1 - node) * 2 + rank % 2
    expected_tokens = [
        token
        for token, ids in enumerate(routes[source])
        if any(int(expert) >= 0 and int(expert) // 24 == node for expert in ids)
    ]
    splits = shard["wire_splits"][0]
    assert int(splits[24]) == len(expected_tokens)
    for node_expert in range(24):
        assert int(splits[node_expert]) == int(
            (routes[source] == node * 24 + node_expert).sum()
        )
    actual_by_token = {}
    for message in shard["wire_messages"][0, : len(expected_tokens)]:
        payload = bytes(message.tolist())
        packed = int.from_bytes(payload[:8], "little")
        assert packed >> 32 == (source << 16) | 0xFFFF
        token = packed & 0xFFFFFFFF
        assert token not in actual_by_token
        actual_by_token[token] = payload
    assert set(actual_by_token) == set(expected_tokens)
    layout = shard["wire_layout"]
    for token in expected_tokens:
        expected = bytearray(layout.message_bytes)
        struct.pack_into(
            "<Q",
            expected,
            layout.src_metadata_offset,
            ((source << 16 | 0xFFFF) << 32) | token,
        )
        for topk, expert in enumerate(routes[source][token].tolist()):
            valid = expert >= 0 and expert // 24 == node
            route_word = ((expert % 24) | (topk << 16)) if valid else 0xFFFF
            struct.pack_into(
                "<I", expected, layout.route_words_offset + topk * 4, route_word
            )
            weight = (
                shards[source]["input_weights"][token, topk].item() if valid else 0.0
            )
            struct.pack_into(
                "<f", expected, layout.topk_weights_offset + topk * 4, weight
            )
        data = bytes(shards[source]["input_data"][token].flatten().tolist())
        sf = bytes(shards[source]["input_sf"][token].flatten().tolist())
        expected[layout.token_data_offset : layout.token_data_offset + len(data)] = data
        expected[layout.sf_data_offset : layout.sf_data_offset + len(sf)] = sf
        assert actual_by_token[token] == bytes(expected)
    print(
        f"[rank {rank}] dispatch bytes PASS messages={len(expected_tokens)} hits={len(shard['keys'])}",
        flush=True,
    )
