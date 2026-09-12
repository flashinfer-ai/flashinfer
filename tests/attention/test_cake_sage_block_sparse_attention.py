# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness coverage for the generated SM120 Sage block-sparse backend."""

import json

import pytest
import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
    bsa_attn_sm120_blk64_sage_fwd,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or tuple(int(value) for value in torch.cuda.get_device_capability()) != (12, 0),
    reason="generated Sage block-sparse attention requires compute capability 12.0",
)

_BLOCK = 64
_HEAD_DIM = 128
_WORKSPACE_BYTES = 512
_V_PERM = (0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15)
_V_INVERSE_PERM = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)
_CASES = json.loads(r"""[
    {
        "name": "ut_aligned_full_b1_h2_s128",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 2,
            "position_sensitive": false,
            "seed": 4201,
            "selected_blocks": 2,
            "seqlen_k": 128,
            "seqlen_q": 128,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_aligned_half_b1_h8_s512",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 8,
            "position_sensitive": false,
            "seed": 4202,
            "selected_blocks": 4,
            "seqlen_k": 512,
            "seqlen_q": 512,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_aligned_sparse_b2_h4_s256",
        "params": {
            "batch_size": 2,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 4,
            "position_sensitive": false,
            "seed": 4203,
            "selected_blocks": 1,
            "seqlen_k": 256,
            "seqlen_q": 256,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_aligned_dense_b2_h8_s1024",
        "params": {
            "batch_size": 2,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 8,
            "position_sensitive": false,
            "seed": 4204,
            "selected_blocks": 12,
            "seqlen_k": 1024,
            "seqlen_q": 1024,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_custom_softmax_scale",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 4,
            "position_sensitive": false,
            "seed": 4001,
            "selected_blocks": 2,
            "seqlen_k": 256,
            "seqlen_q": 256,
            "softmax_scale": 0.5
        }
    },
    {
        "name": "ut_ragged_q",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 4,
            "position_sensitive": false,
            "seed": 1301,
            "selected_blocks": 2,
            "seqlen_k": 128,
            "seqlen_q": 100,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_ragged_k",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 4,
            "position_sensitive": false,
            "seed": 1302,
            "selected_blocks": 2,
            "seqlen_k": 100,
            "seqlen_q": 128,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_ragged_qk",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 4,
            "position_sensitive": false,
            "seed": 1303,
            "selected_blocks": 2,
            "seqlen_k": 100,
            "seqlen_q": 100,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_noncontiguous_topk_ragged_k",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "noncontiguous_topk": true,
            "num_heads": 4,
            "position_sensitive": false,
            "seed": 4951,
            "selected_blocks": 16,
            "seqlen_k": 4000,
            "seqlen_q": 4000,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_noncontiguous_topk_ragged_k_unaligned_tail",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "noncontiguous_topk": true,
            "num_heads": 4,
            "position_sensitive": false,
            "seed": 4951,
            "selected_blocks": 16,
            "seqlen_k": 4033,
            "seqlen_q": 128,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_empty_first_row",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": true,
            "num_heads": 4,
            "position_sensitive": false,
            "seed": 4210,
            "selected_blocks": 4,
            "seqlen_k": 256,
            "seqlen_q": 256,
            "softmax_scale": "default"
        }
    },
    {
        "name": "ut_block_sizes_global",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 1,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 2,
            "position_sensitive": false,
            "seed": 7001,
            "selected_blocks": 2,
            "seqlen_k": 128,
            "seqlen_q": 128,
            "softmax_scale": "default",
            "valid_last_block": 40
        }
    },
    {
        "name": "ut_block_sizes_batch",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 2,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 2,
            "position_sensitive": false,
            "seed": 7002,
            "selected_blocks": 2,
            "seqlen_k": 128,
            "seqlen_q": 128,
            "softmax_scale": "default",
            "valid_last_block": 40
        }
    },
    {
        "name": "ut_block_sizes_head",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 3,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 2,
            "position_sensitive": false,
            "seed": 7003,
            "selected_blocks": 2,
            "seqlen_k": 128,
            "seqlen_q": 128,
            "softmax_scale": "default",
            "valid_last_block": 40
        }
    },
    {
        "name": "ut_v_permutation_all_slots",
        "params": {
            "batch_size": 1,
            "benchmark": false,
            "block_sizes_mode": 0,
            "check_correctness": true,
            "empty_first_row": false,
            "num_heads": 2,
            "position_sensitive": true,
            "seed": 0,
            "selected_blocks": 1,
            "seqlen_k": 64,
            "seqlen_q": 64,
            "softmax_scale": 30.0
        }
    }
]""")

# These choices are independent in the public API.  In particular, sorted
# top-k indices still require the non-contiguous route with either count API.
_COUNT_MODES = (
    ("per_row", {"varying_block_counts": True, "empty_first_row": True}),
    ("omitted", {"omit_block_nums": True, "uniform_block_count": False}),
    ("uniform", {"uniform_block_count": True}),
    ("omitted_uniform", {"omit_block_nums": True, "uniform_block_count": True}),
    (
        "empty_omitted",
        {"omit_block_nums": True, "uniform_block_count": False, "block_sparse_num": 0},
    ),
    ("empty_uniform", {"uniform_block_count": True, "block_sparse_num": 0}),
    (
        "empty_contiguous",
        {
            "omit_block_nums": True,
            "uniform_block_count": True,
            "contiguous_block_indices": True,
            "noncontiguous_topk": False,
            "block_sparse_num": 0,
        },
    ),
)
_CASES.extend(
    {
        "name": f"topk_{count_name}_sizes{mode}_k{seqlen_k}",
        "params": {
            "batch_size": 2,
            "num_heads": 2,
            "seqlen_q": 100,
            "seqlen_k": seqlen_k,
            "selected_blocks": 3,
            "softmax_scale": "default",
            "block_sizes_mode": mode,
            "valid_last_block": 40,
            "empty_first_row": False,
            "position_sensitive": False,
            "noncontiguous_topk": True,
            "contiguous_block_indices": False,
            "seed": 508300 + mode,
            **count_options,
        },
    }
    for mode in range(4)
    for seqlen_k in (320, 257)
    for count_name, count_options in _COUNT_MODES
)
_CASES.extend(
    {
        "name": f"topk_scalar_count_below_capacity_uniform{uniform}",
        "params": {
            "batch_size": 1,
            "num_heads": 2,
            "seqlen_q": 100,
            "seqlen_k": 257,
            "selected_blocks": 3,
            "softmax_scale": "default",
            "block_sizes_mode": 0,
            "empty_first_row": False,
            "position_sensitive": False,
            "noncontiguous_topk": True,
            "contiguous_block_indices": False,
            "uniform_block_count": uniform,
            "omit_block_nums": not uniform,
            "block_sparse_num": 2,
            "seed": 508399,
        },
    }
    for uniform in (False, True)
)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _make_block_sizes(params, *, batch: int, heads: int, k_blocks: int, device):
    mode = int(params["block_sizes_mode"])
    if mode == 0:
        return None
    valid_last = int(params["valid_last_block"])
    if mode == 1:
        sizes = torch.full((k_blocks,), _BLOCK, dtype=torch.int32, device=device)
        sizes[-1] = valid_last
    elif mode == 2:
        sizes = torch.full((batch, k_blocks), _BLOCK, dtype=torch.int32, device=device)
        sizes[:, -1] = valid_last
    else:
        sizes = torch.full(
            (batch, heads, k_blocks), _BLOCK, dtype=torch.int32, device=device
        )
        sizes[:, :, -1] = valid_last
    if params.get("noncontiguous_topk", False):
        sizes[..., 0] = 0
        if mode >= 2:
            offsets = torch.arange(batch, device=device) * 7
            if mode == 3:
                offsets = offsets[:, None] + torch.arange(heads, device=device) * 11
            sizes[..., -1] = (valid_last - offsets).clamp(min=0)
    return sizes


def _make_inputs(params, *, device):
    batch = int(params["batch_size"])
    heads = int(params["num_heads"])
    seqlen_q = int(params["seqlen_q"])
    seqlen_k = int(params["seqlen_k"])
    selected_blocks = int(params["selected_blocks"])
    q_blocks = _ceil_div(seqlen_q, _BLOCK)
    k_blocks = _ceil_div(seqlen_k, _BLOCK)
    padded_k = k_blocks * _BLOCK
    generator = torch.Generator(device=device).manual_seed(int(params["seed"]))

    if bool(params["position_sensitive"]):
        walsh = torch.tensor(
            [
                [
                    1 if ((row & column).bit_count() & 1) == 0 else -1
                    for column in range(16)
                ]
                for row in range(16)
            ],
            dtype=torch.int8,
            device=device,
        ).repeat(1, 8)
        q_int8 = walsh[torch.arange(seqlen_q, device=device) % 16].mul(8)
        k_int8 = walsh[torch.arange(seqlen_k, device=device) % 16].mul(8)
        q_int8 = (
            q_int8.view(1, 1, seqlen_q, _HEAD_DIM)
            .expand(batch, heads, seqlen_q, _HEAD_DIM)
            .contiguous()
        )
        k_int8 = (
            k_int8.view(1, 1, seqlen_k, _HEAD_DIM)
            .expand(batch, heads, seqlen_k, _HEAD_DIM)
            .contiguous()
        )
        q_scale = torch.full(
            (batch, heads, _ceil_div(seqlen_q, 128) * 4),
            1.0 / 64.0,
            dtype=torch.float32,
            device=device,
        )
        k_scale = torch.full(
            (batch, heads, k_blocks),
            1.0 / 64.0,
            dtype=torch.float32,
            device=device,
        )
        token_code = ((torch.arange(padded_k, device=device) % 16).float() - 7.5) / 4.0
        channel_code = (torch.arange(_HEAD_DIM, device=device).float() % 8) / 32.0
        logical_v = token_code.view(1, 1, 1, padded_k) + channel_code.view(
            1, 1, _HEAD_DIM, 1
        )
        logical_v = logical_v.expand(batch, heads, _HEAD_DIM, padded_k).contiguous()
        v_scale = torch.ones(
            (batch, heads, _HEAD_DIM), dtype=torch.float32, device=device
        )
    else:
        q_int8 = torch.randint(
            -32,
            33,
            (batch, heads, seqlen_q, _HEAD_DIM),
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
        k_int8 = torch.randint(
            -32,
            33,
            (batch, heads, seqlen_k, _HEAD_DIM),
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
        q_scale = (
            torch.rand(
                (batch, heads, _ceil_div(seqlen_q, 128) * 4),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.01
            + 0.002
        )
        k_scale = (
            torch.rand(
                (batch, heads, k_blocks),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.01
            + 0.002
        )
        logical_v = torch.randn(
            (batch, heads, _HEAD_DIM, padded_k),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).mul_(0.25)
        v_scale = (
            torch.rand(
                (batch, heads, _HEAD_DIM),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.75
            + 0.25
        )

    block_sizes = _make_block_sizes(
        params, batch=batch, heads=heads, k_blocks=k_blocks, device=device
    )
    if bool(params.get("noncontiguous_topk", False)):
        # Make attention to padded tokens fail the numerical comparison.
        logical_v[..., seqlen_k:] = 64.0
        if block_sizes is not None:
            sizes = block_sizes.reshape(
                batch if block_sizes.ndim >= 2 else 1,
                heads if block_sizes.ndim == 3 else 1,
                k_blocks,
            )
            invalid = torch.arange(_BLOCK, device=device) >= sizes.unsqueeze(-1)
            logical_v.masked_fill_(invalid.reshape(*sizes.shape[:2], 1, padded_k), 64.0)

    inverse_perm = torch.tensor(_V_INVERSE_PERM, dtype=torch.long, device=device)
    v_fp8 = logical_v.view(batch, heads, _HEAD_DIM, k_blocks * 4, 16)
    v_fp8 = v_fp8.index_select(-1, inverse_perm).reshape(
        batch, heads, _HEAD_DIM, padded_k
    )
    v_fp8 = v_fp8.to(torch.float8_e4m3fn).contiguous()

    batch_coord = torch.arange(batch, dtype=torch.int64, device=device).view(
        batch, 1, 1, 1
    )
    head_coord = torch.arange(heads, dtype=torch.int64, device=device).view(
        1, heads, 1, 1
    )
    query_coord = torch.arange(q_blocks, dtype=torch.int64, device=device).view(
        1, 1, q_blocks, 1
    )
    slot = torch.arange(selected_blocks, dtype=torch.int64, device=device).view(
        1, 1, 1, selected_blocks
    )
    q2k = ((batch_coord * 3 + head_coord * 5 + query_coord * 7 + slot) % k_blocks).to(
        torch.int32
    )
    if bool(params.get("noncontiguous_topk", False)):
        # Always select the partial tail and exclude its predecessor so every
        # row has a gap, even if its random top-k happens to be consecutive.
        scores = torch.rand(
            (batch, heads, q_blocks, k_blocks - 2),
            device=device,
            generator=generator,
        )
        selected = scores.topk(selected_blocks - 1, dim=-1).indices
        tail = torch.full(
            (batch, heads, q_blocks, 1),
            k_blocks - 1,
            dtype=torch.int64,
            device=device,
        )
        q2k = torch.cat((selected, tail), dim=-1).sort(dim=-1).values.to(torch.int32)
    q2k = q2k.contiguous()
    block_sparse_num = int(params.get("block_sparse_num", selected_blocks))
    q2k_nums = torch.full(
        (batch, heads, q_blocks),
        block_sparse_num,
        dtype=torch.int32,
        device=device,
    )
    if params.get("varying_block_counts", False):
        q2k_nums = (
            (
                torch.arange(batch * heads * q_blocks, device=device)
                % (selected_blocks + 1)
            )
            .reshape(batch, heads, q_blocks)
            .to(torch.int32)
        )
    if bool(params["empty_first_row"]):
        q2k_nums[0, 0, 0] = 0

    uniform_contiguous = not params["empty_first_row"] and not params.get(
        "noncontiguous_topk", False
    )
    softmax_scale = (
        _HEAD_DIM**-0.5
        if params["softmax_scale"] == "default"
        else float(params["softmax_scale"])
    )
    return {
        **params,
        "Q": q_int8.contiguous(),
        "K": k_int8.contiguous(),
        "V": v_fp8,
        "Q_scale": q_scale.contiguous(),
        "K_scale": k_scale.contiguous(),
        "V_scale": v_scale.contiguous(),
        "q2k_block_index": q2k,
        "q2k_block_nums": None
        if params.get("omit_block_nums", False)
        else q2k_nums.contiguous(),
        "uniform_block_count": params.get("uniform_block_count", uniform_contiguous),
        "contiguous_block_indices": params.get(
            "contiguous_block_indices", uniform_contiguous
        ),
        "block_sizes": block_sizes,
        "block_sparse_num": block_sparse_num,
        "softmax_scale": float(softmax_scale),
        "O": torch.empty(
            (batch, heads, seqlen_q, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        ),
    }


def _logical_v(inputs):
    v_fp8 = inputs["V"]
    groups = int(v_fp8.shape[-1]) // 16
    perm = torch.tensor(_V_PERM, dtype=torch.long, device=v_fp8.device)
    return (
        v_fp8.view(*v_fp8.shape[:-1], groups, 16)
        .index_select(-1, perm)
        .reshape(v_fp8.shape)
        .float()
    )


def _block_valid_tokens(inputs, *, batch: int, head: int, block: int) -> int:
    seqlen_k = int(inputs["seqlen_k"])
    valid = max(0, min(_BLOCK, seqlen_k - block * _BLOCK))
    sizes = inputs["block_sizes"]
    if sizes is None:
        return valid
    mode = int(inputs["block_sizes_mode"])
    if mode == 1:
        declared = int(sizes[block].item())
    elif mode == 2:
        declared = int(sizes[batch, block].item())
    else:
        declared = int(sizes[batch, head, block].item())
    return max(0, min(valid, declared))


def _reference(inputs):
    q = inputs["Q"].float()
    k = inputs["K"].float()
    v = _logical_v(inputs)
    batch, heads, seqlen_q, _ = q.shape
    output = torch.zeros_like(inputs["O"])
    query_rows = torch.arange(seqlen_q, dtype=torch.long, device=q.device)
    q_scale = inputs["Q_scale"].gather(
        2,
        (query_rows // 32).view(1, 1, seqlen_q).expand(batch, heads, seqlen_q),
    )
    q = q * q_scale.unsqueeze(-1)
    v = v * inputs["V_scale"].unsqueeze(-1)

    for batch_idx in range(batch):
        for head_idx in range(heads):
            for q_block in range(_ceil_div(seqlen_q, _BLOCK)):
                q_start = q_block * _BLOCK
                q_end = min(seqlen_q, q_start + _BLOCK)
                counts = inputs["q2k_block_nums"]
                count = (
                    int(inputs["block_sparse_num"])
                    if counts is None or inputs["uniform_block_count"]
                    else int(counts[batch_idx, head_idx, q_block].item())
                )
                if count == 0:
                    continue
                k_parts = []
                v_parts = []
                selected = inputs["q2k_block_index"][
                    batch_idx, head_idx, q_block, :count
                ]
                for block_tensor in selected:
                    block = int(block_tensor.item())
                    valid = _block_valid_tokens(
                        inputs, batch=batch_idx, head=head_idx, block=block
                    )
                    if valid == 0:
                        continue
                    tokens = torch.arange(
                        block * _BLOCK,
                        block * _BLOCK + valid,
                        dtype=torch.long,
                        device=q.device,
                    )
                    k_parts.append(
                        k[batch_idx, head_idx, tokens]
                        * inputs["K_scale"][batch_idx, head_idx, block]
                    )
                    v_parts.append(
                        v[batch_idx, head_idx].index_select(1, tokens).transpose(0, 1)
                    )
                if not k_parts:
                    continue
                key = torch.cat(k_parts, dim=0)
                value = torch.cat(v_parts, dim=0)
                scores = q[batch_idx, head_idx, q_start:q_end] @ key.transpose(0, 1)
                scores.mul_(float(inputs["softmax_scale"]))
                result = torch.softmax(scores, dim=-1) @ value
                output[batch_idx, head_idx, q_start:q_end] = result.to(output.dtype)
    return output


def _aligned_workspace(*, device):
    storage = torch.full(
        (_WORKSPACE_BYTES + 127,), 0xA5, dtype=torch.uint8, device=device
    )
    offset = (-int(storage.data_ptr())) % 128
    workspace = storage[offset : offset + _WORKSPACE_BYTES]
    assert workspace.numel() == _WORKSPACE_BYTES
    assert workspace.data_ptr() % 128 == 0
    return storage, workspace


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_cake_sage_block_sparse_attention(case):
    params = dict(case["params"])
    inputs = _make_inputs(params, device=torch.device("cuda"))
    workspace_storage, workspace = _aligned_workspace(device=inputs["Q"].device)
    expected = _reference(inputs)
    softmax_scale = (
        None if params["softmax_scale"] == "default" else float(params["softmax_scale"])
    )
    noncontiguous_topk = bool(params.get("noncontiguous_topk", False))
    if noncontiguous_topk:
        q2k = inputs["q2k_block_index"]
        tail_block = _ceil_div(int(params["seqlen_k"]), _BLOCK) - 1
        assert torch.all(q2k[..., -1] == tail_block).item()
        assert torch.all(torch.any(q2k[..., 1:] - q2k[..., :-1] > 1, dim=-1)).item()
    if inputs["uniform_block_count"] and inputs["q2k_block_nums"] is not None:
        # A true uniform guarantee makes the scalar count authoritative, even
        # if the supplied per-row counts describe a different selection.
        inputs["q2k_block_nums"].fill_(1 if inputs["block_sparse_num"] == 0 else 0)
    elif inputs["q2k_block_nums"] is not None:
        slots = torch.arange(
            inputs["q2k_block_index"].shape[-1], device=inputs["Q"].device
        )
        inactive = slots >= inputs["q2k_block_nums"].unsqueeze(-1)
        inputs["q2k_block_index"].masked_fill_(inactive, -1)

    returned = bsa_attn_sm120_blk64_sage_fwd(
        inputs["Q"],
        inputs["K"],
        inputs["V"],
        inputs["Q_scale"],
        inputs["K_scale"],
        inputs["V_scale"],
        inputs["q2k_block_index"],
        inputs["block_sparse_num"],
        block_sizes=inputs["block_sizes"],
        q2k_block_nums=inputs["q2k_block_nums"],
        softmax_scale=softmax_scale,
        out=inputs["O"],
        tma_descriptor_workspace=workspace,
        uniform_block_count=inputs["uniform_block_count"],
        contiguous_block_indices=inputs["contiguous_block_indices"],
        backend="cake",
    )

    # The compatibility workspace is unused by every exported specialization.
    assert torch.all(workspace_storage == 0xA5).item()
    assert returned.data_ptr() == inputs["O"].data_ptr()
    if bool(params["empty_first_row"]):
        first_q_block = returned[0, 0, : min(_BLOCK, int(params["seqlen_q"]))]
        assert torch.count_nonzero(first_q_block).item() == 0
    if inputs["block_sparse_num"] == 0:
        assert torch.count_nonzero(returned).item() == 0
    atol, rtol = (
        (3.0e-3, 0.0) if bool(params["position_sensitive"]) else (1.0e-2, 1.0e-2)
    )
    torch.testing.assert_close(returned.float(), expected.float(), atol=atol, rtol=rtol)
