# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Recorded cases of the sparse MLA classes: integer-derived inputs and the digests of their outputs.

Every value is a multiple of 1/8 in [-2, 2] computed from 64-bit integer arithmetic, so it is exact in bf16 and e4m3
and identical on every host and device. A request's cache rows and query rows are functions of its seed and the token
position alone, whatever pool or batch they sit in. ``DIGESTS`` holds sha256 of the output and LSE bits of
:func:`run_digest_cases`; SM100, SM103 and SM107 give the same bits, so one table serves the three architectures.

    python -m flashinfer.batch_invariant.mla.cases     # print the digests of the current device
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

import torch

from . import spec

SOFTMAX_SCALE = 0.0625
PAGE_SIZE = 64
_MASK = 0xFFFFFFFF


def _mix(x: torch.Tensor, seed: int) -> torch.Tensor:
    x = (x * 2654435761 + seed * 40503 + 12345) & _MASK
    x = ((x ^ (x >> 15)) * 2246822519) & _MASK
    x = ((x ^ (x >> 13)) * 3266489917) & _MASK
    return x ^ (x >> 16)


def values(count: int, seed: int, start: int = 0) -> torch.Tensor:
    """``count`` fp32 multiples of 1/8 in [-2, 2], a function of ``(seed, start + i)``."""
    idx = torch.arange(start, start + count, dtype=torch.int64)
    return ((_mix(idx, seed) % 33).to(torch.float32) - 16.0) / 8.0


def torch_dtype(dtype) -> torch.dtype:
    return torch.float8_e4m3fn if spec.dtype_name(dtype) == "e4m3" else torch.bfloat16


def request_keys(seed: int, start: int, count: int, head_dim: int) -> torch.Tensor:
    """Cache rows ``[count, head_dim]`` of tokens ``start .. start + count`` of request ``seed`` (fp32)."""
    return values(count * head_dim, seed, start * head_dim).view(count, head_dim)


def request_query(seed: int, position: int, heads: int, head_dim: int) -> torch.Tensor:
    """Query rows ``[heads, head_dim]`` of the token at ``position`` of request ``seed`` (fp32)."""
    return values(heads * head_dim, seed * 31 + 17, position * 7919).view(
        heads, head_dim
    )


def selection(seed: int, visible: int, topk: int) -> List[int]:
    """Ascending token positions a query that sees ``[0, visible)`` selects: all of them up to ``topk``, else one
    position out of each of ``topk`` equal strides."""
    if visible <= topk:
        return list(range(visible))
    jitter = _mix(torch.arange(topk, dtype=torch.int64), seed * 13 + visible)
    return [
        (i * visible) // topk
        + int(jitter[i]) % (((i + 1) * visible) // topk - (i * visible) // topk)
        for i in range(topk)
    ]


class Pool:
    """A paged KV pool holding several requests on scattered pages.

    ``slots[r][p]`` is the storage row (``page * PAGE_SIZE + offset``) of token ``p`` of request ``r``."""

    def __init__(
        self,
        requests: Sequence[Tuple[int, int]],
        head_dim: int,
        dtype,
        device,
        scramble: int = 7,
    ):
        self.requests = list(requests)
        self.head_dim = head_dim
        self.dtype = torch_dtype(dtype)
        self.device = torch.device(device)
        pages = [-(-length // PAGE_SIZE) for _, length in self.requests]
        total = sum(pages)
        order = torch.argsort(
            _mix(torch.arange(total, dtype=torch.int64), scramble), stable=True
        )
        kv = torch.zeros(total, PAGE_SIZE, head_dim, dtype=torch.float32)
        self.slots: List[torch.Tensor] = []
        offset = 0
        for (seed, length), n in zip(self.requests, pages, strict=True):
            page_ids = order[offset : offset + n]
            offset += n
            rows = torch.zeros(n * PAGE_SIZE, head_dim)
            rows[:length] = request_keys(seed, 0, length, head_dim)
            kv[page_ids] = rows.view(n, PAGE_SIZE, head_dim)
            pos = torch.arange(length)
            self.slots.append(
                (page_ids[pos // PAGE_SIZE] * PAGE_SIZE + pos % PAGE_SIZE).to(
                    torch.int32
                )
            )
        self.kv = kv.to(self.device).to(self.dtype)

    def row(
        self, request: int, visible: int, topk: int, width: int
    ) -> Tuple[torch.Tensor, int]:
        """The index row ``[width]`` and live count of a query of ``request`` that sees ``[0, visible)``."""
        seed = self.requests[request][0]
        picked = selection(seed, visible, topk)
        out = torch.full((width,), -1, dtype=torch.int32)
        out[: len(picked)] = self.slots[request][
            torch.tensor(picked, dtype=torch.int64)
        ]
        return out, len(picked)

    def query(self, request: int, position: int, heads: int) -> torch.Tensor:
        seed = self.requests[request][0]
        return (
            request_query(seed, position, heads, self.head_dim)
            .to(self.device)
            .to(self.dtype)
        )


#: ``(name, dtype, head_dim, heads)``; each runs every row of ``DIGEST_ROWS`` on one 4,096-token request.
DIGEST_CASES = tuple(
    (f"{dtype}.d{head_dim}.h{heads}", dtype, head_dim, heads)
    for dtype, head_dim, heads in (
        ("e4m3", 576, 64),
        ("e4m3", 576, 8),
        ("e4m3", 512, 16),
        ("bf16", 576, 64),
        ("bf16", 576, 8),
        ("bf16", 512, 16),
    )
)
#: ``(visible tokens, list width)``: one key, tile edges, a short list, full lists, the wider Flash rows.
DIGEST_ROWS = (
    (1, 2048),
    (5, 2048),
    (127, 2048),
    (128, 2048),
    (129, 2048),
    (300, 2048),
    (2048, 2048),
    (4096, 2048),
    (2051, 2052),
    (4096, 2051),
)
DIGEST_REQUEST = (4242, 4096)


def digest(out: torch.Tensor, lse: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(out.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    h.update(lse.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def run_digest_cases(device="cuda") -> Dict[str, str]:
    """Run every digest case on ``device`` as one-row launches of the class's plan."""
    from . import engine

    device = torch.device(device)
    result = {}
    for name, dtype, head_dim, heads in DIGEST_CASES:
        pool = Pool([DIGEST_REQUEST], head_dim, dtype, device)
        plan = engine.plan(
            device,
            heads,
            q_dtype=pool.dtype,
            qk_rope_head_dim=head_dim - spec.LATENT_DIM,
            softmax_scale=SOFTMAX_SCALE,
        )
        for visible, width in DIGEST_ROWS:
            topk = min(width, 2051) if width > 2048 else 2048
            index_row, live = pool.row(0, visible, topk, width)
            out, lse = plan.run(
                pool.query(0, visible - 1, heads).view(1, heads, head_dim),
                pool.kv.view(-1, head_dim),
                index_row.view(1, width).to(device),
                torch.tensor([live], dtype=torch.int32, device=device),
                return_lse=True,
            )
            result[f"{name}.v{visible}.w{width}"] = digest(
                out.view(1, 1, heads, spec.LATENT_DIM), lse.view(1, 1, heads)
            )
    return result


#: One table for SM100, SM103 and SM107 (``tests/batch_invariant/test_mla_sparse.py``).
DIGESTS: Dict[str, str] = {
    "bf16.d512.h16.v1.w2048": "f7353a6c554581e4059ee9ff97ed6115",
    "bf16.d512.h16.v127.w2048": "a60ce86eb91d30bc5fd137a4206cb621",
    "bf16.d512.h16.v128.w2048": "05772489ebd42d0e53ce65086db33d7a",
    "bf16.d512.h16.v129.w2048": "51bae6b7d5f94c483034a60c03a9e1a8",
    "bf16.d512.h16.v2048.w2048": "ca05c3cb671fdad3f476d88da18e6385",
    "bf16.d512.h16.v2051.w2052": "10c0b699ffa5984439160c3e6f4d0a02",
    "bf16.d512.h16.v300.w2048": "b0f5e700c4a10bf3ea8cb63898c4da01",
    "bf16.d512.h16.v4096.w2048": "ce9234f425ad5444e4576a1868a5f97f",
    "bf16.d512.h16.v4096.w2051": "b737f11d485dccbfce79100a4c45420c",
    "bf16.d512.h16.v5.w2048": "aba0c7f9b0daa157117927f1775c1581",
    "bf16.d576.h64.v1.w2048": "8728d7a0880ffb03e90cc20575b18504",
    "bf16.d576.h64.v127.w2048": "b2998953cb067b2eab072429773c5d12",
    "bf16.d576.h64.v128.w2048": "9bc9c524e3731189530466dcb1a4c321",
    "bf16.d576.h64.v129.w2048": "bd7fd36dbf200ba5a227959776447df4",
    "bf16.d576.h64.v2048.w2048": "e55d05566cc4f30914cfa28d72600818",
    "bf16.d576.h64.v2051.w2052": "c6f8f027bd1c41309f7caf2bb78c0439",
    "bf16.d576.h64.v300.w2048": "e1076e406fd5aa11701e2490013b34bd",
    "bf16.d576.h64.v4096.w2048": "78f27afb9d7e36c91c9ece56b8f3355c",
    "bf16.d576.h64.v4096.w2051": "63dd83c123fb0497a051a7a5c65ee4e7",
    "bf16.d576.h64.v5.w2048": "0f1ff4f04901551d1970aad72eda07f6",
    "bf16.d576.h8.v1.w2048": "11e6251301386b8cba83b844578839a6",
    "bf16.d576.h8.v127.w2048": "82483cbf1e07eeb9f391a4ba699dca06",
    "bf16.d576.h8.v128.w2048": "982e8e6a739bcaf5c02c4caa247ce2aa",
    "bf16.d576.h8.v129.w2048": "3333fc9b13f42e85cd0f67fb13f21ea8",
    "bf16.d576.h8.v2048.w2048": "45b2816797f275e88677745388b64e34",
    "bf16.d576.h8.v2051.w2052": "1b8f0772757e9414294cf73132c28eed",
    "bf16.d576.h8.v300.w2048": "3337677fe64c84a8c6e3312273d06e96",
    "bf16.d576.h8.v4096.w2048": "ae0a1427915de3d590bc228af4eb142e",
    "bf16.d576.h8.v4096.w2051": "02a134e153b149985d7e1486ab66e779",
    "bf16.d576.h8.v5.w2048": "4056fe16a3095997064578f8211d6225",
    "e4m3.d512.h16.v1.w2048": "b0ca2ddcca410f513ebacc0218b12c5f",
    "e4m3.d512.h16.v127.w2048": "6f462fee06128ddbabeba3981e77289c",
    "e4m3.d512.h16.v128.w2048": "3a8fe168ad4b848fa54c3a30e2254c66",
    "e4m3.d512.h16.v129.w2048": "91dd0707e63a79b7b16a7fc3313c3de7",
    "e4m3.d512.h16.v2048.w2048": "f66ea6c35007bc7d6b1b7287ed1e3534",
    "e4m3.d512.h16.v2051.w2052": "c81618ccbd6f83552c3aff52aae77a80",
    "e4m3.d512.h16.v300.w2048": "3d55cee5c92afcd87db23e831ecf72d1",
    "e4m3.d512.h16.v4096.w2048": "0334c04aea316aa56d17e892bba2265e",
    "e4m3.d512.h16.v4096.w2051": "dd343a2d60355556f0094e37b5d96a7d",
    "e4m3.d512.h16.v5.w2048": "b1dd74c51d7839d02be8c8a4690c7eec",
    "e4m3.d576.h64.v1.w2048": "f038bbd8e669da502098fd1fcf69ff68",
    "e4m3.d576.h64.v127.w2048": "2518e28fccbb83ab853314a9f5ef0174",
    "e4m3.d576.h64.v128.w2048": "5a07e3409ad26a4ac93968f47155e677",
    "e4m3.d576.h64.v129.w2048": "c3c23065302a01ce0a53f2e82347cac2",
    "e4m3.d576.h64.v2048.w2048": "7c836307f0f83aa9d7ab2d58032b6f7f",
    "e4m3.d576.h64.v2051.w2052": "08cb656d3ec14272c4320d941199ec21",
    "e4m3.d576.h64.v300.w2048": "519d3d3456ddc897f7f66139b463f5dd",
    "e4m3.d576.h64.v4096.w2048": "b33e7c3570eb43366d42d4ec08dc465e",
    "e4m3.d576.h64.v4096.w2051": "45c3b1debe955f17269e8b969156391e",
    "e4m3.d576.h64.v5.w2048": "a3a350c241a637e5bedbb507bb03ac6c",
    "e4m3.d576.h8.v1.w2048": "61a037d17b236bbbcd708e667c027d91",
    "e4m3.d576.h8.v127.w2048": "1b90359abc53b8f7673ab75ccd2e033d",
    "e4m3.d576.h8.v128.w2048": "f1dbebac25e609ae7f258fb941c0bf4b",
    "e4m3.d576.h8.v129.w2048": "2d975f465c904d69ae95d04148a66c40",
    "e4m3.d576.h8.v2048.w2048": "9d8285efa3361fefec78f5b251ad1f88",
    "e4m3.d576.h8.v2051.w2052": "b33f5280ee8ab3dd8969fbe7c17687c4",
    "e4m3.d576.h8.v300.w2048": "98522f1160b1884223351c2f9824b5be",
    "e4m3.d576.h8.v4096.w2048": "73e8e1465df5f1b13a536faaf5a955c1",
    "e4m3.d576.h8.v4096.w2051": "c91cf01d802d3d45c214b312b6dfe281",
    "e4m3.d576.h8.v5.w2048": "df9e7ed76070c0671f627d1bc88f05e3",
}


if __name__ == "__main__":
    import json

    print(json.dumps(run_digest_cases(), indent=4, sort_keys=True))
