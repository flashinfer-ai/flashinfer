# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM device kernels in a standalone low-token BF16 research harness.

Unchanged pinned vLLM GEMM device functions, with either naive route assignment
or expert grouping using FI GPU sorting and a padding adapter. All adapters are
included in timed calls. Preserves typed preactivation and pre-cast weighting.
The grouped policy is a standalone comparison, not stock vLLM integration.
Activation and sum are harness Triton kernels, not vLLM CUDA implementations.
Supports SwiGLU and a standalone ReLU2 wrapper. No vLLM framework performance or bitwise Frost equivalence is asserted.
"""

import ast
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import sys

import torch
import triton
import triton.language as tl

VLLM_COMMIT = "71c20f73f2f1764576fed799c152319c9fd1c820"
VLLM_SOURCE_SHA256 = "21860340e4b409b01666e2ba0f0edae66b0f97b859756b174b573bab895cfaf7"


def _extract_device_source(source):
    """Read the pinned upstream file without importing vLLM or fetching code."""
    data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != VLLM_SOURCE_SHA256:
        raise ValueError(f"Expected fused_moe.py from vLLM commit {VLLM_COMMIT}")
    text = data.decode()
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    parts = []
    for name in ("write_zeros_to_output", "fused_moe_kernel"):
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        parts.append("".join(lines[start - 1 : node.end_lineno]))
    return (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        f"# Unmodified device functions from vLLM {VLLM_COMMIT}.\n"
        "import triton\nimport triton.language as tl\n\n" + "\n\n".join(parts)
    )


@triton.jit
def activation(
    pre, mid, I: tl.constexpr, R: tl.constexpr, B: tl.constexpr, GATED: tl.constexpr
):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    r = idx // I
    c = idx % I
    if GATED:
        up = tl.load(pre + r * I * 2 + c, idx < R * I, other=0).to(tl.float32)
        gate = tl.load(pre + r * I * 2 + I + c, idx < R * I, other=0).to(tl.float32)
        value = up * gate / (1 + tl.exp(-gate))
    else:
        up = tl.load(pre + idx, idx < R * I, other=0).to(tl.float32)
        value = tl.maximum(up, 0.0) * tl.maximum(up, 0.0)
    tl.store(mid + idx, value, idx < R * I)


@triton.jit
def combine(
    raw, out, H: tl.constexpr, K: tl.constexpr, T: tl.constexpr, B: tl.constexpr
):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    token = idx // H
    col = idx % H
    acc = tl.full((B,), 0, tl.float32)
    for slot in range(K):
        acc += tl.load(raw + (token * K + slot) * H + col, idx < T * H, other=0).to(
            tl.float32
        )
    tl.store(out + idx, acc, idx < T * H)


@triton.jit
def pad_sorted_indices(
    permuted,
    limits,
    active_tiles,
    sorted_ids,
    R: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    tile = idx // M
    active = tile < tl.load(active_tiles)
    limit = tl.load(limits + tile, (idx < R * M) & active, other=0)
    valid = (idx < R * M) & active & (idx < limit)
    # FI only initializes valid entries. The upstream GEMM masks indices >=R,
    # so never expose negative padding or stale entries outside each tile limit.
    route = tl.load(permuted + idx, valid, other=R)
    tl.store(
        sorted_ids + idx,
        tl.where(valid & (route >= 0) & (route < R), route, R),
        idx < R * M,
    )


class VllmTritonComparator:
    def __init__(self, x, w1, w2, ids, scores, source, gated=True, route_policy="both"):
        if route_policy not in ("both", "naive", "grouped"):
            raise ValueError("route_policy must be both, naive or grouped")
        self.gated = gated
        # Triton inspects source files while compiling; keep this file alive with the
        # comparator rather than exec-ing a source string with no inspectable file.
        extracted = _extract_device_source(source)
        self._source_dir = tempfile.TemporaryDirectory(prefix="frost-vllm-device-")
        module_path = Path(self._source_dir.name) / "device.py"
        module_path.write_text(extracted)
        spec = importlib.util.spec_from_file_location(
            "frost_vllm_pinned_device", module_path
        )
        self._module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self._module
        spec.loader.exec_module(self._module)
        self._gemm = self._module.fused_moe_kernel
        self.x, self.w1, self.w2, self.ids, self.scores = x, w1, w2, ids, scores
        self.t, self.k = ids.shape
        self.h = x.shape[1]
        self.i = w2.shape[-1]
        self.r = self.t * self.k
        if self.r * 4 > w1.shape[0]:
            raise ValueError(
                "The comparator is limited to low-token workloads with routes * 4 <= experts"
            )
        if x.dtype not in (torch.bfloat16, torch.float16) or any(
            tensor.dtype != x.dtype for tensor in (w1, w2)
        ):
            raise ValueError("The comparator requires matching FP16/BF16 tensors")
        self.pre = torch.empty(
            (self.r, (2 if self.gated else 1) * self.i), device=x.device, dtype=x.dtype
        )
        self.mid = torch.empty((self.r, self.i), device=x.device, dtype=x.dtype)
        self.raw = torch.empty((self.r, self.h), device=x.device, dtype=x.dtype)
        self.out = torch.empty_like(x)
        self.padded = torch.empty(1, device=x.device, dtype=torch.int32)
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            allocate_moe_sort_buffers,
            moe_sort,
        )

        self._sort = moe_sort
        self.sort_buffers = allocate_moe_sort_buffers(
            self.t, w1.shape[0], self.k, tile_tokens_dim=16, device=x.device
        )
        self.sorted_ids = torch.empty(self.r * 16, device=x.device, dtype=torch.int32)
        self._grouped = False
        self.tactics = [
            (16, n, k, stage, grouped)
            for grouped in (
                {"both": (False, True), "naive": (False,), "grouped": (True,)}[
                    route_policy
                ]
            )
            for n in (32, 64, 128)
            for k in (32, 64, 128)
            for stage in (2, 3, 4)
        ]

    def get_valid_tactics(self, *args):
        return self.tactics

    def gemm(self, a, b, c, tactic, mul, topk):
        m, n, k, stage = tactic[:4]
        em = self.r * m
        self._gemm[(self.r * triton.cdiv(b.shape[1], n),)](
            a,
            b,
            c,
            None,
            None,
            None,
            self.scores,
            self.sorted_ids if self._grouped else None,
            self.sort_buffers["out_tile_idx_to_expert_idx"]
            if self._grouped
            else self.ids,
            self.sort_buffers["out_total_num_padded_tokens"]
            if self._grouped
            else self.padded,
            b.shape[1],
            b.shape[2],
            em,
            self.r,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(2),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            naive_block_assignment=not self._grouped,
            BLOCK_SIZE_M=m,
            BLOCK_SIZE_N=n,
            BLOCK_SIZE_K=k,
            GROUP_SIZE_M=1,
            SPLIT_K=1,
            MUL_ROUTED_WEIGHT=mul,
            top_k=topk,
            compute_type=tl.float16 if self.x.dtype == torch.float16 else tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            per_channel_quant=False,
            HAS_BIAS=False,
            SWAP_AB=False,
            USE_TD=False,
            num_warps=4,
            num_stages=stage,
        )

    def forward(self, tactic):
        # Include route preparation in every call, including graph replay.
        tactic = (*tactic, False) if len(tactic) == 4 else tuple(tactic)
        if tactic not in self.tactics:
            raise ValueError("Tactic is outside the configured comparator search")
        self._grouped = bool(tactic[4])
        if self._grouped:
            self._sort(
                self.ids,
                self.scores,
                self.w1.shape[0],
                self.k,
                tile_tokens_dim=16,
                enable_pdl=False,
                **self.sort_buffers,
            )
            pad_sorted_indices[(triton.cdiv(self.r * 16, 256),)](
                self.sort_buffers["out_permuted_idx_to_expanded_idx"],
                self.sort_buffers["out_tile_idx_to_mn_limit"],
                self.sort_buffers["out_num_non_exiting_tiles"],
                self.sorted_ids,
                self.r,
                16,
                256,
            )
        else:
            self.padded.fill_(self.r * tactic[0])
        self.gemm(self.x, self.w1, self.pre, tactic, False, self.k)
        activation[(triton.cdiv(self.r * self.i, 256),)](
            self.pre, self.mid, self.i, self.r, 256, self.gated
        )
        self.gemm(self.mid, self.w2, self.raw, tactic, True, 1)
        combine[(triton.cdiv(self.t * self.h, 256),)](
            self.raw, self.out, self.h, self.k, self.t, 256
        )
        return self.out

    def reference(self):
        raw = []
        for token in range(self.t):
            for slot in range(self.k):
                expert = int(self.ids[token, slot])
                pre = (
                    (self.w1[expert].double() @ self.x[token].double())
                    .to(self.x.dtype)
                    .float()
                )
                if self.gated:
                    up, gate = pre.chunk(2)
                    mid = (up * torch.nn.functional.silu(gate)).to(self.x.dtype)
                else:
                    mid = torch.relu(pre).square().to(self.x.dtype)
                partial = (
                    self.w2[expert].double() @ mid.double()
                ).float() * self.scores[token, slot]
                raw.append(partial.to(self.x.dtype))
        raw = torch.stack(raw).view(self.t, self.k, self.h)
        out = torch.zeros_like(self.x, dtype=torch.float32)
        for slot in range(self.k):
            out += raw[:, slot].float()
        return out.to(self.x.dtype)
