# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM device kernels in a standalone low-token BF16 research harness.

This preserves native typed preactivation and pre-cast router weighting.
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


class VllmTritonComparator:
    def __init__(self, x, w1, w2, ids, scores, source, gated=True):
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
            raise ValueError("Only the upstream naive route policy is reproduced")
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
        self.tactics = [
            (16, n, k, stage)
            for n in (32, 64, 128)
            for k in (32, 64, 128)
            for stage in (2, 3, 4)
        ]

    def get_valid_tactics(self, *args):
        return self.tactics

    def gemm(self, a, b, c, tactic, mul, topk):
        m, n, k, stage = tactic
        em = self.r * m
        self._gemm[(self.r * triton.cdiv(b.shape[1], n),)](
            a,
            b,
            c,
            None,
            None,
            None,
            self.scores,
            None,
            self.ids,
            self.padded,
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
            naive_block_assignment=True,
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
        # The upstream low-token path fills this scalar per call; include it in graphs.
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
