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

"""Prepared rank-local W4A8 MoE with caller-owned output and reusable workspace."""

import torch

from .cake_jit import load


def _tile_n(tokens):
    return 16 if tokens < 224 else (64 if tokens >= 640 else 32)


def _pack_weights(w1, s1, w2, s2):
    e = 32
    gate, up = w1.chunk(2, dim=1)
    a = torch.stack((gate, up), dim=2).flatten(1, 2).contiguous()
    gate, up = s1.chunk(2, dim=1)
    sa = torch.stack((gate, up), dim=2).flatten(1, 2).contiguous()
    a = a.reshape(e, 320, 8, 4, 1536).permute(0, 1, 3, 2, 4).contiguous()
    sa = sa.reshape(e, 320, 8, 4, 96).permute(0, 1, 3, 2, 4).contiguous()
    sa = (
        sa.reshape(e, 80, 4, 4, 8, 12, 2, 4)
        .permute(0, 1, 5, 3, 6, 4, 2, 7)
        .contiguous()
        .reshape(e, -1, 32, 32)
    )
    a = (
        a.reshape(e, 80, 128, 12, 2, 64)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .reshape(e, 10240, 1536)
    )
    b = w2.reshape(e, 96, 8, 4, 2560).permute(0, 1, 3, 2, 4).contiguous()
    sb = s2.reshape(e, 96, 8, 4, 160).permute(0, 1, 3, 2, 4).contiguous()
    b = (
        b.reshape(e, 24, 128, 20, 2, 64)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .reshape(e, 3072, 2560)
    )
    sb = (
        sb.reshape(e, 24, 4, 4, 8, 20, 2, 4)
        .permute(0, 1, 5, 3, 6, 4, 2, 7)
        .contiguous()
        .reshape(e, -1, 32, 32)
    )
    return a, sa, b, sb


class PreparedRoutedMoE:
    def __init__(
        self,
        x,
        x_scale,
        w1,
        w1_scale,
        w2,
        w2_scale,
        ids,
        scores,
        *,
        local_expert_offset,
        out,
    ):
        tokens = x.shape[0]
        if not 1 <= tokens <= 8192:
            raise ValueError("received token count must be in 1..8192")
        expected = (
            (x, (tokens, 3072), torch.float8_e4m3fn),
            (w1, (32, 10240, 1536), torch.uint8),
            (w1_scale, (32, 10240, 96), torch.uint8),
            (w2, (32, 3072, 2560), torch.uint8),
            (w2_scale, (32, 3072, 160), torch.uint8),
            (ids, (tokens, 8), torch.int32),
            (scores, (tokens, 8), torch.float32),
            (out, (tokens, 3072), torch.bfloat16),
        )
        for tensor, shape, dtype in expected:
            if tensor.shape != shape or tensor.dtype != dtype:
                raise ValueError(f"expected shape {shape} and dtype {dtype}")
        for tensor in (x, x_scale, w1, w1_scale, w2, w2_scale, ids, scores, out):
            if (
                not tensor.is_cuda
                or tensor.device != x.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    "tensors must be contiguous, 16-byte aligned and on the same CUDA device"
                )
        if x_scale.dtype != torch.uint8 or x_scale.shape != (tokens, 96):
            raise ValueError(
                "x_scale must use the linear [received_tokens, H/32] layout"
            )
        if (
            type(local_expert_offset) is not int
            or local_expert_offset % 32
            or not 0 <= local_expert_offset <= 480
        ):
            raise ValueError(
                "local_expert_offset must select one of 16 contiguous 32-expert ranges"
            )
        out_begin, out_end = (
            out.data_ptr(),
            out.data_ptr() + out.numel() * out.element_size(),
        )
        for tensor in (x, x_scale, w1, w1_scale, w2, w2_scale, ids, scores):
            begin, end = (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            )
            if max(begin, out_begin) < min(end, out_end):
                raise ValueError("output must not overlap any input")
        props = torch.cuda.get_device_properties(x.device)
        if (props.major, props.minor, props.multi_processor_count) != (10, 3, 152):
            raise RuntimeError("this W4A8 MoE route requires a 152-SM GB300")
        self.inputs = dict(
            x=x,
            x_scale=x_scale,
            ids=ids,
            scores=scores,
            local_expert_offset=local_expert_offset,
        )
        self.output, self.tokens = out, tokens
        self.tile_n = _tile_n(tokens)
        self.padding_log2 = self.tile_n.bit_length() - 1
        self.capacity = ((tokens * 8 + 32 * (self.tile_n - 1) + 127) // 128) * 128
        self.tiles = self.capacity // self.tile_n
        with torch.cuda.device(x.device):
            self.modules = []
            for stage in (
                "routing",
                f"fc1_n{self.tile_n}",
                f"fc2_n{self.tile_n}",
                "finalize",
            ):
                wrapper, executor = load(stage)
                self.modules.append(wrapper(executor))
            self.w1, self.w1_scale, self.w2, self.w2_scale = _pack_weights(
                w1, w1_scale, w2, w2_scale
            )
            self.metadata = tuple(
                torch.empty(n, dtype=torch.int32, device=x.device)
                for n in (
                    tokens * 8,
                    self.capacity,
                    self.capacity,
                    self.tiles,
                    self.tiles,
                    1,
                    3,
                )
            )
            self.activation = torch.empty(
                (self.capacity, 5120), dtype=torch.uint8, device=x.device
            )
            self.activation_scale = torch.empty(
                self.capacity * 160, dtype=torch.uint8, device=x.device
            )
            self.expert_output = torch.empty(
                (self.capacity, 3072), dtype=torch.bfloat16, device=x.device
            )
        self._bind_stage_views()

    def _bind_stage_views(self):
        self.x_bytes = self.inputs["x"].view(torch.uint8)
        self.x_scale_words = self.inputs["x_scale"].view(torch.uint32)
        narrow = self.tile_n == 16
        self.fc1_output = (
            self.activation.view(torch.uint16) if narrow else self.activation
        )
        self.fc2_output = (
            self.expert_output.view(torch.uint64) if narrow else self.expert_output
        )
        self.fc2_scale = (
            self.activation_scale.view(torch.uint32)
            if narrow
            else self.activation_scale
        )
        self.fc1_grid = (152, 1, 1) if narrow else (80, self.tiles, 1)
        self.fc2_grid = (128, 1, 1) if narrow else (24, self.tiles, 1)

    def run(self):
        """Run on the bound device's current stream and return the local output."""
        with torch.cuda.device(self.output.device):
            routing, fc1, fc2, finalize = self.modules
            forward, tokens, inverse, experts, limits, size, nonexit = self.metadata
            routing.run(
                self.inputs["ids"],
                *self.metadata,
                self.tokens,
                self.inputs["local_expert_offset"],
                self.padding_log2,
                8,
                1,
                1,
            )
            fc1.run(
                self.w1,
                self.x_bytes,
                self.w1_scale,
                self.x_scale_words,
                self.fc1_output,
                self.activation_scale,
                tokens,
                experts,
                limits,
                nonexit,
                self.tokens,
                *self.fc1_grid,
            )
            fc2.run(
                self.w2,
                self.activation,
                self.w2_scale,
                self.fc2_scale,
                self.fc2_output,
                experts,
                limits,
                nonexit,
                *self.fc2_grid,
            )
            finalize.run(
                self.expert_output,
                self.inputs["scores"],
                forward,
                self.output,
                self.tokens * 2,
                1,
                1,
            )
        return self.output
