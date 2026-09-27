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

"""Run the experimental W4A8 received-token API on a 152-SM GB300.

    python -m flashinfer.experimental.cake_w4a8_received_tokens.cake_example

Weight preparation and the independent PyTorch reference are intentionally
outside the captured call. This example does not measure performance.
"""

import torch

from flashinfer import mxfp8_quantize
from flashinfer.fused_moe import prepare_fp4_block_scale_routed_moe
from flashinfer.quantization.fp4_quantization import mxfp4_quantize
from flashinfer.tllm_enums import SfLayout


def make_inputs(rows=384, pattern="mixed", device="cuda"):
    rng = torch.Generator(device=device).manual_seed(17)
    x = torch.randn((rows, 3072), dtype=torch.bfloat16, device=device, generator=rng)
    q, s = mxfp8_quantize(x, is_sf_swizzled_layout=False)
    inputs = [q.view(torch.float8_e4m3fn), s.view(torch.uint8).reshape(rows, 96)]
    for shape in ((32, 10240, 3072), (32, 3072, 5120)):
        w = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=rng)
        w.mul_(shape[-1] ** -0.5)
        qw, sw = mxfp4_quantize(w.flatten(0, 1), sfLayout=SfLayout.layout_linear)
        inputs.extend(
            (
                qw.view(torch.uint8).reshape(*shape[:-1], shape[-1] // 2),
                sw.view(torch.uint8).reshape(*shape[:-1], shape[-1] // 32),
            )
        )
        del w
    ids = torch.randint(512, (rows, 8), device=device, dtype=torch.int32, generator=rng)
    # Each received row has at least one expert on the destination rank.
    ids[:, 0] %= 32
    if pattern == "duplicates":
        ids[:, 1] = ids[:, 0]
    elif pattern == "all_hot":
        ids.fill_(0)
    elif pattern != "mixed":
        raise ValueError(pattern)
    scores = torch.randn((rows, 8), device=device, generator=rng).softmax(-1)
    return (*inputs, ids, scores)


def _scales(scale):
    return torch.exp2(scale.float() - 127).repeat_interleave(32, -1)


def _fp4(weight, scale):
    codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2).long()
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
        dtype=torch.float32,
    )
    return values[codes] * _scales(scale)


def _activation_round(x):
    """FP32 SwiGLU -> E8M0 RP / E4M3 RNE, including zero and subnormal blocks."""
    blocks = x.reshape(*x.shape[:-1], -1, 32)
    tiny = torch.finfo(torch.float32).tiny
    maximum = blocks.abs().amax(-1)
    maximum = torch.where(maximum < tiny, 0.0, maximum)
    normalized = torch.where(maximum == 0, 448.0, maximum) * (1.0 / 448.0)
    bits = normalized.contiguous().view(torch.int32)
    exponent, mantissa = (bits >> 23) & 255, bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    code = (exponent + bump.int()).clamp(0, 254)
    inverse = ((254 - code) << 23).contiguous().view(torch.float32)

    def ftz(value):
        return torch.where(
            value.abs() < tiny, torch.copysign(torch.zeros_like(value), value), value
        )

    scaled = ftz(ftz(blocks) * ftz(inverse.unsqueeze(-1)))
    q = scaled.clamp(-448, 448).flatten(-2).to(torch.float8_e4m3fn)
    return q.float() * _scales(code)


def reference(inputs, offset=0):
    x, xs, w1, s1, w2, s2, ids, scores = inputs
    x = x.float() * _scales(xs)
    parts = torch.zeros((*ids.shape, 3072), dtype=torch.float32, device=x.device)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for expert in range(32):
            token, slot = torch.where(ids == offset + expert)
            if token.numel() == 0:
                continue
            gate, up = (x[token] @ _fp4(w1[expert], s1[expert]).T).chunk(2, -1)
            activation = _activation_round(torch.nn.functional.silu(gate) * up)
            value = (activation @ _fp4(w2[expert], s2[expert]).T).bfloat16().float()
            parts[token, slot] = value * scores[token, slot, None]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    result = torch.zeros_like(x)
    for slot in range(8):
        result += parts[:, slot]
    return result.bfloat16()


def assert_correct(actual, expected):
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    norm = torch.linalg.vector_norm(expected.float())
    assert bool(error < 0.02 * norm) if bool(norm) else bool(error == 0)


def main():
    inputs = make_inputs()
    output = torch.empty((len(inputs[0]), 3072), dtype=torch.bfloat16, device="cuda")
    prepared = prepare_fp4_block_scale_routed_moe(
        *inputs, local_expert_offset=0, output=output, backend="cake"
    )
    prepared.run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.run()
    graph.replay()
    assert_correct(output, reference(inputs))
    print(f"Verified {output.shape[0]} received rows with CUDA Graph replay.")


if __name__ == "__main__":
    main()
