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

"""Multi-GPU tests of :func:`flashinfer.fused_moe.tp_moe`.

Each (model, tp_size) case spawns ``tp_size`` ranks that build the layer from one synthetic checkpoint (seeded MXFP4 routed experts,
block-FP8 shared expert).  Per M:

* the gate entry against an fp32 reference of the whole layer (the reference gate's routing, x unquantized);
* every rank returns the same output, and a repeated call returns the same bits;
* the ids entry, fed the reference gate's ids / weights, agrees with the gate entry.

M = 65 (above the op's 64-token limit) must raise.

Once per case: a shared-dominant layer (all routed weights zero) against the fp32 shared expert, and for DeepSeek-V4.1 a batch with
image tokens, which select their experts with the vision bias.

Run: ``pytest tests/moe/test_tp_moe.py`` (needs 4 or 8 GPUs of one supported arch).
"""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

MS = (1, 8, 16, 33, 64)
IMAGE_TOKEN_ID = 128815
# fp32-reference tolerances: x and the activations are FP8-quantized inside the layer (Hopper: observed rel ~0.025).  B200
# (SM100) quantizes them to MXFP8, power-of-two UE8M0 scales per 32 elements as flashinfer's trtllm-gen MoE: observed rel
# 0.046 - 0.050, within 0.004 of the trtllm-gen MoE baseline
REL_TOL = {9: 0.05, 10: 0.06}
COS_TOL = 0.998


def _supported_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    caps = {
        torch.cuda.get_device_capability(d) for d in range(torch.cuda.device_count())
    }
    return len(caps) == 1 and caps.pop() in ((9, 0), (10, 0))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------------------------------------------- synthetic checkpoint
def _routed(E, H, I, dev, seed, zero=False):
    """MXFP4 checkpoint tensors: packed codes [E, 2I, H/2] / [E, H, I/2] (two e2m1 per byte), UE8M0 scales 2^-8 .. 2^-3"""
    g = torch.Generator(device=dev).manual_seed(seed)
    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g
    )
    s13 = torch.randint(
        119, 125, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g
    )
    w2 = torch.randint(
        0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=g
    )
    s2 = torch.randint(
        119, 125, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g
    )
    if zero:
        w13.zero_()
        w2.zero_()
    return w13, s13, w2, s2


def _block_fp8(w, blk):
    """[N, K] fp32 -> e4m3 [N, K] + fp32 power-of-two scales [N / blk, K / blk]"""
    n, k = w.shape
    v = w.view(n // blk, blk, k // blk, blk)
    s = torch.exp2(
        torch.ceil(torch.log2(v.abs().amax(dim=(1, 3)).clamp_min(1e-30) / 448.0))
    )
    q = (v / s[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
    return q.contiguous(), s.contiguous()


def _shared(H, Is, blk, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w13 = torch.randn(2 * Is, H, device=dev, generator=g) * 0.02
    w2 = torch.randn(H, Is, device=dev, generator=g) * 0.02
    return _block_fp8(w13, blk) + _block_fp8(w2, blk)


# ----------------------------------------------------------------------------------------------------------- fp32 reference
_E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _dequant_mxfp4(packed, scale):
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    return lut[codes] * torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=-1)


def _dequant_fp8(w, s):
    blk = w.shape[0] // s.shape[0]
    return w.float() * s.repeat_interleave(blk, 0).repeat_interleave(blk, 1)


def _swiglu(h):
    g, u = h.chunk(2, dim=-1)
    return F.silu(g.clamp(max=10.0)) * u.clamp(-10.0, 10.0)


def _gate_ref(logits, bias, top_k):
    """sqrtsoftplus scores, top-k of score + bias, weights = renormalized scores (bias: [E] or per token [M, E])"""
    s = torch.sqrt(F.softplus(logits.float()))
    ids = torch.topk(s + bias, top_k, dim=-1).indices
    w = s.gather(1, ids)
    return ids.to(torch.int32), (w / w.sum(-1, keepdim=True)).float()


def _layer_ref(x, ids, wts, factor, routed, shared):
    w13, s13, w2, s2 = routed
    xf = x.float()
    out = torch.zeros_like(xf)
    for e in torch.unique(ids).tolist():
        rows, slot = torch.nonzero(ids == e, as_tuple=True)
        y = (
            _swiglu(xf[rows] @ _dequant_mxfp4(w13[e], s13[e]).t())
            @ _dequant_mxfp4(w2[e], s2[e]).t()
        )
        out.index_add_(0, rows, y * wts[rows, slot, None])
    sw13, ss13, sw2, ss2 = shared
    return (
        out * factor
        + _swiglu(xf @ _dequant_fp8(sw13, ss13).t()) @ _dequant_fp8(sw2, ss2).t()
    )


def _close(out, ref, what):
    o, r = out.float(), ref.float()
    rel = float((o - r).norm() / r.norm())
    cos = float(F.cosine_similarity(o.flatten(), r.flatten(), dim=0))
    tol = REL_TOL[torch.cuda.get_device_capability(out.device)[0]]
    assert rel < tol and cos > COS_TOL, f"{what}: rel {rel:.4f} cos {cos:.5f}"


# ----------------------------------------------------------------------------------------------------------- worker
def _worker(rank, world, port, model):
    from flashinfer.fused_moe import tp_moe as T

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        device_id=dev,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        cfg = T.TpMoeConfig(model, world)
        sh = cfg.shape
        H, I, E, K = sh.hidden_size, sh.intermediate_size, sh.num_experts, sh.top_k
        blk = 128 if model == "dsv4_flash" else 32
        routed = _routed(E, H, I, dev, seed=1)
        shared = _shared(H, sh.shared_intermediate_size, blk, dev, seed=2)
        ws = T.TpMoeWorkspace(cfg, dist.group.WORLD, dev)
        weights = T.tp_moe_prepare_weights(cfg, rank, *routed, *shared, device=dev)
        g = torch.Generator(device=dev).manual_seed(3)
        bias = torch.randn(E, device=dev, generator=g) * 0.05
        vbias = torch.randn(E, device=dev, generator=g) * 0.05
        vision = model == "dsv41_flash"

        def gate_entry(x, logits, input_ids=None):
            kw = {}
            if vision:
                ii = (
                    input_ids
                    if input_ids is not None
                    else torch.zeros(x.shape[0], dtype=torch.int32, device=dev)
                )
                kw = dict(
                    gate_bias_vision=vbias, input_ids=ii, image_token_id=IMAGE_TOKEN_ID
                )
            return T.tp_moe(x, weights, ws, router_logits=logits, gate_bias=bias, **kw)

        for m in MS:
            x = (torch.randn(m, H, device=dev, generator=g) * 0.5).bfloat16()
            logits = torch.randn(m, E, device=dev, generator=g)
            out = gate_entry(x, logits)
            again = gate_entry(x, logits)
            assert torch.equal(out, again), (
                f"M {m}: repeated call not bitwise identical"
            )
            allo = [torch.empty_like(out) for _ in range(world)]
            dist.all_gather(allo, out)
            assert all(torch.equal(allo[0], o) for o in allo), f"M {m}: ranks differ"
            assert torch.isfinite(out.float()).all(), f"M {m}: non-finite output"
            ids, wts = _gate_ref(logits, bias, K)
            _close(
                out,
                _layer_ref(x, ids, wts, cfg.routed_scaling_factor, routed, shared),
                f"M {m} gate entry vs fp32 reference",
            )
            by_ids = T.tp_moe(x, weights, ws, topk_ids=ids, topk_weights=wts)
            rel = float((by_ids.float() - out.float()).norm() / out.float().norm())
            assert rel < 2e-3, f"M {m}: ids entry vs gate entry rel {rel:.2e}"

        # shared-dominant layer: zero routed weights -> the output is the shared expert alone
        w0 = T.tp_moe_prepare_weights(
            cfg, rank, *_routed(E, H, I, dev, seed=1, zero=True), *shared, device=dev
        )
        for m in (8, 64):
            x = (torch.randn(m, H, device=dev, generator=g) * 0.5).bfloat16()
            logits = torch.randn(m, E, device=dev, generator=g)
            kw = {}
            if vision:
                kw = dict(
                    gate_bias_vision=vbias,
                    input_ids=torch.zeros(m, dtype=torch.int32, device=dev),
                    image_token_id=IMAGE_TOKEN_ID,
                )
            out = T.tp_moe(x, w0, ws, router_logits=logits, gate_bias=bias, **kw)
            sw13, ss13, sw2, ss2 = shared
            ref = (
                _swiglu(x.float() @ _dequant_fp8(sw13, ss13).t())
                @ _dequant_fp8(sw2, ss2).t()
            )
            _close(out, ref, f"M {m} shared-dominant")

        # DeepSeek-V4.1 image tokens: every third token selects its experts with the vision bias
        if vision:
            for m in (8, 33, 64):
                x = (torch.randn(m, H, device=dev, generator=g) * 0.5).bfloat16()
                logits = torch.randn(m, E, device=dev, generator=g)
                input_ids = torch.randint(
                    0, 100000, (m,), dtype=torch.int32, device=dev, generator=g
                )
                img = torch.arange(m, device=dev) % 3 == 0
                input_ids[img] = IMAGE_TOKEN_ID
                out = gate_entry(x, logits, input_ids)
                ids, wts = _gate_ref(logits, torch.where(img[:, None], vbias, bias), K)
                ref = _layer_ref(x, ids, wts, cfg.routed_scaling_factor, routed, shared)
                _close(out, ref, f"M {m} image tokens vs fp32 reference")
                text_ids, _ = _gate_ref(logits, bias, K)
                assert not torch.equal(ids[img], text_ids[img]), (
                    "the vision bias did not change any image token's experts"
                )
                by_ids = T.tp_moe(x, weights, ws, topk_ids=ids, topk_weights=wts)
                rel = float((by_ids.float() - out.float()).norm() / out.float().norm())
                assert rel < 2e-3, (
                    f"M {m} image tokens: ids entry vs gate entry rel {rel:.2e}"
                )
        # the op serves at most 64 tokens per call: M = 65 is rejected before any launch
        x65 = torch.zeros(65, H, dtype=torch.bfloat16, device=dev)
        with pytest.raises(ValueError):
            gate_entry(x65, torch.zeros(65, E, device=dev))
        torch.cuda.synchronize()
        ws.destroy()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _supported_arch(), reason="tp_moe needs SM90a or SM100a GPUs")
@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize("model", ["dsv4_flash", "dsv41_flash"])
def test_tp_moe(model, tp_size):
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"needs {tp_size} GPUs")
    mp.spawn(_worker, args=(tp_size, _free_port(), model), nprocs=tp_size, join=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
