"""Concrete model projection calls; numerical oracles are test-time only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch

from b12x.preparation import PreparationRequest, PreparedCall


@dataclass
class _Expected:
    source: torch.Tensor
    weight: object
    recipe: str
    bias: torch.Tensor | None = None


def _text(metadata):
    return metadata.get("text_config", metadata)


def projection_roles(metadata: Mapping) -> list[tuple[str, int, int, str]]:
    """(role, logical K, TP-local N, checkpoint recipe), not shape buckets."""
    c = _text(metadata)
    tp, h = int(metadata.get("_tp", 2)), int(c["hidden_size"])
    glm = "glm" in str(c.get("model_type", metadata.get("_model_id", ""))).lower()
    roles = []

    def add(name, k, n, recipe="mxfp8"):
        roles.append((name, int(k), int(n), recipe))

    if glm:
        heads, d = int(c["linear_num_heads"]), int(c["linear_head_dim"])
        p = heads * d // tp
        add("kda.in_proj_qkvgfab", h, 3 * p + heads // tp + d)
        add("kda.f_b_proj+g_b_proj", d, p)
        add("kda.g_a_proj", h, d)
        add("kda.o_proj", p, h)
        q, kv = int(c["q_lora_rank"]), int(c["kv_lora_rank"])
        nh = int(c["num_attention_heads"]) // tp
        add("mla+mtp.fused_qkv_a_proj", h, q + kv + int(c["qk_rope_head_dim"]))
        add("mla+mtp.q_b_proj", q, nh * int(c["qk_head_dim"]))
        add(
            "mla+mtp.kv_b_proj",
            kv,
            nh * (int(c["qk_nope_head_dim"]) + int(c["v_head_dim"])),
        )
        add("mla+mtp.o_proj", nh * int(c["v_head_dim"]), h)
        add("indexer+mtp.wq_b", q, int(c["index_n_heads"]) * int(c["index_head_dim"]))
        add("indexer+mtp.wk", h, int(c["index_head_dim"]))
        add("indexer+mtp.weights_proj", h, int(c["index_n_heads"]))
        shared = int(c["moe_intermediate_size"]) * int(c["n_shared_experts"]) // tp
        shared_name = "shared+mtp"
    else:
        kd = int(c["linear_num_key_heads"]) * int(c["linear_key_head_dim"])
        vd = int(c["linear_num_value_heads"]) * int(c["linear_value_head_dim"])
        add("linear.in_proj_qkvz", h, (2 * kd + 2 * vd) // tp)
        add("linear.in_proj_ba", h, 2 * int(c["linear_num_value_heads"]) // tp)
        add("linear.out_proj", vd // tp, h)
        nh, d = int(c["num_attention_heads"]) // tp, int(c["head_dim"])
        nk = max(1, int(c["num_key_value_heads"]) // tp)
        add("qsa.qkv_proj", h, (2 * nh + 2 * nk) * d)
        add("qsa.o_proj", nh * d, h)
        add(
            "qsa.index_qk_proj",
            h,
            (int(c["indexer_n_heads"]) + int(c["indexer_kv_heads"]))
            * int(c["indexer_head_dim"]),
        )
        add("main+mtp.router", h, int(c["num_experts"]), "bf16_gemv")
        add("main+mtp.shared_expert_gate", h, 1, "bf16_gemv")
        add(
            "mtp.index_qk_proj",
            h,
            (int(c["indexer_n_heads"]) + int(c["indexer_kv_heads"]))
            * int(c["indexer_head_dim"]),
            "bf16_gemv",
        )
        shared = int(c["shared_expert_intermediate_size"]) // tp
        shared_name = "shared"
    add(shared_name + ".gate_up_proj", h, 2 * shared)
    add(shared_name + ".down_proj", shared, h)
    # Native vocabulary GEMV is a single-row route, unlike the Torch large-M path.
    add("lm_head", h, ((int(c["vocab_size"]) + 63) // 64 * 64) // tp, "bf16_vocab")
    full = metadata.get("_full_config", metadata)
    vision = full.get("vision_config")
    if vision and not glm:
        vt = int(metadata.get("_vision_tp", tp))
        vh, vi = int(vision["hidden_size"]), int(vision["intermediate_size"])
        add("vision.attn.qkv", vh, 3 * vh // vt)
        add("vision.attn.proj", vh // vt, vh)
        add("vision.mlp.linear_fc1", vh, vi // vt)
        add("vision.mlp.linear_fc2", vi // vt, vh, "nvfp4_a16")
        merged = vh * int(vision["spatial_merge_size"]) ** 2
        add("vision.merger.linear_fc1", merged, merged // vt)
        add("vision.merger.linear_fc2", merged // vt, int(vision["out_hidden_size"]))
    return roles


def _prepare_weight(k: int, n: int, recipe: str, device: torch.device):
    """Build one immutable fixture using the checkpoint's native packing recipe."""
    from b12x.gemm.blockscaled import api

    if recipe in ("bf16_vocab", "bf16_gemv"):
        return torch.randn((n, k), device=device, dtype=torch.bfloat16).mul_(k**-0.5)
    padded = (k + 127) // 128 * 128
    source = torch.randn((n, padded), device=device, dtype=torch.bfloat16).mul_(k**-0.5)
    if padded != k:
        source[:, k:].zero_()
    if recipe == "nvfp4_a16":
        from b12x.quantization.nvfp4 import api as quant

        padded_n = (n + 127) // 128 * 128
        if padded_n != n:
            source = torch.nn.functional.pad(source, (0, 0, 0, padded_n - n))
        scale = torch.full((1,), 256.0, device=device)
        quant_plan = quant.plan(padded_n, padded)
        outputs = quant.allocate_outputs(quant_plan, device=device)
        quant.run(plan=quant_plan, x=source, global_scale=scale, outputs=outputs)
        weight = api.pack_weight(
            outputs.packed_a_storage.view(padded_n, padded // 2)[:n],
            outputs.scale_storage,
            recipe="nvfp4",
            global_scale=scale,
            global_scale_kind="reciprocal",
        )
        return replace(weight, in_features=k)
    from b12x._lib.intrinsics import as_grouped_scale_view_mx
    from b12x.quantization import mxfp8

    values = torch.empty((n, padded), device=device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((n, padded // 32), device=device, dtype=torch.float8_e8m0fnu)
    physical = torch.empty(
        ((n + 127) // 128 * (padded // 128) * 512,), device=device, dtype=torch.uint8
    )
    scale_mma = as_grouped_scale_view_mx(physical.view(1, -1), n, padded)
    quant_plan = mxfp8.plan(
        mxfp8.query_from_call(source, values, scales, scale_mma, expected_m=n)
    )
    mxfp8.quantize_rows(source, values, scales, scale_mma, plan=quant_plan)
    weight = api.pack_weight(values, scales, recipe="mxfp8")
    return replace(weight, in_features=k)


def _packed_query(k: int, n: int, m: int, recipe: str):
    from b12x.gemm.blockscaled import BlockscaledQuery

    return BlockscaledQuery(
        recipe="nvfp4" if recipe == "nvfp4_a16" else "mxfp8",
        num_tokens=m,
        in_features=k,
        padded_in_features=(k + 127) // 128 * 128,
        out_features=n,
        activation_mode="a16" if recipe == "nvfp4_a16" else "auto",
        activation_scale_available=False,
        global_scale_kind="reciprocal" if recipe == "nvfp4_a16" else "none",
        source_contiguous=True,
        source_aligned=True,
        output_mode="functional",
        workspace_form="owned",
        expected_m=m,
    )


def _packed_call(state, source: torch.Tensor, weight, recipe: str, bias):
    if recipe == "nvfp4_a16":
        result = state.run(
            source,
            weight.values,
            weight.scale_mma,
            weight.global_scale,
            activation_scale=None,
        )
    else:
        result = state.run(
            source,
            weight.weight.values,
            weight.weight.scale_mma,
            None,
            activation_scale=None,
        )
    return result if bias is None else result + bias


def make_benchmark_requests(
    metadata: Mapping, *, device: torch.device, rows: tuple[int, ...]
) -> list[PreparationRequest]:
    """Declare every real projection role with independently owned benchmark calls."""
    from b12x.gemm.bf16_gemv import api as gemv
    from b12x.gemm.bf16_vocab_projection import api as vocab
    from b12x.gemm.blockscaled import api as packed
    from vllm import _custom_ops as ops

    device = torch.device(device)
    eps = float(_text(metadata).get("rms_norm_eps", 1e-6))
    roles = projection_roles(metadata)
    # Fixture construction is input setup, never a candidate producer or timed path.
    weights = {
        key: _prepare_weight(*key, device)
        for key in dict.fromkeys((k, n, recipe) for _, k, n, recipe in roles)
    }
    requests: list[PreparationRequest] = []

    for role, k, n, recipe in roles:
        for m in rows:
            if recipe == "bf16_vocab" and m != 1:
                continue
            if recipe == "bf16_gemv" and m > 8:
                continue
            key = (k, n, recipe)
            if recipe == "bf16_gemv":
                query = gemv.GemvQuery(
                    source_dtype="bfloat16",
                    weight_dtype="bfloat16",
                    max_rows=m,
                    in_features=k,
                    out_features=n,
                    source_contiguous=True,
                    source_aligned=True,
                    weight_contiguous=True,
                    weight_aligned=True,
                )
                declaration = gemv.plan(query)
            elif recipe == "bf16_vocab":
                declaration = vocab.plan(
                    vocab.Caps(
                        device=device,
                        max_tokens=m,
                        in_features=k,
                        out_features=n,
                    )
                )
            else:
                declaration = packed.plan(_packed_query(k, n, m, recipe))

            def call_factory(
                state,
                *,
                key=key,
                m=m,
                k=k,
                recipe=recipe,
                role=role,
            ):
                weight = weights[key]
                seed = torch.randn((m, k), device=device, dtype=torch.bfloat16)
                source = torch.empty_like(seed)
                norm = torch.ones(k, device=device, dtype=torch.bfloat16)
                bias = (
                    torch.randn((weight.out_features,), device=device, dtype=torch.bfloat16).mul_(0.01)
                    if role.startswith("vision.")
                    else None
                )

                def produce():
                    ops.rms_norm(source, seed, norm, eps)

                if recipe == "bf16_gemv":
                    run = lambda: state.run(source, weight)
                elif recipe == "bf16_vocab":
                    run = lambda: state.run(source, weight)
                else:
                    run = lambda: _packed_call(state, source, weight, recipe, bias)
                return PreparedCall(
                    run=run,
                    produce=produce,
                    owners=(_Expected(source, weight, recipe, bias), seed, norm),
                )

            requests.append(
                declaration.request(
                    name=f"dense.{role}.m{m}",
                    prepare_call=call_factory,
                    benchmark_call=call_factory,
                    retain_benchmark_call=True,
                )
            )
    return requests


def test_expected(call: PreparedCall) -> torch.Tensor:
    context = next(owner for owner in call.owners if isinstance(owner, _Expected))
    if context.recipe in ("bf16_vocab", "bf16_gemv"):
        weight = context.weight.float()
    elif context.recipe == "mxfp8":
        packed = context.weight
        scale = (
            packed.weight.scale_rows.reshape(packed.out_features, -1)
            .float()
            .repeat_interleave(32, -1)
        )
        weight = packed.weight.values.float() * scale
        weight = weight[:, : context.source.shape[1]]
    else:
        packed = context.weight
        codes = packed.values
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=codes.device,
        )
        values = torch.stack(
            (lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1
        ).flatten(-2)
        n, k = values.shape
        physical = packed.scale_mma.permute(5, 2, 4, 0, 1, 3).contiguous().view(-1)
        r = torch.arange(n, device=codes.device)[:, None]
        g = torch.arange(k // 16, device=codes.device)[None, :]
        offsets = (
            (r // 128 * ((k // 16 + 3) // 4) + g // 4) * 512
            + r % 32 * 16
            + r // 32 % 4 * 4
            + g % 4
        )
        scales = physical.view(torch.float8_e4m3fn)[offsets].float()
        weight = values * scales.repeat_interleave(16, -1) / packed.global_scale
        weight = weight[:, : context.source.shape[1]]
    result = (context.source.float() @ weight.T).to(torch.bfloat16)
    return result if context.bias is None else result + context.bias
