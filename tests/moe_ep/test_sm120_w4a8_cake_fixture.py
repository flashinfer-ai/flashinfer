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

"""Real-weight cross-check of the Cake SM120 hybrid MegaMoE backend (opt-in).

Set ``CAKE_SM120_MEGAMOE_FIXTURE_DIR`` to a released-checkpoint fixture directory
(``meta.json``, ``inputs.pt``, ``rank<r>_weights.pt``; schema
``megamoe-v4flash-real-weight-fixture-v1``, with the FP8 shared-expert planes)
whose rank count equals the launch world size, then run it in its own torchrun
session:

    CAKE_SM120_MEGAMOE_FIXTURE_DIR=/path/to/fixture torchrun --nproc_per_node=4 \
        -m pytest tests/moe_ep/test_sm120_w4a8_cake_fixture.py -m "gpu_4 and arch_sm120" -v

The fixture stores the weights in the *kernel* layout (gate/up rows interleaved
by 8, UE8M0 scales packed four K-blocks per word).  The test rebuilds the
canonical FlashInfer layout from it, checks that ``preprocess_weights``
reproduces the fixture bytes exactly (and the generator harness's scale-vector
operand), feeds the fixture's pre-quantized activations
(``quantize_input=False``) and real gate routing, and compares the kernel output
with the pure-torch oracle at the generator validators' limits (relative L2 <=
0.01, max-abs / reference-max <= 0.02).

The layer is built with the generator harness's row capacity (8192 tokens per rank;
``CAKE_SM120_MEGAMOE_FIXTURE_CAPACITY`` overrides it), i.e. the exact qualified device
build of the parametric source.

If ``CAKE_SM120_MEGAMOE_HARNESS_DUMP`` names a directory with the generator
harness's own ``final_output`` for the same fixture and row count
(``<variant>/rank<r>.pt``), the FlashInfer output must equal it bit for bit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tests.moe_ep.test_moe_ep_sm120_w4a8_cake_mega import (
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    TOP_K,
    decode_fp4,
    decode_fp8,
    init_distributed,
    reference_output,
    table_spec,
)

FIXTURE_ENV = "CAKE_SM120_MEGAMOE_FIXTURE_DIR"
DUMP_ENV = "CAKE_SM120_MEGAMOE_HARNESS_DUMP"
CAPACITY = int(os.environ.get("CAKE_SM120_MEGAMOE_FIXTURE_CAPACITY", "8192"))
REL_L2_MAX = 0.01
MAX_ABS_REFERENCE_FRACTION_MAX = 0.02


def _fixture_dir() -> Path:
    root = os.environ.get(FIXTURE_ENV)
    if not root:
        pytest.skip(f"set {FIXTURE_ENV} to the real-weight fixture directory")
    return Path(root)


def _canonical_from_kernel_layout(fx: dict, device: torch.device, local: int):
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_weights import (
        gate_up_interleave_index,
        unpack_scales_kblock_major,
    )

    index = gate_up_interleave_index(INTERMEDIATE).to(device)
    inverse = torch.empty_like(index)
    inverse[index] = torch.arange(index.numel(), device=device)

    def unpack(sf: torch.Tensor, experts: int, k_words: int, n: int) -> torch.Tensor:
        return unpack_scales_kblock_major(
            sf.reshape(experts * k_words, n, 4)
            .view(torch.uint32)
            .reshape(experts * k_words, n),
            experts=experts,
        )

    w1_fp4 = fx["w1_fp4"].to(device)
    out = {
        "w13": w1_fp4.index_select(1, inverse).contiguous(),
        "w2": fx["w2_fp4"].to(device).contiguous(),
        "w13_scale": unpack(
            fx["w1_sf"].to(device), local, HIDDEN // 128, 2 * INTERMEDIATE
        )
        .index_select(1, inverse)
        .contiguous(),
        "w2_scale": unpack(fx["w2_sf"].to(device), local, INTERMEDIATE // 128, HIDDEN),
    }
    if "shared_w1_fp8" in fx:
        out["shared_w13"] = (
            fx["shared_w1_fp8"].to(device).index_select(0, inverse).contiguous()
        )
        out["shared_w13_scale"] = (
            unpack(fx["shared_w1_sf8"].to(device), 1, HIDDEN // 128, 2 * INTERMEDIATE)[
                0
            ]
            .index_select(0, inverse)
            .contiguous()
        )
        out["shared_w2"] = fx["shared_w2_fp8"].to(device).contiguous()
        out["shared_w2_scale"] = unpack(
            fx["shared_w2_sf8"].to(device), 1, INTERMEDIATE // 128, HIDDEN
        )[0].contiguous()
    return out


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
@pytest.mark.parametrize(
    "variant",
    [
        v.strip()
        for v in os.environ.get(
            "CAKE_SM120_MEGAMOE_VARIANTS", "noshared,fp8shared"
        ).split(",")
        if v.strip()
    ],
)
def test_real_fixture_matches_oracle_and_harness(variant: str) -> None:
    root = _fixture_dir()
    rank, world_size = init_distributed()
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("schema") != "megamoe-v4flash-real-weight-fixture-v1":
        pytest.skip("unsupported fixture schema")
    if int(meta.get("ranks", 8)) != world_size:
        pytest.skip(
            f"fixture was built for EP{meta.get('ranks')}, launched EP{world_size}"
        )
    rows = int(os.environ.get("CAKE_SM120_MEGAMOE_FIXTURE_ROWS", "512"))
    shared = variant == "fp8shared"

    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        CakePrequantizedMoEWeightsWithSharedExpert,
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_staging import (
        dequantize_activations_k32,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_weights import (
        scale_vector_layout_inverse,
        transform_prequantized_weights,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    local = EXPERTS // world_size
    fx = torch.load(root / f"rank{rank}_weights.pt", map_location="cpu")
    if shared and "shared_w1_fp8" not in fx:
        pytest.skip("fixture carries no FP8 shared-expert planes")
    c = _canonical_from_kernel_layout(fx, device, local)
    shared_kwargs = (
        {
            k: c[k]
            for k in ("shared_w13", "shared_w2", "shared_w13_scale", "shared_w2_scale")
        }
        if shared
        else {}
    )

    # -- canonical -> kernel layout reproduces the generator harness's operands --
    transformed = transform_prequantized_weights(
        c["w13"],
        c["w2"],
        c["w13_scale"],
        c["w2_scale"],
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        **shared_kwargs,
    )
    assert torch.equal(transformed.w1_fp4, fx["w1_fp4"].to(device))
    assert torch.equal(transformed.w2_fp4, fx["w2_fp4"].to(device))
    w1_seg = scale_vector_layout_inverse(transformed.w1_sfb, columns=2 * INTERMEDIATE)
    w2_seg = scale_vector_layout_inverse(transformed.w2_sfb, columns=HIDDEN)
    kb1, kb2 = HIDDEN // 128, INTERMEDIATE // 128
    assert torch.equal(
        w1_seg[: local * kb1].view(torch.uint8).reshape(fx["w1_sf"].shape),
        fx["w1_sf"].to(device),
    )
    assert torch.equal(
        w2_seg[: local * kb2].view(torch.uint8).reshape(fx["w2_sf"].shape),
        fx["w2_sf"].to(device),
    )
    if shared:
        assert torch.equal(transformed.w1_b8, fx["shared_w1_fp8"].to(device))
        assert torch.equal(transformed.w2_b8, fx["shared_w2_fp8"].to(device))
        assert torch.equal(
            w1_seg[local * kb1 :].view(torch.uint8).reshape(fx["shared_w1_sf8"].shape),
            fx["shared_w1_sf8"].to(device),
        )
        assert torch.equal(
            w2_seg[local * kb2 :].view(torch.uint8).reshape(fx["shared_w2_sf8"].shape),
            fx["shared_w2_sf8"].to(device),
        )
    del transformed

    # -- fixture inputs (pre-quantized activations, real gate routing) --------
    inputs = torch.load(root / "inputs.pt", map_location="cpu")
    x_fp8 = inputs["x_fp8"][rank, :rows].to(device).contiguous()
    x_sf = inputs["x_sf"][rank, :rows].to(device).contiguous()
    topk_ids = inputs["topk_idx"][rank, :rows].to(torch.int64).to(device).contiguous()
    topk_weights = inputs["topk_weights"][rank, :rows].float().to(device).contiguous()
    routed = (
        c["w13"],
        c["w2"],
        c["w13_scale"].view(torch.float8_e8m0fnu),
        c["w2_scale"].view(torch.float8_e8m0fnu),
    )
    weights = (
        CakePrequantizedMoEWeightsWithSharedExpert(
            *routed,
            shared_w13=c["shared_w13"].view(torch.float8_e4m3fn),
            shared_w2=c["shared_w2"].view(torch.float8_e4m3fn),
            shared_w13_scale=c["shared_w13_scale"].view(torch.float8_e8m0fnu),
            shared_w2_scale=c["shared_w2_scale"].view(torch.float8_e8m0fnu),
        )
        if shared
        else MoEWeightPack(*routed)
    )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size, rank=rank, device=device.index
        ),
        fleet_params=FleetParams(
            num_experts=EXPERTS,
            max_tokens_per_rank=CAPACITY,
            token_hidden_size=HIDDEN,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=TOP_K,
                shared_expert="fp8_in_kernel" if shared else None,
                transport_table=table_spec(),
            ),
            quantize_input=False,
            preprocess_weights=True,
        ),
    )
    tensors = MoEEpTensors(
        hidden_states=x_fp8, topk_ids=topk_ids, topk_weights=topk_weights, scales=x_sf
    )
    layer.warmup(tensors)
    out = layer(tensors).clone()
    again = layer(tensors).clone()
    torch.cuda.synchronize()
    assert layer._kernel.protocol_error(layer._workspace) == 0
    assert torch.equal(out, again)
    layer.destroy()  # collective: release the transport session on every rank

    # -- bit-exact parity with the generator harness (optional) ---------------
    dump = os.environ.get(DUMP_ENV)
    if dump and (Path(dump) / variant / f"rank{rank}.pt").is_file():
        expected = torch.load(
            Path(dump) / variant / f"rank{rank}.pt", map_location="cpu"
        )
        assert int(expected["rows"]) == rows
        assert torch.equal(out.cpu(), expected["final_output"]), (
            "differs from the harness output"
        )

    # -- oracle over the global dequantized weights ---------------------------
    def gather(t: torch.Tensor) -> torch.Tensor:
        parts = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim=0)

    g_w13, g_w13_sf = gather(c["w13"]), gather(c["w13_scale"])
    g_w2, g_w2_sf = gather(c["w2"]), gather(c["w2_scale"])
    shared_dense = (
        (
            decode_fp8(c["shared_w13"], c["shared_w13_scale"]),
            decode_fp8(c["shared_w2"], c["shared_w2_scale"]),
        )
        if shared
        else None
    )
    reference = reference_output(
        dequantize_activations_k32(x_fp8, x_sf),
        topk_ids,
        topk_weights,
        lambda e: decode_fp4(g_w13[e], g_w13_sf[e]),
        lambda e: decode_fp4(g_w2[e], g_w2_sf[e]),
        shared_dense,
    )
    assert torch.isfinite(out).all()
    difference = (out.float() - reference.float()).abs()
    rel_l2 = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    )
    max_fraction = float(difference.max()) / float(
        reference.float().abs().max().clamp_min(1e-12)
    )
    assert rel_l2 <= REL_L2_MAX, f"relative L2 {rel_l2:.5f} > {REL_L2_MAX}"
    assert max_fraction <= MAX_ABS_REFERENCE_FRACTION_MAX, (
        f"max-abs / reference-max {max_fraction:.5f} > {MAX_ABS_REFERENCE_FRACTION_MAX}"
    )
    dist.barrier()
