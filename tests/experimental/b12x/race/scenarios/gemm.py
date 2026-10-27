"""Dense GEMM race scenarios (b12x blockscaled and related engines)."""

from __future__ import annotations

import random

import torch

from ._base import Scenario, _generator, _rows_for_waves


def dense_blockscaled_mxfp8(device, case) -> Scenario:
    """b12x blockscaled MXFP8 GEMM at a fixed plan mode (quantized or a16).

    Defaults are GLM-5.3 Flash TP4's KDA input projection at 96 rows, the shape
    that exposed the stage-reuse race (local-inference-lab/flashinfer#16).
    """
    from b12x.gemm import blockscaled
    from b12x.gemm.blockscaled._a16 import _weight_parts
    from b12x.preparation import PreparationSession, PreparedCall

    n, k = case.get("n", 6288), case.get("k", 4096)
    mode = case.get("mode", "quantized")
    rows = case.get("rows") or _rows_for_waves(device, n, case.get("waves", 3))
    gen = _generator(case)
    weight = (torch.randn((n, k), generator=gen) * 64).clamp(-448, 448)
    weight = weight.to(device).to(torch.float8_e4m3fn)
    scales = torch.randint(122, 127, (n, k // 32), generator=gen, dtype=torch.int32)
    packed = blockscaled.pack_weight(weight, scales.to(torch.uint8).to(device))
    source = (torch.randn((rows, k), generator=gen) * 0.5).to(torch.bfloat16).to(device)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)

    query = blockscaled.query_from_call(source, packed, activation_mode=mode, out=out)
    plan = blockscaled.plan(query)
    values, scale_parts, gain, _ = _weight_parts(packed)

    def call(state):
        return PreparedCall(
            run=lambda: state.run(source, values, scale_parts, gain, out=out)
        )

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    session.__enter__()
    session.prepare((plan.request(name="race.dense_mxfp8", prepare_call=call),))
    session.freeze()

    def forward():
        blockscaled.mm(source, packed, out, plan=plan)

    return Scenario(
        forward=forward,
        outputs={"out": out},
        reset=lambda: out.fill_(float("nan")),
        close=lambda: session.__exit__(None, None, None),
        keepalive=[session, plan, packed, source],
        description=f"dense MXFP8 {mode} {rows}x{k}->{n}",
    )


FAMILIES = {
    "dense_blockscaled_mxfp8": dense_blockscaled_mxfp8,
}

CASES = {
    "dense_blockscaled_mxfp8": [
        # GLM-5.3 Flash TP4 KDA input/output and MLA qkv_a projections.
        {"n": 6288, "k": 4096, "mode": "quantized"},
        {"n": 4096, "k": 2048, "mode": "quantized"},
        {"n": 2208, "k": 4096, "mode": "quantized"},
        {"rows": 96, "n": 6288, "k": 4096, "mode": "quantized"},
        {"n": 4096, "k": 1536, "mode": "a16"},
        {"rows": 16, "n": 4096, "k": 1536, "mode": "a16"},
    ],
}


def _fuzz_dense_blockscaled_mxfp8(rng: random.Random) -> dict:
    return {
        "rows": rng.choice((17, 32, 64, 96, 128, 200, 256, 512, None)),
        "n": 128 * rng.randint(4, 64),
        "k": 128 * rng.randint(4, 40),
        "mode": rng.choice(("quantized", "a16")),
        "seed": rng.getrandbits(31),
    }


FUZZ = {
    "dense_blockscaled_mxfp8": _fuzz_dense_blockscaled_mxfp8,
}


# Fixed races put back by mutations.py: delete the fix's lines, then expect a scenario to fail.
MUTATIONS = {
    # local-inference-lab/flashinfer#16: TMA refill of a stage consumers may still be reading.
    "dense_gemm_stage_fence": {
        "module": "b12x._lib.dense_gemm",
        "replace": [("cute.arch.fence_view_async_shared()", "pass")],
        "family": "dense_blockscaled_mxfp8",
        "cases": [
            {"rows": 512, "n": 6288, "k": 4096, "mode": "quantized"},
            {"n": 6288, "k": 4096, "mode": "quantized"},
        ],
        "delay": False,
    },
}
