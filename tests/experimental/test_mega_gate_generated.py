# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""Routing values, physical mapping, held-out token counts, current-stream and changed-input replay."""

import re

import pytest
import torch
import torch.nn.functional as F
from flashinfer.experimental.deepgemm_mega_gate import mega_gate as _runtime
from flashinfer.mega_gate import prepare_mega_gate

K, E, TOPK = (
    _runtime.EXPORTED["K"],
    _runtime.EXPORTED["E"],
    _runtime.EXPORTED["num_topk"],
)
EXPORTED_ROWS = [
    (1, False, True),
    (3, False, True),
    (16, False, True),
    (128, False, True),
    (512, False, True),
    (1024, False, True),
    (2048, False, True),
    (4096, False, True),
    (8192, False, True),
    (16, True, True),
    (16, False, False),
    (1, False, False),
]
HELD_OUT_TOKEN_COUNTS = (2, 7, 33, 100, 400, 777, 3000, 5000)


def _skip_unless_exported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        _runtime.device_facts(torch.cuda.current_device())[0]
    except RuntimeError as error:
        pytest.skip(str(error))


def _mapping(physical):
    counts = torch.full((E,), 2, dtype=torch.int32, device="cuda") if physical else None
    mapping = (
        torch.stack(
            (
                torch.arange(E, device="cuda", dtype=torch.int32),
                torch.arange(E, device="cuda", dtype=torch.int32) + E,
            ),
            1,
        )
        if physical
        else None
    )
    return mapping, counts


def test_every_token_count_selects_an_exported_template():
    for sms in (148, 152):
        for deterministic in (False, True):
            for M in (*range(1, 4097), 8192, 16384, 65536, 1 << 20):
                index, launch_ctas, workers = _runtime.select_template(
                    M, sms, deterministic=deterministic
                )
                template = _runtime.TEMPLATES[index]
                assert template["program"] in _runtime.cake_jit.MODULES
                if template["static_tokens"]:
                    assert _runtime._tile_geometry(
                        template, template["static_tokens"]
                    ) == _runtime._tile_geometry(template, M)
                assert (
                    launch_ctas
                    % (
                        template["num_mma_ctas"]
                        * template["num_expert_groups"]
                        * template["num_split_k"]
                    )
                    == 0
                )
                assert workers >= 1 and launch_ctas >= 1
                assert (not deterministic) or template["num_split_k"] == 1
                if deterministic and M <= 16:
                    # The deterministic route of at most 16 tokens runs the paired-CTA 32-token block (one tile of
                    # 3 x 2 CTAs, worker stride 1; a template outside DeepGEMM's own selector).
                    assert (
                        template["block_tokens"],
                        template["num_mma_ctas"],
                        launch_ctas,
                        workers,
                    ) == (32, 2, 6, 1)
                if deterministic and M == 33:
                    # The held-out 33-token deterministic count keeps the single-CTA 16-token block.
                    assert (template["block_tokens"], template["num_mma_ctas"]) == (
                        16,
                        1,
                    )
    for M, deterministic, physical in EXPORTED_ROWS:
        unmapped = deterministic or not physical
        index_100, _, _ = _runtime.select_template(
            M,
            148,
            has_physical_map=physical,
            unmapped_output=unmapped,
            deterministic=deterministic,
        )
        index_103, _, _ = _runtime.select_template(
            M,
            152,
            has_physical_map=physical,
            unmapped_output=unmapped,
            deterministic=deterministic,
        )
        assert index_100 == index_103
        exact = _runtime.TEMPLATES[index_100]["static_tokens"]
        # The small physical-map production rows and the one-tile logical-output rows run their exact-shape
        # program (one 16-token tile serves M 2..16, one program per route); every other row runs the runtime program.
        if physical and not deterministic:
            static = {1: 1, 3: 16, 16: 16, 128: 128, 512: 512}
        elif not physical and not deterministic:
            static = {m: 16 for m in range(2, 17)}
        else:
            static = {}
        assert exact == static.get(M, 0)


@pytest.mark.parametrize("M,deterministic,physical", EXPORTED_ROWS)
def test_routing_values_stream_and_replay(M, deterministic, physical):
    _skip_unless_exported()
    x = torch.ones((M, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((E, K), dtype=torch.bfloat16, device="cuda")
    bias = torch.arange(E, dtype=torch.float32, device="cuda")
    mapping, counts = _mapping(physical)
    unmapped = (
        torch.empty((M, TOPK), dtype=torch.int64, device="cuda")
        if deterministic or not physical
        else None
    )
    ep_rank = 7 if deterministic else 0
    plan = prepare_mega_gate(
        x,
        weight,
        TOPK,
        bias=bias,
        to_physical_map=mapping,
        logical_count=counts,
        unmapped_topk_idx=unmapped,
        ep_rank=ep_rank,
        deterministic=deterministic,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for changed in (False, True, False):
        with torch.cuda.stream(stream):
            bias.copy_(
                torch.arange(E, device="cuda", dtype=torch.float32).flip(0)
                if changed
                else torch.arange(E, device="cuda", dtype=torch.float32)
            )
            plan.outputs[0].fill_(-1)
            plan.outputs[1].fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        logical = (
            torch.arange(TOPK, device="cuda", dtype=torch.int64)
            if changed
            else torch.arange(E - 1, E - TOPK - 1, -1, device="cuda")
        )
        expected = logical[None].expand(M, -1)
        if physical:
            duplicate = ((ep_rank + torch.arange(M, device="cuda") * 23333) % 2)[
                :, None
            ]
            expected = expected + duplicate * E
        assert torch.equal(plan.outputs[0], expected)
        torch.testing.assert_close(
            plan.outputs[1],
            torch.full_like(plan.outputs[1], 0.25),
            atol=1e-5,
            rtol=1e-5,
        )
        if unmapped is not None:
            assert torch.equal(unmapped, logical[None].expand(M, -1))


def _reference(x, weight, bias):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        logits = x.float() @ weight.float().T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    scores = F.softplus(logits).sqrt()
    return scores, scores + bias


def _check_routing(
    indices,
    weights,
    unmapped,
    *,
    scores,
    ranking,
    mapping,
    counts,
    ep_rank,
    tokens=None,
):
    M = scores.shape[0]
    if unmapped is not None:
        logical = unmapped
    elif mapping is not None:
        logical = torch.where(indices < E, indices, indices - E)
    else:
        logical = indices
    assert bool(((logical >= 0) & (logical < E)).all())
    selected = scores.gather(1, logical)
    expected_weights = selected / (selected.sum(1, keepdim=True) + 1e-20) * 1.5
    torch.testing.assert_close(weights, expected_weights, atol=1e-4, rtol=1e-4)
    chosen = ranking.gather(1, logical)
    omitted = ranking.scatter(1, logical, float("-inf")).amax(1)
    crossing = omitted - chosen.amin(1)
    assert bool((crossing <= 1e-5 + 1e-5 * ranking.abs().amax(1)).all())
    if mapping is not None:
        if tokens is None:
            tokens = torch.arange(M, device=indices.device, dtype=torch.int64)
        duplicate = (ep_rank + tokens[:, None] * 23333) % counts[logical].to(
            torch.int64
        )
        assert torch.equal(indices, mapping[logical, duplicate].to(torch.int64))
    else:
        assert torch.equal(indices, logical)


@pytest.mark.parametrize("M", HELD_OUT_TOKEN_COUNTS)
@pytest.mark.parametrize(
    "deterministic,physical",
    [(False, True), (True, True), (False, False)],
    ids=["physical", "deterministic_unmapped", "logical_unmapped"],
)
def test_held_out_token_counts_match_reference(M, deterministic, physical):
    _skip_unless_exported()
    generator = torch.Generator(device="cuda").manual_seed(1000 + M)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda", generator=generator)
    weight = torch.randn(
        (E, K), dtype=torch.bfloat16, device="cuda", generator=generator
    ).mul_(K**-0.5)
    bias = torch.randn(
        (E,), dtype=torch.float32, device="cuda", generator=generator
    ).mul_(0.1)
    mapping, counts = _mapping(physical)
    unmapped = (
        torch.full((M, TOPK), -2, dtype=torch.int64, device="cuda")
        if deterministic or not physical
        else None
    )
    ep_rank = 3
    plan = prepare_mega_gate(
        x,
        weight,
        TOPK,
        bias=bias,
        to_physical_map=mapping,
        logical_count=counts,
        unmapped_topk_idx=unmapped,
        ep_rank=ep_rank,
        deterministic=deterministic,
    )
    assert M not in {row[0] for row in EXPORTED_ROWS}
    indices, weights = plan.run()
    torch.cuda.synchronize()
    scores, ranking = _reference(x, weight, bias)
    _check_routing(
        indices,
        weights,
        unmapped,
        scores=scores,
        ranking=ranking,
        mapping=mapping,
        counts=counts,
        ep_rank=ep_rank,
    )
    if deterministic:
        snapshot = (indices.clone(), weights.clone(), unmapped.clone())
        for _ in range(3):
            plan.run()
        torch.cuda.synchronize()
        assert all(
            torch.equal(a, b)
            for a, b in zip((indices, weights, unmapped), snapshot, strict=True)
        )


MASK_ROWS = (1, 3, 16, 100, 256, 320, 384, 640, 960, 2048)


@pytest.mark.parametrize("enable_pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("physical", [True, False], ids=["physical", "logical"])
@pytest.mark.parametrize("M", MASK_ROWS)
def test_token_mask_and_image_bias_match_reference(M, physical, enable_pdl):
    _skip_unless_exported()
    generator = torch.Generator(device="cuda").manual_seed(2000 + M)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda", generator=generator)
    weight = torch.randn(
        (E, K), dtype=torch.bfloat16, device="cuda", generator=generator
    ).mul_(K**-0.5)
    bias = torch.randn(
        (E,), dtype=torch.float32, device="cuda", generator=generator
    ).mul_(0.1)
    # The image bias reorders the experts of an image token: a large shift
    # makes a ranking that ignores it fail the crossing check.
    image_bias = torch.randn(
        (E,), dtype=torch.float32, device="cuda", generator=generator
    ).mul_(2.0)
    mask = torch.rand((M,), device="cuda", generator=generator) > 0.25
    mask[0] = True
    image_token_mask = torch.rand((M,), device="cuda", generator=generator) > 0.5
    mapping, counts = _mapping(physical)
    unmapped = torch.full((M, TOPK), -2, dtype=torch.int64, device="cuda")
    ep_rank = 5
    plan = prepare_mega_gate(
        x,
        weight,
        TOPK,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        mask=mask,
        to_physical_map=mapping,
        logical_count=counts,
        unmapped_topk_idx=unmapped,
        ep_rank=ep_rank,
        enable_pdl=enable_pdl,
    )
    assert plan.route["route_flags"] & (
        _runtime.ROUTE_FLAG_TOKEN_MASK | _runtime.ROUTE_FLAG_IMAGE_BIAS
    ) == (_runtime.ROUTE_FLAG_TOKEN_MASK | _runtime.ROUTE_FLAG_IMAGE_BIAS)
    assert plan.route["enable_pdl"] is enable_pdl
    plan.outputs[0].fill_(-3)
    plan.outputs[1].fill_(float("nan"))
    indices, weights = plan.run()
    torch.cuda.synchronize()
    scores, ranking = _reference(x, weight, bias)
    ranking = torch.where(image_token_mask[:, None], scores + image_bias, ranking)
    masked = ~mask
    assert torch.equal(indices[masked], torch.full_like(indices[masked], -1))
    assert torch.equal(weights[masked], torch.zeros_like(weights[masked]))
    assert torch.equal(unmapped[masked], torch.full_like(unmapped[masked], -1))
    kept = mask.nonzero().squeeze(1)
    _check_routing(
        indices[kept],
        weights[kept],
        unmapped[kept],
        scores=scores[kept],
        ranking=ranking[kept],
        mapping=mapping,
        counts=None if counts is None else counts,
        ep_rank=ep_rank,
        tokens=kept,
    )


def test_image_bias_and_its_token_mask_come_together():
    _skip_unless_exported()
    x = torch.zeros((4, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((E, K), dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros((E,), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="together"):
        prepare_mega_gate(x, weight, TOPK, bias=bias, image_bias=bias.clone())
    with pytest.raises(ValueError, match="bool"):
        prepare_mega_gate(x, weight, TOPK, bias=bias, mask=torch.ones(4, device="cuda"))
    with pytest.raises(NotImplementedError):
        prepare_mega_gate(
            x,
            weight,
            TOPK,
            bias=bias,
            fix_routing_mask=torch.ones(4, dtype=torch.bool, device="cuda"),
        )


def test_decode_rows_select_deepgemm_blocks():
    """The 193..960-token rows run DeepGEMM's CTA pairing and split-K at its own
    block or the next exported one, at most one 16-token step larger, on 148 and
    152 SMs."""
    for sms in (148, 152):
        for M in range(193, 961):
            block, ctas, split = _runtime.deepgemm_config(M, K, E, sms)[:3]
            index, _, _ = _runtime.select_template(M, sms)
            template = _runtime.TEMPLATES[index]
            assert (template["num_mma_ctas"], template["num_split_k"]) == (
                ctas,
                split,
            ), (sms, M)
            assert block <= template["block_tokens"] <= block + 16, (sms, M)


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library; the manifest
    holds one entry per (library, architecture) with the package's sources, their
    digests and an empty nvcc flag list."""
    from pathlib import Path

    from flashinfer.experimental.deepgemm_mega_gate import cake_build, cake_jit

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == "flashinfer.experimental.deepgemm_mega_gate"
    assert set(manifest["arch_flags"]) == set(cake_jit.ARCHES)
    assert cake_jit.arch_for_capability((10, 7)) == "sm_107a"
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in cake_jit.MODULES.items():
        assert record["library"] == record["kernel"] == name
        assert re.fullmatch(r"cake_deepgemm_mega_gate_[0-9a-f]{20}", name)
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["ffi_entry"] == "run"
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] == "runtime"
        assert launch["dynamic_smem_bytes"] > 0
        plan = record["arg_plan"]
        assert plan[-4:] == [
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ]
        template = cake_jit.template(name)
        arches = (
            ["sm_100a"]
            if template["cluster_tokens"]
            else ["sm_100a", "sm_103a", "sm_107a"]
        )
        assert list(record["arches"]) == arches, name
        for arch in record["arches"]:
            library = cake_jit.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch and entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{name}_kernel.cu",
                f"csrc/{name}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert entry["nvcc_flags"] == [], library
    assert referenced == set(libraries)
    assert len(cake_jit.MODULES) == 23 and len(libraries) == 65
    assert (package / "DEEPGEMM_NOTICE.txt").is_file()
    for source in package.glob("csrc/*_kernel.cu"):
        head = source.read_text(encoding="utf-8").split("*/", 1)[0]
        assert "by FlashInfer team" in head and "DEEPGEMM_NOTICE.txt" in head, source
