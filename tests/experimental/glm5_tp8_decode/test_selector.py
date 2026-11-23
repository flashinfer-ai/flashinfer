# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The library's long-context top-k selector, tested apart from score generation."""

import hashlib
from pathlib import Path

import pytest
import torch

from ._selector_reference import check_selection, make_scores
from .moe_reference import bitwise

pytestmark = pytest.mark.node_exclusive

_ADVERSARIAL = (
    ("fp32_collision", 1048580),
    ("negative_collision", 1048580),
    ("wide", 524289),
    ("signed_zero", 1048580),
    ("high_ties", 1048580),
    ("all_equal", 524292),
    ("group_spikes", 1052672),
    ("subnormal", 202761),
)
_SMALL = (4, 32, 2048, 2049, 202760, 202761, 262144, 524288)


def _module():
    from flashinfer.experimental.glm5_tp8_decode import jit
    from flashinfer.jit.core import gen_jit_spec, sm103a_nvcc_flags

    source = Path(__file__).with_name("selector_binding.cu")
    include = jit._CSRC / "attention"
    header = include / "long_topk.cuh"
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[:20]
    return gen_jit_spec(
        f"test_glm5_long_selector_sm103a_{digest}",
        [source],
        extra_include_paths=[include, *jit._header_dirs()],
        extra_cuda_cflags=[
            *sm103a_nvcc_flags,
            "--expt-relaxed-constexpr",
            "-ftz=false",
            "-prec-div=true",
            "-prec-sqrt=true",
        ],
        use_fast_math=False,
    ).build_and_load()


def test_library_selector_adversarial_keys_and_replay(collective, collective_watchdog):
    # Deterministically distribute sixteen cases once over the eight GPUs.
    # Each process calls the library selector with its own independent data.
    module = collective.checked(_module)
    collective.checked(module.initialize)

    def assigned_cases():
        device = collective.device
        capacity = 1052672 + 8
        scratch_storage = torch.full(
            (16 * capacity + 512,), 0x6D, dtype=torch.uint8, device=device
        )
        # The same scratch size the operator allocates, between guard bytes.
        scratch = scratch_storage[256:-256]
        output_storage = torch.full(
            (4 * 2048 + 128,), -17, dtype=torch.int32, device=device
        )
        selected = output_storage[64:-64].view(4, 2048)
        logits = torch.empty((4, capacity), dtype=torch.float32, device=device)
        position = torch.zeros(1, dtype=torch.int32, device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graphs = {}
        for grouped in (False, True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                module.run(logits, position, selected, scratch, grouped)
            graphs[grouped] = graph
            del graph
        cases = (_ADVERSARIAL[collective.rank], ("ascending", _SMALL[collective.rank]))
        for pattern, length in cases:
            for phase, past in enumerate((length - 4, max(0, length - 5), length - 4)):
                expected = make_scores(pattern, capacity, past, device, phase)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    logits.copy_(expected)
                    position.fill_(past)
                for grouped in (False, True):
                    with torch.cuda.stream(stream):
                        if phase == 0:
                            module.run(logits, position, selected, scratch, grouped)
                        else:
                            graphs[grouped].replay()
                    stream.synchronize()
                    check_selection(logits, selected, past)
                    assert bitwise(logits, expected), "selector modified input scores"
                    assert position.item() == past
                    assert bool((scratch_storage[:256] == 0x6D).all())
                    assert bool((scratch_storage[-256:] == 0x6D).all())
                    assert bool((output_storage[:64] == -17).all())
                    assert bool((output_storage[-64:] == -17).all())
        graphs.clear()

    collective.checked(assigned_cases)
