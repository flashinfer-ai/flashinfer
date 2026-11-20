"""
Copyright (c) 2024 by FlashInfer team.

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

import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Barrier

import pytest
import torch

import flashinfer
from flashinfer import _top_p_renorm_specialized
from flashinfer.utils import device_support_pdl
from tests.infra.test_helpers.parametrize import (
    parametrize_product,
    pairwise_product_cases,
)
from tests.utils.test_helpers.pdl import late_store
from tests.utils.test_helpers.sampling import (
    required_num_collision_rows,
    required_num_trials,
)


# H100 CI peak GPU memory: 39.5 GiB.
pytestmark = pytest.mark.gpu_exclusive


@pytest.mark.parametrize("use_default_generator", [False, True])
def test_get_seed_and_offset_threaded_reservations(use_default_generator):
    device = torch.device("cuda", torch.cuda.current_device())
    increments = (1, 4, 5, 13)
    calls_per_thread = 256
    start = Barrier(len(increments))

    # Preserve the process-default generator for the remaining sampling tests.
    with torch.random.fork_rng(devices=[device.index]):
        generator = (
            torch.cuda.default_generators[device.index]
            if use_default_generator
            else torch.Generator(device=device)
        )
        generator.manual_seed(1234)
        generator.set_offset(128)

        def reserve(increment):
            start.wait(timeout=10)
            ranges = []
            for _ in range(calls_per_thread):
                seed, offset = flashinfer.sampling.get_seed_and_offset(
                    increment,
                    None if use_default_generator else generator,
                    device,
                )
                assert seed == 1234
                ranges.append((offset, (increment + 3) // 4 * 4))
            return ranges

        with ThreadPoolExecutor(max_workers=len(increments)) as executor:
            ranges = [
                item
                for group in executor.map(reserve, increments, timeout=30)
                for item in group
            ]

        # Every reservation owns one disjoint range, with no lost advancement.
        next_offset = 128
        for offset, length in sorted(ranges):
            assert offset == next_offset
            next_offset += length
        assert generator.get_offset() == next_offset


def normal_distribution(std):
    def normal_noise(shape, device):
        return torch.randn(shape, device=device) * std

    normal_noise.__name__ = f"normal_distribution(std={std})"
    return normal_noise


def gumbel_distribution(beta):
    def gumbel_noise(shape, device):
        U = torch.rand(shape, device=device)
        eps = 1e-20
        return torch.log(-torch.log(U + eps) + eps) / beta

    gumbel_noise.__name__ = f"gumbel_distribution(beta={beta})"
    return gumbel_noise


@pytest.mark.parametrize("vocab_size", [4095, 4096, 4097, 16384, 16387])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("deterministic", [False, True])
def test_sampling_last_valid_fallback(vocab_size, indices_dtype, deterministic):
    # Force the rounding fallback: the total mass is below any Philox uniform.
    # The last nonzero can be in an earlier tile, or in the final partial vector.
    last = torch.tensor([0, 31, 1023, 2047, 4094, vocab_size - 1], device="cuda")
    probs = torch.zeros(8, vocab_size, device="cuda")
    probs[:6, 0] = 1e-20
    probs[torch.arange(6, device="cuda"), last] = 1e-20
    probs[7] = float("nan")
    indices = torch.tensor(
        [5, 0, 6, 4, 2, 7, 3, 1, 5], device="cuda", dtype=indices_dtype
    )
    expected = torch.cat([last, torch.zeros(2, device="cuda", dtype=last.dtype)])
    expected_valid = torch.arange(8, device="cuda") < 6

    samples, valid = flashinfer.sampling.sampling_from_probs(
        probs,
        indices=indices,
        deterministic=deterministic,
        seed=42,
        offset=0,
        return_valid=True,
    )
    torch.testing.assert_close(
        samples, expected[indices].to(indices_dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(valid, expected_valid[indices])


@pytest.mark.parametrize("vocab_size", [4095, 4096, 4097, 16387])
@pytest.mark.parametrize(
    "sampling_type,mass",
    [(name, 1.0) for name in ["plain", "top_k", "top_p", "min_p", "joint"]]
    + [(name, 1e-20) for name in ["plain", "top_k", "top_p", "joint"]],
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_sampling_masked_tiles_graph(vocab_size, sampling_type, mass, deterministic):
    # Tiny mass forces fallback; min-p scales its uniform by the eligible mass,
    # so its one-hot input only covers normal selection.
    # k=2 prevents a bad zero-probability fallback from being hidden by a retry.
    probs = torch.zeros(4, vocab_size, device="cuda")
    positions = torch.tensor([0, 1023, 2047, vocab_size - 1], device="cuda")
    rows = torch.arange(4, device="cuda")
    probs[rows, positions] = mass
    ops = {
        "plain": (flashinfer.sampling.sampling_from_probs, {}),
        "top_k": (flashinfer.sampling.top_k_sampling_from_probs, {"top_k": 2}),
        "top_p": (flashinfer.sampling.top_p_sampling_from_probs, {"top_p": 0.9}),
        "min_p": (flashinfer.sampling.min_p_sampling_from_probs, {"min_p": 0.1}),
        "joint": (
            flashinfer.sampling.top_k_top_p_sampling_from_probs,
            {"top_k": 2, "top_p": 0.9, "filter_apply_order": "joint"},
        ),
    }
    op, kwargs = ops[sampling_type]

    def sample():
        return op(probs, deterministic=deterministic, seed=42, offset=0, **kwargs)

    torch.testing.assert_close(sample(), positions.to(torch.int32), atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        samples = sample()
    graph.replay()
    torch.testing.assert_close(samples, positions.to(torch.int32), atol=0, rtol=0)

    # A later replay must not retain the previous call's local fallback candidate.
    probs.zero_()
    positions = vocab_size - 1 - positions
    probs[rows, positions] = mass
    graph.replay()
    torch.testing.assert_close(samples, positions.to(torch.int32), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(1, 4096), (5, 129280), (230, 129280)])
@pytest.mark.parametrize("per_row_temperature", [False, True])
def test_softmax_low_temperature_normalization(shape, per_row_temperature):
    logits = torch.full(shape, -20.0, device="cuda:0")
    logits[:, 123] = 42.0
    temperature = 0.001
    if per_row_temperature:
        temperature = torch.full((shape[0],), temperature, device=logits.device)
    probs = flashinfer.sampling.softmax(logits, temperature=temperature)
    expected = torch.zeros_like(logits)
    expected[:, 123] = 1.0
    torch.testing.assert_close(probs, expected, atol=1e-6, rtol=0)


def test_softmax_rejects_cpu_before_cuda_state(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("CUDA workspace or capability state was accessed for CPU logits")

    monkeypatch.setattr(flashinfer.sampling, "_get_cache_buf", fail_if_called)
    monkeypatch.setattr(flashinfer.sampling, "device_support_pdl", fail_if_called)
    monkeypatch.setattr(
        flashinfer.sampling, "_supports_blackwell_softmax", fail_if_called
    )

    with pytest.raises(ValueError, match="logits must be a CUDA tensor"):
        flashinfer.sampling.softmax(torch.randn(2, 3))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_validate_softmax_temperature_accepts_floating_row_tensor(dtype):
    logits = torch.empty((3, 5), dtype=torch.float32)
    temperature = torch.ones(3, dtype=dtype)

    temperature_arr, temperature_val, temperature_is_none = (
        flashinfer.sampling._validate_softmax_temperature(logits, temperature)
    )

    assert temperature_arr is temperature
    assert temperature_val == 0.0
    assert not temperature_is_none


@pytest.mark.parametrize(
    "temperature,expected_value,is_none",
    [(None, 1.0, True), (1, 1.0, False), (0.5, 0.5, False)],
)
def test_validate_softmax_temperature_accepts_scalar(
    temperature, expected_value, is_none
):
    logits = torch.empty((3, 5), dtype=torch.float32)

    temperature_arr, temperature_val, temperature_is_none = (
        flashinfer.sampling._validate_softmax_temperature(logits, temperature)
    )

    assert temperature_arr is None
    assert temperature_val == expected_value
    assert temperature_is_none is is_none


@pytest.mark.parametrize(
    "temperature,error_match",
    [
        (torch.tensor(1.0), "must be 1D"),
        (torch.ones(3, 1), "must be 1D"),
        (torch.ones(2), "length must equal logits.size\\(0\\)"),
        (torch.ones(3, dtype=torch.int32), "must have dtype"),
        (torch.ones(3, dtype=torch.bool), "must have dtype"),
    ],
)
def test_validate_softmax_temperature_rejects_invalid_tensor(temperature, error_match):
    logits = torch.empty((3, 5), dtype=torch.float32)

    with pytest.raises(ValueError, match=error_match):
        flashinfer.sampling._validate_softmax_temperature(logits, temperature)


def test_validate_softmax_temperature_rejects_other_device():
    logits = torch.empty((3, 5), dtype=torch.float32)
    temperature = torch.ones(3, device="meta")

    with pytest.raises(ValueError, match="same device as logits"):
        flashinfer.sampling._validate_softmax_temperature(logits, temperature)


@pytest.mark.parametrize("temperature", [True, "1.0", object()])
def test_validate_softmax_temperature_rejects_non_numeric_scalar(temperature):
    logits = torch.empty((3, 5), dtype=torch.float32)

    with pytest.raises(TypeError, match="temperature must be None, a real scalar"):
        flashinfer.sampling._validate_softmax_temperature(logits, temperature)


@pytest.mark.parametrize("rows", [128, 256, 384, 512, 1024])
def test_blackwell_32k_rows_take_cached_cluster_route(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    logits = torch.randn((rows, 32000), device="cuda", dtype=torch.float32)
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=None, enable_pdl=False
    )
    actual = flashinfer.sampling.softmax(logits, temperature=None, enable_pdl=False)
    expected = torch.softmax(logits, dim=-1)

    assert route == flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER
    assert actual.data_ptr() != logits.data_ptr()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_blackwell_scalar_one_pdl_b64_v32000_takes_cached_cluster_route():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    logits = torch.randn((64, 32000), device="cuda", dtype=torch.float32)
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=1.0, enable_pdl=True
    )
    actual = flashinfer.sampling.softmax(logits, temperature=1.0, enable_pdl=True)
    expected = torch.softmax(logits, dim=-1)

    assert route == flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER
    assert actual.data_ptr() != logits.data_ptr()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rows,vocab_size,temperature,enable_pdl,expected_route",
    [
        (1, 111, 0.5, False, 1),
        (256, 32004, None, False, 2),
        (1, 32004, None, False, 3),
        (256, 32000, None, False, 5),
        (512, 32000, None, False, 5),
        (256, 32000, 0.5, False, 5),
        (256, 64000, None, False, 5),
        (1, 32000, None, False, 5),
        (1, 256000, None, False, 5),
        (48, 256000, None, False, 6),
        (100, 128000, 0.5, True, 6),
        (512, 256000, None, False, 7),
        (1024, 128000, 0.5, True, 7),
    ],
)
def test_blackwell_softmax_all_negative_infinity_matches_public_semantics(
    rows, vocab_size, temperature, enable_pdl, expected_route
):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    logits = torch.full((rows, vocab_size), -torch.inf, device="cuda")
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    actual = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    expected = torch.softmax(logits, dim=-1)

    assert route == expected_route
    assert actual.data_ptr() != logits.data_ptr()
    assert torch.isnan(expected).all()
    assert torch.isnan(actual).all()


# One shape per Blackwell Softmax route: (rows, vocab_size, temperature,
# enable_pdl, route).
_BLACKWELL_SOFTMAX_ROUTE_SHAPES = [
    (1, 111, 0.5, False, 1),
    (64, 256, 0.5, False, 1),
    (256, 32004, None, False, 2),
    (16, 16380, None, False, 2),
    # An odd vocabulary misaligns rows: unaligned head and tail elements.
    (200, 32003, 0.5, False, 2),
    (1, 32004, None, False, 3),
    # One tile per row: a single split per row.
    (200, 250, None, False, 3),
    (256, 32000, None, False, 5),
    (1, 256000, None, False, 5),
    (48, 256000, None, False, 6),
    (100, 128000, 0.5, True, 6),
    (512, 256000, None, False, 7),
]


def _blackwell_softmax_route_and_output(logits, temperature, enable_pdl):
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    actual = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    return route, actual


@pytest.mark.parametrize(
    "rows,vocab_size,temperature,enable_pdl,expected_route",
    _BLACKWELL_SOFTMAX_ROUTE_SHAPES,
)
@pytest.mark.parametrize("offset", [1.0e7, 3.0e9, -1.0e10, 1.0e30])
@pytest.mark.parametrize("scale", [1.0, 1.0e6])
def test_blackwell_softmax_large_magnitude_logits_match_torch(
    rows, vocab_size, temperature, enable_pdl, expected_route, offset, scale
):
    # Row maxima from 2^30 on take the exact exponent shift m; every route
    # must still match torch.
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    torch.manual_seed(11)
    logits = torch.randn(rows, vocab_size, device="cuda") * scale + offset
    route, actual = _blackwell_softmax_route_and_output(logits, temperature, enable_pdl)
    expected = torch.softmax(
        logits if temperature is None else logits / temperature, dim=-1
    )

    assert route == expected_route
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-3)
    significant = expected > 1e-6
    torch.testing.assert_close(
        actual[significant], expected[significant], atol=0.0, rtol=1e-5
    )
    torch.testing.assert_close(
        actual.sum(-1), torch.ones(rows, device="cuda"), atol=1e-5, rtol=0.0
    )


_FLOAT32_1E38 = float(torch.tensor(1.0e38, dtype=torch.float32))


@pytest.mark.parametrize(
    "rows,vocab_size,temperature,enable_pdl,expected_route",
    _BLACKWELL_SOFTMAX_ROUTE_SHAPES,
)
@pytest.mark.parametrize(
    "low,high,fill",
    [
        # Maxima of magnitude below 2^30 and from 2^30 on.
        (2.0**30 - 64, 2.0**30, 0.0),
        (-(2.0**30) - 128, -(2.0**30) + 64, -1.0e10),
        # Maxima near and past 2^127: one twice the other, adjacent floats,
        # and both past 2^127 with a fill whose difference overflows.
        (_FLOAT32_1E38, 2 * _FLOAT32_1E38, 0.0),
        (-2 * _FLOAT32_1E38, -_FLOAT32_1E38, -3.0e38),
        (2.0**127 - 2.0**103, 2.0**127, 0.0),
        (3.0e38, torch.finfo(torch.float32).max, -3.0e38),
    ],
)
def test_blackwell_softmax_extreme_maxima_match_torch(
    rows, vocab_size, temperature, enable_pdl, expected_route, low, high, fill
):
    # Each row holds `low` before and after three copies of `high`. Where a
    # thread's chunk spans 2048 columns, thread 0 sees `low` at column 0 and
    # then `high` at column 2048 (one tile, group or stride later) while its
    # warp neighbour keeps `low` from column 8, so thread, warp, cluster and
    # output rescales go from `low` to `high`. The result is 1/3 at `high`
    # and 0 elsewhere.
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    values = torch.full((rows, vocab_size), fill, device="cuda")
    low_cols = [0, 8, vocab_size // 3, vocab_size - 2]
    high_cols = [min(2048, vocab_size - 3), vocab_size // 2, vocab_size - 1]
    values[:, low_cols] = low
    values[:, high_cols] = high
    # Powers of two keep the scaled logits exact.
    logits = values if temperature is None else values * temperature
    route, actual = _blackwell_softmax_route_and_output(logits, temperature, enable_pdl)
    expected = torch.zeros_like(values)
    expected[:, high_cols] = 1.0 / 3.0

    assert route == expected_route
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        actual, torch.softmax(values, dim=-1), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize(
    "rows,vocab_size,enable_pdl",
    [(64, 256, False), (256, 32000, False), (100, 128000, True), (512, 256000, False)],
)
def test_blackwell_softmax_low_temperature_huge_logits_match_torch(
    rows, vocab_size, enable_pdl
):
    # A temperature of 0.005 takes logits near 1e36 past 2^127.
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    torch.manual_seed(17)
    logits = torch.rand(rows, vocab_size, device="cuda") * 1.0e36
    logits[:, vocab_size // 2] = 1.5e36
    actual = flashinfer.sampling.softmax(
        logits, temperature=0.005, enable_pdl=enable_pdl
    )
    expected = torch.softmax(logits / 0.005, dim=-1)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_blackwell_route_observer_falls_back_off_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("This test exercises the non-Blackwell observer gate")

    logits = torch.empty((1, 111), device="cuda")
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=0.5, enable_pdl=False
    )

    assert route == flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_FALLBACK


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    "rows,vocab_size,temperature_kind",
    [
        (4, 151552, "scalar"),  # a speculative-decoding verify step's rows
        (37, 262144, "per_row"),  # the widest cached row: 16 CTAs x 8 tiles
        (512, 128000, "none"),
        (8, 4096, "per_row"),
    ],
)
def test_blackwell_softmax_on_sm107_takes_blackwell_routes(
    rows, vocab_size, temperature_kind, enable_pdl
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("This test exercises the SM107 gate")

    generator = torch.Generator(device="cuda").manual_seed(rows * vocab_size)
    logits = torch.randn(rows, vocab_size, device="cuda", generator=generator)
    if temperature_kind == "none":
        temperature, scale = None, torch.ones(rows, 1, device="cuda")
    elif temperature_kind == "scalar":
        temperature, scale = 0.7, torch.full((rows, 1), 0.7, device="cuda")
    else:
        temperature = torch.rand(rows, device="cuda", generator=generator) + 0.25
        scale = temperature[:, None]
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    actual = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    expected = torch.softmax(logits / scale, dim=-1)

    assert route != flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_FALLBACK
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rows,vocab_size,temperature_kind,enable_pdl,expected_route",
    [
        (1, 32000, "none", False, "row"),
        (1, 262144, "none", False, "row"),  # widest cached row: 16 CTAs x 8 tiles
        (8, 4096, "scalar", False, "row"),
        (64, 32000, "scalar", True, "row"),
        (8, 256000, "none", False, "row"),  # 8-tile rows in one wave
        (16, 256000, "none", False, "shared_tiles"),  # under 4/5 of persistent trips
        (48, 151936, "scalar", False, "shared_tiles"),  # 5-tile chunks
        (48, 256000, "per_row", True, "persistent"),  # persistent 16-CTA clusters
        (100, 128000, "none", False, "persistent"),  # persistent 8-CTA clusters
        (64, 128000, "none", False, "persistent"),  # 4/5 of the persistent trips
        (200, 256000, "per_row", True, "persistent"),  # < 8 rows per shared cluster
        (96, 100008, "none", False, "persistent"),  # a partial last tile in every row
        (1024, 128000, "none", False, "shared_tiles"),  # 7+ rows per C8 shared cluster
        (1024, 256000, "per_row", True, "shared_tiles"),
        (989, 128256, "per_row", True, "shared_tiles"),  # rows not a multiple of 8
        (600, 100008, "scalar", False, "shared_tiles"),  # the last tile unreached
    ],
)
def test_blackwell_cached_cluster_route_is_selected_and_correct(
    rows, vocab_size, temperature_kind, enable_pdl, expected_route
):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    torch.manual_seed(7)
    logits = torch.randn(rows, vocab_size, device="cuda")
    if temperature_kind == "none":
        temperature = None
        probs_ref = torch.softmax(logits, dim=-1)
    elif temperature_kind == "scalar":
        temperature = 0.5
        probs_ref = torch.softmax(logits / temperature, dim=-1)
    else:
        temperature = torch.rand(rows, device="cuda") * 0.9 + 0.1
        probs_ref = torch.softmax(logits / temperature[:, None], dim=-1)

    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )

    routes = {
        "row": flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER,
        "persistent": flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_PERSISTENT,
        "shared_tiles": flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_SHARED_TILES,
    }
    assert route == routes[expected_route]
    # Only sm_103a launches with PDL: the row-caching and shared-tile kernels
    # through their _pdl twins, the persistent kernels through a flag.
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    on_sm103 = torch.cuda.get_device_capability() == (10, 3)
    pdl_twin = enable_pdl and on_sm103 and expected_route != "persistent"
    assert kernel[2] == int(pdl_twin)
    assert probs.data_ptr() != logits.data_ptr()
    torch.testing.assert_close(probs, probs_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rows,vocab_size",
    [
        (240, 256000),  # 16-CTA persistent clusters, about 18 rows each
        (480, 128000),  # 8-CTA persistent clusters, about 13 rows each
    ],
)
def test_blackwell_persistent_softmax_every_launch_reads_its_own_rows(rows, vocab_size):
    # The persistent route keeps clusters resident and walks each one over many
    # rows; a shared-memory tile slot takes the next row's bulk copy once every
    # thread has read it. Every launch must match the softmax of its own rows.
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    torch.manual_seed(7)
    logits = torch.randn(rows, vocab_size, device="cuda")
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=None, enable_pdl=False
    )
    assert (
        route == flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_PERSISTENT
    )
    probs_ref = torch.softmax(logits.double(), dim=-1).float()
    launches = 4000
    bad = torch.zeros((), dtype=torch.int64, device="cuda")
    for _ in range(launches):
        probs = flashinfer.sampling.softmax(logits, temperature=None, enable_pdl=False)
        bad += ~((probs - probs_ref).abs() <= 1e-8).all()
    assert bad.item() == 0, f"{bad.item()} of {launches} launches differ"


@pytest.mark.parametrize(
    "rows,vocab_size",
    [
        (31, 32000),  # B200: as many rows as resident 16-CTA clusters
        (32, 32000),  # B200: fewer resident 16-CTA clusters than rows
        (33, 32000),  # GB300: as many rows as resident 16-CTA clusters
        (34, 32000),  # B200 and GB300
        (36, 64000),
        (71, 24576),  # B200: as many rows as resident 8-CTA clusters
        (72, 24576),  # B200: fewer resident 8-CTA clusters than rows
        (73, 24576),
        (74, 24576),  # GB300: as many rows as resident 8-CTA clusters
        (75, 24576),  # GB300: fewer resident 8-CTA clusters than rows
    ],
)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_blackwell_cached_cluster_spread_keeps_rows_resident(
    rows, vocab_size, enable_pdl
):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    torch.manual_seed(7)
    logits = torch.randn(rows, vocab_size, device="cuda")
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=0.5, enable_pdl=enable_pdl
    )
    probs = flashinfer.sampling.softmax(logits, temperature=0.5, enable_pdl=enable_pdl)

    _, max_tiles, pdl_twin, resident_clusters, next_resident_clusters, _ = kernel
    on_sm103 = torch.cuda.get_device_capability() == (10, 3)
    # Spreading a row over more CTAs stops before a cluster size whose
    # resident clusters are fewer than the rows, so all rows run in one wave,
    # and not earlier: twice the CTAs would leave rows for a second wave.
    assert max_tiles == 4
    assert rows <= resident_clusters
    assert rows > next_resident_clusters
    assert pdl_twin == int(enable_pdl and on_sm103)
    torch.testing.assert_close(
        probs, torch.softmax(logits / 0.5, dim=-1), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize(
    "rows,vocab_size,cluster_ctas",
    [
        (36, 80000, 16),  # 5-tile 8-CTA chunks
        (36, 128000, 16),  # 8-tile 8-CTA chunks
        (73, 32000, 8),  # 4-tile 4-CTA chunks; B200: past the resident 8-CTA clusters
    ],
)
def test_blackwell_cached_cluster_spread_of_wide_chunks_ignores_residency(
    rows, vocab_size, cluster_ctas
):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    torch.manual_seed(7)
    logits = torch.randn(rows, vocab_size, device="cuda")
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=0.5
    )
    probs = flashinfer.sampling.softmax(logits, temperature=0.5)

    # Rows whose chunks are wider than the capped spreads allow keep spreading
    # even when some clusters then run in a second wave, as measured faster.
    assert kernel[:2] == (cluster_ctas, 4)
    torch.testing.assert_close(
        probs, torch.softmax(logits / 0.5, dim=-1), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize(
    "vocab_size",
    [
        151936,  # 7-tile 12-CTA chunks, 5-tile 16-CTA chunks
        196608,  # 8-tile 12-CTA chunks, 6-tile 16-CTA chunks
        204800,  # 9-tile 12-CTA chunks, 10-tile 10-CTA chunks
        229376,  # 10-tile 12-CTA chunks, 12-tile 10-CTA chunks
        262144,  # 11-tile 12-CTA chunks, 13-tile 10-CTA chunks
    ],
)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_blackwell_cached_cluster_one_row_over_takes_wide_kernel(
    vocab_size, enable_pdl
):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    torch.manual_seed(7)
    kernels = {}
    for rows in range(4, 13):
        logits = torch.randn(rows, vocab_size, device="cuda")
        kernels[rows] = (
            flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
                logits, temperature=0.5, enable_pdl=enable_pdl
            )
        )
    # A batch with one row more than 16-CTA clusters fit one CTA per SM takes
    # the wide kernel of 12 CTAs, or of 10 CTAs when the 16-CTA chunk exceeds
    # 6 tiles, with the narrowest register chunk that covers the row; one row
    # fewer or more keeps 16 CTAs.
    wide = [rows for rows, kernel in kernels.items() if kernel[0] in (10, 12)]
    assert len(wide) <= 1
    sixteen_chunk = -(-vocab_size // 16)
    sixteen_tiles = -(-sixteen_chunk // 2048)
    if torch.cuda.get_device_properties(0).multi_processor_count in (148, 152):
        # Known behavior: B200, GB200 and GB300 fit seven 16-CTA clusters
        # one CTA per SM. GB200 and GB300 fit eight 12-CTA clusters; B200
        # fits only seven, but fits eight 10-CTA clusters for chunks > 6 tiles.
        # For a new device/topology, verify its routing and expand this block
        # rather than accepting multiple routes and hiding a slower kernel.
        if "GB200" in torch.cuda.get_device_name(0):  # noqa: SIM114
            assert wide == [8]
        elif torch.cuda.get_device_capability() == (10, 3) or sixteen_tiles > 6:
            assert wide == [8]
        else:
            assert wide == []
    for rows, kernel in kernels.items():
        if rows not in wide:
            assert kernel[:2] == (16, 8)
    for rows in wide:
        cluster_ctas, max_tiles = kernels[rows][:2]
        assert cluster_ctas == 12 or sixteen_tiles > 6
        chunk = -(-vocab_size // cluster_ctas)
        tiles = -(-chunk // 2048)
        narrowest = (
            8 if tiles <= 8 else 10 if tiles <= 10 else {12: 11, 10: 13}[cluster_ctas]
        )
        assert max_tiles == narrowest
        assert rows <= kernels[rows][3]
        logits = torch.randn(rows, vocab_size, device="cuda")
        probs = flashinfer.sampling.softmax(
            logits, temperature=0.5, enable_pdl=enable_pdl
        )
        torch.testing.assert_close(
            probs, torch.softmax(logits / 0.5, dim=-1), atol=1e-3, rtol=1e-3
        )


def _check_cached_cluster_route(rows, vocab_size, sm100_route, sm103_route):
    torch.manual_seed(7)
    logits = torch.randn(rows, vocab_size, device="cuda")
    capability = torch.cuda.get_device_capability()
    expected = sm100_route if capability == (10, 0) else sm103_route
    routes = {
        "row": flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER,
        "persistent": flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_PERSISTENT,
        "shared_tiles": flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_SHARED_TILES,
    }
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=0.5
    )
    assert route == routes[expected]
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=0.5
    )
    if expected == "row":
        assert kernel[:2] == (16, 4)
    probs = flashinfer.sampling.softmax(logits, temperature=0.5)
    torch.testing.assert_close(
        probs, torch.softmax(logits / 0.5, dim=-1), atol=1e-3, rtol=1e-3
    )


def _skip_unless_b200_or_gb300():
    capability = torch.cuda.get_device_capability()
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if (capability, sm_count) not in (((10, 0), 148), ((10, 3), 152)):
        pytest.skip("These expectations are stated for B200 and GB300")


@pytest.mark.parametrize(
    "rows,vocab_size,sm100_route,sm103_route",
    [
        # 8-tile 8-CTA chunks: the 4-tile 16-CTA kernel in two waves.
        (40, 128000, "row", "row"),
        # 7-tile 8-CTA chunks: two waves on sm_100, the shared-tile kernel on sm_103.
        (40, 100000, "row", "shared_tiles"),
        # 7-tile chunks past one wave of shared-tile clusters: the persistent kernel.
        (84, 100000, "persistent", "persistent"),
        # 6-tile chunks: the trip rule on sm_100; on sm_103 under 2.5 rows per
        # persistent cluster, except from 20/9 rows per persistent cluster on.
        (79, 90000, "persistent", "shared_tiles"),
        (80, 90000, "persistent", "persistent"),
        # 7-tile 16-CTA chunks: the trip rule on sm_100, under 2.5 rows per persistent cluster on sm_103.
        (24, 212992, "persistent", "shared_tiles"),
        # 5-tile 8-CTA chunks: the persistent kernel from three persistent trips on
        # when the fifth tile is at least 3/8 full, otherwise from 2.5 rows per
        # persistent cluster on (sm_100).
        (256, 80000, "persistent", "persistent"),
        # ... whose fifth tile is more than 1/4 (sm_100) or at least 3/8 (sm_103) full.
        (160, 70000, "persistent", "shared_tiles"),
        (128, 68000, "shared_tiles", "shared_tiles"),
        (320, 69632, "shared_tiles", "shared_tiles"),
        (160, 71680, "persistent", "persistent"),
        # 5-tile 16-CTA chunks filling 9/10 of three persistent trips.
        (40, 151936, "persistent", "persistent"),
        # Full 16-CTA chunks: 10 (sm_100) or 8 (sm_103) rows per shared-tile cluster.
        (256, 256000, "persistent", "persistent"),
        (352, 256000, "shared_tiles", "shared_tiles"),
    ],
)
def test_blackwell_cached_cluster_route_rules_per_arch(
    rows, vocab_size, sm100_route, sm103_route
):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")
    _skip_unless_b200_or_gb300()
    _check_cached_cluster_route(rows, vocab_size, sm100_route, sm103_route)


@pytest.mark.parametrize(
    "rows,vocab_size,sm100_route,sm103_route",
    [
        # 8-tile 8-CTA chunks: the 4-tile 16-CTA kernel in two waves up to 9/5
        # (sm_100) of its resident clusters; on sm_103 past 7/10 of the resident
        # 8-CTA shared-tile clusters up to 5/3.
        (51, 128000, "row", "shared_tiles"),
        (52, 128000, "row", "row"),
        (56, 128000, "persistent", "persistent"),
        # 6-tile 8-CTA chunks: two waves up to 10/7 on sm_100; on sm_103 past 3/4
        # of the resident 8-CTA shared-tile clusters up to 7/4.
        (44, 90000, "row", "shared_tiles"),
        (45, 90000, "shared_tiles", "shared_tiles"),
        (55, 90000, "persistent", "shared_tiles"),
        (56, 90000, "persistent", "row"),
        (57, 90000, "persistent", "row"),
        (58, 90000, "persistent", "shared_tiles"),
        # 7-tile 8-CTA chunks: the trip rule.
        (48, 100000, "shared_tiles", "shared_tiles"),
        (52, 100000, "persistent", "shared_tiles"),
        # 8-tile 8-CTA chunks: the trip rule.
        (68, 128000, "shared_tiles", "persistent"),
        # 6-tile chunks: the trip rule (sm_100), under 2.5 rows per persistent
        # cluster (sm_103) unless the rows reach 20/9 rows per persistent cluster.
        (64, 90000, "persistent", "shared_tiles"),
        (68, 90000, "shared_tiles", "shared_tiles"),
        (73, 90000, "persistent", "shared_tiles"),
        (89, 90000, "persistent", "persistent"),
        (90, 90000, "persistent", "persistent"),
        (79, 84000, "persistent", "shared_tiles"),
        (80, 84000, "persistent", "persistent"),
        # 5-tile 8-CTA chunks whose rows fill at least 4/5 of the persistent trips:
        # the persistent kernel from three persistent trips on when the fifth tile
        # is at least 3/8 full (otherwise from 2.5 rows per persistent cluster on
        # sm_100), below 7 rows per shared-tile cluster.
        (57, 80000, "shared_tiles", "shared_tiles"),
        (58, 80000, "shared_tiles", "shared_tiles"),
        (62, 70000, "shared_tiles", "shared_tiles"),
        (66, 80000, "shared_tiles", "shared_tiles"),
        (70, 80000, "shared_tiles", "shared_tiles"),
        (80, 80000, "persistent", "shared_tiles"),
        (86, 80000, "persistent", "shared_tiles"),
        (88, 80000, "persistent", "persistent"),
        (496, 80000, "persistent", "persistent"),
        (497, 80000, "shared_tiles", "persistent"),
        (544, 80000, "shared_tiles", "shared_tiles"),
        # On sm_100 also at three persistent trips from 10/9 rows per shared-tile
        # cluster on, and below 15/2 rows per shared-tile cluster when the fifth
        # tile is at most 279/1024 full.
        (78, 70000, "shared_tiles", "shared_tiles"),
        (79, 70000, "persistent", "shared_tiles"),
        (82, 70000, "persistent", "shared_tiles"),
        (78, 76000, "shared_tiles", "shared_tiles"),
        (79, 76000, "persistent", "shared_tiles"),
        (496, 70000, "persistent", "shared_tiles"),
        (497, 70000, "persistent", "shared_tiles"),
        (532, 70000, "persistent", "shared_tiles"),
        (533, 70000, "shared_tiles", "shared_tiles"),
        (497, 70008, "shared_tiles", "shared_tiles"),
        # 6- and 7-tile 16-CTA chunks on sm_100: the persistent kernel past 4 rows
        # per shared-tile cluster up to 20 (6 tiles) or 22 (7 tiles) persistent
        # trips, or up to 24 trips when the last tile holds at most 6/8 (6 tiles)
        # or 4/8 (7 tiles) of a tile.
        (123, 180000, "persistent", "persistent"),
        (124, 180000, "persistent", "persistent"),
        (280, 180000, "persistent", "shared_tiles"),
        (281, 180000, "persistent", "shared_tiles"),
        (336, 180000, "persistent", "shared_tiles"),
        (337, 180000, "shared_tiles", "shared_tiles"),
        (281, 188416, "persistent", "shared_tiles"),
        (281, 188432, "shared_tiles", "shared_tiles"),
        (336, 192000, "shared_tiles", "shared_tiles"),
        (280, 196608, "persistent", "shared_tiles"),
        (281, 196608, "shared_tiles", "shared_tiles"),
        (124, 210000, "persistent", "persistent"),
        (308, 210000, "persistent", "shared_tiles"),
        (309, 210000, "persistent", "shared_tiles"),
        (336, 210000, "persistent", "shared_tiles"),
        (337, 210000, "shared_tiles", "shared_tiles"),
        (309, 212992, "persistent", "shared_tiles"),
        (309, 213008, "shared_tiles", "shared_tiles"),
        (336, 224000, "shared_tiles", "shared_tiles"),
        (308, 229376, "persistent", "shared_tiles"),
        (309, 229376, "shared_tiles", "shared_tiles"),
        # 5-tile 16-CTA chunks of four persistent trips keep the shared-tile kernel.
        (48, 151936, "shared_tiles", "shared_tiles"),
        (56, 163840, "shared_tiles", "shared_tiles"),
        # Full 16-CTA chunks: 10 (sm_100) or 8 (sm_103) rows per shared-tile cluster.
        (263, 256000, "persistent", "persistent"),
        (264, 256000, "persistent", "shared_tiles"),
        (309, 256000, "persistent", "shared_tiles"),
        (310, 256000, "shared_tiles", "shared_tiles"),
        (320, 256000, "shared_tiles", "shared_tiles"),
        (336, 256000, "shared_tiles", "shared_tiles"),
    ],
)
def test_blackwell_cached_cluster_route_boundaries_per_arch(
    rows, vocab_size, sm100_route, sm103_route
):
    # These rows sit at the boundaries set by the resident clusters of each part.
    _skip_unless_b200_or_gb300()
    _check_cached_cluster_route(rows, vocab_size, sm100_route, sm103_route)


@pytest.mark.parametrize(
    "rows,vocab_size", [(281, 180000), (336, 180000), (309, 210000), (336, 210000)]
)
@pytest.mark.parametrize("temperature_kind", ["scalar", "scalar_pdl", "per_row"])
def test_blackwell_cached_cluster_partial_tile_trip_cap_per_arch(
    rows, vocab_size, temperature_kind
):
    # On sm_100, 6- and 7-tile 16-CTA chunks whose last tile holds at most 6/8
    # (6 tiles) or 4/8 (7 tiles) of a tile keep the persistent kernel up to 24
    # persistent trips, unless the temperature is per row; sm_103 takes the
    # shared-tile kernel.
    _skip_unless_b200_or_gb300()
    torch.manual_seed(7)
    logits = torch.randn(rows, vocab_size, device="cuda")
    enable_pdl = temperature_kind == "scalar_pdl"
    if temperature_kind == "per_row":
        temperature = torch.rand(rows, device="cuda") * 0.9 + 0.1
        probs_ref = torch.softmax(logits / temperature[:, None], dim=-1)
    else:
        temperature = 0.5
        probs_ref = torch.softmax(logits / temperature, dim=-1)
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    persistent = (
        torch.cuda.get_device_capability() == (10, 0) and temperature_kind != "per_row"
    )
    assert route == (
        flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_PERSISTENT
        if persistent
        else flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_SHARED_TILES
    )
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    torch.testing.assert_close(probs, probs_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "vocab_size,ten_cta_tiles",
    [
        (131080, 8),  # 5-tile 16-CTA chunks, 7-tile 10-CTA chunks
        (163840, 8),  # 8-tile 10-CTA chunks
        (180000, 10),  # 9-tile 10-CTA chunks
        (204800, 10),  # 7-tile 16-CTA chunks, 10-tile 10-CTA chunks
        (204808, 0),  # 11-tile 10-CTA chunks keep 16 CTAs
    ],
)
@pytest.mark.parametrize("rows", [21, 22, 23, 24, 28, 29])
@pytest.mark.parametrize("temperature_kind", ["scalar", "scalar_pdl", "per_row"])
def test_blackwell_cached_cluster_ten_cta_rows_per_arch(
    vocab_size, ten_cta_tiles, rows, temperature_kind
):
    # On sm_103 these batches take the 10-CTA kernel from 11/14 of its resident
    # clusters up to all of them (22 to 28 rows on GB300), unless the
    # temperature is per row; on sm_100 they keep 16 CTAs.
    _skip_unless_b200_or_gb300()
    torch.manual_seed(7)
    logits = torch.randn(rows, vocab_size, device="cuda")
    enable_pdl = temperature_kind == "scalar_pdl"
    if temperature_kind == "per_row":
        temperature = torch.rand(rows, device="cuda") * 0.9 + 0.1
        probs_ref = torch.softmax(logits / temperature[:, None], dim=-1)
    else:
        temperature = 0.5
        probs_ref = torch.softmax(logits / temperature, dim=-1)
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    ten = (
        torch.cuda.get_device_capability() == (10, 3)
        and temperature_kind != "per_row"
        and ten_cta_tiles > 0
        and 22 <= rows <= 28
    )
    assert kernel[:2] == ((10, ten_cta_tiles) if ten else (16, 8))
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    torch.testing.assert_close(probs, probs_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rows,vocab_size,temperature_kind,sm100_chunk,sm103_chunk",
    [
        # 4-tile 16-CTA row kernel past one wave or with a cut of more than 1/16.
        (8, 100000, "none", 6272, 6272),
        (8, 100000, "scalar", 6272, 6272),
        (28, 48000, "none", 3008, 3008),
        (55, 120000, "none", 7520, 7520),
        # 4-tile 4- and 8-CTA row kernels with a cut of more than 1/16 (142
        # and 71 resident clusters on B200, 148 and 74 on GB300): 4-tile chunks
        # of 4 CTAs within five waves on sm_100 and in one wave on sm_103 (8
        # CTAs keep whole tiles).
        (96, 24584, "none", 6176, 6176),
        (128, 28672, "scalar", 7168, 7168),
        (512, 24584, "none", 6176, 8192),
        (710, 24584, "none", 6176, 8192),
        (711, 24584, "none", 8192, 8192),
        (72, 24584, "none", 4096, 4096),
        (74, 28672, "scalar_pdl", 4096, 4096),
        (71, 28672, "none", 4096, 4096),
        (75, 24584, "none", 6176, 4096),
        # 8-CTA shared-tile and persistent kernels (on sm_103 the persistent
        # kernel only with a cut of at least 1/10).
        (45, 81928, "none", 10272, 10272),
        (320, 69632, "none", 8704, 8704),
        (82, 70000, "none", 8768, 8768),
        (52, 100000, "scalar_pdl", 12512, 12512),
        (370, 66000, "none", 8256, 8256),  # five waves of shared-tile clusters (GB300)
        (180, 71680, "none", 8960, 8960),  # five persistent trips (GB300)
        # On sm_100 also past five waves, up to seven 8-CTA waves or trips.
        (371, 66000, "none", 8256, 10240),
        (181, 71680, "none", 8960, 10240),
        (48, 76000, "none", 9504, 9504),
        (82, 76000, "none", 9504, 9504),
        (50, 110000, "none", 14336, 13760),
        (52, 90000, "scalar_pdl", 11264, 11264),
        (89, 73728, "none", 9216, 9216),  # a cut of exactly 1/10
        (128, 100000, "none", 12512, 12512),
        (96, 100000, "scalar_pdl", 12512, 12512),
        (87, 76000, "none", 9504, 10240),
        (85, 90000, "none", 11264, 12288),
        (54, 105000, "none", 13152, 14336),
        (64, 120000, "none", 15008, 16384),
        # 16-CTA shared-tile and persistent kernels with a cut of at least 1/10,
        # or on sm_100 past one wave.
        (30, 140000, "none", 8768, 8768),
        (70, 201088, "none", 12576, 12576),
        (32, 151936, "none", 9504, 10240),
        (40, 151936, "none", 9504, 10240),
        (64, 180000, "none", 11264, 12288),
        # Whole tiles: the wide row kernel, a one-wave 16-CTA chunk with a cut
        # under 1/10, a 4-tile row kernel with a cut of at most 1/16 in one
        # wave, the 4-tile row kernels of 4 and 8 CTAs with other chunk tile
        # counts, the 4-tile row kernels of 1 and 2 CTAs, the 8-tile 16-CTA row
        # kernel, rows past five waves (on sm_100 past seven for the 8-CTA
        # kernels) or past one wave with a cut of at most 1/16.
        (8, 256000, "none", 26624, 22528),
        (24, 151936, "none", 10240, 16384),
        (100, 30720, "none", 8192, 8192),
        (96, 32000, "none", 8192, 8192),
        (1, 7168, "none", 2048, 2048),
        (200, 20480, "scalar", 6144, 6144),
        (64, 14848, "scalar_pdl", 2048, 2048),
        (100, 50304, "none", 8192, 8192),
        (200, 10240, "none", 6144, 6144),
        (300, 7168, "none", 8192, 4096),
        (741, 24584, "none", 8192, 8192),
        (371, 50304, "none", 8192, 8192),
        (8, 126464, "none", 8192, 8192),
        (1, 151936, "none", 10240, 10240),
        (232, 71680, "none", 10240, 10240),
        (498, 66000, "none", 10240, 10240),
        (76, 92000, "none", 12288, 12288),
    ],
)
def test_blackwell_cached_cluster_balanced_chunk_per_arch(
    rows, vocab_size, temperature_kind, sm100_chunk, sm103_chunk
):
    # These kernels cover ceil(V / C) rounded up to 32 elements per CTA instead
    # of whole tiles; the expected chunk depends on the GPU's resident clusters.
    _skip_unless_b200_or_gb300()
    torch.manual_seed(11)
    logits = torch.randn(rows, vocab_size, device="cuda")
    temperature = None if temperature_kind == "none" else 0.7
    enable_pdl = temperature_kind == "scalar_pdl"
    probs_ref = torch.softmax(
        logits if temperature is None else logits / temperature, dim=-1
    )
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    on_sm100 = torch.cuda.get_device_capability() == (10, 0)
    assert kernel[5] == (sm100_chunk if on_sm100 else sm103_chunk)
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    torch.testing.assert_close(probs, probs_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rows,vocab_size",
    [
        (32, 151936),  # on B200 a full wave of 16-CTA clusters and one more
        (16, 151936),
        (48, 151936),
        (64, 151936),
        (32, 128000),
        (32, 152064),
        (1024, 151936),  # past two waves of shared-tile clusters
        (62, 70000),  # 8-CTA shared-tile kernel within two waves
        (128, 68000),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blackwell_softmax_unit_temperature_relative_error(rows, vocab_size, seed):
    # A scalar temperature of 1 and no temperature run the same kernel and give
    # identical outputs, each within 1e-6 relative error of float64 softmax.
    _skip_unless_b200_or_gb300()
    torch.manual_seed(seed)
    logits = torch.randn(rows, vocab_size, device="cuda")
    expected = torch.softmax(logits.double(), dim=-1)
    unit = flashinfer.sampling.softmax(logits, temperature=1.0)
    none = flashinfer.sampling.softmax(logits, temperature=None)

    assert torch.equal(unit, none)
    rel = (unit.double() - expected).abs() / expected
    assert rel.max().item() <= 1e-6


@pytest.mark.parametrize(
    "rows,prefix_rows,vocab_size",
    [
        # 16-CTA shared-tile kernel: five waves of resident clusters on B200
        # (the last with the balanced chunk for a per-row temperature), and a
        # prefix within two waves that takes the low-latency kernel.
        (155, 32, 151936),
        (155, 62, 151936),
        # 8-CTA shared-tile kernel.
        (355, 62, 66000),
    ],
)
@pytest.mark.parametrize("temperature_kind", ["none", "scalar", "per_row"])
def test_blackwell_softmax_shared_tile_kernels_match_across_waves(
    rows, prefix_rows, vocab_size, temperature_kind
):
    # Rows give the same bits whether the batch launches the shared-tile kernel
    # for more than two waves or the low-latency kernel for at most two waves.
    _skip_unless_b200_or_gb300()
    torch.manual_seed(11)
    logits = torch.randn(rows, vocab_size, device="cuda")
    if temperature_kind == "per_row":
        temperature = torch.rand(rows, device="cuda") * 0.9 + 0.1
        prefix_temperature = temperature[:prefix_rows].contiguous()
    else:
        temperature = None if temperature_kind == "none" else 0.7
        prefix_temperature = temperature
    prefix = logits[:prefix_rows].contiguous()
    shared_tiles = (
        flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_SHARED_TILES
    )
    for x, t in ((logits, temperature), (prefix, prefix_temperature)):
        route = flashinfer.sampling._blackwell_softmax_route_for_testing(
            x, temperature=t
        )
        assert route == shared_tiles
    full_kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=temperature
    )
    prefix_kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        prefix, temperature=prefix_temperature
    )
    # Same cluster size and chunk, so the same per-row arithmetic.
    assert full_kernel[0] == prefix_kernel[0]
    assert full_kernel[5] == prefix_kernel[5]
    full = flashinfer.sampling.softmax(logits, temperature=temperature)
    part = flashinfer.sampling.softmax(prefix, temperature=prefix_temperature)
    assert torch.equal(full[:prefix_rows], part)


@pytest.mark.parametrize(
    "rows,vocab_size,cluster_ctas,sm100_chunk,sm103_chunk",
    [
        # 16-CTA shared-tile kernel: up to 9 waves and from 9 up to 13 waves
        # of 31 resident clusters on B200.
        (156, 151936, 16, 9504, 10240),
        (279, 151936, 16, 9504, 10240),
        (280, 151936, 16, 9504, 10240),
        (403, 151936, 16, 9504, 10240),
        (404, 151936, 16, 10240, 10240),
        (337, 201088, 16, 12576, 14336),
        # 16-CTA persistent kernel: up to 18 trips of 14 resident clusters.
        (71, 201088, 16, 12576, 14336),
        (252, 201088, 16, 12576, 14336),
        (253, 201088, 16, 14336, 14336),
        # 8-CTA kernels: up to 7 waves of 71 resident shared-tile clusters or
        # 7 trips of 33 resident persistent clusters.
        (356, 66000, 8, 8256, 8256),
        (497, 66000, 8, 8256, 10240),
        (498, 66000, 8, 10240, 10240),
        (166, 71680, 8, 8960, 8960),
        (231, 71680, 8, 8960, 10240),
        (232, 71680, 8, 10240, 10240),
    ],
)
@pytest.mark.parametrize("temperature_kind", ["scalar", "scalar_pdl", "per_row"])
def test_blackwell_cached_cluster_balanced_chunk_past_five_waves_per_arch(
    rows, vocab_size, cluster_ctas, sm100_chunk, sm103_chunk, temperature_kind
):
    # On sm_100 the 8-tile 8- and 16-CTA shared-tile and persistent kernels
    # keep the balanced chunk past five waves up to these bounds, unless the
    # temperature is per row; sm_103 keeps its five-wave rule.
    _skip_unless_b200_or_gb300()
    torch.manual_seed(7)
    logits = torch.randn(rows, vocab_size, device="cuda")
    enable_pdl = temperature_kind == "scalar_pdl"
    if temperature_kind == "per_row":
        temperature = torch.rand(rows, device="cuda") * 0.9 + 0.1
        probs_ref = torch.softmax(logits / temperature[:, None], dim=-1)
    else:
        temperature = 0.5
        probs_ref = torch.softmax(logits / temperature, dim=-1)
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    on_sm100 = torch.cuda.get_device_capability() == (10, 0)
    per_cta = -(-vocab_size // cluster_ctas)
    whole = -(-per_cta // 2048) * 2048
    assert kernel[0] == cluster_ctas
    assert kernel[5] == (
        (whole if temperature_kind == "per_row" else sm100_chunk)
        if on_sm100
        else sm103_chunk
    )
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    torch.testing.assert_close(probs, probs_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rows,vocab_size,cluster_ctas,sm100_chunk,sm103_chunk",
    [
        (96, 24584, 4, 6176, 6176),
        (512, 24584, 4, 6176, 8192),
        (72, 24584, 8, 4096, 4096),
        (71, 28672, 8, 4096, 4096),
        (1, 7168, 4, 2048, 2048),
        (100, 50304, 8, 8192, 8192),
    ],
)
@pytest.mark.parametrize("temperature_kind", ["scalar", "scalar_pdl", "per_row"])
def test_blackwell_cached_cluster_four_tile_rows_balanced_chunk_per_arch(
    rows, vocab_size, cluster_ctas, sm100_chunk, sm103_chunk, temperature_kind
):
    # The 4-tile row kernels of 4 and 8 CTAs take the balanced chunk for the
    # chunk tile counts set per architecture in one wave and past it, unless
    # the temperature is per row; other tile counts keep whole tiles.
    _skip_unless_b200_or_gb300()
    torch.manual_seed(7)
    logits = torch.randn(rows, vocab_size, device="cuda")
    enable_pdl = temperature_kind == "scalar_pdl"
    if temperature_kind == "per_row":
        temperature = torch.rand(rows, device="cuda") * 0.9 + 0.1
        probs_ref = torch.softmax(logits / temperature[:, None], dim=-1)
    else:
        temperature = 0.5
        probs_ref = torch.softmax(logits / temperature, dim=-1)
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    on_sm100 = torch.cuda.get_device_capability() == (10, 0)
    per_cta = -(-vocab_size // cluster_ctas)
    whole = -(-per_cta // 2048) * 2048
    assert kernel[:2] == (cluster_ctas, 4)
    assert kernel[5] == (
        whole
        if temperature_kind == "per_row"
        else sm100_chunk
        if on_sm100
        else sm103_chunk
    )
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    torch.testing.assert_close(probs, probs_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rows,vocab_size,temperature_kind,sm100_chunk,sm103_chunk",
    [
        # 8-CTA shared-tile kernel with a cut of at least 1/10: 6-, 7- and
        # 8-tile chunks, per-row temperatures included.
        (79, 81928, "none", 10272, 10272),
        (45, 81928, "per_row", 10272, 10272),
        (296, 81928, "scalar", 10272, 12288),
        (370, 88320, "scalar_pdl", 11040, 12288),
        (371, 88320, "none", 11040, 12288),
        (53, 98312, "none", 12320, 12320),
        (73, 98312, "per_row", 12320, 12320),
        (370, 103168, "none", 12896, 12896),
        (371, 103168, "scalar", 12896, 14336),
        (50, 114696, "scalar_pdl", 7200, 14368),
        (74, 117760, "none", 14720, 14720),
        # 8-CTA shared-tile kernel with a cut under 1/10: 5- to 8-tile chunks.
        (82, 73736, "none", 9248, 9248),
        (83, 73736, "scalar_pdl", 9248, 9248),
        (115, 76544, "per_row", 9568, 9568),
        (116, 76544, "none", 9568, 10240),
        (79, 88328, "scalar", 11072, 11072),
        (296, 88328, "none", 11072, 12288),
        (370, 91904, "per_row", 12288, 12288),
        (371, 91904, "scalar_pdl", 11488, 12288),
        (53, 103176, "none", 12928, 12928),
        (73, 103176, "scalar", 12928, 12928),
        (370, 107264, "scalar_pdl", 13408, 13408),
        (371, 107264, "per_row", 14336, 14336),
        (50, 117768, "none", 7392, 14752),
        (51, 117768, "per_row", 7392, 14752),
        (74, 126720, "scalar", 16384, 15840),
        # 8-CTA persistent kernel: 6-, 7- and 8-tile chunks.
        (92, 81928, "scalar_pdl", 10272, 10272),
        (93, 81928, "per_row", 10272, 10272),
        (93, 81928, "none", 10272, 10272),
        (180, 88320, "scalar", 11040, 11040),
        (181, 88320, "per_row", 12288, 12288),
        (128, 98312, "none", 12320, 12320),
        (129, 98312, "scalar_pdl", 12320, 12320),
        (180, 103168, "per_row", 14336, 12896),
        (181, 103168, "none", 12896, 14336),
        (56, 114696, "per_row", 14368, 14368),
        (180, 117760, "scalar_pdl", 14720, 14720),
        (181, 117760, "scalar", 14720, 16384),
    ],
)
def test_blackwell_cached_cluster_eight_cta_balanced_chunk_rows_per_arch(
    rows, vocab_size, temperature_kind, sm100_chunk, sm103_chunk
):
    # The 8-CTA kernels of a 5- to 8-tile chunk take the balanced chunk up to
    # five waves or trips, on sm_100 up to seven unless the temperature is per
    # row (371 rows of 88320 pass five shared-tile waves on sm_103). On sm_103,
    # 296 to 443 rows of a 5- or 6-tile chunk whose CTA share exceeds 5 tiles
    # run the persistent kernel with whole tiles instead.
    _skip_unless_b200_or_gb300()
    torch.manual_seed(13)
    logits = torch.randn(rows, vocab_size, device="cuda")
    enable_pdl = temperature_kind == "scalar_pdl"
    if temperature_kind == "per_row":
        temperature = torch.rand(rows, device="cuda") * 0.9 + 0.1
        probs_ref = torch.softmax(logits / temperature[:, None], dim=-1)
    else:
        temperature = None if temperature_kind == "none" else 0.5
        probs_ref = torch.softmax(
            logits if temperature is None else logits / temperature, dim=-1
        )
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    on_sm100 = torch.cuda.get_device_capability() == (10, 0)
    assert kernel[5] == (sm100_chunk if on_sm100 else sm103_chunk)
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=enable_pdl
    )
    torch.testing.assert_close(probs, probs_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("order", ["four_first", "eight_first"])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_blackwell_cached_cluster_row_residency_per_cluster_size(order, enable_pdl):
    # The resident clusters of the 4-tile row kernels are kept per cluster
    # size: querying the 4-CTA kernel (512 rows) before or after the spread of
    # 72 to 75 rows to 8 CTAs (B200: 71 resident 8-CTA clusters, GB300: 74)
    # changes neither launch.
    _skip_unless_b200_or_gb300()
    torch.manual_seed(7)
    shapes = [(512, 24584), (72, 24576), (73, 24576), (74, 24576), (75, 24576)]
    if order == "eight_first":
        shapes = shapes[1:] + shapes[:1]
    on_sm100 = torch.cuda.get_device_capability() == (10, 0)
    four = (4, 4, 6176) if on_sm100 else (4, 4, 8192)
    eight = (4, 4, 6144) if on_sm100 else (8, 4, 4096)
    expected = {512: four, 72: eight, 73: eight, 74: eight, 75: (4, 4, 6144)}
    for _ in range(2):
        for rows, vocab_size in shapes:
            logits = torch.randn(rows, vocab_size, device="cuda")
            kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
                logits, temperature=0.5, enable_pdl=enable_pdl
            )
            assert (kernel[0], kernel[1], kernel[5]) == expected[rows]
            probs = flashinfer.sampling.softmax(
                logits, temperature=0.5, enable_pdl=enable_pdl
            )
            torch.testing.assert_close(
                probs, torch.softmax(logits / 0.5, dim=-1), atol=1e-3, rtol=1e-3
            )


@pytest.mark.parametrize(
    "extra_rows,vocab_size,sm100_ctas",
    [
        (0, 32000, 8),  # rows * 4 CTAs == 2 * SMs: sm_100 keeps spreading to 8 CTAs
        (0, 28672, 8),  # from 14 tiles of vocabulary
        (0, 28664, 4),
        (2, 32768, 8),  # up to 8 CTAs past it from 15 tiles of vocabulary
        (2, 30712, 4),
        (3, 32768, 4),
    ],
)
def test_blackwell_cached_cluster_four_tile_chunks_spread_at_twice_the_sms(
    extra_rows, vocab_size, sm100_ctas
):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if sm_count % 2:
        pytest.skip("The rule is stated for an even SM count")

    torch.manual_seed(7)
    rows = sm_count // 2 + extra_rows
    logits = torch.randn(rows, vocab_size, device="cuda")
    kernel = flashinfer.sampling._blackwell_softmax_cached_kernel_for_testing(
        logits, temperature=0.5
    )
    expected = sm100_ctas if capability == (10, 0) else 4
    assert kernel[:2] == (expected, 4)
    probs = flashinfer.sampling.softmax(logits, temperature=0.5)
    torch.testing.assert_close(
        probs, torch.softmax(logits / 0.5, dim=-1), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize(
    "rows,vocab_size",
    [
        (1, 32004),  # vocabulary not a multiple of 8
        (256, 262152),  # wider than the 16 x 8 x 2048 cache
    ],
)
def test_blackwell_cached_cluster_route_declines_unaligned_rows(rows, vocab_size):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    logits = torch.randn(rows, vocab_size, device="cuda")
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=None, enable_pdl=False
    )
    probs = flashinfer.sampling.softmax(logits, temperature=None, enable_pdl=False)

    assert route not in (
        flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_FALLBACK,
        flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER,
        flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_PERSISTENT,
        flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_CACHED_CLUSTER_SHARED_TILES,
    )
    torch.testing.assert_close(
        probs, torch.softmax(logits, dim=-1), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("rows,vocab_size", [(16, 32000), (256, 32004), (1, 111)])
def test_blackwell_misaligned_base_pointer_takes_fallback(rows, vocab_size):
    # Every generated route reads 128/256-bit vectors relative to the buffer
    # base; a storage-offset view with a 16-byte aligned base must not reach
    # them (the bootstrap kernel faults with cudaErrorMisalignedAddress).
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blackwell Softmax routes require SM100 or SM103")

    storage = torch.randn(rows * vocab_size + 4, device="cuda")
    logits = storage[4:].view(rows, vocab_size)  # contiguous, 16-byte aligned base
    assert logits.is_contiguous() and logits.data_ptr() % 32 == 16

    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=None, enable_pdl=False
    )
    probs = flashinfer.sampling.softmax(logits, temperature=None, enable_pdl=False)

    assert route == flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_FALLBACK
    torch.testing.assert_close(
        probs, torch.softmax(logits, dim=-1), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("temperature", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("temperature_arr", [True, False])
@pytest.mark.parametrize("neg_inf_input", [True, False])
def test_softmax(
    batch_size, vocab_size, distribution, temperature, temperature_arr, neg_inf_input
):
    torch.manual_seed(42)
    logits = distribution((batch_size, vocab_size), "cuda:0")
    if neg_inf_input:
        # assign random logits to -inf
        num_inf = torch.randint(0, logits.numel() - 1, (), device=logits.device).item()
        inf_idx = torch.randperm(logits.numel(), device=logits.device)[:num_inf]
        logits.view(-1).index_fill_(0, inf_idx, float("-inf"))
        torch.cuda.synchronize()  # wait for the index_fill_ to finish because it can overlap with the softmax kernel

    if temperature_arr:
        temperature_arr = torch.full((batch_size,), temperature, device="cuda:0")
        probs = flashinfer.sampling.softmax(logits, temperature=temperature_arr)
        logits_scaled = logits / temperature_arr.unsqueeze(-1)
    else:
        probs = flashinfer.sampling.softmax(logits, temperature=temperature)
        logits_scaled = logits / temperature

    probs_ref = torch.softmax(logits_scaled, dim=-1)

    assert torch.allclose(probs, probs_ref, atol=1e-5)


@pytest.mark.parametrize("api", ["online_softmax", "sampling", "logits_pipe"])
def test_softmax_concurrent_streams(api):
    torch.manual_seed(42)
    shape = (128, 131072)
    logits = [
        torch.randn(shape, device="cuda"),
        torch.randn(shape, device="cuda") * 2 + 3,
    ]
    references = [value.softmax(dim=-1) for value in logits]

    if api == "online_softmax":
        # Public APIs dispatch to a different kernel on Blackwell. Exercise
        # OnlineSoftmax directly with the same workspace for both graphs.
        workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
        run = partial(
            flashinfer.sampling.get_sampling_module().softmax,
            workspace,
            maybe_temperature_arr=None,
            temperature_val=1.0,
            enable_pdl=False,
        )
    elif api == "sampling":
        run = partial(flashinfer.sampling.softmax, enable_pdl=False)
    else:
        from flashinfer.logits_processor import LogitsPipe, Softmax, Temperature

        pipe = LogitsPipe([Temperature(), Softmax(enable_pdl=False)], compile=True)
        run = partial(pipe, temperature=1.0)

    run(logits[0])  # Compile before capture.
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    outputs = []
    for graph, stream, value in zip(graphs, streams, logits, strict=True):
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(run(value))

    for _ in range(10):
        for graph, stream in zip(graphs, streams, strict=True):
            with torch.cuda.stream(stream):
                graph.replay()
        for stream in streams:
            stream.synchronize()
        for output, reference in zip(outputs, references, strict=True):
            torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("offset", [0, 4, -4, 32768])
def test_softmax_split_output_overlap(offset):
    from flashinfer.jit.sampling import gen_sampling_module

    vocab_size = 32768
    storage = torch.randn(vocab_size + abs(offset), device="cuda")
    logits = storage.narrow(0, max(-offset, 0), vocab_size).view(1, vocab_size)
    output = storage.narrow(0, max(offset, 0), vocab_size).view(1, vocab_size)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    softmax = gen_sampling_module().build_and_load().softmax
    if offset == vocab_size:
        softmax(workspace, logits, output, None, 1.0, False)
        torch.testing.assert_close(output, logits.softmax(dim=-1), atol=1e-5, rtol=1e-5)
    else:
        with pytest.raises(
            RuntimeError, match="OnlineSoftmax failed with error code invalid argument"
        ):
            softmax(workspace, logits, output, None, 1.0, False)


# Batches of at most 128 rows with a vocabulary of at least 24576 take the vocab-splitting
# map-reduce kernels; the rest take the fused kernel.
@pytest.mark.parametrize("batch_size,vocab_size", [(128, 4096), (4, 32768)])
def test_softmax_pdl_waits_for_temperature(batch_size, vocab_size):
    logits = torch.randn(batch_size, vocab_size, device="cuda")
    if not device_support_pdl(logits.device):
        pytest.skip("PDL not supported on this device")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    softmax = flashinfer.sampling.get_sampling_module().softmax
    temperature_value = 0.5

    temperature = torch.full((batch_size,), temperature_value, device="cuda")
    expected = softmax(workspace, logits, temperature, 0.0, False)

    # Row 0's temperature is what the previous kernel is still producing.
    # No other kernel may run between that producer and the softmax kernel.
    temperature[0] = temperature_value * 4
    late_store(temperature, temperature_value)
    probs = softmax(workspace, logits, temperature, 0.0, True)

    torch.testing.assert_close(probs, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch_size,vocab_size",
    [
        (8, 111),  # warp-packed route
        (8, 1001),  # rowwise route (vocab not a multiple of 8)
        (256, 32004),  # cooperative bootstrap route
        (1, 32000),  # row-caching cluster route, _pdl twin
        (16, 151552),  # shared-tile cluster route, _pdl twin
        (48, 256000),  # persistent 16-CTA clusters
        (1024, 256000),  # shared-tile 16-CTA clusters, _pdl twin
    ],
)
def test_softmax_blackwell_pdl_waits_for_temperature(batch_size, vocab_size):
    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("Cake softmax routes launch with PDL on SM103 and SM107")
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda")
    temperature_value = 0.5
    temperature = torch.full((batch_size,), temperature_value, device="cuda")
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=temperature, enable_pdl=True
    )
    assert route != flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_FALLBACK
    expected = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=False
    )

    # Row 0's temperature is what the previous kernel is still producing.
    # No other kernel may run between that producer and the softmax kernel.
    temperature[0] = temperature_value * 4
    late_store(temperature, temperature_value)
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=True
    )

    torch.testing.assert_close(probs, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch_size,vocab_size,temperature_kind",
    [
        (1, 32000, "none"),  # cached cluster route, one row spread over 16 CTAs
        (256, 32000, "none"),  # cached cluster route, 4-CTA clusters
        (256, 64000, "none"),  # cached cluster route, 8-CTA clusters
        (512, 64000, "none"),  # cached cluster route, 8-CTA clusters
        (1024, 64000, "none"),  # cached cluster route, 8-CTA clusters
        (48, 256000, "none"),  # persistent 16-CTA non-portable clusters
        (1024, 256000, "none"),  # shared-tile 16-CTA non-portable clusters
        (989, 128256, "per_row"),  # shared-tile 8-CTA clusters, per-row temperature
        (1, 32004, "none"),  # cooperative bootstrap route (vocab not a multiple of 8)
        (256, 32004, "none"),  # rowwise route (vocab not a multiple of 8)
        (1, 111, "scalar"),  # scalar-temperature warp-packed route
        (1, 111, "per_row"),  # per-row-temperature warp-packed route
    ],
)
def test_softmax_blackwell_routes(batch_size, vocab_size, temperature_kind):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Cake softmax routes require SM100 or SM103")

    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda")
    if temperature_kind == "none":
        temperature = None
        probs_ref = torch.softmax(logits, dim=-1)
    elif temperature_kind == "scalar":
        temperature = 0.5
        probs_ref = torch.softmax(logits / temperature, dim=-1)
    else:
        temperature = torch.full((batch_size,), 0.5, device="cuda")
        probs_ref = torch.softmax(logits / temperature[:, None], dim=-1)
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=False
    )

    assert torch.allclose(probs, probs_ref, atol=1e-5)


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_softmax_blackwell_random_per_row_temperature_contract(vocab_size):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Cake softmax routes require SM100 or SM103")

    torch.manual_seed(42)
    rows = 989
    logits = torch.randn(rows, vocab_size, device="cuda")
    temperature = torch.rand(rows, device="cuda")
    route = flashinfer.sampling._blackwell_softmax_route_for_testing(
        logits, temperature=temperature, enable_pdl=True
    )
    probs = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=True
    )
    probs_ref = torch.softmax(logits / temperature[:, None], dim=-1)

    assert route != flashinfer.sampling._BLACKWELL_SOFTMAX_ROUTE_FALLBACK
    assert probs.data_ptr() != logits.data_ptr()
    torch.testing.assert_close(
        probs,
        probs_ref,
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("zero_ratio", [0.0, 0.5, 0.9])
def test_sampling_freq(vocab_size, distribution, zero_ratio):
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    zero_indices = torch.randperm(vocab_size)[: int(vocab_size * zero_ratio)]
    logits[:, zero_indices] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    num_trials = required_num_trials(probs, compare_two_samples=False)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)

    samples = flashinfer.sampling.sampling_from_probs(
        probs, indices=torch.zeros(num_trials, dtype=torch.int32, device=logits.device)
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials

    assert torch.all(counter[zero_indices] == 0)
    similarity = torch.cosine_similarity(freq, probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling_freq(vocab_size, distribution, p):
    # use torch profiler to check the performance of the code
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    sorted_prob, indices = torch.sort(probs, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(1, vocab_size, dtype=torch.int32, device=logits.device)
    mask.scatter_add_(1, indices, (cdf > (1 - p)).int())

    renorm_probs = flashinfer.sampling.top_p_renorm_probs(probs, p)
    num_trials = required_num_trials(renorm_probs, compare_two_samples=False)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
    samples = flashinfer.sampling.top_p_sampling_from_probs(
        probs,
        p,
        indices=torch.zeros(num_trials, dtype=torch.int32, device=logits.device),
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials
    assert torch.all(mask[torch.arange(1), samples] == 1)
    similarity = torch.cosine_similarity(freq, renorm_probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling_freq(vocab_size, distribution, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    sorted_prob, _ = torch.sort(probs, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (probs >= pivot.unsqueeze(-1)).int()

    renorm_probs = flashinfer.sampling.top_k_renorm_probs(probs, k)
    num_trials = required_num_trials(renorm_probs, compare_two_samples=False)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
    samples = flashinfer.sampling.top_k_sampling_from_probs(
        probs,
        k,
        indices=torch.zeros(num_trials, dtype=torch.int32, device=logits.device),
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials
    assert torch.all(mask[torch.arange(1), samples] == 1)
    similarity = torch.cosine_similarity(freq, renorm_probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling(batch_size, vocab_size):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    num_trails = 5000
    for _ in range(num_trails):
        samples = flashinfer.sampling.sampling_from_probs(normalized_prob)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling_from_logits(batch_size, vocab_size):
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0")
    num_trails = 5000
    for _ in range(num_trails):
        samples = flashinfer.sampling.sampling_from_logits(logits)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
def test_sampling_from_logits_freq(vocab_size, distribution):
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    num_trials = required_num_trials(probs, compare_two_samples=False)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
    samples = flashinfer.sampling.sampling_from_logits(
        logits, indices=torch.zeros(num_trials, dtype=torch.int32, device=logits.device)
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials
    similarity = torch.cosine_similarity(freq, probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_p_sampling_from_probs(normalized_prob, p)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_k_sampling_from_probs(normalized_prob, k)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling_with_variable_k(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    k = torch.randint(1, k + 1, (batch_size,), device="cuda:0")
    pivot = sorted_prob[torch.arange(batch_size), k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_k_sampling_from_probs(normalized_prob, k)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
def test_min_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    # scale min-p
    top_probs = sorted_prob[:, -1].unsqueeze(-1)
    scaled_p = p * top_probs
    # min-p mask
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (sorted_prob >= scaled_p).int())
    min_p_tensor = torch.full((batch_size,), p, device="cuda:0")

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.min_p_sampling_from_probs(
            normalized_prob,
            min_p_tensor,
        )

        assert torch.all(mask[torch.arange(batch_size), samples] == 1), samples[
            torch.nonzero(mask[torch.arange(batch_size), samples] == 0)
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    # top-p mask
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask_top_p = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask_top_p.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    # top-k mask
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask_top_k = (normalized_prob >= pivot.unsqueeze(-1)).int()
    # overall mask
    mask = torch.minimum(mask_top_p, mask_top_k)
    top_p_tensor = torch.full((batch_size,), p, device="cuda:0")
    top_k_tensor = torch.full((batch_size,), k, device="cuda:0")

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            normalized_prob,
            top_k_tensor,
            top_p_tensor,
            filter_apply_order="joint",
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize(
    "batch_size,vocab_size",
    [(1, 8192), (20, 32000), (3, 50257), (7, 151936), (2, 262144)],
)
@pytest.mark.parametrize("mode", ["top_p", "top_k", "joint"])
def test_split_row_rejection_sampling(batch_size, vocab_size, mode):
    # Small batches with vocabularies of 8192 or more take the split-row (thread block
    # cluster) kernels on SM90; these shapes cover cluster sizes 2, 4 and 8, an odd vocabulary
    # (scalar staging), per-row thresholds, a row whose leading chunks have no mass, and k = 1.
    torch.manual_seed(42)
    eps = 1e-4
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * 5
    logits[0, : vocab_size // 2] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    top_k = torch.randint(1, 1000, (batch_size,), dtype=torch.int32, device="cuda:0")
    top_k[-1] = 1
    top_p = torch.rand(batch_size, device="cuda:0") * 0.9 + 0.05

    sorted_prob, order = torch.sort(probs, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask_top_p = torch.zeros_like(probs, dtype=torch.int32)
    mask_top_p.scatter_add_(1, order, (cdf > (1 - top_p.unsqueeze(-1)) - eps).int())
    pivot = sorted_prob.flip(-1).gather(1, (top_k.long() - 1).unsqueeze(-1))
    mask_top_k = (probs >= pivot).int()
    mask = {
        "top_p": mask_top_p,
        "top_k": mask_top_k,
        "joint": torch.minimum(mask_top_p, mask_top_k),
    }[mode]

    def sample(offset, indices=None, per_row=True):
        k, p = (top_k, top_p) if per_row else (int(top_k[0]), float(top_p[0]))
        if mode == "top_p":
            return flashinfer.sampling.top_p_sampling_from_probs(
                probs, p, indices, seed=1234, offset=offset, return_valid=True
            )
        if mode == "top_k":
            return flashinfer.sampling.top_k_sampling_from_probs(
                probs, k, indices, seed=1234, offset=offset, return_valid=True
            )
        return flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs,
            k,
            p,
            indices,
            filter_apply_order="joint",
            seed=1234,
            offset=offset,
            return_valid=True,
        )

    rows = torch.arange(batch_size, device="cuda:0")
    for offset in range(0, 200, 4):
        samples, valid = sample(offset)
        assert torch.all(valid)
        assert torch.all(mask[rows, samples] == 1)
        # Same seed and offset give the same samples.
        assert torch.equal(sample(offset)[0], samples)

    # Outputs mapped to rows through indices, with scalar thresholds (the row-0 mask).
    indices = torch.tensor([0, batch_size - 1, 0], dtype=torch.int32, device="cuda:0")
    if mode == "top_p":
        mask_row0 = mask_top_p[0]
    elif mode == "top_k":
        mask_row0 = mask_top_k[0]
    else:
        mask_row0 = torch.minimum(mask_top_p[0], mask_top_k[0])
    for offset in range(0, 40, 4):
        samples, valid = sample(offset, indices, per_row=False)
        assert torch.all(valid)
        assert mask_row0[samples[0]] == 1 and mask_row0[samples[2]] == 1
        assert 0 <= samples[1] < vocab_size


@pytest.mark.parametrize(
    "batch_size,vocab_size", [(256, 8192), (32, 32000), (256, 151936)]
)
@pytest.mark.parametrize("mode", ["top_p", "top_k", "joint"])
def test_split_row_rejection_sampling_freq(batch_size, vocab_size, mode):
    # The split-row kernels (cluster sizes 2, 4 and 8 for these shapes on SM90) partition each
    # row into per-CTA chunks and per-warp segments. Put the probability mass on a sparse set
    # that includes both sides of every such boundary for cluster sizes 2, 4 and 8, and compare
    # the sample frequencies with the renormalized distribution.
    torch.manual_seed(42)
    num_warps = 8
    support = {0, vocab_size - 1}
    for cluster_size in [2, 4, 8]:
        chunk = (-(-vocab_size // cluster_size) + 3) // 4 * 4
        seg = (-(-chunk // num_warps) + 3) // 4 * 4
        for start in range(0, vocab_size, chunk):
            for b in [start + w * seg for w in range(num_warps)]:
                support.update(i for i in [b - 1, b] if 0 <= i < vocab_size)
    support.update(torch.randint(0, vocab_size, (32,)).tolist())
    support = torch.tensor(sorted(support), device="cuda:0")
    probs_row = torch.zeros(vocab_size, device="cuda:0")
    probs_row[support] = torch.rand(len(support), device="cuda:0") + 0.1
    probs_row /= probs_row.sum()
    probs = probs_row.repeat(batch_size, 1)

    k, p = len(support) // 2, 0.7
    sorted_prob, order = torch.sort(probs_row)
    mask_top_p = torch.zeros(vocab_size, dtype=torch.bool, device="cuda:0")
    mask_top_p[order] = torch.cumsum(sorted_prob, 0) > 1 - p
    mask_top_k = probs_row >= sorted_prob[-k]
    mask = {
        "top_p": mask_top_p,
        "top_k": mask_top_k,
        "joint": mask_top_p & mask_top_k,
    }[mode]
    expected = torch.where(mask, probs_row, 0.0)
    expected /= expected.sum()

    generator = torch.Generator(device="cuda:0").manual_seed(0)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
    num_calls = 51200 // batch_size
    for _ in range(num_calls):
        if mode == "top_p":
            samples = flashinfer.sampling.top_p_sampling_from_probs(
                probs, p, generator=generator
            )
        elif mode == "top_k":
            samples = flashinfer.sampling.top_k_sampling_from_probs(
                probs, k, generator=generator
            )
        else:
            samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, k, p, filter_apply_order="joint", generator=generator
            )
        counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    num_draws = num_calls * batch_size
    freq = counter.float() / num_draws
    assert torch.all(counter[~mask] == 0)
    tol = 6 * torch.sqrt(expected * (1 - expected) / num_draws) + 1e-6
    assert torch.all((freq - expected).abs() <= tol), (freq - expected).abs().max()


@parametrize_product(
    {
        "batch_size": [1, 99, 989],
        "vocab_size": [111, 32000, 128256],
        "k": [100],
        "p": [0.1, 0.5],
    },
    regular=pairwise_product_cases,
)
def test_top_k_top_p_sampling_from_probs_logits_alignment(batch_size, vocab_size, k, p):
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * 5
    generator_logits = torch.Generator("cuda:0")
    generator_probs = generator_logits.clone_state()
    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, filter_apply_order="top_k_first", generator=generator_logits
    )
    samples_ref = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        torch.softmax(logits, dim=-1),
        k,
        p,
        filter_apply_order="top_k_first",
        generator=generator_probs,
    )

    num_matches = (samples == samples_ref).sum().item()
    match_rate = num_matches / samples.numel()

    # NOTE(Zihao): Applying softmax followed by top_k_renorm (softmax -> top_k_renorm)
    # does not guarantee bitwise-identical results compared to top_k_mask followed by softmax (top_k_mask -> softmax).
    # This may cause slight differences in subsequent top-p sampling.
    # Additionally, ties at the k-th position may be resolved differently.
    # We tolerate up to a 5% mismatch rate.
    assert match_rate >= 0.95, (
        f"Sample match rate {match_rate:.2%} is below threshold "
        f"({samples.numel() - num_matches}/{samples.numel()} mismatches, expected <=5%)"
    )


def _top_k_top_p_sample(x, top_k, top_p, from_logits, **kwargs):
    if from_logits:
        return flashinfer.sampling.top_k_top_p_sampling_from_logits(
            x, top_k, top_p, **kwargs
        )
    return flashinfer.sampling.top_k_top_p_sampling_from_probs(
        torch.softmax(x, dim=-1), top_k, top_p, **kwargs
    )


@pytest.mark.parametrize("batch_size", [1, 99])
@pytest.mark.parametrize("top_k_max", [64, 256])
@pytest.mark.parametrize("from_logits", [False, True])
def test_top_k_top_p_sampling_per_row_k_with_top_k_max(
    batch_size, top_k_max, from_logits
):
    torch.manual_seed(42)
    vocab_size = 128256
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * 5
    top_k = torch.randint(1, top_k_max + 1, (batch_size,), device="cuda:0")
    # Rows above the bound, including "no top-k" rows, behave as top_k == top_k_max.
    top_k[0] = vocab_size
    top_k[-1] = -1
    expected_k = torch.where(top_k < 0, top_k_max, top_k.clamp(max=top_k_max))
    sorted_logits, _ = torch.sort(logits, descending=True)
    pivot = sorted_logits[torch.arange(batch_size), expected_k - 1]
    mask = logits >= pivot.unsqueeze(-1)

    generator = torch.Generator("cuda:0").manual_seed(0)
    for _ in range(100):
        samples = _top_k_top_p_sample(
            logits, top_k, 0.9, from_logits, top_k_max=top_k_max, generator=generator
        )
        assert torch.all(mask[torch.arange(batch_size), samples])


@pytest.mark.parametrize("k", [16, 200])
@pytest.mark.parametrize("from_logits", [False, True])
def test_top_k_top_p_sampling_per_row_k_with_top_k_max_freq(k, from_logits):
    torch.manual_seed(42)
    p = 0.9
    vocab_size, num_rows, num_iters = 128256, 1000, 50
    logits = torch.randn(1, vocab_size, device="cuda:0") * 2
    probs = torch.softmax(logits, dim=-1)
    ref = flashinfer.sampling.top_p_renorm_probs(
        flashinfer.sampling.top_k_renorm_probs(probs, k), p
    )[0]

    # Reuse the unique source row through the indexed fast path.
    x = logits
    top_k = torch.tensor([k], dtype=torch.int32, device="cuda:0")
    indices = torch.zeros(num_rows, dtype=torch.int32, device="cuda:0")
    generator = torch.Generator("cuda:0").manual_seed(0)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
    for _ in range(num_iters):
        samples = _top_k_top_p_sample(
            x,
            top_k,
            p,
            from_logits,
            indices=indices,
            top_k_max=256,
            generator=generator,
        )
        counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / (num_rows * num_iters)
    assert torch.all(ref[freq > 0] > 0)
    similarity = torch.cosine_similarity(freq, ref, dim=0)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("from_logits", [False, True])
def test_top_k_first_indexed_fast_path_matches_expanded_call(from_logits):
    vocab_size = 65536
    logits = torch.full((2, vocab_size), -float("inf"), device="cuda:0")
    # Both rows have a tie at the exact-k boundary.
    logits[0, :3] = torch.tensor([0.0, 0.0, -1.0], device="cuda:0")
    logits[1, :4] = torch.tensor([2.0, 1.0, 1.0, 0.0], device="cuda:0")
    top_k = torch.tensor([1, 2], dtype=torch.int32, device="cuda:0")
    top_p = torch.tensor([1.0, 0.75], device="cuda:0")
    indices = torch.tensor([0, 1, 1, 0] * 16, dtype=torch.int64, device="cuda:0")

    indexed = _top_k_top_p_sample(
        logits,
        top_k,
        top_p,
        from_logits,
        indices=indices,
        top_k_max=2,
        seed=20261031,
        offset=17,
    )
    expanded = _top_k_top_p_sample(
        logits.index_select(0, indices),
        top_k.index_select(0, indices),
        top_p.index_select(0, indices),
        from_logits,
        top_k_max=2,
        seed=20261031,
        offset=17,
    )

    assert indexed.dtype == torch.int64
    torch.testing.assert_close(indexed, expanded.to(torch.int64), rtol=0, atol=0)
    # Exact top-k retains only one of row 0's tied maxima.
    assert torch.unique(indexed[indices == 0]).numel() == 1


@pytest.mark.parametrize("from_logits", [False, True])
def test_top_k_first_indexed_fast_path_validates_top_p_length(from_logits):
    logits = torch.zeros((2, 65536), device="cuda:0")
    top_k = torch.tensor([1, 2], dtype=torch.int32, device="cuda:0")
    indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda:0")
    output_sized_top_p = torch.full((indices.numel(),), 0.9, device="cuda:0")

    with pytest.raises(
        ValueError, match="Sampling parameter tensor batch size mismatch"
    ):
        _top_k_top_p_sample(
            logits,
            top_k,
            output_sized_top_p,
            from_logits,
            indices=indices,
            top_k_max=2,
        )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_logits(batch_size, vocab_size, p):
    torch.manual_seed(42)
    logits = torch.rand(batch_size, vocab_size, device="cuda:0") * 5
    generator_logits = torch.Generator("cuda:0")
    generator_probs = generator_logits.clone_state()
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")

    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, filter_apply_order="joint", generator=generator_logits
    )

    samples_ref = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        torch.softmax(logits, dim=-1),
        k,
        p,
        filter_apply_order="joint",
        generator=generator_probs,
    )
    assert torch.all(samples == samples_ref)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("is_deterministic", [False, True])
def test_top_p_renorm_probs(batch_size, vocab_size, p, is_deterministic):
    if is_deterministic and vocab_size < 2048:
        pytest.skip(
            "rows narrower than the radix histogram use the ternary search path"
        )
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf >= (1 - p)).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = flashinfer.sampling.top_p_renorm_probs(
        normalized_prob, p, is_deterministic=is_deterministic
    )
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )
    if is_deterministic:
        again = flashinfer.sampling.top_p_renorm_probs(
            normalized_prob, p, is_deterministic=True
        )
        assert torch.equal(renorm_prob, again)


@pytest.mark.parametrize("vocab_size", [(1 << 20) - 1, 1 << 20])
def test_top_p_renorm_probs_deterministic_wide_rows(vocab_size):
    # Rows below 2^20 elements accumulate split 32-bit histogram words; wider
    # rows keep 64-bit words. Both must keep the sorted reference's tokens.
    torch.manual_seed(42)
    p = 0.9
    pre_norm_prob = torch.rand(1, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    prob64 = normalized_prob.double()
    sorted_prob, indices = torch.sort(prob64, descending=False)
    expected_kept = torch.zeros_like(prob64, dtype=torch.bool)
    expected_kept.scatter_(1, indices, torch.cumsum(sorted_prob, dim=-1) >= (1 - p))

    renorm_prob = flashinfer.sampling.top_p_renorm_probs(
        normalized_prob, p, is_deterministic=True
    )
    again = flashinfer.sampling.top_p_renorm_probs(
        normalized_prob, p, is_deterministic=True
    )
    assert torch.equal(renorm_prob, again)
    kept = renorm_prob > 0
    # Rounding at the cumulative-mass boundary may move a single token.
    assert (kept != expected_kept).sum().item() <= 1
    torch.testing.assert_close(
        renorm_prob[kept].double(),
        prob64[kept] / prob64[kept].sum(),
        rtol=1e-3,
        atol=0,
    )


@pytest.mark.parametrize("batch_size", [4, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_top_p_renorm_probs_per_request(batch_size, vocab_size):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # Per-request top_p values varying across the batch
    top_p_arr = torch.linspace(0.1, 0.9, batch_size, device="cuda:0")

    # Compute ground truth per row
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf >= (1 - top_p_arr.unsqueeze(1))).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = flashinfer.sampling.top_p_renorm_probs(normalized_prob, top_p_arr)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


# top_p_renorm_probs on compute capability 10.3 (B300 / GB300): float32 ``probs`` with a per-row
# float32 ``top_p`` tensor and ``is_deterministic=False`` are served by a thread-block-cluster
# kernel (``flashinfer/_top_p_renorm_specialized.py``) for the exact (batch_size, vocab_size)
# rows of ``flashinfer/top_p_renorm_probs_specialized_workloads.json``; every other call keeps
# the AIR top-p path. The calls it can never serve return before any counter or diagnostics; the
# kill-switch / fallback lines of the others reach stderr only for calls on an SM103 device. The
# guard and marker tests use CPU tensors (some reporting a fake CUDA device); the shape and
# never-served-call tests run on every GPU (off SM103 they check that the call stays on the AIR
# path and off stderr); the remaining tests need SM103.

_TOP_P_SPECIALIZED_KILL_SWITCH = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
_TOP_P_SPECIALIZED_SHAPES = [
    (5, 154880),
    (6, 154880),
    (12, 154880),
    (18, 154880),
    (24, 154880),
    (30, 154880),
    (36, 154880),
    (54, 154880),
    (4, 248320),
    (7, 248320),
    (12, 248320),
    (16, 248320),
    (20, 248320),
    (24, 248320),
    (28, 248320),
    (32, 248320),
    (68, 248320),
]


def _is_sm103():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)


def _skip_unless_sm103():
    if not _is_sm103():
        pytest.skip("The specialized top-p renorm kernel dispatches on SM103 only")


def _top_p_specialized_stats():
    return _top_p_renorm_specialized._top_p_renorm_probs_specialized_stats()


@pytest.fixture
def top_p_specialized_enabled(monkeypatch):
    monkeypatch.delenv(_TOP_P_SPECIALIZED_KILL_SWITCH, raising=False)


def _top_p_specialized_target(probs, top_p):
    """The nucleus target of each row, as on the AIR path (blockSum * p): top_p times the row's
    mass (float64), so a row that does not sum to 1 keeps the same entries."""
    p = probs.double()
    return top_p.double().reshape(-1, 1).to(p.device) * p.clamp(min=0).sum(
        dim=-1, keepdim=True
    )


def _top_p_specialized_reference(probs, top_p):
    """Exact semantics in float64: keep p >= threshold, where threshold is the smallest value of
    the shortest descending prefix whose mass reaches top_p times the row's mass (ties kept);
    renormalise, unless the kept mass is <= 1e-8 (that row is left as it is, like on AIR)."""
    p = probs.double()
    sp, _ = torch.sort(p, dim=-1, descending=True)
    excl = torch.cumsum(sp, dim=-1) - sp
    target = _top_p_specialized_target(p, top_p)
    k = (excl < target).sum(dim=-1, keepdim=True).clamp(min=1)
    thr = sp.gather(-1, k - 1)
    kept = torch.where(p >= thr, p, torch.zeros_like(p))
    mass = kept.sum(dim=-1, keepdim=True)
    cut = float(torch.tensor(1e-8, dtype=torch.float32))
    return torch.where(mass > cut, kept / mass, kept).float()


def _top_p_specialized_edge_margin(probs, top_p):
    """Distance of each row's nucleus edge from its target, relative to the row's mass
    (float64); rows whose edge entry is at most 1e-6 of the mass are exempt (flipping it moves
    no output beyond the tolerance)."""
    sp, _ = torch.sort(probs.double(), dim=-1, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    excl = cum - sp
    tp = _top_p_specialized_target(probs, top_p)
    mass = sp.clamp(min=0).sum(dim=-1, keepdim=True).clamp(min=1e-300)
    k = (excl < tp).sum(dim=-1, keepdim=True).clamp(min=1)
    inc = cum.gather(-1, k - 1)
    exc = excl.gather(-1, k - 1)
    edge = sp.gather(-1, k - 1)
    m = torch.minimum((inc - tp).abs(), (exc - tp).abs()) / mass
    return torch.where(edge > 1e-6 * mass, m, torch.ones_like(m)).reshape(-1)


def _top_p_specialized_robust_top_p(probs, top_p):
    """Lower top_p (2e-4 steps) on rows whose nucleus edge lies within 1e-5 of the row's mass
    from the target, so the kept set does not depend on the float32 summation order of the path
    under test."""
    tp = top_p.clone()
    for _ in range(50):
        bad = _top_p_specialized_edge_margin(probs, tp) < 1e-5
        if not bool(bad.any()):
            break
        tp = torch.where(bad, tp - 2e-4, tp)
    return tp


def _top_p_specialized_row(kind, vocab_size, g):
    """One float64 probability row of a serving-like or adversarial kind."""
    v = vocab_size
    if kind == 0:
        # truncated upstream (top-k 20): a few peaked values, exact zeros elsewhere
        vals = torch.softmax(torch.randn(20, generator=g, dtype=torch.float64) * 4.0, 0)
        row = torch.zeros(v, dtype=torch.float64)
        row[torch.randperm(v, generator=g)[:20]] = vals
    elif kind == 1:
        # one dominant token over a smooth log-normal bulk (nucleus of 1-2 tokens)
        row = torch.exp(torch.randn(v, generator=g, dtype=torch.float64) * 3.0)
        row = row / row.sum() * 0.03
        row[int(torch.randint(0, v, (1,), generator=g))] += 0.97
    elif kind == 2:
        # wide nucleus: the entries >= 2^-8 cannot reach top_p (exact radix fallback)
        row = torch.softmax(torch.randn(v, generator=g, dtype=torch.float64), 0)
    elif kind == 3:
        # dominant token plus a heavy bulk: the nucleus reaches deep into tiny entries
        row = torch.exp(torch.randn(v, generator=g, dtype=torch.float64) * 2.0)
        row = row / row.sum() * 0.25
        row[int(torch.randint(0, v, (1,), generator=g))] += 0.75
    else:
        # about 100 entries >= 2^-8 holding 97 % of the mass (block-wide resolve of the head set)
        idx = torch.randperm(v, generator=g)
        head = 0.005 + 0.01 * torch.rand(100, generator=g, dtype=torch.float64)
        tail = torch.rand(2000, generator=g, dtype=torch.float64)
        row = torch.zeros(v, dtype=torch.float64)
        row[idx[:100]] = head / head.sum() * 0.97
        row[idx[100:2100]] = tail / tail.sum() * 0.03
    return row / row.sum()


def _top_p_specialized_inputs(
    batch_size, vocab_size, seed, top_p=0.95, kinds=(0, 1, 0, 2, 1, 3, 4)
):
    g = torch.Generator().manual_seed(seed)
    probs = torch.stack(
        [
            _top_p_specialized_row(kinds[b % len(kinds)], vocab_size, g)
            for b in range(batch_size)
        ]
    ).float()
    top_ps = top_p if isinstance(top_p, (list, tuple)) else [top_p]
    tp = torch.tensor(
        [float(top_ps[b % len(top_ps)]) for b in range(batch_size)],
        dtype=torch.float32,
    )
    return probs.cuda(), _top_p_specialized_robust_top_p(probs, tp).cuda()


def _assert_top_p_specialized_contract(out, ref):
    """The specialized kernel's contract: the float64 reference within atol 1e-6, rtol 1e-5."""
    assert out.dtype == torch.float32 and out.shape == ref.shape
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref.to(out.device), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("batch_size,vocab_size", _TOP_P_SPECIALIZED_SHAPES)
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 1.0, "per_request"])
def test_top_p_renorm_probs_specialized_shapes(
    batch_size, vocab_size, p, top_p_specialized_enabled
):
    # test_top_p_renorm_probs / test_top_p_renorm_probs_per_request on the rows the SM103 kernel
    # serves, with top_p passed as a per-row float32 tensor (the served signature).
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    if p == "per_request":
        top_p_arr = torch.linspace(0.1, 0.9, batch_size, device="cuda:0")
    else:
        top_p_arr = torch.full((batch_size,), p, device="cuda:0")
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf >= (1 - top_p_arr.unsqueeze(1))).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    dispatched = _top_p_specialized_stats()["dispatch_count"]
    renorm_prob = flashinfer.sampling.top_p_renorm_probs(normalized_prob, top_p_arr)
    # SM103 serves the call with the specialized kernel; every other GPU keeps the AIR path.
    assert _top_p_specialized_stats()["dispatch_count"] == dispatched + int(_is_sm103())
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


def test_top_p_renorm_probs_specialized_allowlist():
    assert _top_p_renorm_specialized._load_allowlist() == frozenset(
        _TOP_P_SPECIALIZED_SHAPES
    )


@pytest.fixture
def top_p_specialized_cpu_call():
    batch_size, vocab_size = _TOP_P_SPECIALIZED_SHAPES[0]
    return torch.rand(batch_size, vocab_size), torch.full((batch_size,), 0.95)


def _top_p_specialized_unlisted_batch(vocab_size):
    return next(
        b
        for b in range(1, 1024)
        if (b, vocab_size) not in set(_TOP_P_SPECIALIZED_SHAPES)
    )


def test_top_p_renorm_probs_specialized_guard_accepts_until_the_device_check(
    top_p_specialized_cpu_call, top_p_specialized_enabled
):
    probs, top_p = top_p_specialized_cpu_call
    # every semantic and layout gate passes; a CPU tensor stops at the device gate
    assert _top_p_renorm_specialized._reject_reason(probs, top_p, False) == "device"


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda p, t: (p, t, True), "deterministic"),
        (lambda p, t: (p, 0.95, False), "scalar_top_p"),
        (lambda p, t: (p.bfloat16(), t, False), "dtype"),
        (lambda p, t: (p, t.double(), False), "dtype"),
        (lambda p, t: (p, t[1:], False), "shape"),
        (lambda p, t: (p.unsqueeze(0), t, False), "shape"),
        (lambda p, t: (p.t().contiguous().t(), t, False), "non_contiguous"),
        (lambda p, t: (p, torch.empty(2 * t.numel())[::2], False), "non_contiguous"),
        (
            lambda p, t: (
                p.new_empty(_top_p_specialized_unlisted_batch(p.shape[1]), p.shape[1]),
                t.new_empty(_top_p_specialized_unlisted_batch(p.shape[1])),
                False,
            ),
            "shape_not_allowlisted",
        ),
        (
            lambda p, t: (p[:, : p.shape[1] - 4].contiguous(), t, False),
            "shape_not_allowlisted",
        ),
        (
            lambda p, t: (torch.empty(p.numel() + 1)[1:].view(p.shape), t, False),
            "alignment",
        ),
    ],
)
def test_top_p_renorm_probs_specialized_guard_rejections(
    top_p_specialized_cpu_call, top_p_specialized_enabled, mutate, reason
):
    probs, top_p, is_deterministic = mutate(*top_p_specialized_cpu_call)
    assert (
        _top_p_renorm_specialized._reject_reason(probs, top_p, is_deterministic)
        == reason
    )


def test_top_p_renorm_probs_specialized_kill_switch_is_read_at_call_time(
    top_p_specialized_cpu_call, monkeypatch
):
    probs, top_p = top_p_specialized_cpu_call
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    assert (
        _top_p_renorm_specialized._reject_reason(probs, top_p, False) == "kill_switch"
    )
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "0")
    assert _top_p_renorm_specialized._reject_reason(probs, top_p, False) == "device"


def test_top_p_renorm_probs_specialized_ineligible_call_returns_none(
    top_p_specialized_cpu_call, top_p_specialized_enabled
):
    # A call the kernel can never serve (here: CPU tensors) is not counted.
    probs, top_p = top_p_specialized_cpu_call
    before = _top_p_specialized_stats()
    assert (
        _top_p_renorm_specialized.top_p_renorm_probs_specialized(probs, top_p, False)
        is None
    )
    after = _top_p_specialized_stats()
    for key in ("calls", "dispatch_count", "fallbacks"):
        assert after[key] == before[key]
    assert after["fallback_reasons"] == before["fallback_reasons"]


class _TopPSpecializedLogRecorder:
    """Stand-in for the ``flashinfer.jit`` logger of the specialized path: (level, message)."""

    def __init__(self):
        self.records = []

    def _record(self, level, msg, *args):
        self.records.append((level, msg % args if args else msg))

    def debug(self, msg, *args):
        self._record("debug", msg, *args)

    def info(self, msg, *args):
        self._record("info", msg, *args)

    def warning(self, msg, *args):
        self._record("warning", msg, *args)


@pytest.fixture
def top_p_specialized_markers(monkeypatch):
    """Fresh one-time marker state; returns the recorder standing in for the logger."""
    monkeypatch.setattr(_top_p_renorm_specialized, "_markers_emitted", set())
    monkeypatch.setattr(_top_p_renorm_specialized, "_markers_logged_debug", set())
    recorder = _TopPSpecializedLogRecorder()
    monkeypatch.setattr(_top_p_renorm_specialized, "logger", recorder)
    return recorder


class _TopPSpecializedFakeCudaDevices:
    """Stand-in for torch.cuda's current device and per-device stream capture (no GPU needed):
    ``is_current_stream_capturing()`` reports the current device's stream, as torch does."""

    def __init__(self, monkeypatch, current=0, capturing=()):
        self.current = current
        self.capturing = set(capturing)
        fake = self

        class _Device:
            def __init__(self, device):
                self.index = (
                    device if isinstance(device, int) else torch.device(device).index
                )

            def __enter__(self):
                self.previous, fake.current = fake.current, self.index

            def __exit__(self, *exc):
                fake.current = self.previous
                return False

        monkeypatch.setattr(torch.cuda, "device", _Device)
        monkeypatch.setattr(
            torch.cuda,
            "is_current_stream_capturing",
            lambda: fake.current in fake.capturing,
        )


class _TopPSpecializedFakeCudaTensor(torch.Tensor):
    """A CPU tensor that reports a CUDA device, for the guard's device gates without a GPU (the
    device is a class attribute, so tensors derived from it report it too)."""

    _fake_device = torch.device("cuda", 0)

    @property
    def device(self):
        return self._fake_device


def _top_p_specialized_fake_cuda_call(index, shape=None):
    batch_size, vocab_size = shape or _TOP_P_SPECIALIZED_SHAPES[0]
    fake = type(
        f"_TopPSpecializedFakeCuda{index}Tensor",
        (_TopPSpecializedFakeCudaTensor,),
        {"_fake_device": torch.device("cuda", index)},
    )
    probs = torch.rand(batch_size, vocab_size).as_subclass(fake)
    top_p = torch.full((batch_size,), 0.95).as_subclass(fake)
    return probs, top_p


@pytest.fixture
def top_p_specialized_fake_devices(monkeypatch):
    """Fake CUDA devices for fake-CUDA tensors: cuda:0 and cuda:2 of compute capability 10.3,
    cuda:1 of 10.0, none capturing; fresh per-device caches."""
    devices = _TopPSpecializedFakeCudaDevices(monkeypatch, current=0)
    capabilities = {0: (10, 3), 1: (10, 0), 2: (10, 3)}
    monkeypatch.setattr(
        _top_p_renorm_specialized,
        "get_compute_capability",
        lambda device: capabilities[device.index],
    )
    monkeypatch.setattr(_top_p_renorm_specialized, "_device_capability", {})
    monkeypatch.setattr(_top_p_renorm_specialized, "_device_supported", {})
    monkeypatch.setattr(_top_p_renorm_specialized, "_launch_failed_devices", set())
    return devices


def test_top_p_renorm_probs_specialized_target_device_gate(monkeypatch):
    # stderr is reserved for calls on a CUDA device of compute capability 10.3, read from a
    # per-device cache that is never filled while that device's stream is capturing a CUDA graph.
    capabilities = {0: (10, 3), 1: (10, 0), 2: (10, 3), 3: (10, 3)}
    queried = []

    def get_compute_capability(device):
        queried.append(device.index)
        return capabilities[device.index]

    devices = _TopPSpecializedFakeCudaDevices(monkeypatch, current=0)
    monkeypatch.setattr(
        _top_p_renorm_specialized, "get_compute_capability", get_compute_capability
    )
    monkeypatch.setattr(_top_p_renorm_specialized, "_device_capability", {})
    gate = _top_p_renorm_specialized._is_target_device
    assert gate(torch.device("cuda", 0)) and gate(torch.device("cuda", 0))
    assert not gate(torch.device("cuda", 1))
    assert queried == [0, 1]
    devices.capturing = {0, 2}
    assert gate(torch.device("cuda", 0))
    # a device not seen before the capture counts as another device, without a query
    assert not gate(torch.device("cuda", 2))
    # ... also while another device (not capturing) is current
    devices.capturing = {2}
    assert not gate(torch.device("cuda", 2))
    assert queried == [0, 1]
    # a capture on the current device does not hide a device whose stream is not capturing
    devices.capturing = {0}
    assert gate(torch.device("cuda", 3))
    assert queried == [0, 1, 3]
    assert devices.current == 0
    assert not gate(torch.device("cpu")) and not gate(None)


def test_top_p_renorm_probs_specialized_capture_check_reads_the_tensor_device(
    top_p_specialized_enabled, monkeypatch
):
    # Before the module is built, a capture on the stream of probs' device keeps the
    # (capture-safe) stock path, also while another device is current:
    # torch.cuda.is_current_stream_capturing() alone reads the current device's stream.
    probs, top_p = _top_p_specialized_fake_cuda_call(1)
    devices = _TopPSpecializedFakeCudaDevices(monkeypatch, current=0, capturing={1})
    monkeypatch.setattr(
        _top_p_renorm_specialized, "get_compute_capability", lambda device: (10, 3)
    )
    monkeypatch.setattr(_top_p_renorm_specialized, "_device_capability", {})
    monkeypatch.setattr(_top_p_renorm_specialized, "_launch_failed_devices", set())
    monkeypatch.setattr(_top_p_renorm_specialized, "_module", None)
    reject = _top_p_renorm_specialized._reject_reason
    assert reject(probs, top_p, False) == "capturing_before_jit"
    assert devices.current == 0
    devices.current = 1
    assert reject(probs, top_p, False) == "capturing_before_jit"
    devices.current = 0
    devices.capturing = set()
    assert reject(probs, top_p, False) is None


def test_top_p_renorm_probs_specialized_fallback_lines_are_silent_off_target(
    top_p_specialized_cpu_call,
    top_p_specialized_enabled,
    top_p_specialized_markers,
    monkeypatch,
    capsys,
):
    # Calls the kernel can never serve (here: CPU tensors, a scalar top_p), also with the kill
    # switch set, write nothing at all. The only lines off a compute capability 10.3 device are
    # those of calls whose arguments the first checks cannot read: once each, at DEBUG level on
    # the flashinfer.jit logger, never on stderr.
    probs, top_p = top_p_specialized_cpu_call
    specialized = _top_p_renorm_specialized.top_p_renorm_probs_specialized
    for _ in range(2):
        assert specialized(probs, top_p, False) is None
        assert specialized(probs, 0.95, False) is None
        assert specialized(object(), top_p, False) is None
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    assert specialized(probs, top_p, False) is None
    assert specialized(probs, top_p, False) is None
    assert capsys.readouterr().err == ""
    assert _top_p_renorm_specialized._markers_emitted == set()
    records = top_p_specialized_markers.records
    assert [level for level, _ in records] == ["debug"]
    marker = _top_p_renorm_specialized.FALLBACK_MARKER
    assert records[0][1].startswith(f"{marker}: guard_error")


def test_top_p_renorm_probs_specialized_never_served_calls_skip_the_guard(
    top_p_specialized_enabled,
    top_p_specialized_markers,
    top_p_specialized_fake_devices,
    monkeypatch,
    capsys,
):
    # The calls the kernel can never serve (is_deterministic, a scalar top_p, other dtypes, a
    # device other than compute capability 10.3, a shape off the allowlist) return before the
    # environment, the counters and the diagnostics; all but the last also before the
    # torch.compile check, which the allowlist check follows (dynamo would specialize the batch
    # size on it). The capability is read once per device.
    specialized = _top_p_renorm_specialized.top_p_renorm_probs_specialized
    probs, top_p = _top_p_specialized_fake_cuda_call(0)
    other_probs, other_top_p = _top_p_specialized_fake_cuda_call(1)
    vocab_size = probs.shape[1]
    unlisted_batch = _top_p_specialized_unlisted_batch(vocab_size)
    unlisted_probs, unlisted_top_p = _top_p_specialized_fake_cuda_call(
        0, (unlisted_batch, vocab_size)
    )
    # the first call on each device that passes the device-independent checks records its
    # capability (a scalar top_p does not get that far)
    assert specialized(probs, 0.95, False) is None
    assert _top_p_renorm_specialized._device_supported == {}
    assert specialized(unlisted_probs, unlisted_top_p, False) is None
    assert specialized(other_probs, other_top_p, False) is None
    cpu_probs, cpu_top_p = torch.rand(probs.shape), torch.full((probs.shape[0],), 0.95)
    assert specialized(cpu_probs, cpu_top_p, False) is None
    assert _top_p_renorm_specialized._device_supported == {
        torch.device("cuda", 0): True,
        torch.device("cuda", 1): False,
        torch.device("cpu"): False,
    }

    def must_not_run(*args, **kwargs):
        raise AssertionError("a call the kernel never serves reached the guard")

    for name in (
        "_reject_reason",
        "_bump_fallback",
        "_report_once",
        "_emit_once",
        "get_compute_capability",
    ):
        monkeypatch.setattr(_top_p_renorm_specialized, name, must_not_run)
    compile_checks = []
    monkeypatch.setattr(
        torch.compiler, "is_compiling", lambda: bool(compile_checks.append(1))
    )
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    before = _top_p_specialized_stats()
    for args in [
        (probs, top_p, True),
        (probs, 0.95, False),
        (probs.double(), top_p, False),
        (probs, top_p.double(), False),
        (probs.bfloat16(), top_p, False),
        (other_probs, other_top_p, False),
        (cpu_probs, cpu_top_p, False),
    ]:
        assert specialized(*args) is None
    assert compile_checks == []
    assert specialized(unlisted_probs, unlisted_top_p, False) is None
    assert compile_checks == [1]
    after = _top_p_specialized_stats()
    for key in ("calls", "dispatch_count", "fallbacks"):
        assert after[key] == before[key]
    assert after["fallback_reasons"] == before["fallback_reasons"]
    assert top_p_specialized_markers.records == []
    assert capsys.readouterr().err == ""


def test_top_p_renorm_probs_specialized_never_served_checks_trace_under_torch_compile(
    top_p_specialized_cpu_call,
):
    # The first checks run before the torch.compile check, so dynamo traces them: no graph break
    # (fullgraph), and a call it can never serve keeps the stock path.
    probs, top_p = top_p_specialized_cpu_call
    specialized = _top_p_renorm_specialized.top_p_renorm_probs_specialized

    def call(p, t):
        out = specialized(p, t, False)
        return p * 2 if out is None else out

    compiled = torch.compile(call, backend="eager", fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(probs, top_p), probs * 2)
    torch.testing.assert_close(compiled(probs[:3], top_p[:3]), probs[:3] * 2)


def test_top_p_renorm_probs_specialized_fallback_reason_is_written_once(
    top_p_specialized_enabled,
    top_p_specialized_markers,
    top_p_specialized_fake_devices,
    capsys,
):
    # On a compute capability 10.3 device the first fallback of each reason of a call the kernel
    # would serve is one stderr line (whatever the host's logging configuration), later ones are
    # silent.
    probs, top_p = _top_p_specialized_fake_cuda_call(0)
    strided = probs.t().contiguous().t()
    specialized = _top_p_renorm_specialized.top_p_renorm_probs_specialized
    assert specialized(strided, top_p, False) is None
    assert specialized(strided, top_p, False) is None
    assert specialized(probs, top_p[1:], False) is None
    marker = _top_p_renorm_specialized.FALLBACK_MARKER
    lines = [line for line in capsys.readouterr().err.splitlines() if marker in line]
    assert len(lines) == 2
    assert lines[0].startswith(
        f"{marker}: non_contiguous (shape={tuple(probs.shape)}, dtype=torch.float32, "
        "device=cuda:0"
    )
    assert lines[1].startswith(f"{marker}: shape")
    assert top_p_specialized_markers.records == [("info", line) for line in lines]


def test_top_p_renorm_probs_specialized_kill_switch_marker_is_written_once(
    top_p_specialized_markers,
    top_p_specialized_fake_devices,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    probs, top_p = _top_p_specialized_fake_cuda_call(0)
    _top_p_renorm_specialized.top_p_renorm_probs_specialized(probs, top_p, False)
    _top_p_renorm_specialized.top_p_renorm_probs_specialized(probs, top_p, False)
    err = capsys.readouterr().err
    assert err.count(_top_p_renorm_specialized.KILL_SWITCH_MARKER) == 1
    assert _top_p_renorm_specialized.FALLBACK_MARKER not in err


@pytest.fixture
def top_p_specialized_missing_allowlist(monkeypatch):
    monkeypatch.setattr(
        _top_p_renorm_specialized, "_WORKLOAD_FILE", "missing_workloads.json"
    )
    _top_p_renorm_specialized._load_allowlist.cache_clear()
    yield
    monkeypatch.undo()
    _top_p_renorm_specialized._load_allowlist.cache_clear()
    assert _top_p_renorm_specialized._load_allowlist()
    assert _top_p_renorm_specialized._allowlist_error is None


@pytest.mark.parametrize("device_index", [0, 1])
def test_top_p_renorm_probs_specialized_missing_allowlist_is_reported(
    device_index,
    top_p_specialized_enabled,
    top_p_specialized_markers,
    top_p_specialized_fake_devices,
    top_p_specialized_missing_allowlist,
    capsys,
):
    # Without its allowlist no call is served; the load error is reported once, on stderr (after
    # a warning), by the first call the kernel would otherwise serve: a call on a compute
    # capability 10.3 device (cuda:0). A call on another device (cuda:1) never reads the
    # allowlist and writes nothing.
    probs, top_p = _top_p_specialized_fake_cuda_call(device_index)
    specialized = _top_p_renorm_specialized.top_p_renorm_probs_specialized
    assert specialized(probs, top_p, False) is None
    assert specialized(probs, top_p, False) is None
    assert _top_p_specialized_stats()["allowlist_error"].startswith("FileNotFoundError")
    marker = _top_p_renorm_specialized.FALLBACK_MARKER
    err = capsys.readouterr().err
    records = top_p_specialized_markers.records
    if device_index == 0:
        lines = [line for line in err.splitlines() if marker in line]
        assert len(lines) == 1
        assert lines[0].startswith(
            f"{marker}: allowlist_unavailable (FileNotFoundError"
        )
        assert [level for level, _ in records] == ["warning", "info"]
        assert "allowlist unavailable (FileNotFoundError" in records[0][1]
    else:
        assert err == ""
        assert records == []


def test_top_p_renorm_probs_specialized_guard_exception_keeps_the_stock_path(
    top_p_specialized_enabled,
):
    assert (
        _top_p_renorm_specialized.top_p_renorm_probs_specialized(
            object(), torch.ones(1), False
        )
        is None
    )
    stats = _top_p_specialized_stats()
    assert stats["fallback_reasons"].get("guard_error", 0) >= 1
    assert stats["guard_error"]


def test_top_p_renorm_probs_specialized_stats_report_the_compile_footprint():
    stats = _top_p_specialized_stats()
    assert stats["allowlist_rows"] == len(_TOP_P_SPECIALIZED_SHAPES)
    assert stats["compiled_variants"] in (0, 1)
    assert stats["kernel_instantiations"] == 18
    # the allowlisted rows (float4 rows) reach 8 of the 9 launch configurations (all but the
    # short-row one)
    assert stats["distinct_kernels_for_allowlist"] == 8
    configs = {
        (c["cluster_size"], c["num_threads"], c["min_blocks"], c["unroll"])
        for c in stats["allowlist_kernel_configs"]
    }
    assert configs == set(_top_p_renorm_specialized._LAUNCH_CONFIGS[1:])
    assert all(c["vectorized"] for c in stats["allowlist_kernel_configs"])


def test_top_p_renorm_probs_specialized_launch_plan_mirror():
    # Spot checks of the B x V launch table (serving rows) against the configurations it names.
    plan = _top_p_renorm_specialized._plan_config
    assert plan(7, 248320) == (16, 1024, 1, 4, True)
    assert plan(6, 154880) == (16, 1024, 1, 3, True)
    assert plan(12, 154880) == (16, 512, 3, 5, True)
    assert plan(16, 248320) == (16, 512, 3, 8, True)
    assert plan(30, 154880) == (16, 256, 4, 10, True)
    assert plan(36, 154880) == (8, 512, 3, 8, True)
    assert plan(32, 248320) == (8, 512, 2, 16, True)
    assert plan(68, 248320) == (8, 512, 4, 8, True)
    assert plan(4, 32000) == (8, 256, 6, 8, True)


def test_top_p_renorm_probs_specialized_never_served_calls_are_silent(
    top_p_specialized_enabled, top_p_specialized_markers, capsys
):
    # Calls the kernel never serves (a scalar top_p, a shape off the allowlist, bfloat16 rows)
    # take the AIR path on every GPU without a counter, a marker or a log line.
    probs = torch.rand(4, 1024, device="cuda:0")
    probs = probs / probs.sum(dim=-1, keepdim=True)
    top_p = torch.full((4,), 0.95, device="cuda:0")
    before = _top_p_specialized_stats()
    for _ in range(2):
        flashinfer.sampling.top_p_renorm_probs(probs, 0.95)
        flashinfer.sampling.top_p_renorm_probs(probs, top_p)
        flashinfer.sampling.top_p_renorm_probs(probs.bfloat16(), top_p)
    after = _top_p_specialized_stats()
    for key in ("calls", "dispatch_count", "fallbacks"):
        assert after[key] == before[key]
    assert _top_p_renorm_specialized.FALLBACK_MARKER not in capsys.readouterr().err
    assert top_p_specialized_markers.records == []


@pytest.mark.parametrize("batch_size,vocab_size", _TOP_P_SPECIALIZED_SHAPES)
def test_top_p_renorm_probs_specialized_matches_reference(
    batch_size, vocab_size, top_p_specialized_enabled, monkeypatch
):
    _skip_unless_sm103()
    probs, top_p = _top_p_specialized_inputs(
        batch_size, vocab_size, seed=batch_size * 7 + vocab_size
    )
    ref = _top_p_specialized_reference(probs, top_p)
    before = _top_p_specialized_stats()["dispatch_count"]
    out = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    assert _top_p_specialized_stats()["dispatch_count"] == before + 1
    _assert_top_p_specialized_contract(out, ref)

    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    stock = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    assert _top_p_specialized_stats()["dispatch_count"] == before + 1
    # the AIR path's own test tolerance (float32 accumulation)
    torch.testing.assert_close(out, stock, atol=1e-3, rtol=1e-3)


def test_top_p_renorm_probs_specialized_dispatch_marker_is_written_once(
    top_p_specialized_enabled, top_p_specialized_markers, capsys
):
    _skip_unless_sm103()
    batch_size, vocab_size = _TOP_P_SPECIALIZED_SHAPES[0]
    probs, top_p = _top_p_specialized_inputs(batch_size, vocab_size, seed=11)
    flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    err = capsys.readouterr().err
    assert err.count(_top_p_renorm_specialized.DISPATCH_MARKER) == 1
    assert f"(batch_size={batch_size}, vocab_size={vocab_size}" in err
    assert _top_p_renorm_specialized.FALLBACK_MARKER not in err


def test_top_p_renorm_probs_specialized_kill_switch_takes_the_stock_path(monkeypatch):
    _skip_unless_sm103()
    batch_size, vocab_size = _TOP_P_SPECIALIZED_SHAPES[0]
    probs, top_p = _top_p_specialized_inputs(batch_size, vocab_size, seed=1)
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    before = _top_p_specialized_stats()
    out = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    after = _top_p_specialized_stats()
    assert after["dispatch_count"] == before["dispatch_count"]
    assert (
        after["fallback_reasons"]["kill_switch"]
        == before["fallback_reasons"].get("kill_switch", 0) + 1
    )
    torch.testing.assert_close(
        out, _top_p_specialized_reference(probs, top_p).cuda(), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("batch_size,vocab_size", [(8, 248320), (40, 154880)])
def test_top_p_renorm_probs_specialized_unlisted_shape_takes_the_stock_path(
    batch_size, vocab_size, top_p_specialized_enabled
):
    _skip_unless_sm103()
    probs, top_p = _top_p_specialized_inputs(batch_size, vocab_size, seed=2)
    before = _top_p_specialized_stats()["dispatch_count"]
    out = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    assert _top_p_specialized_stats()["dispatch_count"] == before
    torch.testing.assert_close(
        out, _top_p_specialized_reference(probs, top_p).cuda(), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("batch_size,vocab_size", [(7, 248320), (54, 154880)])
def test_top_p_renorm_probs_specialized_uninitialised_output(
    batch_size, vocab_size, top_p_specialized_enabled
):
    # The kernel must not rely on a zero-filled output: every element is written.
    _skip_unless_sm103()
    probs, top_p = _top_p_specialized_inputs(batch_size, vocab_size, seed=3)
    module = _top_p_renorm_specialized._get_module()
    assert module is not None
    out = torch.full_like(probs, float("nan"))
    module.top_p_renorm_probs_specialized(probs, top_p, out)
    _assert_top_p_specialized_contract(out, _top_p_specialized_reference(probs, top_p))


@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("top_p", [0.95, 0.5, 0.999, 1.0])
def test_top_p_renorm_probs_specialized_row_kinds(
    kind, top_p, top_p_specialized_enabled
):
    # Head-set rows (0, 1: one-entry exit and warp-level resolve; 4: block-wide resolve) and
    # exact radix-fallback rows (2, 3) at several thresholds.
    _skip_unless_sm103()
    probs, tp = _top_p_specialized_inputs(
        6, 154880, seed=10 + kind, top_p=top_p, kinds=(kind,)
    )
    out = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    _assert_top_p_specialized_contract(out, _top_p_specialized_reference(probs, tp))


def test_top_p_renorm_probs_specialized_mixed_per_row_top_p(top_p_specialized_enabled):
    _skip_unless_sm103()
    probs, tp = _top_p_specialized_inputs(
        12,
        248320,
        seed=4,
        top_p=[0.95, 0.1, 1.0, 0.5, 0.999, 0.8],
        kinds=(0, 1, 2, 3, 4),
    )
    out = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    _assert_top_p_specialized_contract(out, _top_p_specialized_reference(probs, tp))


def test_top_p_renorm_probs_specialized_ties_at_the_threshold(
    top_p_specialized_enabled,
):
    _skip_unless_sm103()
    batch_size, vocab_size = 5, 154880
    probs = torch.zeros(batch_size, vocab_size)
    probs[:, :5] = 0.125  # five exact ties
    probs[:, 5:3005] = 0.375 / 3000  # the rest of the mass below 2^-8
    top_p = torch.full((batch_size,), 0.3)
    out = flashinfer.sampling.top_p_renorm_probs(probs.cuda(), top_p.cuda())
    ref = _top_p_specialized_reference(probs, top_p)
    assert (ref[:, :5] > 0).all() and (ref[:, 5:] == 0).all()
    _assert_top_p_specialized_contract(out, ref)


def test_top_p_renorm_probs_specialized_tiny_top_p(top_p_specialized_enabled):
    # A top_p below the fixed-point resolution keeps the row maximum.
    _skip_unless_sm103()
    probs, _ = _top_p_specialized_inputs(4, 248320, seed=5, kinds=(1, 2))
    tp = torch.full((4,), 1e-30, device="cuda")
    out = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    _assert_top_p_specialized_contract(out, _top_p_specialized_reference(probs, tp))


@pytest.mark.parametrize("top_p_value", [0.0, 0.5, 0.95])
def test_top_p_renorm_probs_specialized_zero_mass_rows(
    top_p_value, top_p_specialized_enabled, monkeypatch
):
    # A row whose kept mass is <= 1e-8 (all zeros, or vanishing values) is left unnormalised,
    # exactly like the AIR path (norm 1.0), never 0/0 = NaN or x/0 = inf. The nucleus target is
    # relative to the row's mass there too: of the different vanishing values in row 2, top_p 0.5
    # keeps 2^-28 alone, like AIR (an absolute target would keep both).
    _skip_unless_sm103()
    batch_size, vocab_size = 4, 248320
    probs = torch.zeros(batch_size, vocab_size, device="cuda")
    probs[1] = 1e-14  # every entry vanishing: total mass 2.5e-9
    probs[2, 7] = 2.0**-28
    probs[2, vocab_size // 2] = 2.0**-29
    probs[3, ::7] = 1e-13
    tp = torch.full((batch_size,), top_p_value, device="cuda")
    out = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    stock = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, stock, atol=0.0, rtol=0.0)
    assert int((out[2] > 0).sum()) == (1 if top_p_value <= 0.5 else 2)


def test_top_p_renorm_probs_specialized_nan_entries_are_never_kept(
    top_p_specialized_enabled, monkeypatch
):
    # A NaN entry has no mass and is never kept, on the radix path (whose bit-prefix test alone
    # would keep it: +NaN's bits sort above every finite value) as on the head paths (a NaN
    # converts to 2^63 fixed-point units, so a NaN head entry or histogram entry would carry that
    # mass): the output is the float64 reference of the row with its NaNs set to 0. The AIR path
    # writes no NaN either, but its float32 sums become NaN, so it keeps every finite positive
    # entry of such a row (top_p is lost); only the absence of NaN is compared with it.
    _skip_unless_sm103()
    vocab_size = 154880  # 6 rows: an allowlisted shape
    g = torch.Generator().manual_seed(31)
    # radix, radix (keep all at top_p 1), radix, and the head paths: warp-level (a +NaN is a
    # head candidate: its bits are >= those of 2^-8) and block-wide
    kinds = [2, 2, 3, 1, 4]
    rows = [_top_p_specialized_row(kind, vocab_size, g) for kind in kinds]
    rows.append(torch.zeros(vocab_size, dtype=torch.float64))
    probs = torch.stack(rows).float()
    assert probs.shape == (6, vocab_size)
    nan_at = torch.arange(5) * (vocab_size // 5) + 3  # one NaN in several CTAs' slices
    probs[:5, nan_at] = float("nan")
    # NaNs with the sign bit set (bits 0xffffffff) in every CTA's slice too
    neg_nan_at = nan_at + vocab_size // 10
    probs[:5, neg_nan_at] = torch.tensor(-1, dtype=torch.int32).view(torch.float32)
    probs[5] = float("nan")  # all NaN: no mass, stays all zero
    clean = torch.nan_to_num(probs, nan=0.0)
    tp = torch.tensor([0.9, 1.0, 0.95, 0.95, 0.9, 0.5])
    tp = _top_p_specialized_robust_top_p(clean, tp)
    before = _top_p_specialized_stats()["dispatch_count"]
    out = flashinfer.sampling.top_p_renorm_probs(probs.cuda(), tp.cuda())
    assert _top_p_specialized_stats()["dispatch_count"] == before + 1
    assert not torch.isnan(out).any()
    assert (out[:5, nan_at] == 0).all() and (out[:5, neg_nan_at] == 0).all()
    assert not out[5].any()
    _assert_top_p_specialized_contract(out, _top_p_specialized_reference(clean, tp))
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    stock = flashinfer.sampling.top_p_renorm_probs(probs.cuda(), tp.cuda())
    assert not torch.isnan(stock).any()


def test_top_p_renorm_probs_specialized_unnormalised_cutoff_is_exact(
    top_p_specialized_enabled, monkeypatch
):
    # Rows whose kept mass lies within the 2^-56 fixed-point rounding of the 1e-8 cutoff: the
    # kernel decides on the exact kept mass, not on the rounded fixed-point sum.
    _skip_unless_sm103()
    batch_size, vocab_size = 5, 154880
    below = 1.9531246411852443e-11  # 512 x = 9.999998e-9; its fixed-point sum rounds above 1e-8
    above = 3.300330148059416e-11  # 303 x = 1e-8 + 29.5 * 2^-56; its fixed-point sum rounds below

    def spread(n):  # n positions across every CTA's slice of the row
        return torch.arange(n) * (vocab_size // n)

    probs = torch.zeros(batch_size, vocab_size, dtype=torch.float64)
    probs[0, spread(512)] = below
    probs[1, spread(303)] = above
    # row 2 stays all zero
    # row 3: far above the cutoff, decided by the fixed-point sum alone
    probs[3, spread(513)] = below
    # row 4: the row-0 entries plus a tail outside the nucleus (top_p 5e-9 keeps the ties)
    probs[4, spread(512)] = below
    probs[4, torch.arange(1000) * 154 + 1] = 1e-15
    probs = probs.float().cuda()
    tp = torch.tensor([0.5, 0.5, 0.5, 0.5, 5e-9], device="cuda")
    out = flashinfer.sampling.top_p_renorm_probs(probs, tp)

    # float64 reference: keep the nucleus, divide by its mass unless that mass is <= 1e-8f
    p = probs.double().cpu()
    sp, _ = torch.sort(p, dim=-1, descending=True)
    excl = torch.cumsum(sp, dim=-1) - sp
    k = (
        (excl < _top_p_specialized_target(p, tp.cpu()))
        .sum(dim=-1, keepdim=True)
        .clamp(min=1)
    )
    kept = torch.where(p >= sp.gather(-1, k - 1), p, torch.zeros_like(p))
    mass = kept.sum(dim=-1, keepdim=True)
    cut = float(torch.tensor(1e-8, dtype=torch.float32))
    assert mass[0].item() <= cut < mass[1].item() and mass[4].item() <= cut
    ref = torch.where(mass > cut, kept / mass, kept).float()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=1e-6)

    # The stock path leaves the below-cutoff and all-zero rows unnormalised too (its float32
    # sum of these equal entries is exact).
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    stock = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    torch.testing.assert_close(out[[0, 2]], stock[[0, 2]], atol=0.0, rtol=0.0)


def _top_p_specialized_non_unit_rows(vocab_size, g):
    """Rows that do not sum to 1, with their top_p (float64)."""
    v = vocab_size
    rows, top_ps = [], []
    # the review's row: AIR's target is 0.5 * 0.5 = 0.25, so 0.375 alone is kept (output 1)
    row = torch.zeros(v, dtype=torch.float64)
    row[0], row[1] = 0.375, 0.125
    rows.append(row)
    top_ps.append(0.5)
    # vanishing, different values: 2^-28 alone is kept, left unnormalised (kept mass <= 1e-8)
    row = torch.zeros(v, dtype=torch.float64)
    row[7], row[v // 2] = 2.0**-28, 2.0**-29
    rows.append(row)
    top_ps.append(0.5)
    # vanishing everywhere (mass 7e-4): the 2^-28 half is kept and renormalised
    rows.append(
        torch.where(torch.arange(v) % 2 == 0, 2.0**-28, 2.0**-29).to(torch.float64)
    )
    top_ps.append(0.5)
    # a serving row at half mass (one-entry nucleus) and at mass 3.7 (block-wide head resolve)
    rows.append(_top_p_specialized_row(1, v, g) * 0.5)
    top_ps.append(0.95)
    rows.append(_top_p_specialized_row(4, v, g) * 3.7)
    top_ps.append(0.9)
    # an unnormalised uniform row (mass ~v/2): too heavy for the 2^-56 fixed point
    rows.append(torch.rand(v, generator=g, dtype=torch.float64))
    top_ps.append(0.5)
    # a wide nucleus at mass 1e-3 (exact radix path)
    rows.append(_top_p_specialized_row(2, v, g) * 1e-3)
    top_ps.append(0.8)
    return torch.stack(rows).float(), torch.tensor(top_ps, dtype=torch.float32)


def test_top_p_renorm_probs_specialized_non_unit_mass_rows_match_air(
    top_p_specialized_enabled, monkeypatch
):
    # The nucleus target is top_p times the row's own mass, as on the AIR path (blockSum * p),
    # so rows that do not sum to 1 keep the same entries as there.
    _skip_unless_sm103()
    batch_size, vocab_size = 7, 248320
    probs, tp = _top_p_specialized_non_unit_rows(
        vocab_size, torch.Generator().manual_seed(12)
    )
    assert probs.shape[0] == batch_size
    tp = _top_p_specialized_robust_top_p(probs, tp)
    probs, tp = probs.cuda(), tp.cuda()
    before = _top_p_specialized_stats()["dispatch_count"]
    out = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    assert _top_p_specialized_stats()["dispatch_count"] == before + 1
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    stock = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    assert torch.isfinite(out).all()
    # the same entries as the AIR path, the same values up to its float32 sums
    assert torch.equal(out > 0, stock > 0)
    torch.testing.assert_close(out, stock, atol=1e-6, rtol=1e-4)
    _assert_top_p_specialized_contract(out, _top_p_specialized_reference(probs, tp))
    # the review's rows, exactly
    assert out[0, 0].item() == 1.0 and int((out[0] > 0).sum()) == 1
    assert out[1, 7].item() == 2.0**-28 and int((out[1] > 0).sum()) == 1
    assert int((out[2] > 0).sum()) == vocab_size // 2


def _top_p_specialized_vanishing_rows(vocab_size, g):
    """Rows whose positive entries all lie below the head paths' 2^-57 resolution, with their
    top_p (float32): the radix path's fixed point follows the row's own mass."""
    v = vocab_size
    rows, top_ps = [], []
    # the review's row: 2^-60 alone reaches top_p * mass, left unnormalised
    row = torch.zeros(v, dtype=torch.float64)
    row[0], row[1] = 2.0**-60, 2.0**-61
    rows.append(row)
    top_ps.append(0.5)
    # the same values spread over every CTA's slice
    row = torch.zeros(v, dtype=torch.float64)
    row[::150] = 2.0**-60
    row[75::150] = 2.0**-61
    rows.append(row)
    top_ps.append(0.5)
    # vanishing everywhere
    rows.append(
        torch.where(torch.arange(v) % 2 == 0, 2.0**-70, 2.0**-71).to(torch.float64)
    )
    top_ps.append(0.5)
    # a wide serving-like row scaled to mass 1e-25 (a nucleus deep in the tail)
    rows.append(_top_p_specialized_row(2, v, g) * 1e-25)
    top_ps.append(0.8)
    # subnormal entries (AIR's -use_fast_math build flushes them: compared with float64 only)
    row = torch.zeros(v, dtype=torch.float64)
    row[3], row[v - 5] = 2.0**-140, 2.0**-141
    rows.append(row)
    top_ps.append(0.5)
    # all zero: stays all zero
    rows.append(torch.zeros(v, dtype=torch.float64))
    top_ps.append(0.5)
    return torch.stack(rows).float(), torch.tensor(top_ps, dtype=torch.float32)


def test_top_p_renorm_probs_specialized_vanishing_rows_match_air(
    top_p_specialized_enabled, monkeypatch
):
    # Positive entries too small for the head paths' 2^-56 fixed point still carry mass: the
    # radix path selects in a per-row power-of-two fixed point (the row's mass in [2^58, 2^61)
    # units), so these rows keep the same entries as AIR and the float64 reference, left
    # unnormalised (kept mass <= 1e-8).
    _skip_unless_sm103()
    batch_size, vocab_size = 6, 154880
    probs, tp = _top_p_specialized_vanishing_rows(
        vocab_size, torch.Generator().manual_seed(21)
    )
    assert probs.shape[0] == batch_size
    assert (probs[4] > 0).sum() == 2 and float(probs[4].max()) < 2.0**-126
    tp = _top_p_specialized_robust_top_p(probs, tp)
    probs, tp = probs.cuda(), tp.cuda()
    before = _top_p_specialized_stats()["dispatch_count"]
    out = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    assert _top_p_specialized_stats()["dispatch_count"] == before + 1
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    stock = flashinfer.sampling.top_p_renorm_probs(probs, tp)
    ref = _top_p_specialized_reference(probs, tp).to(out.device)
    assert torch.isfinite(out).all()
    air_rows = [0, 1, 2, 3, 5]
    torch.testing.assert_close(out[air_rows], stock[air_rows], atol=0.0, rtol=0.0)
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
    assert out[0, 0].item() == 2.0**-60 and int((out[0] > 0).sum()) == 1
    assert int((out[1] > 0).sum()) == int((probs[1] == 2.0**-60).sum())
    assert int((out[2] > 0).sum()) == vocab_size // 2
    assert out[4, 3].item() == 2.0**-140 and int((out[4] > 0).sum()) == 1
    assert not out[5].any()


def test_top_p_renorm_probs_specialized_exact_mass_contract(top_p_specialized_enabled):
    # The nucleus target is top_p times the row's fixed-point mass (exact for these entries,
    # independent of any summation order), so the kept set is the float64 reference's. Float32
    # [0.5, 0.3, 0.2] sums to 1 + 2^-26 exactly: at top_p 0.5 the target is just above 0.5, so 0.3
    # is kept too and the
    # output is [0.625, 0.375]. AIR's float32 row sum rounds to 1.0 here and keeps 0.5 alone; at
    # such ties AIR's float32 result depends on its accumulation order (float atomics across
    # CTAs), so the kernel follows exact arithmetic instead (documented in the kernel header).
    _skip_unless_sm103()
    batch_size, vocab_size = 5, 154880
    probs = torch.zeros(batch_size, vocab_size)
    probs[0, :3] = torch.tensor([0.5, 0.3, 0.2])
    # the same row with its entries in other CTAs' slices
    probs[1, [vocab_size - 1, 0, vocab_size // 2]] = torch.tensor([0.5, 0.3, 0.2])
    # the same tie at mass 2 + 2^-25
    probs[2, :3] = torch.tensor([1.0, 0.6, 0.4])
    # an exact tie: the prefix {0.5} reaches the target 0.5 exactly, so 0.25 is not kept
    probs[3, :3] = torch.tensor([0.5, 0.25, 0.25])
    # row 4 stays all zero
    tp = torch.full((batch_size,), 0.5)
    out = flashinfer.sampling.top_p_renorm_probs(probs.cuda(), tp.cuda()).cpu()
    expected = torch.zeros(batch_size, vocab_size)
    expected[0, :2] = torch.tensor([0.625, 0.375])
    expected[1, [vocab_size - 1, 0]] = torch.tensor([0.625, 0.375])
    expected[2, :2] = torch.tensor([0.625, 0.375])
    expected[3, 0] = 1.0
    torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        out, _top_p_specialized_reference(probs, tp), atol=0.0, rtol=0.0
    )


def test_top_p_renorm_probs_specialized_fixed_point_target_and_resolution(
    top_p_specialized_enabled,
):
    # The radix path's masses are whole units of 2^-q of the row's scale (q = 59 for these rows of
    # mass ~1): an entry of at least 2^-35 of the mass converts exactly, a smaller one rounds to
    # the nearest unit, and the target top_p * mass is rounded up to a whole unit.
    _skip_unless_sm103()
    batch_size, vocab_size = 5, 154880
    probs = torch.zeros(batch_size, vocab_size, dtype=torch.float64)
    # row 0: 2^-16, then 131070 x 2^-17 and one 2^-46, at top_p 2^-16. The exact target is
    # 2^-16 + 2^-62, above the 2^-16 entry alone, so every 2^-17 entry is kept (their mass with
    # 2^-16 is exactly 1, so the output is the input there). The target rounded to the nearest
    # unit (2^43 + 1/8 -> 2^43) kept 2^-16 alone; rounded up (2^43 + 1) it does not.
    probs[0, 0] = 2.0**-16
    probs[0, 1:131071] = 2.0**-17
    probs[0, 131071] = 2.0**-46
    # row 1: [0.5, 0.25, 0.25] plus 131072 x 2^-61 (2^-44 in all) at top_p 0.5. Exact arithmetic
    # keeps the three entries (target 0.5 + 2^-45), but each 2^-61 entry is a quarter unit and
    # rounds to 0: the fixed-point target is 0.5, reached by 0.5 alone. The tail mass is within
    # the documented bound (V/2 units, 2^-41 of the row's mass), so this row is pinned to the
    # fixed-point result, which is also AIR's float32 result.
    probs[1, :3] = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    probs[1, 3:131075] = 2.0**-61
    # row 2: the same row with one 2^-35 entry (2^24 units, converted exactly): target
    # 0.5 + 2^-36, so the 0.25 ties are kept, as exact arithmetic keeps them (AIR's float32 sum
    # rounds 1 + 2^-35 to 1 and keeps 0.5 alone).
    probs[2, :3] = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    probs[2, vocab_size - 1] = 2.0**-35
    # row 3: an exact tie at the target (0.5 alone reaches 0.5 * 1.0); row 4: all zero
    probs[3, :3] = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    probs = probs.float()
    tp = torch.tensor([2.0**-16, 0.5, 0.5, 0.5, 0.5])
    out = flashinfer.sampling.top_p_renorm_probs(probs.cuda(), tp.cuda()).cpu()
    ref = _top_p_specialized_reference(probs, tp)
    exact_rows = [0, 2, 3, 4]
    torch.testing.assert_close(out[exact_rows], ref[exact_rows], atol=0.0, rtol=0.0)
    assert int((out[0] > 0).sum()) == 131071
    torch.testing.assert_close(out[0, :131071], probs[0, :131071], atol=0.0, rtol=0.0)
    assert out[0, 131071].item() == 0.0
    expected_1 = torch.zeros(vocab_size)
    expected_1[0] = 1.0
    torch.testing.assert_close(out[1], expected_1, atol=0.0, rtol=0.0)
    # exact arithmetic (an exactly rounded sum, which is exact here) puts the target above 0.5
    assert math.fsum(probs[1].double().tolist()) == 1.0 + 2.0**-44
    assert out[2, :3].tolist() == [0.5, 0.25, 0.25] and out[3, 0].item() == 1.0


def test_top_p_renorm_probs_specialized_launch_plan_matches_the_kernel(
    top_p_specialized_enabled,
):
    _skip_unless_sm103()
    module = _top_p_renorm_specialized._get_module()
    for b, v in _TOP_P_SPECIALIZED_SHAPES + [
        (1, 1000),
        (200, 32000),
        (9, 262144),
        (21, 200001),
        (41, 200001),
    ]:
        c, nt, mb, unr, vec = _top_p_renorm_specialized._plan_config(b, v)
        assert module.top_p_renorm_probs_specialized_plan(b, v) == (
            c * 100000000 + nt * 10000 + mb * 1000 + unr * 10 + int(vec)
        )


def test_top_p_renorm_probs_specialized_cuda_graph_after_eager_dispatch(
    top_p_specialized_enabled,
):
    _skip_unless_sm103()
    batch_size, vocab_size = 7, 248320
    probs, top_p = _top_p_specialized_inputs(batch_size, vocab_size, seed=6)
    # eager: builds and loads the module
    flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    assert _top_p_specialized_stats()["precompiled"]
    before = _top_p_specialized_stats()["dispatch_count"]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    torch.cuda.current_stream().wait_stream(s)
    assert _top_p_specialized_stats()["dispatch_count"] == before + 1

    new_probs, new_top_p = _top_p_specialized_inputs(batch_size, vocab_size, seed=7)
    probs.copy_(new_probs)
    top_p.copy_(new_top_p)
    graph.replay()
    torch.cuda.synchronize()
    _assert_top_p_specialized_contract(
        out, _top_p_specialized_reference(new_probs, new_top_p)
    )


def test_top_p_renorm_probs_specialized_capture_before_jit(
    top_p_specialized_enabled, monkeypatch
):
    _skip_unless_sm103()
    probs, top_p = _top_p_specialized_inputs(6, 154880, seed=8)
    # warm the AIR path eagerly so its module is loaded outside the capture
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    monkeypatch.delenv(_TOP_P_SPECIALIZED_KILL_SWITCH)
    monkeypatch.setattr(_top_p_renorm_specialized, "_module", None)

    def _must_not_build():
        raise AssertionError("the specialized module was built inside a capture")

    monkeypatch.setattr(_top_p_renorm_specialized, "_get_module", _must_not_build)
    before = _top_p_specialized_stats()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    torch.cuda.current_stream().wait_stream(s)
    after = _top_p_specialized_stats()
    assert after["dispatch_count"] == before["dispatch_count"]
    assert (
        after["fallback_reasons"]["capturing_before_jit"]
        == before["fallback_reasons"].get("capturing_before_jit", 0) + 1
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out, _top_p_specialized_reference(probs, top_p).cuda(), atol=1e-3, rtol=1e-3
    )


def test_top_p_renorm_probs_specialized_capture_on_another_device(
    top_p_specialized_enabled, monkeypatch
):
    # A graph captured on cuda:1 while cuda:0 is current, before the module is built: the call
    # on cuda:1 takes the (capture-safe) stock path instead of building inside the capture.
    _skip_unless_sm103()
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability(1) != (10, 3):
        pytest.skip("needs two SM103 GPUs")
    probs, top_p = _top_p_specialized_inputs(6, 154880, seed=13)
    probs, top_p = probs.to("cuda:1"), top_p.to("cuda:1")
    # warm the AIR path on cuda:1 eagerly so its module is loaded outside the capture
    monkeypatch.setenv(_TOP_P_SPECIALIZED_KILL_SWITCH, "1")
    flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    monkeypatch.delenv(_TOP_P_SPECIALIZED_KILL_SWITCH)
    monkeypatch.setattr(_top_p_renorm_specialized, "_module", None)

    def _must_not_build():
        raise AssertionError("the specialized module was built inside a capture")

    monkeypatch.setattr(_top_p_renorm_specialized, "_get_module", _must_not_build)
    before = _top_p_specialized_stats()
    with torch.cuda.device(1):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s), torch.cuda.device(0):
            assert not torch.cuda.is_current_stream_capturing()
            out = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
        torch.cuda.current_stream().wait_stream(s)
    after = _top_p_specialized_stats()
    assert after["dispatch_count"] == before["dispatch_count"]
    assert (
        after["fallback_reasons"]["capturing_before_jit"]
        == before["fallback_reasons"].get("capturing_before_jit", 0) + 1
    )
    graph.replay()
    torch.cuda.synchronize(1)
    torch.testing.assert_close(
        out,
        _top_p_specialized_reference(probs, top_p).to(out.device),
        atol=1e-3,
        rtol=1e-3,
    )


def test_top_p_renorm_probs_specialized_other_compute_capability(
    top_p_specialized_enabled, monkeypatch
):
    _skip_unless_sm103()
    batch_size, vocab_size = _TOP_P_SPECIALIZED_SHAPES[0]
    probs, top_p = _top_p_specialized_inputs(batch_size, vocab_size, seed=9)
    monkeypatch.setattr(
        _top_p_renorm_specialized, "get_compute_capability", lambda device: (10, 0)
    )
    # keep the faked capability out of the per-device cache other tests read
    monkeypatch.setattr(_top_p_renorm_specialized, "_device_capability", {})
    assert (
        _top_p_renorm_specialized._reject_reason(probs, top_p, False)
        == "compute_capability"
    )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_renorm_probs(batch_size, vocab_size, k, distribution, dtype):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = distribution((batch_size, vocab_size), "cuda:0")
    normalized_prob_fp32 = torch.softmax(logits, dim=-1)
    normalized_prob = normalized_prob_fp32.to(dtype)

    renorm_prob = flashinfer.sampling.top_k_renorm_probs(normalized_prob, k)

    # Check output dtype matches input
    assert renorm_prob.dtype == dtype

    # Check that the output sums to 1
    sums = renorm_prob.float().sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-2, atol=1e-2)

    # Count non-zero elements in output
    nonzero_counts = (renorm_prob > 0).sum(dim=-1)

    # Find the pivot value (k-th largest) and count ties
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]

    # Count how many elements are strictly greater than pivot
    num_greater = (normalized_prob > pivot.unsqueeze(-1)).sum(dim=-1)
    # Count how many elements equal the pivot (ties)
    num_ties = (normalized_prob == pivot.unsqueeze(-1)).sum(dim=-1)

    # Valid range: [num_greater, num_greater + num_ties]
    # The kernel must keep all elements > pivot, and may keep some/all/none of the ties
    # But it must keep exactly k elements total (if there are enough)
    nonzero_input = (normalized_prob > 0).sum(dim=-1)
    expected_k = torch.minimum(
        torch.full_like(nonzero_input, k, dtype=torch.int64), nonzero_input
    )

    # Check: nonzero_counts should be in valid range considering ties
    max_valid = num_greater + num_ties

    # The actual count should be >= k (we keep at least k) and within tie range
    # Due to floating point, allow small tolerance
    assert torch.all(nonzero_counts >= torch.clamp(expected_k - 1, min=0)), (
        f"Some rows have fewer non-zero elements than expected. "
        f"nonzero_counts min: {nonzero_counts.min()}, expected_k min: {expected_k.min()}"
    )
    assert torch.all(nonzero_counts <= max_valid + 1), (
        f"Some rows have more non-zero elements than allowed by ties. "
        f"nonzero_counts max: {nonzero_counts.max()}, max_valid max: {max_valid.max()}"
    )


@pytest.mark.parametrize("top_k_arr", [True, False])
@pytest.mark.parametrize("is_deterministic", [None, False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_renorm_probs_custom_op(top_k_arr, is_deterministic, dtype):
    torch.manual_seed(42)
    probs = torch.softmax(torch.randn(4, 128, device="cuda:0"), dim=-1).to(dtype)
    if top_k_arr:
        top_k = torch.tensor([1, 10, 50, 128], dtype=torch.int32, device=probs.device)
        maybe_top_k_arr, top_k_val = top_k, 0
    else:
        top_k = 10
        maybe_top_k_arr, top_k_val = None, top_k
    row_states_buffer = torch.zeros(1024 * 1024, dtype=torch.uint8, device=probs.device)
    op = flashinfer.sampling.get_sampling_module().top_k_renorm_probs

    if is_deterministic is None:
        # Preserve the four-argument call used by ProbsTopKOp.
        actual = op(probs, maybe_top_k_arr, top_k_val, row_states_buffer)
    else:
        actual = op(
            probs, maybe_top_k_arr, top_k_val, row_states_buffer, is_deterministic
        )

    expected = flashinfer.sampling.top_k_renorm_probs(
        probs, top_k, is_deterministic=bool(is_deterministic)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 500])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_renorm_probs_deterministic(batch_size, vocab_size, k, dtype):
    """is_deterministic=True must be bitwise reproducible across invocations.

    The default multi-CTA path accumulates the renormalization sum with float
    atomicAdd, so CTA arrival order changes the last-bit rounding and repeated
    calls on identical input differ bitwise. Replicated execution (e.g.
    tensor-parallel ranks running the sampler redundantly on identical inputs)
    then diverges, which desynchronizes downstream state. The deterministic
    mode accumulates in fixed-point integer atomics instead.
    """
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = normal_distribution(1)((batch_size, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1).to(dtype)

    base = flashinfer.sampling.top_k_renorm_probs(probs, k, is_deterministic=True)
    for _ in range(10):
        again = flashinfer.sampling.top_k_renorm_probs(probs, k, is_deterministic=True)
        assert torch.equal(again, base), "deterministic renorm differed across calls"

    # Same kept support as the default kernel; values match within the
    # fixed-point quantization noise (far below the test tolerance).
    ref = flashinfer.sampling.top_k_renorm_probs(probs, k)
    assert torch.equal(base > 0, ref > 0)
    torch.testing.assert_close(base, ref, rtol=1e-3, atol=1e-3)

    sums = base.float().sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_renorm_probs_mixed_k_persistent_loop(dtype):
    """Test top_k_renorm_probs with mixed k values in persistent loop (multi-CTA mode).

    This test catches a specific bug where:
    - Large batch size triggers the persistent loop (multiple iterations per CTA group)
    - Large vocab_size triggers multi-CTA mode (multiple CTAs per row)
    - Mixed k values: some rows have k >= vocab_size (skip radix select),
      others have k < vocab_size (use radix select)

    The bug was that k >= vocab_size iterations would skip radix select
    without clearing the histogram buffers, leaving stale data that corrupted
    subsequent k < vocab_size iterations.
    """
    batch_size = 1024  # Large batch to trigger persistent loop
    vocab_size = 128 * 1024  # Large vocab to trigger multi-CTA mode

    torch.manual_seed(42)
    generator = torch.Generator(device="cuda:0").manual_seed(42)

    # Generate random logits
    logits = torch.rand((batch_size, vocab_size), device="cuda:0", generator=generator)

    # Generate k values: mix of small k and k == vocab_size
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    k_values = torch.randint(
        1, 1000, (batch_size,), device="cuda:0", generator=generator
    )

    # Randomly set some rows to k == vocab_size (about 50%)
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    mask = torch.randint(
        0, 2, (batch_size,), generator=generator, dtype=torch.bool, device="cuda:0"
    )
    k_values.masked_fill_(mask, vocab_size)

    # Convert to probs
    probs = torch.softmax(logits, dim=-1).to(dtype)

    # Run FlashInfer top_k_renorm_probs
    renorm_probs = flashinfer.sampling.top_k_renorm_probs(probs, k_values)

    # Verify output dtype
    assert renorm_probs.dtype == dtype

    # Verify sum to 1
    sums = renorm_probs.float().sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-2, atol=1e-2)

    # Verify non-zero count matches k for each row
    nonzero_counts = (renorm_probs > 0).sum(dim=-1)

    # For rows with k >= vocab_size, all elements should be non-zero
    # For rows with k < vocab_size, non-zero count should be >= k (may be more due to ties)
    for i in range(batch_size):
        k = k_values[i].item()
        count = nonzero_counts[i].item()

        if k >= vocab_size:
            # All elements should be non-zero
            assert count == vocab_size, (
                f"Row {i}: k >= vocab_size but count={count} != {vocab_size}"
            )
        else:
            # Count should be at least k (may be more due to ties at the threshold)
            row_probs = probs[i].float()
            topk_vals, _ = torch.topk(row_probs, k, sorted=True)
            threshold = topk_vals[-1]
            expected_ge_threshold = (row_probs >= threshold).sum().item()

            # Allow small tolerance for floating point
            assert count >= k - 1, f"Row {i}: k={k} but only {count} non-zero elements"
            assert count <= expected_ge_threshold + 1, (
                f"Row {i}: k={k}, expected at most {expected_ge_threshold} but got {count}"
            )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("neginf_input", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_mask_logits(
    batch_size, vocab_size, k, distribution, neginf_input, dtype
):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = distribution((batch_size, vocab_size), "cuda:0")
    if neginf_input:
        num_neginf = torch.randint(1, vocab_size * batch_size, (1,)).item()
        idxs = torch.randperm(batch_size * vocab_size, device="cuda:0")[:num_neginf]
        logits[idxs // vocab_size, idxs % vocab_size] = -float("inf")

    logits = logits.to(dtype)
    masked_logits = flashinfer.sampling.top_k_mask_logits(logits, k)

    # Check output dtype matches input
    assert masked_logits.dtype == dtype

    # Check that softmax of masked logits sums to 1
    probs = torch.softmax(masked_logits.float(), dim=-1)
    sums = probs.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-3, atol=1e-3)

    # Count finite elements in output
    finite_counts = torch.isfinite(masked_logits).sum(dim=-1)

    # Find the pivot value (k-th largest among finite values) and count ties
    # Replace -inf with a very small value for sorting
    logits_for_sort = logits.clone()
    logits_for_sort[~torch.isfinite(logits_for_sort)] = -float("inf")
    sorted_logits, _ = torch.sort(logits_for_sort, descending=True)

    # Count finite inputs per row
    finite_inputs = torch.isfinite(logits).sum(dim=-1)

    # For each row, find the pivot (k-th largest if enough finite values)
    effective_k = torch.minimum(
        torch.full_like(finite_inputs, k, dtype=torch.int64), finite_inputs
    )

    # Get pivot for each row (handle case where effective_k might be 0)
    pivot = torch.zeros(batch_size, dtype=dtype, device=logits.device)
    for i in range(batch_size):
        ek = effective_k[i].item()
        if ek > 0:
            pivot[i] = sorted_logits[i, ek - 1]
        else:
            pivot[i] = float("-inf")

    # Count how many elements are strictly greater than pivot
    num_greater = (logits > pivot.unsqueeze(-1)).sum(dim=-1)
    # Count how many elements equal the pivot (ties) - only among finite values
    num_ties = ((logits == pivot.unsqueeze(-1)) & torch.isfinite(logits)).sum(dim=-1)

    # Valid range considering ties
    max_valid = num_greater + num_ties

    # Check: finite_counts should be >= effective_k (we keep at least k finite values)
    # and <= max_valid (we don't keep more than all elements >= pivot)
    # Allow small tolerance for floating point issues
    assert torch.all(finite_counts >= torch.clamp(effective_k - 1, min=0)), (
        f"Some rows have fewer finite elements than expected. "
        f"finite_counts min: {finite_counts.min()}, effective_k min: {effective_k.min()}"
    )
    assert torch.all(finite_counts <= max_valid + 1), (
        f"Some rows have more finite elements than allowed by ties. "
        f"finite_counts max: {finite_counts.max()}, max_valid max: {max_valid.max()}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_mask_logits_mixed_k_persistent_loop(dtype):
    """Test top_k_mask_logits with mixed k values in persistent loop (multi-CTA mode).

    This test catches the same bug as test_top_k_renorm_probs_mixed_k_persistent_loop
    but for the mask_logits variant.
    """
    batch_size = 1024  # Large batch to trigger persistent loop
    vocab_size = 128 * 1024  # Large vocab to trigger multi-CTA mode

    torch.manual_seed(42)
    generator = torch.Generator(device="cuda:0").manual_seed(42)

    # Generate random logits
    logits = torch.rand((batch_size, vocab_size), device="cuda:0", generator=generator)
    logits = logits.to(dtype)

    # Generate k values: mix of small k and k == vocab_size
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    k_values = torch.randint(
        1, 1000, (batch_size,), device="cuda:0", generator=generator
    )

    # Randomly set some rows to k == vocab_size (about 50%)
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    mask = torch.randint(
        0, 2, (batch_size,), generator=generator, dtype=torch.bool, device="cuda:0"
    )
    k_values.masked_fill_(mask, vocab_size)

    # Run FlashInfer top_k_mask_logits
    masked_logits = flashinfer.sampling.top_k_mask_logits(logits, k_values)

    # Verify output dtype
    assert masked_logits.dtype == dtype

    # Verify finite count matches k for each row
    finite_counts = torch.isfinite(masked_logits).sum(dim=-1)

    for i in range(batch_size):
        k = k_values[i].item()
        count = finite_counts[i].item()

        if k >= vocab_size:
            # All elements should be finite
            assert count == vocab_size, (
                f"Row {i}: k >= vocab_size but finite count={count} != {vocab_size}"
            )
        else:
            # Count should be at least k (may be more due to ties at the threshold)
            row_logits = logits[i].float()
            topk_vals, _ = torch.topk(row_logits, k, sorted=True)
            threshold = topk_vals[-1]
            expected_ge_threshold = (row_logits >= threshold).sum().item()

            # Allow small tolerance for floating point
            assert count >= k - 1, f"Row {i}: k={k} but only {count} finite elements"
            assert count <= expected_ge_threshold + 1, (
                f"Row {i}: k={k}, expected at most {expected_ge_threshold} but got {count}"
            )


@pytest.mark.parametrize("initial_offset", [0, 12])
def test_get_seed_and_offset_reserves_current_range(initial_offset):
    seed = 12345
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    generator.set_offset(initial_offset)
    expected_offset = initial_offset

    for increment in [1, 5, 32, 2, 0]:
        actual_seed, actual_offset = flashinfer.sampling.get_seed_and_offset(
            increment, generator
        )
        assert (actual_seed, actual_offset) == (seed, expected_offset)
        expected_offset += (increment + 3) // 4 * 4
        assert generator.get_offset() == expected_offset
        assert generator.initial_seed() == seed


def test_chain_speculative_sampling_reserves_current_range():
    seed = 12345
    generator = torch.Generator(device="cuda:0").manual_seed(seed)

    # Rejection consumes N+2 draws: reserve 8 for N=3, then 4 for N=1.
    for n, offset, next_offset in [(3, 0, 8), (1, 8, 12)]:
        draft_probs = torch.zeros((1, n, 3), device="cuda:0")
        draft_probs[..., 0] = 1
        draft_token_ids = torch.zeros((1, n), dtype=torch.int32, device="cuda:0")
        target_probs = torch.tensor(
            [[[0.0, 0.5, 0.5]] + [[0.5, 0.5, 0.0]] * n], device="cuda:0"
        )
        args = (draft_probs, draft_token_ids, target_probs)

        actual = flashinfer.sampling.chain_speculative_sampling(
            *args, generator=generator
        )
        assert generator.get_offset() == next_offset
        expected = flashinfer.sampling.chain_speculative_sampling(
            *args, generator=generator, seed=seed, offset=offset
        )
        # Explicit seed/offset must bypass generator reservation.
        assert generator.get_offset() == next_offset
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("num_speculate_tokens", [1, 3, 5, 7])
@pytest.mark.parametrize("onehot_target", [False, True])
def test_chain_speculative_sampling(
    batch_size,
    vocab_size,
    num_speculate_tokens,
    onehot_target,
):
    pre_norm_draft_prob = torch.rand(
        batch_size, num_speculate_tokens, vocab_size, device="cuda:0"
    )
    normalized_draft_prob = pre_norm_draft_prob / pre_norm_draft_prob.sum(
        dim=-1, keepdim=True
    )
    draft_token_ids = torch.randint(
        vocab_size, (batch_size, num_speculate_tokens), device="cuda:0"
    )
    if not onehot_target:
        pre_norm_target_prob = torch.rand(
            batch_size, num_speculate_tokens + 1, vocab_size, device="cuda:0"
        )
        target_onehot_prob = pre_norm_target_prob / pre_norm_target_prob.sum(
            dim=-1, keepdim=True
        )
    else:
        target_token_ids = torch.randint(
            vocab_size, (batch_size, num_speculate_tokens + 1), device="cuda:0"
        )
        target_token_ids[..., :num_speculate_tokens] = draft_token_ids
        target_onehot_prob = torch.zeros(
            (batch_size, num_speculate_tokens + 1, vocab_size), device="cuda:0"
        )
        target_onehot_prob.scatter_(2, target_token_ids.unsqueeze(-1), 1)

    # NOTE(Zihao): this is a very simple test that only checks whether output is valid or not.
    for trials in range(10):  # noqa: B007
        accepted_num = torch.zeros(batch_size, dtype=torch.int32, device="cuda:0")
        emitted_num = torch.zeros(batch_size, dtype=torch.int32, device="cuda:0")
        (
            output_token_ids,
            accepted_num,
            emitted_num,
        ) = flashinfer.sampling.chain_speculative_sampling(
            normalized_draft_prob,
            draft_token_ids,
            target_onehot_prob,
            accepted_num,
            emitted_num,
        )
        if onehot_target:
            assert torch.all(output_token_ids == target_token_ids)
        else:
            assert torch.all(output_token_ids[output_token_ids >= 0] < vocab_size)
            assert output_token_ids.shape == (batch_size, num_speculate_tokens + 1)
            matches = output_token_ids[..., :-1] != draft_token_ids
            for row in range(batch_size):
                mismatch_idx = torch.nonzero(matches[row], as_tuple=True)[0]
                if len(mismatch_idx) > 0:
                    # mismatch_idx should be contiguous
                    assert torch.all(mismatch_idx[1:] == mismatch_idx[:-1] + 1)
                    # from the second mismatched token on, the output tokens should be -1
                    assert torch.all(output_token_ids[row, mismatch_idx[0] + 1 :] == -1)

        assert torch.all(emitted_num + 1 == (output_token_ids != -1).sum(dim=1))


@pytest.mark.parametrize(
    "vocab_size", [1023, 1024, 1025, 4095, 4096, 4097, 32000, 32001, 128256, 151937]
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_chain_speculative_sampling_residual_normalizer(vocab_size, deterministic):
    # The residual relu(target - draft) is spread over many indices with unequal mass,
    # so the value of its sum decides where the inverse-CDF walk stops. A reduction
    # that drops tiles, vector lanes, or the cross-thread step samples from a truncated
    # distribution and lowers the mean sampled index. The stride is not a multiple of
    # the vector width, so every lane carries mass.
    batch_size = 512
    draft_probs = torch.zeros(batch_size, 1, vocab_size, device="cuda:0")
    draft_probs[..., 0] = 1.0
    draft_token_ids = torch.zeros((batch_size, 1), device="cuda:0", dtype=torch.int32)
    target_probs = torch.zeros(batch_size, 2, vocab_size, device="cuda:0")
    idx = torch.arange(1, vocab_size, 65, device="cuda:0")
    weights = torch.arange(1, idx.numel() + 1, device="cuda:0", dtype=torch.float32)
    probs = weights / weights.sum()
    target_probs[:, 0, idx] = probs
    target_probs[:, 1, 0] = 1.0
    expected_mean = (idx.double() * probs.double()).sum()
    variance = ((idx.double() - expected_mean) ** 2 * probs.double()).sum()

    samples = []
    for seed in (12345, 999, 7):
        output_token_ids, accepted_num, _ = (
            flashinfer.sampling.chain_speculative_sampling(
                draft_probs,
                draft_token_ids,
                target_probs,
                deterministic=deterministic,
                seed=seed,
                offset=17,
            )
        )
        # The draft token has no target mass, so every row resamples the residual.
        assert torch.all(accepted_num == 0)
        assert torch.all(output_token_ids[:, 1:] == -1)
        assert torch.all(
            torch.isin(output_token_ids[:, 0], idx.to(output_token_ids.dtype))
        )
        samples.append(output_token_ids[:, 0].double())

    drawn = torch.cat(samples)
    # Six standard errors: a correct kernel fails this about once in 5e8 runs.
    standard_error = (variance / drawn.numel()).sqrt()
    assert (drawn.mean() - expected_mean).abs() < 6 * standard_error


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
def test_tensor_validation_min_p(batch_size, vocab_size, p):
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # 1: Float p works and returns samples of shape (batch_size,).
    samples = flashinfer.sampling.min_p_sampling_from_probs(normalized_prob, p)
    assert samples.shape == (batch_size,)

    # 2: 2D tensor raises error.
    with pytest.raises(
        ValueError, match=r"Expected a 1D tensor or scalar.*got a 2D tensor"
    ):
        flashinfer.sampling.min_p_sampling_from_probs(
            normalized_prob,
            torch.tensor(
                [[p] * vocab_size] * batch_size, dtype=torch.float32, device="cuda:0"
            ),
        )

    # 3: 0D tensor raises error.
    with pytest.raises(
        ValueError,
        match=r"Expected a 1D tensor of shape \(batch_size,\) or scalar.*got a 0-dimensional tensor",
    ):
        flashinfer.sampling.min_p_sampling_from_probs(
            normalized_prob, torch.tensor(p, dtype=torch.float32, device="cuda:0")
        )

    # 4: 1D tensor with a broken batch size raises error (only when batch_size > 1).
    if batch_size > 1:
        with pytest.raises(
            ValueError, match="Sampling parameter tensor batch size mismatch"
        ):
            flashinfer.sampling.min_p_sampling_from_probs(
                normalized_prob, torch.tensor([p], dtype=torch.float32, device="cuda:0")
            )

    # 5: 1D tensor with the correct batch size works.
    samples = flashinfer.sampling.min_p_sampling_from_probs(
        normalized_prob,
        torch.tensor([p] * batch_size, dtype=torch.float32, device="cuda:0"),
    )
    assert samples.shape == (batch_size,)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_check_tensor_param_top_p(batch_size, vocab_size, p):
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # 1: Float p has the same shape as probs.
    samples = flashinfer.sampling.top_p_renorm_probs(normalized_prob, p)
    assert samples.shape == normalized_prob.shape

    # 2: 2D tensor raises error.
    with pytest.raises(
        ValueError, match=r"Expected a 1D tensor or scalar.*got a 2D tensor"
    ):
        flashinfer.sampling.top_p_renorm_probs(
            normalized_prob,
            torch.tensor(
                [[p] * vocab_size] * batch_size, dtype=torch.int, device="cuda:0"
            ),
        )

    # 3: 0D tensor raises error.
    with pytest.raises(
        ValueError,
        match=r"Expected a 1D tensor of shape \(batch_size,\) or scalar.*got a 0-dimensional tensor",
    ):
        flashinfer.sampling.top_p_renorm_probs(
            normalized_prob, torch.tensor(p, dtype=torch.int, device="cuda:0")
        )

    # 4: 1D tensor with a broken batch size raises error (only when batch_size > 1).
    if batch_size > 1:
        with pytest.raises(ValueError, match="Sampling parameter.*batch size mismatch"):
            flashinfer.sampling.top_p_renorm_probs(
                normalized_prob, torch.tensor([p], dtype=torch.int, device="cuda:0")
            )

    # 5: 1D tensor with the correct batch size works.
    samples = flashinfer.sampling.top_p_renorm_probs(
        normalized_prob,
        torch.tensor([p] * batch_size, dtype=torch.int, device="cuda:0"),
    )
    assert samples.shape == normalized_prob.shape


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_check_tensor_param_top_k(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # 1: Scalar k has the same shape as probs.
    samples = flashinfer.sampling.top_k_renorm_probs(normalized_prob, k)
    assert samples.shape == normalized_prob.shape

    # 2: 2D tensor raises error.
    with pytest.raises(
        ValueError, match=r"Expected a 1D tensor or scalar.*got a 2D tensor"
    ):
        flashinfer.sampling.top_k_renorm_probs(
            normalized_prob,
            torch.tensor(
                [[k] * vocab_size] * batch_size, dtype=torch.int, device="cuda:0"
            ),
        )

    # 3: 0D tensor raises error.
    with pytest.raises(
        ValueError,
        match=r"Expected a 1D tensor of shape \(batch_size,\) or scalar.*got a 0-dimensional tensor",
    ):
        flashinfer.sampling.top_k_renorm_probs(
            normalized_prob, torch.tensor(k, dtype=torch.int, device="cuda:0")
        )

    # 4: 1D tensor with a wrong shape raises error (only when batch_size > 1).
    if batch_size > 1:
        with pytest.raises(ValueError, match="Sampling parameter.*batch size mismatch"):
            flashinfer.sampling.top_k_renorm_probs(
                normalized_prob, torch.tensor([k], dtype=torch.int, device="cuda:0")
            )

    # 5: 1D tensor with the correct batch size works.
    samples = flashinfer.sampling.top_k_renorm_probs(
        normalized_prob,
        torch.tensor([k] * batch_size, dtype=torch.int, device="cuda:0"),
    )
    assert samples.shape == normalized_prob.shape


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling_from_probs_seed_offset_reproducibility(batch_size, vocab_size):
    """Test that explicit seed/offset produces reproducible results."""
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    seed, offset = 12345, 0

    samples1 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=seed, offset=offset
    )
    samples2 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=seed, offset=offset
    )

    assert torch.all(samples1 == samples2), (
        "Same seed/offset should produce identical samples"
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling_from_logits_seed_offset_reproducibility(batch_size, vocab_size):
    """Test that explicit seed/offset produces reproducible results."""
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0")

    seed, offset = 12345, 0

    samples1 = flashinfer.sampling.sampling_from_logits(
        logits, seed=seed, offset=offset
    )
    samples2 = flashinfer.sampling.sampling_from_logits(
        logits, seed=seed, offset=offset
    )

    assert torch.all(samples1 == samples2), (
        "Same seed/offset should produce identical samples"
    )


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling_different_seed_offset_produces_different_results(vocab_size):
    """Test that different seed/offset values produce different samples."""
    torch.manual_seed(42)
    # Size the two all-match checks before drawing outputs. Larger batch shapes
    # remain covered by the seed/offset reproducibility tests above.
    batch_size = 16
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    num_rows = required_num_collision_rows(normalized_prob)
    normalized_prob = normalized_prob[:num_rows]

    samples_seed1 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=12345, offset=0
    )
    samples_seed2 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=67890, offset=0
    )

    samples_offset1 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=12345, offset=0
    )
    samples_offset2 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=12345, offset=1000
    )

    seed_match_rate = (samples_seed1 == samples_seed2).float().mean().item()
    offset_match_rate = (samples_offset1 == samples_offset2).float().mean().item()

    assert seed_match_rate < 1, (
        f"Different seeds should produce mostly different samples, "
        f"got {seed_match_rate:.2%} match rate"
    )
    assert offset_match_rate < 1, (
        f"Different offsets should produce mostly different samples, "
        f"got {offset_match_rate:.2%} match rate"
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000])
@pytest.mark.parametrize(
    "sampling_type",
    ["from_probs", "from_logits", "top_p", "top_k", "min_p", "top_k_top_p"],
)
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_int64_indices_sampling(batch_size, vocab_size, sampling_type, indices_dtype):
    """Test that all sampling functions work with int64 indices."""
    torch.manual_seed(42)

    logits = torch.randn(batch_size, vocab_size, device="cuda:0")
    probs = torch.softmax(logits, dim=-1)
    indices = torch.arange(batch_size, dtype=indices_dtype, device="cuda:0")

    if sampling_type == "from_probs":
        samples = flashinfer.sampling.sampling_from_probs(probs, indices=indices)
    elif sampling_type == "from_logits":
        samples = flashinfer.sampling.sampling_from_logits(logits, indices=indices)
    elif sampling_type == "top_p":
        samples = flashinfer.sampling.top_p_sampling_from_probs(
            probs, 0.9, indices=indices
        )
    elif sampling_type == "top_k":
        k = min(100, vocab_size)
        samples = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, indices=indices
        )
    elif sampling_type == "min_p":
        samples = flashinfer.sampling.min_p_sampling_from_probs(
            probs, 0.1, indices=indices
        )
    elif sampling_type == "top_k_top_p":
        k = min(100, vocab_size)
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, k, 0.9, indices=indices, filter_apply_order="joint"
        )

    assert samples.dtype == indices_dtype, (
        f"Output dtype {samples.dtype} doesn't match indices dtype {indices_dtype}"
    )
    assert samples.shape == (batch_size,)
    assert torch.all(samples < vocab_size) and torch.all(samples >= 0)


@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("vocab_size", [111, 32000])
def test_sampling_with_default_device_cuda(batch_size, vocab_size):
    """Test that sampling works correctly when torch.set_default_device("cuda") is set.

    This is a regression test for issue #2333 where generator.set_state() would fail
    with "RNG state must be a torch.ByteTensor" error when the default device is CUDA.
    """
    torch.manual_seed(42)
    original_device = torch.get_default_device()
    try:
        # Set default device to CUDA
        torch.set_default_device("cuda")

        # Create logits and test top_k_top_p_sampling_from_logits
        logits = torch.randn(batch_size, vocab_size, device="cuda:0")

        # This should not raise "RNG state must be a torch.ByteTensor" error
        samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, top_k=100, top_p=0.9
        )

        assert samples.shape == (batch_size,)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)

        # Also test other sampling functions
        probs = torch.softmax(logits, dim=-1)

        samples = flashinfer.sampling.sampling_from_probs(probs)
        assert samples.shape == (batch_size,)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)

        samples = flashinfer.sampling.top_p_sampling_from_probs(probs, 0.9)
        assert samples.shape == (batch_size,)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)

    finally:
        # Restore original default device
        torch.set_default_device(original_device)


@pytest.mark.parametrize("batch_size", [1, 4, 19])
@pytest.mark.parametrize("vocab_size", [111, 32000])
def test_sampling_nan_input(batch_size, vocab_size):
    torch.manual_seed(42)
    probs = torch.rand(batch_size, vocab_size, device="cuda:0", dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Set NaN at different positions: first, middle, last
    nan_indices = [0]
    if batch_size > 1:
        nan_indices.append(batch_size // 2)
    if batch_size > 2:
        nan_indices.append(batch_size - 1)

    for idx in nan_indices:
        probs[idx, :] = float("nan")

    valid_indices = [i for i in range(batch_size) if i not in nan_indices]

    def check_result(result, valid):
        # NaN rows should return 0 and valid=False
        for idx in nan_indices:
            assert result[idx].item() == 0 and not valid[idx].item()
        # Non-NaN rows should have valid=True and valid token index
        for idx in valid_indices:
            assert valid[idx].item()
            assert 0 <= result[idx].item() < vocab_size

    # sampling_from_probs
    result, valid = flashinfer.sampling.sampling_from_probs(probs, return_valid=True)
    check_result(result, valid)

    # top_k_sampling_from_probs
    result, valid = flashinfer.sampling.top_k_sampling_from_probs(
        probs, top_k=50, return_valid=True
    )
    check_result(result, valid)

    # top_p_sampling_from_probs
    result, valid = flashinfer.sampling.top_p_sampling_from_probs(
        probs, top_p=0.9, return_valid=True
    )
    check_result(result, valid)

    # min_p_sampling_from_probs
    result, valid = flashinfer.sampling.min_p_sampling_from_probs(
        probs, min_p=0.1, return_valid=True
    )
    check_result(result, valid)

    # top_k_top_p_sampling_from_probs (joint mode)
    result, valid = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs, top_k=50, top_p=0.9, filter_apply_order="joint", return_valid=True
    )
    check_result(result, valid)


if __name__ == "__main__":
    # test_sampling_freq(128256, gumbel_distribution(0.1), 0.5)
    test_sampling_from_logits_freq(128256, gumbel_distribution(0.1))
    # test_top_p_sampling_freq(128256, gumbel_distribution(0.1), 0.5)
    # test_top_k_sampling_freq(1, 128256, 10)
    # test_sampling(19, 500)
    # test_sampling(1, 111)
    # test_top_p_sampling(3, 111, 0.9)
    # test_top_k_sampling(3, 111, 10)
    # test_top_p_renorm_probs(3, 111, 0.9)
    # test_top_k_renorm_probs(3, 111, 10)
    # test_top_k_mask_logits(99, 989, 10)
    # test_chain_speculative_sampling(3, 111, 3, False)
    # test_chain_speculative_sampling(3, 111, 3, True)


@pytest.mark.parametrize("api", ["top_k", "top_k_top_p"])
@pytest.mark.parametrize("indices_dtype", [None, torch.int32, torch.int64])
@pytest.mark.parametrize("threshold_dtype", [torch.int32, torch.int64])
def test_per_request_top_k_with_indices_dtype(api, indices_dtype, threshold_dtype):
    """Check the threshold element type without relying on a row-remapping fix.

    Each case uses one batched call and identity indices (or no indices), so
    probability-row and output-position indexing agree even without #5340.
    """
    device = "cuda:0"
    repeats = 1024
    rows = torch.tensor(
        [[0.5, 0.25, 0.125, 0.125]] * 3, dtype=torch.float32, device=device
    )
    probs = rows.repeat(repeats, 1)
    top_k = torch.tensor([1, 2, 4], dtype=threshold_dtype, device=device).repeat(
        repeats
    )
    indices = (
        None
        if indices_dtype is None
        else torch.arange(probs.size(0), dtype=indices_dtype, device=device)
    )
    before_probs, before_top_k = probs.clone(), top_k.clone()
    before_indices = None if indices is None else indices.clone()
    kwargs = dict(indices=indices, seed=20260920, offset=0, return_valid=True)
    if api == "top_k":
        out, valid = flashinfer.sampling.top_k_sampling_from_probs(
            probs, top_k, **kwargs
        )
    else:
        out, valid = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, top_k, 1.0, filter_apply_order="joint", **kwargs
        )
    output_dtype = torch.int32 if indices is None else indices.dtype
    assert out.shape == (probs.size(0),) and out.dtype == output_dtype
    assert valid.shape == out.shape and valid.dtype == torch.bool
    assert bool(valid.all())
    assert bool(((out >= 0) & (out < rows.size(1))).all())
    assert torch.equal(probs, before_probs)
    assert torch.equal(top_k, before_top_k)
    if indices is not None:
        assert torch.equal(indices, before_indices)
    samples = out.reshape(repeats, 3)
    expected = [[0], [0, 1], [0, 1, 2, 3]]
    for lane, support in enumerate(expected):
        observed = torch.unique(samples[:, lane]).tolist()
        assert observed == support, (
            api,
            indices_dtype,
            threshold_dtype,
            lane,
            observed,
        )


@pytest.mark.parametrize(
    "api,vocab_size", [("top_k", 4), ("min_p", 4), ("top_k", 8192)]
)
@pytest.mark.parametrize("remap", ["keep_all", "argmax", "mixed"])
@pytest.mark.parametrize("expanded", [False, True], ids=["equal_rows", "more_outputs"])
def test_per_request_param_indexed_by_row(api, vocab_size, remap, expanded):
    """A per-request threshold must be read from the probs row ``indices`` selects.

    The three rows have different argmax classes and binary-exact probabilities, so
    the expected answer for each row is an exact set. Each lane's threshold either
    keeps everything or keeps the argmax alone, and the remapping gives neighbouring
    lanes different rows, so reading the threshold by output position changes at
    least one lane's answer in every case here.

    ``equal_rows`` repeats the rows and the thresholds so that the output-indexed
    read stays in bounds: it is a negative control with a wrong answer rather than a
    crash. ``more_outputs`` is the shape the fix enables -- fewer probs rows than
    outputs -- where the output-indexed read would run off the threshold array.

    At vocab 8192, top-k takes the SM90 split-row kernel.
    """
    num_classes = 4
    stride = vocab_size // num_classes
    rows = torch.zeros(3, vocab_size, dtype=torch.float32, device="cuda:0")
    rows[:, ::stride] = torch.tensor(
        [
            [0.5, 0.25, 0.125, 0.125],
            [0.125, 0.5, 0.25, 0.125],
            [0.125, 0.125, 0.5, 0.25],
        ],
        device=rows.device,
    )
    lanes = 1024
    # row 1 keeps every class; rows 0 and 2 keep their argmax only.
    if api == "top_k":
        values, dtype = [1, num_classes, 1], torch.int32
    else:
        values, dtype = [0.9, 0.01, 0.9], torch.float32
    threshold = torch.tensor(values, dtype=dtype, device=rows.device)
    mapping = {"keep_all": [1, 1, 1], "argmax": [2, 0, 2], "mixed": [1, 2, 0]}[remap]
    indices = torch.tensor(mapping, dtype=torch.int32, device=rows.device).repeat(lanes)
    if expanded:
        probs = rows
    else:
        probs = rows.repeat(lanes, 1)
        threshold = threshold.repeat(lanes)

    probs_before = probs.clone()
    threshold_before = threshold.clone()
    indices_before = indices.clone()
    fn = (
        flashinfer.sampling.top_k_sampling_from_probs
        if api == "top_k"
        else flashinfer.sampling.min_p_sampling_from_probs
    )
    samples, valid = fn(
        probs, threshold, indices=indices, seed=20260919, offset=0, return_valid=True
    )

    assert samples.shape == indices.shape
    assert samples.dtype == indices.dtype
    assert valid.shape == indices.shape and valid.dtype == torch.bool
    assert bool(valid.all())
    assert bool(((samples >= 0) & (samples < vocab_size)).all())
    assert bool((samples % stride == 0).all())
    assert torch.equal(probs, probs_before)
    assert torch.equal(threshold, threshold_before)
    assert torch.equal(indices, indices_before)

    samples = samples.reshape(lanes, len(mapping)) // stride
    for lane, row_idx in enumerate(mapping):
        counts = torch.bincount(samples[:, lane].long(), minlength=num_classes)
        if row_idx == 1:
            # Every class of row 1 has probability >= 1/8, so a lane that keeps all of
            # them misses one with probability at most 4 * (7/8)**1024 (~1e-59).
            assert bool((counts > 0).all()), (remap, lane, counts)
        else:
            # rows 0 and 2 keep their argmax alone, and row i's argmax is class i.
            assert counts[row_idx].item() == lanes, (remap, lane, counts)
