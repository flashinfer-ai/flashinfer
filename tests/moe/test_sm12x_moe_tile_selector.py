"""SM12x MoE tile selector against an independent host oracle."""

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils.moe_route_meta import (
    Mxfp8Mxfp4TileSelectorRuntime,
    moe_tile_selector,
)
from flashinfer.utils import is_sm120a_supported


TILE_MN = Mxfp8Mxfp4TileSelectorRuntime.TILE_MN
pytestmark = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="cute_dsl not available"
)


def _plain_indices(bms):
    return tuple(
        next(index for index, tile in enumerate(TILE_MN[op]) if tile[0] == bm)
        for op, bm in enumerate(bms)
    )


def _oracle(
    counts, plain_bms=(64, 128), *, hidden_size=4096, intermediate=256, num_sms=188
):
    plain_indices = _plain_indices(plain_bms)
    selected = []
    for op, tiles in enumerate(TILE_MN):
        n = intermediate if op == 0 else hidden_size
        statistics = []
        for bm, bn in tiles:
            if bm <= 0 or bn <= 0:
                statistics.append(None)
                continue
            m_tiles = sum(math.ceil(rows / bm) for rows in counts)
            n_tiles = math.ceil(n / bn)
            work_tiles = m_tiles * n_tiles
            waves = math.ceil(work_tiles / num_sms)
            work = m_tiles * bm * n_tiles * bn
            statistics.append((waves, work))
        plain = plain_indices[op]
        waves_plain, work_plain = statistics[plain]
        scores = [
            None if item is None else item[0] * work_plain + item[1] * waves_plain
            for item in statistics
        ]
        best = min(score for score in scores if score is not None)
        selected.append(plain if scores[plain] == best else scores.index(best))
    return tuple(selected)


@pytest.mark.parametrize(
    "counts",
    ([1], [0, 1, 2, 31, 32, 33, 129], [96] * 256),
)
def test_selector_matches_host_oracle(counts):
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")
    device = torch.device("cuda", torch.cuda.current_device())
    counts_gpu = torch.tensor(counts, dtype=torch.int32, device=device)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            counts_gpu.cumsum(0, dtype=torch.int32),
        )
    )
    tile_mn = torch.tensor(TILE_MN, dtype=torch.int32, device=device)
    plain_bms = (64, 128)
    plain = torch.tensor(_plain_indices(plain_bms), dtype=torch.int32, device=device)
    selected = torch.empty(2, dtype=torch.int32, device=device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    moe_tile_selector(
        offsets,
        tile_mn,
        plain,
        selected,
        hidden_size=4096,
        fc1_inter_size=512,
        num_sms=num_sms,
    )
    assert tuple(selected.cpu().tolist()) == _oracle(counts, plain_bms, num_sms=num_sms)


@pytest.mark.parametrize("intermediate", (2048, 1024, 512, 256))
def test_selector_matches_host_oracle_across_fc1_shapes(intermediate):
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")
    device = torch.device("cuda", torch.cuda.current_device())
    counts = [96] * 256
    counts_gpu = torch.tensor(counts, dtype=torch.int32, device=device)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            counts_gpu.cumsum(0, dtype=torch.int32),
        )
    )
    tile_mn = torch.tensor(TILE_MN, dtype=torch.int32, device=device)
    plain_bms = (64, 128)
    plain = torch.tensor(_plain_indices(plain_bms), dtype=torch.int32, device=device)
    selected = torch.empty(2, dtype=torch.int32, device=device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    moe_tile_selector(
        offsets,
        tile_mn,
        plain,
        selected,
        hidden_size=4096,
        fc1_inter_size=2 * intermediate,
        num_sms=num_sms,
    )
    assert tuple(selected.cpu().tolist()) == _oracle(
        counts, plain_bms, intermediate=intermediate, num_sms=num_sms
    )


def test_runtime_reuses_buffers_and_returns_both_tiles():
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")
    device = torch.device("cuda", torch.cuda.current_device())
    plain_bms = (64, 128)
    runtime = Mxfp8Mxfp4TileSelectorRuntime(device, 4096, 256, plain_bms)
    selected = torch.empty(2, dtype=torch.int32, device=device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    counts = [0, 1, 2, 31, 32, 33, 129]
    counts_gpu = torch.tensor(counts, dtype=torch.int32, device=device)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            counts_gpu.cumsum(0, dtype=torch.int32),
        )
    )
    runtime.launch(offsets, selected, num_sms)
    indices = _oracle(counts, plain_bms, num_sms=num_sms)
    assert runtime.result() == tuple(
        (*TILE_MN[op][index], 128) for op, index in enumerate(indices)
    )
    assert runtime.matches(device, 4096, 256, plain_bms)
    assert not runtime.matches(device, 4096, 256, (32, 128))


def test_selector_rejects_invalid_host_contract():
    with pytest.raises(ValueError, match="m_indptr"):
        moe_tile_selector(
            torch.empty(2, dtype=torch.int64),
            torch.empty((2, 4, 2), dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            hidden_size=4096,
            fc1_inter_size=512,
            num_sms=110,
        )
