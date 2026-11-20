# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flashinfer.prims_ts.batched_gemm.tools.routed import routed_inputs
from flashinfer.prims_ts.batched_gemm.tools.routed_inputs import Fixture

pytestmark = pytest.mark.cpu_only


def _fixture(path):
    arrays = {}
    for name in ("B", "SFB", "route"):
        array = np.arange(16, dtype=np.uint8)
        np.save(path / f"{name}.npy", array)
        arrays[name] = dict(
            sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            bytes=array.nbytes,
            dtype=str(array.dtype),
        )
    (path / "manifest.json").write_text(
        json.dumps(dict(shape=[512, 256, 256, 2], tensors=arrays))
    )
    return Fixture(path, (512, 256, 256, 1))


@pytest.mark.parametrize("name", ["B", "SFB", "route"])
@pytest.mark.parametrize("operation", ["init", "load", "audit"])
def test_prefix_fixture_rejects_corruption_outside_prefix(tmp_path, name, operation):
    fixture = _fixture(tmp_path)
    prefix = fixture.load(name)
    assert prefix.tolist() == list(range(8))
    fixture.audit(name, prefix)
    array = np.load(tmp_path / f"{name}.npy")
    array[-1] ^= 1
    np.save(tmp_path / f"{name}.npy", array)
    with pytest.raises(AssertionError, match="source checksum mismatch"):
        if operation == "init":
            Fixture(tmp_path, (512, 256, 256, 1))
        elif operation == "load":
            fixture.load(name)
        else:
            fixture.audit(name, prefix)


def test_generated_fp4_mapping_has_bounded_temporaries():
    sizes = []

    def indices(shape, *, high, device):
        count = int(np.prod(shape))
        sizes.append(count)
        assert high == 16 and count <= 1 << 20
        return (torch.arange(count) % 16).to(dtype=torch.uint8, device=device)

    stub = SimpleNamespace(
        _make_token_layout=None,
        _random_quantized_tensor=None,
        _random_sf_bytes=None,
        _is_packed_e2m1_dtype=lambda dtype: True,
        _storage_shape_for_dtype=lambda shape, dtype: shape,
        _random_uint8=indices,
    )
    shape = (2, (1 << 20) + 8)
    with routed_inputs(stub, None):
        result = stub._random_quantized_tensor(
            shape, dtype_kind=0, fan_in=1, device="cpu"
        )
    codes = (0xC, 0xA, 0x0, 0x2)
    lookup = torch.tensor(
        [codes[i % 4] | (codes[i // 4] << 4) for i in range(16)], dtype=torch.uint8
    )
    assert result.shape == shape and result.dtype == torch.uint8
    assert torch.equal(result.flatten(), lookup.repeat(result.numel() // 16))
    assert sizes == [1 << 20, 1 << 20, 16]


@pytest.mark.parametrize("shape", [(0, 1), (1, 0), (-2, 4), (2, -4)])
def test_routed_cli_rejects_nonpositive_cluster_dimensions(monkeypatch, capsys, shape):
    import sys
    from flashinfer.prims_ts.batched_gemm.tools import routed

    monkeypatch.setattr(sys, "argv", ["routed", "--cluster-shape", *map(str, shape)])
    with pytest.raises(SystemExit, match="2"):
        routed.main()
    assert "Both cluster-shape dimensions must be positive" in capsys.readouterr().err
