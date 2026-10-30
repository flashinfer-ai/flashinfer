# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the geometry, the support probe (never raises), the version probe and the vendored manifest's digests."""

import hashlib
import json
import pathlib
import pytest
import torch


def test_geometry_validation():
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry

    g = Qwen35BlockGeometry(d_model=4096, num_qo_heads=32, num_kv_heads=2)
    assert (
        g.group == 16
        and g.n_qkvg == 17408
        and g.scale == 0.0625
        and g.unsupported_reason() is None
    )
    assert (
        Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2).n_qkvg
        == 13312
    )
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        Qwen35BlockGeometry(d_model=4096, num_qo_heads=30, num_kv_heads=4)
    with pytest.raises(ValueError, match="rope_dim"):
        Qwen35BlockGeometry(d_model=4096, num_qo_heads=32, num_kv_heads=2, rope_dim=65)
    with pytest.raises(ValueError, match="positive int"):
        Qwen35BlockGeometry(d_model=0, num_qo_heads=32, num_kv_heads=2)
    assert (
        "head_dim=256"
        in Qwen35BlockGeometry(
            d_model=4096, num_qo_heads=32, num_kv_heads=2, head_dim=128
        ).unsupported_reason()
    )
    assert (
        "multiple of 256"
        in Qwen35BlockGeometry(
            d_model=4000, num_qo_heads=32, num_kv_heads=2
        ).unsupported_reason()
    )


def test_geometry_rope_dim_envelope_is_the_shared_gate():
    """The rope_dim rule the probe and prepare share: 64 and 128 at head_dim 256; 32 / 256 are the fused projection's declines, 192
    the lowered decode chain's, 16 both -- every one a typed reason naming the value and the served set, none a ValueError."""
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry
    from flashinfer.experimental.qwen_3_5_block import support

    assert support.ROPE_DIMS == (64, 128)
    for rope in support.ROPE_DIMS:
        assert (
            Qwen35BlockGeometry(
                d_model=4096, num_qo_heads=32, num_kv_heads=2, rope_dim=rope
            ).unsupported_reason()
            is None
        )
    for rope in (16, 32, 192, 256):
        reason = Qwen35BlockGeometry(
            d_model=4096, num_qo_heads=32, num_kv_heads=2, rope_dim=rope
        ).unsupported_reason()
        assert reason is not None and f"rope_dim={rope}" in reason, (rope, reason)
        assert "(64, 128)" in reason
    assert "fused norm + RoPE projection" in support.rope_dim_reason(32)
    assert "lowered decode chain" in support.rope_dim_reason(192)
    # head_dim is judged first: a 128-dim head names the head dim, not the RoPE rule
    assert (
        "head_dim=256"
        in Qwen35BlockGeometry(
            d_model=4096, num_qo_heads=32, num_kv_heads=2, head_dim=128, rope_dim=64
        ).unsupported_reason()
    )


def test_support_probe_on_cpu_never_raises_and_names_the_device():
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry, qwen_3_5_block_support

    g = Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    # the opt-in: calling the decorated function (it warns ExperimentalWarning once per process; the policy test pins the decorator)
    sup = qwen_3_5_block_support(torch.device("cpu"), g)
    assert qwen_3_5_block_support.is_experimental is True
    assert sup.served is False and sup.forms == () and "SM107" in sup.reason
    assert (
        sup.prefill.served is False
        and sup.extend.reason == sup.reason
        and sup.decode.reason == sup.reason
    )
    assert (
        sup.page_sizes == (16, 32, 64, 128)
        and sup.kv_layouts == ("NHD", "HND")
        and sup.decode_max_q_len == 4
        and sup.decode_max_rows == 32
    )
    assert sup.dtypes == (torch.bfloat16,) and sup.cutedsl_required == "4.8.0"
    assert sup.rope_dims == (64, 128)
    assert (
        "bfloat16"
        in qwen_3_5_block_support(torch.device("cpu"), g, dtype=torch.float16).reason
    )
    # the geometry gate speaks before the device: a declined rope_dim is the reason on every device, for every form
    rope32 = qwen_3_5_block_support(
        torch.device("cpu"),
        Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2, rope_dim=32),
    )
    assert not rope32.served and "rope_dim=32" in rope32.reason
    assert all(
        not rope32.form(f).served and rope32.form(f).reason == rope32.reason
        for f in ("prefill", "extend", "decode")
    )
    assert (
        "head_dim=256"
        in qwen_3_5_block_support(
            torch.device("cpu"),
            Qwen35BlockGeometry(
                d_model=2560, num_qo_heads=24, num_kv_heads=2, head_dim=128
            ),
        ).reason
    )


def test_support_probe_on_the_local_device():
    """Wherever this runs: the probe answers; served exactly on SM107 with a Rubin-capable DSL; the per-form records agree."""
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry, qwen_3_5_block_support

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    g = Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    sup = qwen_3_5_block_support(torch.device("cuda", 0), g)
    cc = torch.cuda.get_device_capability(0)
    assert sup.device_cc == cc
    if cc != (10, 7):
        assert not sup.served and f"SM{cc[0]}{cc[1]}" in sup.reason
    elif not sup.cutedsl_ok:
        assert not sup.served and "cutlass-dsl" in sup.reason
    else:
        assert sup.served and sup.forms == ("prefill", "extend", "decode")
        assert (
            sup.extend.declines
            and sup.decode.declines
            and "head_dim" in sup.decode.declines
        )


def test_version_probe_and_vendored_manifest_digests():
    from flashinfer.qwen_3_5_block import qwen_3_5_block_support
    from flashinfer.experimental.qwen_3_5_block import support

    src = support.source()
    # the origin is the cuDNN Frontend repository the pin is reproducible from (vendor.py's ORIGIN; the root NOTICE names it)
    assert src.origin == "https://github.com/NVIDIA/cudnn-frontend", src.origin
    assert (
        src.commit.startswith("977757763")
        and src.version == "1.33.0"
        and src.files == 118
    )
    assert (
        src.validated_cutedsl == "4.8.0"
        and src.validated_compute_capability == "10.7"
        and len(src.manifest_sha256) == 64
    )
    root = pathlib.Path(support.__file__).parent / "_cudnn_frost"
    manifest = json.loads((root / "VENDOR_MANIFEST.json").read_text())
    for rec in manifest["files"]:
        assert (
            hashlib.sha256((root / rec["path"]).read_bytes()).hexdigest()
            == rec["vendored_sha256"]
        ), rec["path"]
        assert rec["license"] in ("MIT", "Apache-2.0", "Apache-2.0 AND MIT"), rec
    assert (
        manifest["binder_module"]
        == "flashinfer.experimental.qwen_3_5_block._sdpa_binder"
    )
    assert qwen_3_5_block_support.is_experimental is True
