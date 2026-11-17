# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

import json

import pytest

from flashinfer.prims_ts.moe.tools.sweep.menu_config import runtime_twin
from flashinfer.prims_ts.moe.tools.sweep.menu_emit import (
    _AUTO_FIELDS,
    _array_spans,
    _compact_members,
    _ordered_keys,
    _resolve_auto_fields,
    _row_configs,
    emit_menu_prefix,
    expand_config_rows,
)
from flashinfer.prims_ts.moe.tools.sweep.menu_retime import (
    menu_candidate_id,
    normalize_activation_options,
)
from flashinfer.prims_ts.moe.tools.sweep.space import Candidate


def test_activation_normalized_identity_is_class_scoped():
    options = {
        "tile_n": 8,
        "tile_k": 512,
        "fused_act": True,
        "act": "swiglu",
        "eltwise_act_type": "none",
    }
    assert normalize_activation_options(options) == {
        "tile_n": 8,
        "tile_k": 512,
    }
    candidate = Candidate(options=options, tile_n=8)
    gated = menu_candidate_id(candidate, fc="fc1", activation_class_name="gated")
    elementwise = menu_candidate_id(
        candidate, fc="fc1", activation_class_name="elementwise"
    )
    assert gated != elementwise


def _base_config():
    return {
        "sm_archs": [107],
        "templates": {
            "Nv": {
                "dtype_a": "e2m1",
                "dtype_c": "e2m1",
                "mma_m": 128,
                "umma_cta_count": 1,
                "tile_scheduler": "persistent",
                "tile_n": 8,
                "tile_k": 512,
                "num_stages_a": 4,
            }
        },
        "configs": [
            {
                "_comment": "old nv fc1",
                "_template": "Nv",
                "route_act": "ldgsts",
                "tile_n,num_stages_a": [[8, 4], [16, 5]],
                "fused_act,act,eltwise_act_type": [
                    [True, "swiglu", "none"],
                    [True, "geglu", "none"],
                    [False, "none", "relu2"],
                ],
                "use_per_token_sf_b": [True, False],
                "per_token_sf_dtype": "fp32",
            },
            {
                "_comment": "old nv fc2",
                "_template": "Nv",
                "dtype_c": "bf16",
                "tile_n,num_stages_a": [[8, 4], [16, 5]],
            },
            {
                "_comment": "unrelated bf16",
                "dtype_a": "bf16",
                "dtype_c": "bf16",
                "route_act": "ldgsts",
                "tile_scheduler": "persistent",
                "tile_n": 8,
                "tile_k": 128,
                "fused_act": True,
                "act": "swiglu",
            },
        ],
    }


def _entry(
    rank,
    *,
    fc,
    tile_n,
    stages,
    activation_class=None,
    drain=False,
):
    options = {
        "dtype_a": "e2m1",
        "dtype_c": "e2m1" if fc == "fc1" else "bf16",
        "mma_m": 128,
        "umma_cta_count": 1,
        "tile_scheduler": "persistent",
        "tile_n": tile_n,
        "tile_k": 512,
        "num_stages_a": stages,
        "use_clc_fast_drain": drain,
        "use_work_throttle": drain,
    }
    if fc == "fc1":
        options["route_act"] = "ldgsts"
        options["use_per_token_sf_b"] = False
    return {
        "rank": rank,
        "candidate_id": f"candidate-{rank}",
        "fc": fc,
        "activation_class": activation_class,
        "tile_n": tile_n,
        "options": options,
    }


def _order(*entries, floor_size=None, dtype="NvFP4xNvFP4"):
    return {
        "floor_size": len(entries) if floor_size is None else floor_size,
        "inputs": {"target": {"dtype": dtype}},
        "entries": list(entries),
    }


def _write_inputs(tmp_path, base, order):
    base_path = tmp_path / "base.json"
    order_path = tmp_path / "order.json"
    base_path.write_text(json.dumps(base, indent=4) + "\n")
    order_path.write_text(json.dumps(order, indent=2) + "\n")
    return base_path, order_path


def _target_rows(data, fc):
    return [
        row
        for row in expand_config_rows(data)
        if row["dtype_a"] == "e2m1"
        and row["dtype_c"] == ("e2m1" if fc == "fc1" else "bf16")
        and bool(row.get("route_act")) == (fc == "fc1")
    ]


def test_emit_replaces_dtype_slice_and_inherits_shared_variants(tmp_path):
    order = _order(
        _entry(1, fc="fc1", tile_n=8, stages=4, activation_class="gated"),
        _entry(2, fc="fc2", tile_n=8, stages=4),
    )
    base_path, order_path = _write_inputs(tmp_path, _base_config(), order)
    base_text = base_path.read_text()
    unrelated_span = _array_spans(base_text, "configs")[2][-1]
    unrelated = base_text[slice(*unrelated_span)]
    out_path = tmp_path / "out.json"

    emit_menu_prefix(
        order_path,
        n=2,
        base_path=base_path,
        out_path=out_path,
        dtype="NvFP4xNvFP4",
    )
    emitted = json.loads(out_path.read_text())
    fc1 = _target_rows(emitted, "fc1")
    assert {(row["fused_act"], row["act"], row["eltwise_act_type"]) for row in fc1} == {
        (True, "swiglu", "none"),
        (True, "geglu", "none"),
        (False, "none", "relu2"),
    }
    assert {row["use_per_token_sf_b"] for row in fc1} == {True, False}
    assert "_menu_metadata" not in emitted
    output_text = out_path.read_text()
    output_span = _array_spans(output_text, "configs")[2][-1]
    assert output_text[slice(*output_span)] == unrelated


def test_emit_per_class_does_not_transfer_gated_geometry_to_relu2(tmp_path):
    order = _order(
        _entry(1, fc="fc1", tile_n=8, stages=4, activation_class="gated"),
        _entry(2, fc="fc2", tile_n=8, stages=4),
    )
    base_path, order_path = _write_inputs(tmp_path, _base_config(), order)
    out_path = tmp_path / "out.json"
    emit_menu_prefix(
        order_path,
        n=2,
        base_path=base_path,
        out_path=out_path,
        dtype="NvFP4xNvFP4",
        activations="per-class",
    )
    fc1 = _target_rows(json.loads(out_path.read_text()), "fc1")
    assert {row["act"] for row in fc1} == {"swiglu", "geglu"}
    assert all(row["eltwise_act_type"] == "none" for row in fc1)


def test_emit_groups_members_and_strips_template_noops(tmp_path):
    order = _order(
        _entry(1, fc="fc1", tile_n=8, stages=4, activation_class="gated"),
        _entry(2, fc="fc2", tile_n=8, stages=4),
        _entry(3, fc="fc1", tile_n=16, stages=5, activation_class="gated"),
        _entry(4, fc="fc2", tile_n=16, stages=5),
    )
    base_path, order_path = _write_inputs(tmp_path, _base_config(), order)
    out_path = tmp_path / "out.json"
    emit_menu_prefix(
        order_path,
        n=4,
        base_path=base_path,
        out_path=out_path,
        dtype="NvFP4xNvFP4",
    )
    data = json.loads(out_path.read_text())
    raw_fc1 = [
        row
        for row in data["configs"]
        if str(row.get("_comment", "")).startswith("FP4_FC1")
    ]
    assert len(raw_fc1) == 1
    grouped_keys = [key for key in raw_fc1[0] if "," in key]
    assert any("tile_n" in key and "num_stages_a" in key for key in grouped_keys)
    assert "mma_m" not in raw_fc1[0]
    assert "tile_k" not in raw_fc1[0]


def test_compact_members_emits_complete_stage_and_register_blocks():
    inherited = {
        "tile_k": 512,
        "num_stages_a": 4,
        "num_stages_b": 4,
        "num_stages_smem_sfa": 4,
        "num_stages_smem_sfb": 4,
        "num_stages_tmem_sfa": 4,
        "num_stages_tmem_sfb": 4,
        "num_stages_tmem_acc": 1,
    }
    common = {
        "tile_k": 512,
        "num_stages_a": 3,
        "num_stages_b": 4,
        "num_stages_smem_sfa": 3,
        "num_stages_smem_sfb": 3,
        "num_stages_tmem_sfa": 1,
        "num_stages_tmem_sfb": 1,
        "mma_regs": 88,
        "epilogue_regs": 144,
    }
    stages = "num_stages_a,num_stages_b,num_stages_smem_sfa,num_stages_smem_sfb"

    def compact(*members):
        return _compact_members(_ordered_keys(members[0]), members, inherited)

    # Constant blocks are single rows and keep template-equal stage values.
    expected = {
        "tile_n": [8, 16],
        stages: [[3, 4, 3, 3]],
        "num_stages_tmem_sfa": 1,
        "num_stages_tmem_sfb": 1,
        "num_stages_tmem_acc": 1,
        "mma_regs,epilogue_regs": [[88, 144]],
    }
    result = compact({**common, "tile_n": 8}, {**common, "tile_n": 16})
    assert list(result.items()) == list(expected.items())
    # A varying stage pulls its whole block into the joint key.
    assert compact(
        {**common, "tile_n": 8},
        {**common, "tile_n": 16, "num_stages_a": 5, "num_stages_tmem_sfb": 2},
    ) == {
        f"tile_n,{stages},num_stages_tmem_sfa,num_stages_tmem_sfb": [
            [8, 3, 4, 3, 3, 1, 1],
            [16, 5, 4, 3, 3, 1, 2],
        ],
        "num_stages_tmem_acc": 1,
        "mma_regs,epilogue_regs": [[88, 144]],
    }
    # A varying accumulator depth joins the stage block.
    assert compact(
        {**common, "tile_n": 8, "num_stages_tmem_acc": 1},
        {**common, "tile_n": 16, "num_stages_tmem_acc": 2},
    ) == {
        f"tile_n,{stages},num_stages_tmem_acc": [
            [8, 3, 4, 3, 3, 1],
            [16, 3, 4, 3, 3, 2],
        ],
        "num_stages_tmem_sfa": 1,
        "num_stages_tmem_sfb": 1,
        "mma_regs,epilogue_regs": [[88, 144]],
    }


_MX_FC1_AUTO_ROW = {
    "dtype_a": "mxe2m1",
    "dtype_b": "mxe4m3",
    "dtype_c": "mxe4m3",
    "bias_type": "m",
    "use_deepseek_fp8": False,
    "transpose_mma_output": True,
    "route_act": "ldgsts",
    "route_sfs_act": "ldgsts",
    "sf_layout_b": "linear",
    "sf_layout_c": "8x4",
    "mma_m": 256,
    "umma_cta_count": 2,
    "mma_n": 32,
    "mma_k": 64,
    "tile_n": 32,
    "epi_tile_m": 128,
    "epi_tile_n": 32,
    "tile_k": 1024,
    "cluster_dim_x": 0,
    "num_stages_a": 3,
    "num_stages_b": 3,
    "num_stages_smem_sfa": 4,
    "num_stages_smem_sfb": 2,
    "num_stages_tmem_sfa": 5,
    "num_stages_tmem_sfb": 5,
    "num_stages_tmem_acc": 2,
    "num_stages_c_smem": 2,
    "num_load_b_warps": 0,
    "num_load_sfb_warps": 2,
    "mma_regs": 48,
    "load_regs": 48,
    "load_a_regs": 0,
    "load_b_regs": 0,
    "load_sf_regs": 48,
    "load_sfa_regs": 0,
    "load_sfb_regs": 0,
    "copy_sf_regs": 48,
    "padding_regs": 48,
    "workid_regs": 48,
    "epilogue_regs": 160,
    "use_tma_store": True,
    "use_tma_oob_opt": True,
    "tile_scheduler": "persistent",
    "use_clc_fast_drain": True,
    "use_work_throttle": True,
}


def test_resolve_auto_fields_pins_zeros_without_changing_kernels():
    def resolve(*geometries):
        return _resolve_auto_fields(
            [
                {"fc": "fc1", "options": options, "classes": {"gated"}, "rank": rank}
                for options, rank in geometries
            ],
            mode="shared",
            activations={"gated": ((True, "swiglu", "none"),)},
            per_token={"fc1": ((None, None),)},
            sm_archs=(107,),
        )

    (resolved,) = resolve((_MX_FC1_AUTO_ROW, 1))
    options = resolved["options"]
    assert {
        field: options[field]
        for field in _AUTO_FIELDS
        if _MX_FC1_AUTO_ROW.get(field) == 0
    } == {
        "cluster_dim_x": 2,
        "num_load_b_warps": 4,
        "load_a_regs": 48,
        "load_b_regs": 48,
        "load_sfa_regs": 48,
        "load_sfb_regs": 48,
    }
    activation = {"fused_act": True, "act": "swiglu", "eltwise_act_type": "none"}
    before = _row_configs({**_MX_FC1_AUTO_ROW, **activation}, fc="fc1", sm_archs=(107,))
    after = _row_configs({**options, **activation}, fc="fc1", sm_archs=(107,))
    assert before and before.keys() == after.keys()
    assert all(
        before[key].compile_signature() == after[key].compile_signature()
        for key in before
    )

    # A zero row and its explicit spelling are one kernel, so they merge.
    (merged,) = resolve((_MX_FC1_AUTO_ROW, 3), (options, 2))
    assert merged["options"] == options
    assert merged["rank"] == 2


def test_emit_deduplicates_explicit_runtime_twin(tmp_path):
    order = _order(
        _entry(1, fc="fc1", tile_n=8, stages=4, activation_class="gated"),
        _entry(2, fc="fc2", tile_n=8, stages=4),
        _entry(
            3,
            fc="fc1",
            tile_n=8,
            stages=4,
            activation_class="gated",
            drain=True,
        ),
        _entry(4, fc="fc2", tile_n=8, stages=4, drain=True),
    )
    base_path, order_path = _write_inputs(tmp_path, _base_config(), order)
    out_path = tmp_path / "out.json"
    emit_menu_prefix(
        order_path,
        n=4,
        base_path=base_path,
        out_path=out_path,
        dtype="NvFP4xNvFP4",
    )
    target = [
        row
        for row in expand_config_rows(json.loads(out_path.read_text()))
        if row["dtype_a"] == "e2m1"
    ]
    assert all(not row.get("use_clc_fast_drain", False) for row in target)


def test_shared_runtime_twin_rule_matches_emitted_scheduler_policy():
    base = {"tile_scheduler": "persistent", "tile_n": 8}
    assert runtime_twin(base) == {
        **base,
        "use_clc_fast_drain": True,
        "use_work_throttle": True,
    }
    assert runtime_twin({"tile_scheduler": "static"}) is None
    assert (
        runtime_twin(
            {
                **base,
                "use_clc_fast_drain": True,
                "use_work_throttle": True,
            }
        )
        is None
    )


def test_emit_rejects_tile_pairing_mismatch_and_mixed_raw_row(tmp_path):
    order = _order(
        _entry(1, fc="fc1", tile_n=8, stages=4, activation_class="gated"),
        _entry(2, fc="fc2", tile_n=16, stages=5),
    )
    base_path, order_path = _write_inputs(tmp_path, _base_config(), order)
    with pytest.raises(ValueError, match="tile_n mismatch"):
        emit_menu_prefix(
            order_path,
            n=2,
            base_path=base_path,
            out_path=tmp_path / "out.json",
            dtype="NvFP4xNvFP4",
        )

    mixed = _base_config()
    mixed["configs"][0]["dtype_a"] = ["e2m1", "bf16"]
    base_path, order_path = _write_inputs(tmp_path, mixed, order)
    with pytest.raises(ValueError, match="mixes NvFP4xNvFP4"):
        emit_menu_prefix(
            order_path,
            n=2,
            base_path=base_path,
            out_path=tmp_path / "mixed.json",
            dtype="NvFP4xNvFP4",
        )


def test_emit_in_place_is_atomic_and_dry_run_writes_nothing(tmp_path):
    order = _order(
        _entry(1, fc="fc1", tile_n=8, stages=4, activation_class="gated"),
        _entry(2, fc="fc2", tile_n=8, stages=4),
    )
    base_path, order_path = _write_inputs(tmp_path, _base_config(), order)
    before = base_path.read_bytes()
    emit_menu_prefix(
        order_path,
        n=2,
        base_path=base_path,
        dtype="NvFP4xNvFP4",
        in_place=True,
        dry_run=True,
    )
    assert base_path.read_bytes() == before

    emit_menu_prefix(
        order_path,
        n=2,
        base_path=base_path,
        dtype="NvFP4xNvFP4",
        in_place=True,
    )
    assert base_path.read_bytes() != before

    bad_order = _order(*order["entries"], floor_size=3)
    order_path.write_text(json.dumps(bad_order))
    after_success = base_path.read_bytes()
    with pytest.raises(ValueError, match="mandatory floor"):
        emit_menu_prefix(
            order_path,
            n=2,
            base_path=base_path,
            dtype="NvFP4xNvFP4",
            in_place=True,
        )
    assert base_path.read_bytes() == after_success


def test_in_place_emits_for_different_dtypes_compose_in_either_order(tmp_path):
    base = _base_config()
    base["templates"]["Mx"] = {
        "dtype_a": "mxe2m1",
        "dtype_b": "mxe4m3",
        "dtype_c": "mxe4m3",
        "mma_m": 128,
        "umma_cta_count": 1,
        "tile_scheduler": "persistent",
        "tile_n": 8,
        "tile_k": 512,
        "num_stages_a": 4,
    }
    base["configs"][2:2] = [
        {
            "_comment": "old mx fc1",
            "_template": "Mx",
            "route_act": "ldgsts",
            "fused_act,act,eltwise_act_type": [
                [True, "swiglu", "none"],
                [True, "geglu", "none"],
                [False, "none", "relu2"],
            ],
        },
        {
            "_comment": "old mx fc2",
            "_template": "Mx",
            "dtype_c": "bf16",
        },
    ]
    nv_order = _order(
        _entry(1, fc="fc1", tile_n=8, stages=4, activation_class="gated"),
        _entry(2, fc="fc2", tile_n=8, stages=4),
    )

    def mx_entry(rank, fc):
        options = {
            "dtype_a": "mxe2m1",
            "dtype_b": "mxe4m3",
            "dtype_c": "mxe4m3" if fc == "fc1" else "bf16",
            "mma_m": 128,
            "umma_cta_count": 1,
            "tile_scheduler": "persistent",
            "tile_n": 8,
            "tile_k": 512,
            "num_stages_a": 4,
            "use_clc_fast_drain": False,
            "use_work_throttle": False,
        }
        if fc == "fc1":
            options["route_act"] = "ldgsts"
        return {
            "rank": rank,
            "candidate_id": f"mx-{rank}",
            "fc": fc,
            "activation_class": "gated" if fc == "fc1" else None,
            "tile_n": 8,
            "options": options,
        }

    mx_order = _order(
        mx_entry(1, "fc1"),
        mx_entry(2, "fc2"),
        dtype="MxFP4xMxFP8",
    )
    nv_order_path = tmp_path / "nv-order.json"
    mx_order_path = tmp_path / "mx-order.json"
    nv_order_path.write_text(json.dumps(nv_order))
    mx_order_path.write_text(json.dumps(mx_order))
    forward = tmp_path / "forward.json"
    reverse = tmp_path / "reverse.json"
    forward.write_text(json.dumps(base, indent=4) + "\n")
    reverse.write_text(json.dumps(base, indent=4) + "\n")

    for path, sequence in (
        (
            forward,
            (
                (nv_order_path, "NvFP4xNvFP4"),
                (mx_order_path, "MxFP4xMxFP8"),
            ),
        ),
        (
            reverse,
            (
                (mx_order_path, "MxFP4xMxFP8"),
                (nv_order_path, "NvFP4xNvFP4"),
            ),
        ),
    ):
        for order_path, dtype in sequence:
            emit_menu_prefix(
                order_path,
                n=2,
                base_path=path,
                dtype=dtype,
                in_place=True,
            )
    assert forward.read_bytes() == reverse.read_bytes()


def test_emit_rejects_prefix_below_floor(tmp_path):
    order = tmp_path / "order.json"
    order.write_text(json.dumps({"floor_size": 2, "entries": []}))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"sm_archs": [107], "templates": {}, "configs": []}))
    with pytest.raises(ValueError, match="mandatory floor"):
        emit_menu_prefix(
            order,
            n=1,
            base_path=base,
            out_path=tmp_path / "out.json",
            dtype="NvFP4xNvFP4",
        )


def test_emit_refuses_to_overwrite_base_or_cross_dtype_boundary(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"sm_archs": [107], "templates": {}, "configs": []}))
    order = tmp_path / "order.json"
    order.write_text(
        json.dumps(
            {
                "floor_size": 0,
                "entries": [],
                "inputs": {"target": {"dtype": "NvFP4xNvFP4"}},
            }
        )
    )

    with pytest.raises(ValueError, match="must not overwrite"):
        emit_menu_prefix(
            order,
            n=0,
            base_path=base,
            out_path=base,
            dtype="NvFP4xNvFP4",
        )
    with pytest.raises(ValueError, match="conflicts"):
        emit_menu_prefix(
            order,
            n=0,
            base_path=base,
            out_path=tmp_path / "out.json",
            dtype="MxFP8xMxFP8",
        )
