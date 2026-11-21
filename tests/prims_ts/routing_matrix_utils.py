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

"""Shared small-test/full-benchmark inputs for the routed/dense matrix."""

from dataclasses import replace

from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import RouteImpl
from tests.prims_ts.test_batched_gemm_routed_e32 import routed_e32_config
from tests.prims_ts.test_batched_gemm_routed_geometry import tma_native_options

CGAS = [(2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4)]
CASES = ("e32", "e256", "e384")


def matrix_shape(case, *, small):
    if case == "e32":
        experts, tokens, top_k, n, k = 32, 4096, 8, 8192, 10240
        if small:
            tokens, n, k = 257, 1280, 2304
    else:
        experts = {"e256": 256, "e384": 384}[case]
        tokens, top_k, n, k = 512, experts, 4096 if experts == 256 else 6144, 7168
        if small:
            experts = 8 if case == "e256" else 12
            tokens, top_k, n, k = 517, experts, 1280, 2304
    return dict(
        num_experts=experts,
        num_tokens=tokens,
        top_k=top_k,
        problem_n=n,
        problem_k=k,
        routing="uniform" if small else "balanced_round_robin",
        seed=1111,
        routing_seed=1111,
        scale_c_value=1.0,
        scale_gate_value=1.0,
    )


def matrix_options(tile_m, cga, *, routed, routed_fallback=True):
    if tile_m == 512:
        options = tma_native_options(cga, warps=8)
    else:
        options = routed_e32_config(RouteImpl.TMA, 8, 4)
        options.update(
            cluster_dim_x=cga[0],
            cluster_dim_y=cga[1],
            routed_native_sfa=1,
        )
        for name in (
            "mma_regs",
            "load_regs",
            "load_sf_regs",
            "copy_sf_regs",
            "sync_regs",
            "workid_regs",
            "padding_regs",
            "load_a_regs",
        ):
            options[name] = 48
    if routed and routed_fallback and cga != (2, 1):
        options.update(
            fallback_cluster_dim_x=2,
            fallback_cluster_dim_y=1,
            dense_specialize_tma_cga=1,
            dense_local_barrier=1,
        )
    if not routed:
        flexible = cga != (2, 1)
        options.update(
            route_act=0,
            route_sfs_act=0,
            routed_native_sfa=0,
            num_load_a_warps=1,
            num_load_sfa_warps=1,
            fallback_cluster_dim_x=2 if flexible else 0,
            fallback_cluster_dim_y=1,
            dense_fuse_tma_pipeline=1,
            dense_split_tma_pipeline=1,
            dense_fuse_tma_loads=1,
            dense_partition_tma_loads=1,
            dense_specialize_tma_cga=int(flexible),
            dense_local_barrier=int(flexible),
            dense_rolled_epilogue=1,
            dense_dkg_store_order=1,
            dense_async_sf=1,
            dense_rolled_mma=1,
            num_stages_workid=1,
            use_work_throttle=0,
        )
        for name in (
            "mma_regs",
            "load_regs",
            "load_sf_regs",
            "copy_sf_regs",
            "sync_regs",
            "workid_regs",
            "padding_regs",
            "load_a_regs",
        ):
            options[name] = 80
        if tile_m == 256:
            options.update(
                num_stages_a=8,
                num_stages_b=8,
                num_stages_smem_sfa=8,
                num_stages_smem_sfb=8,
                num_stages_tmem_acc=2,
                num_stages_c_smem=2,
                epi_tile_n=32,
                tmem_ldst_max_num_regs=32,
                epilogue_regs=176,
            )
    options["routed_sfa_first"] = int(routed)
    return options


def install_matched_inputs(monkeypatch, case, *, zero):
    """Keep B/A/SFA/SFB RNG order identical when SFA becomes pre-expanded.

    The non-routed path ordinarily generates SFB before SFA. Draw compact
    SFA first, like the routed path, then scatter its gathered values into
    the ordinary dense scale layout. No kernel or reference math is patched.
    """
    import torch
    from flashinfer.prims_ts.batched_gemm.batched_gemm_quant import build_sf_buffer

    state = {}
    original_layout = runner._make_token_layout
    original_sf = runner._create_sf_tensor_mma_layout
    original_linear_sf = runner._create_sf_linear

    def token_layout(**kwargs):
        layout = original_layout(**kwargs)
        if case != "e32":
            generator = torch.Generator(device="cpu").manual_seed(
                kwargs["routing_seed"]
            )
            valid_routes = torch.randint(
                0,
                kwargs["num_tokens"],
                (sum(layout.tokens_per_expert),),
                generator=generator,
                dtype=torch.int32,
            ).tolist()
            routes = [0] * layout.total_padded_tokens
            offset = 0
            for start, count in zip(
                layout.padded_starts[:-1], layout.tokens_per_expert, strict=True
            ):
                routes[start : start + count] = valid_routes[offset : offset + count]
                offset += count
            layout = replace(layout, expanded_to_token=routes)
        state["layout"] = layout
        state["num_tokens"] = kwargs["num_tokens"]
        return layout

    def fp4(shape, *, dtype_kind, fan_in, device):
        assert runner._is_packed_e2m1_dtype(dtype_kind)
        codes = (0xC, 0xA, 0x0, 0x2)
        lookup = torch.tensor(
            [codes[i % 4] | (codes[i // 4] << 4) for i in range(16)],
            dtype=torch.uint8,
            device=device,
        )
        indices = runner._random_uint8(
            runner._storage_shape_for_dtype(shape, dtype_kind), high=16, device=device
        )
        result = lookup[indices.long()]
        if zero:
            result.zero_()
        state["b" if len(shape) == 3 else "a_compact"] = result
        return result

    def sf_bytes(shape, *, cfg, device, layout_kind):
        result = runner._random_uint8(shape, high=2, device=device).mul_(8).add_(0x38)
        if zero:
            result.zero_()
        return result

    def linear_sf(**kwargs):
        result = original_linear_sf(**kwargs)
        state["sfa_compact"] = result
        return result

    def mma_sf(**kwargs):
        if "sfb" not in state:
            if not kwargs["cfg"].has_routed_sfs:
                linear_sf(
                    num_tokens=state["num_tokens"],
                    sf_k_dim=kwargs["sf_k"],
                    device=kwargs["device"],
                    cfg=kwargs["cfg"],
                )
            result = original_sf(**kwargs)
            state["sfb"] = result
            return result
        layout = state["layout"]
        assert kwargs["mn"] == layout.total_padded_tokens
        expanded = runner._expand_bf16_activations(
            state["sfa_compact"].view(state["num_tokens"], -1), layout
        )
        sf_k = kwargs["sf_k"]
        result = build_sf_buffer(
            expanded[:, :sf_k],
            total_bytes=((kwargs["mn"] + 127) // 128) * ((sf_k + 3) // 4) * 512,
            layout=kwargs["layout"],
            data_blk_cols=sf_k,
            device=kwargs["device"],
        )
        state["sfa_expanded"] = result
        return result

    monkeypatch.setattr(runner, "_make_token_layout", token_layout)
    monkeypatch.setattr(runner, "_random_quantized_tensor", fp4)
    monkeypatch.setattr(runner, "_random_sf_bytes", sf_bytes)
    monkeypatch.setattr(runner, "_create_sf_linear", linear_sf)
    monkeypatch.setattr(runner, "_create_sf_tensor_mma_layout", mma_sf)
    return state
