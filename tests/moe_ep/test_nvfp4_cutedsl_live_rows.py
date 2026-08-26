"""CPU gates for capacity-backed NVFP4 MegaMoE live-row launches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")


def _frontend_config(*, num_tokens: int = 8):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Config,
    )

    return MegaMoENvfp4Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=num_tokens,
        num_topk=2,
        num_total_experts=4,
        hidden=64,
        intermediate=128,
        in_kernel_fc2_reduce=False,
    )


def _cpu_inputs(*, num_tokens: int = 8):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Inputs,
    )

    return MegaMoENvfp4Inputs(
        activation=torch.empty(num_tokens, 32),
        activation_sf=torch.empty(num_tokens, 4),
        topk_idx=torch.empty(num_tokens, 2, dtype=torch.int64),
        topk_weights=torch.empty(num_tokens, 2),
        fc1_weight=torch.empty(4, 32, 128),
        fc1_weight_sf=torch.empty(4, 1),
        fc2_weight=torch.empty(4, 64, 64),
        fc2_weight_sf=torch.empty(4, 1),
        fc1_alpha=torch.empty(4),
        fc2_alpha=torch.empty(4),
        fc1_norm_const=torch.empty(4),
        output_activation=torch.empty(num_tokens, 64),
    )


def test_standalone_reduce_accepts_zero_partial_and_full_live_extents(monkeypatch):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Frontend,
    )

    frontend = MegaMoENvfp4Frontend(_frontend_config())
    inputs = _cpu_inputs()
    validate = mock.Mock()
    monkeypatch.setattr(frontend, "_validate_inputs", validate)

    assert frontend._prepare_launch_inputs(inputs, num_tokens=0) is None
    validate.assert_not_called()

    partial = frontend._prepare_launch_inputs(inputs, num_tokens=3)
    validate.assert_called_once_with(inputs, num_tokens=3)
    assert partial is not None
    for name in (
        "activation",
        "activation_sf",
        "topk_idx",
        "topk_weights",
        "output_activation",
    ):
        raw = getattr(inputs, name)
        live = getattr(partial, name)
        assert live.shape[0] == 3
        assert live.data_ptr() == raw.data_ptr()
    for name in (
        "fc1_weight",
        "fc1_weight_sf",
        "fc2_weight",
        "fc2_weight_sf",
        "fc1_alpha",
        "fc2_alpha",
        "fc1_norm_const",
    ):
        assert getattr(partial, name) is getattr(inputs, name)

    full = frontend._prepare_launch_inputs(inputs, num_tokens=8)
    assert full is inputs
    assert frontend._prepare_launch_inputs(inputs, num_tokens=None) is inputs

    with pytest.raises(ValueError, match=r"num_tokens must be in \[0, 8\]"):
        frontend._prepare_launch_inputs(inputs, num_tokens=-1)
    with pytest.raises(ValueError, match=r"num_tokens must be in \[0, 8\]"):
        frontend._prepare_launch_inputs(inputs, num_tokens=9)


def test_runtime_kwargs_make_all_token_row_extents_dynamic(monkeypatch):
    import cuda.bindings.driver as cuda_driver

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Frontend,
    )

    frontend = MegaMoENvfp4Frontend(_frontend_config())
    inputs = _cpu_inputs(num_tokens=3)
    calls: dict[int, tuple[int, ...]] = {}

    def record_to_cute(
        tensor,
        assumed_align=16,
        *,
        static_layout=False,
        dynamic_compact_shape_modes=(),
    ):
        del assumed_align, static_layout
        calls[id(tensor)] = dynamic_compact_shape_modes
        return ("cute", id(tensor))

    monkeypatch.setattr(frontend, "_to_cute", record_to_cute)
    monkeypatch.setattr(frontend, "_to_cute_ptr", lambda tensor: ("ptr", id(tensor)))
    monkeypatch.setattr(cuda_driver, "CUstream", lambda handle: ("stream", handle))
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=17),
    )
    mega = SimpleNamespace(
        symmetric_base=0,
        peer_offsets_list=[0],
        local_workspace=object(),
        shared_workspace=object(),
    )

    kwargs = frontend._build_mega_runtime_kwargs(inputs, mega)

    for name in (
        "activation",
        "activation_sf",
        "topk_idx",
        "topk_weights",
        "output_activation",
    ):
        assert calls[id(getattr(inputs, name))] == (0,)
        assert kwargs[name] == ("cute", id(getattr(inputs, name)))
    for name in (
        "fc1_weight",
        "fc1_weight_sf",
        "fc2_weight",
        "fc2_weight_sf",
        "fc1_alpha",
        "fc2_alpha",
        "fc1_norm_const",
    ):
        assert calls[id(getattr(inputs, name))] == ()


def test_live_token_extents_share_one_dynamic_cutedsl_signature():
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Frontend,
    )

    frontend = MegaMoENvfp4Frontend(_frontend_config())
    backing = torch.empty(8, 4)
    full = frontend._to_cute(backing, dynamic_compact_shape_modes=(0,))
    partial = frontend._to_cute(backing[:3], dynamic_compact_shape_modes=(0,))

    assert full.__cache_key__ == partial.__cache_key__
    assert (
        frontend._to_cute(backing, static_layout=True).__cache_key__
        != frontend._to_cute(backing[:3], static_layout=True).__cache_key__
    )


def test_backend_thunk_cache_preserves_capacity_extent(monkeypatch):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.backend import (
        Nvfp4CutedslMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.config import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    compiled = object()

    class FakeFrontend:
        def __init__(self, *, fc2_reduces_topk=False):
            self._mega = SimpleNamespace(compiled=compiled)
            self.config = SimpleNamespace(fc2_reduces_topk=fc2_reduces_topk)
            self.builds: list[int | None] = []
            self.launches: list[int | None] = []

        def set_gate_up_clamp(self, clamp):
            raise AssertionError(f"unexpected clamp {clamp}")

        def make_launch_thunk(self, inputs, *, num_tokens=None):
            del inputs
            self.builds.append(num_tokens)

            def thunk():
                self.launches.append(num_tokens)

            return thunk

    frontend = FakeFrontend()
    out_buf = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    workspace = SimpleNamespace(
        _frontend=frontend,
        x=torch.empty(8, 2),
        x_sf=torch.empty(8, 1),
        topk_idx=torch.empty(8, 2, dtype=torch.int64),
        topk_weights=torch.empty(8, 2),
        fc1_alpha=torch.empty(1),
        fc2_alpha=torch.empty(1),
        fc1_norm_const=torch.empty(1),
        output_activation=out_buf,
    )
    transformed_weights = (
        (torch.empty(1), torch.empty(1)),
        (torch.empty(1), torch.empty(1)),
    )
    backend = Nvfp4CutedslMegaKernelBackend(
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=64,
            top_k=2,
        )
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=23),
    )

    full_a = torch.empty_like(out_buf)
    full_b = torch.empty_like(out_buf)
    partial_a = torch.empty(3, 4)
    partial_b = torch.empty(3, 4)
    full_c = torch.empty_like(out_buf)
    empty = torch.empty(0, 4)
    oversized = torch.empty(9, 4)

    with pytest.raises(ValueError, match="exceeds workspace capacity"):
        backend.compute(workspace, transformed_weights, output=oversized)
    assert backend.compute(workspace, transformed_weights, output=empty) is empty
    assert backend.compute(workspace, transformed_weights, output=full_a) is full_a
    assert backend.compute(workspace, transformed_weights, output=full_b) is full_b
    assert (
        backend.compute(workspace, transformed_weights, output=partial_a) is partial_a
    )
    assert (
        backend.compute(workspace, transformed_weights, output=partial_b) is partial_b
    )
    assert backend.compute(workspace, transformed_weights, output=full_c) is full_c

    # The EP communication workspace is indexed with capacity*topk strides.
    # Partial output rows are copied back to the caller, but the persistent
    # kernel must retain capacity-sized descriptors so sender and receiver use
    # the same metadata addresses on every rank.
    assert frontend.builds == [None]
    assert frontend.launches == [None, None, None, None, None, None]
    torch.testing.assert_close(partial_a, out_buf[:3])
    torch.testing.assert_close(full_c, out_buf)

    # In-kernel reduction preserves the capacity-padded launch for every live
    # extent, including zero: independently scheduled EP peers require every
    # rank to participate physically in the persistent kernel each round.
    ikr_frontend = FakeFrontend(fc2_reduces_topk=True)
    ikr_workspace = SimpleNamespace(**vars(workspace))
    ikr_workspace._frontend = ikr_frontend
    ikr_backend = Nvfp4CutedslMegaKernelBackend(
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=64,
            top_k=2,
            # Deliberately stale relative to the effective frontend config,
            # as after an autotune winner flips this compile-time tactic.
            in_kernel_fc2_reduce=False,
        )
    )
    assert (
        ikr_backend.compute(ikr_workspace, transformed_weights, output=empty) is empty
    )
    assert (
        ikr_backend.compute(ikr_workspace, transformed_weights, output=partial_a)
        is partial_a
    )
    assert (
        ikr_backend.compute(ikr_workspace, transformed_weights, output=full_a) is full_a
    )
    assert ikr_frontend.builds == [None]
    assert ikr_frontend.launches == [None, None, None]
