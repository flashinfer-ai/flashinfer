from __future__ import annotations

import torch

from flashinfer.experimental.sm12x.comm.pcie.pcie_oneshot import (
    PCIeOneshotAllReduce,
    PCIeOneshotAllReducePool,
    parse_pcie_oneshot_max_size,
)


class _FakeExt:
    def __init__(self):
        self.init_calls = []
        self.register_pcie_buffers_calls = []
        self.register_buffer_calls = []
        self.all_reduce_calls = []
        self.all_reduce_fused_add_rms_norm_calls = []
        self.dispose_calls = []
        self.register_graph_buffers_calls = []
        self.handle_bytes = [1, 2, 3]
        self.offsets = [0, 64]

    def init_custom_ar(self, signal_ptrs, rank_data, rank):
        self.init_calls.append((tuple(signal_ptrs), rank_data.device.type, rank))
        return 12345

    def register_pcie_buffers(self, ptr, ptrs0, ptrs1):
        self.register_pcie_buffers_calls.append((ptr, tuple(ptrs0), tuple(ptrs1)))

    def register_buffer(self, ptr, peer_input_ptrs):
        self.register_buffer_calls.append((ptr, tuple(peer_input_ptrs)))

    def all_reduce(self, ptr, inp, out, reg_buffer, reg_buffer_bytes):
        self.all_reduce_calls.append(
            (
                ptr,
                int(inp.data_ptr()),
                int(out.data_ptr()),
                reg_buffer,
                reg_buffer_bytes,
            )
        )
        out.copy_(inp)

    def all_reduce_fused_add_rms_norm(
        self,
        ptr,
        inp,
        residual,
        weight,
        out,
        residual_out,
        epsilon,
        reg_buffer,
        reg_buffer_bytes,
    ):
        self.all_reduce_fused_add_rms_norm_calls.append(
            (
                ptr,
                int(inp.data_ptr()),
                int(residual.data_ptr()),
                int(weight.data_ptr()),
                int(out.data_ptr()),
                int(residual_out.data_ptr()),
                epsilon,
                reg_buffer,
                reg_buffer_bytes,
            )
        )
        residual_value = residual.clone()
        residual_out.copy_(inp)
        residual_out.add_(residual_value)
        variance = residual_out.float().square().mean(dim=-1, keepdim=True)
        normalized = residual_out.float() * torch.rsqrt(variance + epsilon)
        out.copy_((normalized * weight.float()).to(out.dtype))

    def dispose(self, ptr):
        self.dispose_calls.append(ptr)

    def meta_size(self):
        return 256

    def get_graph_buffer_ipc_meta(self, ptr):
        return list(self.handle_bytes), list(self.offsets)

    def register_graph_buffers(self, ptr, handles, offsets):
        self.register_graph_buffers_calls.append((ptr, handles, offsets))


def _make_runtime(
    *,
    rank=0,
    world_size=2,
    exchange_group=None,
    max_size=8 * 1024 * 1024,
    eager=False,
    ext=None,
    stream_affine=True,
):
    ext = ext or _FakeExt()
    kwargs = {}
    if eager:
        kwargs["eager_buffer_ptrs0"] = tuple(range(200, 200 + world_size))
        kwargs["eager_buffer_ptrs1"] = tuple(range(300, 300 + world_size))
    return PCIeOneshotAllReduce(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        signal_ptrs=tuple(range(100, 100 + world_size)),
        exchange_group=exchange_group,
        max_size=max_size,
        ext_module=ext,
        stream_affine=stream_affine,
        **kwargs,
    )


def test_parse_pcie_oneshot_max_size_accepts_auto_and_suffixes():
    assert parse_pcie_oneshot_max_size(None) is None
    assert parse_pcie_oneshot_max_size("auto") is None
    assert parse_pcie_oneshot_max_size("64KB") == 64 * 1024
    assert parse_pcie_oneshot_max_size("2m") == 2 * 1024 * 1024
    assert parse_pcie_oneshot_max_size(4096) == 4096


def test_register_buffer_is_idempotent_for_same_mapping():
    runtime = _make_runtime()
    ext = runtime._ext

    runtime.register_buffer((111, 222))
    runtime.register_buffer((111, 222))

    assert ext.register_buffer_calls == [(12345, (111, 222))]


def test_all_reduce_registers_explicit_peer_ptrs_once():
    runtime = _make_runtime()
    ext = runtime._ext
    inp = torch.arange(8, dtype=torch.bfloat16)

    out0 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))
    out1 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))

    assert torch.equal(out0, inp)
    assert torch.equal(out1, inp)
    assert ext.register_buffer_calls == [(12345, (inp.data_ptr(), 222))]
    assert len(ext.all_reduce_calls) == 2


def test_eager_buffers_allow_all_reduce_without_peer_ptrs():
    runtime = _make_runtime(eager=True)
    ext = runtime._ext
    inp = torch.arange(8, dtype=torch.bfloat16)

    out = runtime.all_reduce(inp)

    assert torch.equal(out, inp)
    assert ext.register_pcie_buffers_calls == [(12345, (200, 201), (300, 301))]
    assert ext.register_buffer_calls == []
    assert len(ext.all_reduce_calls) == 1


def test_fused_add_rms_norm_returns_norm_and_residual_outputs():
    runtime = _make_runtime(eager=True)
    ext = runtime._ext
    inp = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 8
    residual = torch.linspace(-0.5, 0.5, 16, dtype=torch.bfloat16).reshape(2, 8)
    weight = torch.linspace(0.75, 1.25, 8, dtype=torch.bfloat16)

    out, residual_out = runtime.all_reduce_fused_add_rms_norm(
        inp,
        residual,
        weight,
        1e-6,
    )

    expected_residual = inp + residual
    variance = expected_residual.float().square().mean(dim=-1, keepdim=True)
    expected_out = (
        expected_residual.float() * torch.rsqrt(variance + 1e-6) * weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual_out, expected_residual)
    torch.testing.assert_close(out, expected_out)
    assert len(ext.all_reduce_fused_add_rms_norm_calls) == 1


def test_world_size_10_is_supported_by_eager_pool():
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(rank=3, world_size=10, eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=3,
        world_size=10,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )

    runtime = pool.for_stream()

    assert runtime.world_size == 10
    assert runtime._ext.init_calls == [
        ((100, 101, 102, 103, 104, 105, 106, 107, 108, 109), "cpu", 3)
    ]
    assert runtime._ext.register_pcie_buffers_calls == [
        (
            12345,
            (200, 201, 202, 203, 204, 205, 206, 207, 208, 209),
            (300, 301, 302, 303, 304, 305, 306, 307, 308, 309),
        )
    ]
    assert created == [(None, runtime)]
