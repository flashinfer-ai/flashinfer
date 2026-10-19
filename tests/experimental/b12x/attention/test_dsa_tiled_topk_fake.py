import torch
import pytest

from b12x._lib.compile_plan import compile_only_launches
from b12x.attention.dsa_indexer import contiguous_kernel, tiled_topk
from b12x.attention.dsa_indexer.tiled_topk import _same_data_address


def test_indexer_dlpack_helpers_skip_fake_storage(monkeypatch) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    def fail(*args, **kwargs):
        raise AssertionError("fake tensor reached DLPack")

    monkeypatch.setattr(contiguous_kernel, "from_dlpack", fail)
    monkeypatch.setattr(tiled_topk, "from_dlpack", fail)
    with FakeTensorMode(), compile_only_launches():
        values = torch.empty((2, 4), device="cuda")
        contiguous_kernel._to_kernel_tensor(values, contiguous_kernel.cutlass.Float32)
        tiled_topk._to_kernel_tensor(values, tiled_topk.cutlass.Float32)
        assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_tiled_topk_alias_check_preserves_fake_aliases(monkeypatch, dtype) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    def fail(*args, **kwargs):
        raise AssertionError("fake tensor reached data_ptr")

    monkeypatch.setattr(torch.Tensor, "data_ptr", fail)
    with FakeTensorMode():
        values = torch.empty((4,), dtype=dtype, device="cuda")
        separate = torch.empty_like(values)
        assert _same_data_address(values, values)
        assert _same_data_address(values, values.view(2, 2))
        assert not _same_data_address(values, values.narrow(0, 1, 2))
        assert not _same_data_address(values, separate)


def test_tiled_topk_alias_check_preserves_real_aliases() -> None:
    values = torch.empty((4,))
    assert _same_data_address(values, values.view(2, 2))
    assert not _same_data_address(values, values.narrow(0, 1, 2))
    assert not _same_data_address(values, torch.empty_like(values))
