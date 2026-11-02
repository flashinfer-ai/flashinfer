import os
import subprocess
import sys
import textwrap


def _run_offline_worker(code):
    prelude = textwrap.dedent(
        """
        import multiprocessing
        import torch
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        from b12x._lib import compile_pool

        def initialize_worker():
            compile_pool._initialize_worker(
                0, (12, 1), "synthetic", "GB10", 48, 232448, 232448,
                multiprocessing.get_context("spawn").Array("q", (0, 0)),
            )
        """
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fake_cuda_indexing_stays_offline():
    _run_offline_worker(
        """
        cases = [
            slice(1, None, 2),
            1,
            (slice(None), slice(1, 4)),
            (Ellipsis, None, 2),
            True,
            [1, 3],
            (slice(None), [1, 3]),
        ]

        def operand(device, transpose):
            tensor = torch.empty(200, dtype=torch.float32, device=device).narrow(
                0, 3, 100
            ).view(20, 5)
            return tensor.t() if transpose else tensor

        def geometry(tensor):
            return tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset()

        expected = {}
        for transpose in (False, True):
            real = operand("cpu", transpose)
            assert not isinstance(real, FakeTensor)
            expected[transpose] = [geometry(real[index]) for index in cases]

        initialize_worker()
        with FakeTensorMode():
            for transpose in (False, True):
                fake = operand("cuda", transpose)
                for index, reference in zip(cases, expected[transpose]):
                    got = fake[index]
                    assert geometry(got) == reference, (transpose, index)
                    assert got.device.type == "cuda"
            fake[:2] = 1.0
            fake[[1, 3]] = 2.0
        assert not torch.cuda.is_initialized()
        """
    )


def test_fake_cuda_assignment_validates_broadcasting():
    _run_offline_worker(
        """
        cases = [
            (slice(0, 2), (2, 5), None),
            (slice(0, 2), (5,), None),
            (slice(0, 2), (1, 2, 5), None),
            (slice(0, 2), (3, 5), RuntimeError),
            (slice(0, 2), (2, 4), RuntimeError),
            ([1, 3], (2, 5), None),
            ([1, 3], (5,), None),
            ([1, 3], (1, 2, 5), None),
            ([1, 3], (3, 5), RuntimeError),
            ([1, 3], (2, 4), RuntimeError),
            ((slice(None), [1, 3]), (4, 2), None),
            ((slice(None), [1, 3]), (4, 3), RuntimeError),
        ]

        def assign(destination, index, value):
            try:
                destination[index] = value
            except (RuntimeError, ValueError, IndexError) as error:
                return type(error)
            return None

        for index, shape, expected_error in cases:
            destination = torch.empty(4, 5)
            value = torch.empty(shape)
            assert not isinstance(destination, FakeTensor)
            assert assign(destination, index, value) is expected_error, (index, shape)

        initialize_worker()
        with FakeTensorMode():
            for index, shape, expected_error in cases:
                destination = torch.empty(4, 5, device="cuda")
                value = torch.empty(shape, device="cuda")
                assert assign(destination, index, value) is expected_error, (index, shape)
        real = torch.zeros(4, 5)
        real[:2] = 3.0
        assert torch.equal(real[:2], torch.full((2, 5), 3.0))
        assert not torch.cuda.is_initialized()
        """
    )
