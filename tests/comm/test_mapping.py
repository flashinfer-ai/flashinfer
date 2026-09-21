import pytest

from flashinfer.comm.mapping import Mapping


@pytest.mark.parametrize(
    ("kwargs", "invalid_name"),
    [
        ({"tp_size": 2}, "tp_size"),
        ({"pp_size": 2}, "pp_size"),
        ({"cp_size": 2}, "cp_size"),
    ],
)
def test_auto_parallel_rejects_explicit_parallel_sizes(kwargs, invalid_name):
    with pytest.raises(ValueError, match=invalid_name):
        Mapping(world_size=2, auto_parallel=True, **kwargs)


def test_auto_parallel_accepts_default_parallel_sizes():
    mapping = Mapping(world_size=2, auto_parallel=True)
    assert mapping.auto_parallel
