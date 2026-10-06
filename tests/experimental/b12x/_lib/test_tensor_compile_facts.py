"""Tensor compile facts preserve cache identity without transient policies."""

import importlib

import pytest

compiler = importlib.import_module("b12x._lib.compiler")


@pytest.mark.parametrize(
    "shape,stride",
    [((), ()), ((0,), (1,)), ((7, 32), (32, 1)), ((3, 7, 32), (224, 1, 7))],
)
@pytest.mark.parametrize("dynamic", [(), (0,), (0, 1, 2)])
def test_tensor_facts_preserve_keys_without_dimension_objects(
    monkeypatch, shape, stride, dynamic
):
    """Implicit dimension policies emit identical facts without transient objects."""
    torch = pytest.importorskip("torch")
    tensor = torch.empty_strided(shape, stride, dtype=torch.bfloat16)
    explicit = compiler.tensor_compile_fact(
        "source",
        tensor,
        dims=tuple(
            compiler.DimKey.dynamic() if i in dynamic else compiler.DimKey.exact(n)
            for i, n in enumerate(shape)
        ),
        strides=tuple(
            compiler.DimKey.dynamic() if i in dynamic else compiler.DimKey.exact(n)
            for i, n in enumerate(stride)
        ),
        align=16,
        layout={"order": [0, 1]},
    )
    expected_spec = compiler.KernelCompileSpec.from_facts("test.tensor", 1, explicit)

    def forbid_dimension_object(*args, **kwargs):
        raise AssertionError("Implicit tensor facts must not construct DimKey")

    monkeypatch.setattr(compiler.DimKey, "exact", forbid_dimension_object)
    monkeypatch.setattr(compiler.DimKey, "dynamic", forbid_dimension_object)
    actual = compiler.tensor_compile_fact(
        "source",
        tensor,
        dynamic_dims=dynamic,
        dynamic_strides=dynamic,
        align=16,
        layout={"order": [0, 1]},
    )
    actual_spec = compiler.KernelCompileSpec.from_facts("test.tensor", 1, actual)
    assert actual == explicit
    assert actual_spec.json_key == expected_spec.json_key
    assert actual_spec.hash_key == expected_spec.hash_key


def test_tensor_facts_keep_live_dynamic_rows_out_of_compile_key():
    """Dynamic shape facts retain geometry without retaining tensor identity."""
    torch = pytest.importorskip("torch")
    facts = [
        compiler.tensor_compile_fact("source", torch.empty(rows, 32), dynamic_dims=(0,))
        for rows in (1, 3, 8, 4096)
    ]
    assert all(value == facts[0] for value in facts)
    assert (
        compiler.tensor_compile_fact("source", torch.empty(3, 64), dynamic_dims=(0,))
        != facts[0]
    )


@pytest.mark.parametrize("keyword,label", [("dims", "dim"), ("strides", "stride")])
def test_tensor_facts_reject_explicit_policy_rank_mismatch(keyword, label):
    torch = pytest.importorskip("torch")
    with pytest.raises(
        ValueError, match=f"{label} policy rank 1 does not match tensor rank 2"
    ):
        compiler.tensor_compile_fact("source", torch.empty(3, 32), **{keyword: (1,)})
