# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tracing for source-neutral sparse MLA with caller-prepared metadata."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


def _template(
    state,
    *,
    primary_page=1,
    extra_page=1,
    primary_rank=3,
    extra_rank=3,
    tensor_scalars=(),
):
    prefix = ["rows"] if state["packed"] else ["batch_size", "seq_len_q"]
    axes = dict(
        batch_size=Const(abbrev="b", value=state["batch"]),
        seq_len_q=Const(abbrev="sq", value=state["max_q"]),
        rows=Var(),
        num_heads=Const(abbrev="h", value=state["heads"]),
        head_dim=Const(abbrev="d", value=512),
        primary_page_size=Const(abbrev="ps", value=primary_page),
        extra_page_size=Const(abbrev="pe", value=extra_page),
        primary_capacity=Const(abbrev="k", value=state["ks"]),
        extra_capacity=Const(abbrev="ke", value=state["kc"]),
        route_capacity=Const(abbrev="", value=state["capacity"]),
        primary_pages=Var(),
        extra_pages=Var(),
        passes=Var(),
        scale_words=Var(),
        singleton=Const(abbrev="", value=1),
        qo_offsets=Const(abbrev="", value=state["batch"] + 1),
    )

    def pool(name, rank):
        dims = [name + "_pages", name + "_page_size", "head_dim"]
        if rank == 4:
            dims.insert(1 if state["kv_layout"] == "HND" else 2, "singleton")
        return dims

    inputs = {
        "query": Tensor(prefix + ["num_heads", "head_dim"]),
        "kv_cache": Tensor(pool("primary", primary_rank)),
        "extra_kv_cache": Tensor(pool("extra", extra_rank), optional=state["kc"] == 0),
        "qo_indptr": Tensor(
            ["qo_offsets"], dtype="int32", optional=not state["packed"]
        ),
        "sinks": Tensor(
            ["num_heads"], dtype="float32", optional=not state["has_sinks"]
        ),
        "validate": Scalar("bool", optional=True),
    }
    fields = (
        ("indices", ["rows", "primary_capacity"], "int32", False),
        ("lengths", ["rows"], "int32", False),
        ("extra_indices", ["rows", "extra_capacity"], "int32", state["kc"] == 0),
        ("extra_lengths", ["rows"], "int32", state["kc"] == 0),
        ("routes", ["passes", "rows", "route_capacity"], "int32", True),
        ("execution_lengths", ["passes", "rows"], "int32", True),
        ("valid_counts", ["passes", "rows"], "int32", True),
        ("scale_params", ["passes", "scale_words"], "float32", True),
    )
    for index, (name, dims, dtype, optional) in enumerate(fields):
        inputs[name] = Tensor(
            dims, dtype=dtype, optional=optional, param="metadata", tuple_idx=index
        )
    for name in (
        "softmax_scale",
        "q_scale",
        "kv_scale",
        "extra_kv_scale",
        "output_scale",
    ):
        shape = dict(tensor_scalars).get(name)
        inputs[name] = (
            Tensor(list(shape), dtype="float32", optional=True)
            if shape is not None
            else Scalar("float32", optional=True)
        )
    outputs = {
        "output": Tensor(
            prefix + ["num_heads", "head_dim"], dtype="bfloat16", param="out"
        )
    }
    if state["return_lse"]:
        outputs["lse"] = Tensor(prefix + ["num_heads"], dtype="float32", param="lse")
    return TraceTemplate(
        op_type="mla_paged",
        name_prefix="prims_ts_sparse_mla_"
        + ("fp8" if state.get("dtype") == torch.float8_e4m3fn else "bf16")
        + ("_packed_q" if state["packed"] else "")
        + ("_valid_prefix" if state.get("assume_valid_prefix", False) else "")
        + ("_sink" if state["has_sinks"] else "")
        + ("_lse" if state["return_lse"] else ""),
        description="Native D512 sparse attention with caller-prepared storage-row metadata. One primary KV pool and optional extra pool; preparation excluded. LSE excludes sinks.",
        axes=axes,
        inputs=inputs,
        outputs=outputs,
        constraints=["head_dim == 512"],
        tags=[
            "backend:prims-ts",
            "status:experimental",
            "mla",
            "sparse",
            "metadata:prepared",
            "two-source" if state["kc"] else "single-source",
        ]
        + (["indices:valid-prefix"] if state.get("assume_valid_prefix", False) else []),
    )


def sparse_mla_wrapper_trace_dispatch(**kwargs):
    state = getattr(getattr(kwargs.get("self"), "_impl", None), "_state", None)
    if state is None:
        raise RuntimeError("plan() must precede tracing run()")
    primary, extra = kwargs.get("kv_cache"), kwargs.get("extra_kv_cache")
    if not isinstance(primary, torch.Tensor) or not isinstance(
        kwargs.get("metadata"), tuple
    ):
        raise ValueError(
            "prepared tracing requires the primary pool and prepared metadata"
        )
    if state["kc"] and not isinstance(extra, torch.Tensor):
        raise ValueError("prepared tracing requires the extra pool")

    def page(cache):
        if cache is None:
            return 1
        return cache.shape[2 if cache.ndim == 4 and state["kv_layout"] == "HND" else 1]

    scalars = tuple(
        (name, tuple("singleton" for _ in value.shape))
        for name in (
            "softmax_scale",
            "q_scale",
            "kv_scale",
            "extra_kv_scale",
            "output_scale",
        )
        if isinstance((value := kwargs.get(name)), torch.Tensor)
    )
    return _template(
        state,
        primary_page=page(primary),
        extra_page=page(extra),
        primary_rank=primary.ndim,
        extra_rank=3 if extra is None else extra.ndim,
        tensor_scalars=scalars,
    )


sparse_mla_wrapper_trace_dispatch.templates = [  # type: ignore[attr-defined]
    _template(
        dict(
            batch=2,
            max_q=1,
            heads=16,
            packed=False,
            ks=128,
            kc=64,
            capacity=256,
            kv_layout="NHD",
            has_sinks=True,
            return_lse=True,
        )
    )
]


def sparse_mla_one_shot_trace_dispatch(**kwargs):
    from types import SimpleNamespace

    query, metadata = kwargs["query"], kwargs["metadata"]
    packed = query.ndim == 3
    offsets = kwargs.get("qo_indptr")
    if packed and offsets is None:
        raise ValueError("packed queries require qo_indptr")
    batch = offsets.numel() - 1 if packed else query.shape[0]
    max_q = kwargs.get("max_seq_len_q")
    if max_q is None:
        max_q = (
            max(1, int((offsets[1:] - offsets[:-1]).max().item()))
            if packed
            else query.shape[1]
        )
    ks = metadata.indices.shape[-1]
    kc = 0 if metadata.extra_indices is None else metadata.extra_indices.shape[-1]
    state = dict(
        dtype=query.dtype,
        batch=batch,
        max_q=max_q,
        heads=query.shape[-2],
        packed=packed,
        ks=ks,
        kc=kc,
        capacity=max(256, ((ks + 127) // 128 + (kc + 127) // 128) * 128),
        kv_layout=kwargs.get("kv_layout", "NHD"),
        has_sinks=kwargs.get("sinks") is not None,
        assume_valid_prefix=kwargs.get("assume_valid_prefix", False),
        return_lse=kwargs.get("return_lse", False),
    )
    return sparse_mla_wrapper_trace_dispatch(
        **{**kwargs, "self": SimpleNamespace(_impl=SimpleNamespace(_state=state))}
    )


sparse_mla_one_shot_trace_dispatch.templates = (  # type: ignore[attr-defined]
    sparse_mla_wrapper_trace_dispatch.templates  # type: ignore[attr-defined]
)
