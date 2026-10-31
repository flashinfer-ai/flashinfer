# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Metadata-only traces for finalized, precomputed MoELayer calls.

Packs contain backend-native tensors. Preserve every view and its storage
dimensions rather than pretending packed FP4 is a logical BF16 matrix.
"""

import functools
import hashlib

import torch

from ..template import Const, Tensor, TraceTemplate, Var


def _pack_tensors(act_pack, weight_pack):
    activations = tuple(
        (name, value)
        for name in (
            "hidden_states_q",
            "hidden_states_scale",
            "topk_ids",
            "topk_weights",
            "per_token_scale",
        )
        if isinstance(value := getattr(act_pack, name, None), torch.Tensor)
    )
    weights = tuple(
        (f"{backend}_{name}", value)
        for backend, view in sorted(weight_pack.native_views.items())
        for name, value in sorted(view.items())
        if isinstance(value, torch.Tensor)
    )
    return activations, weights


class _PackedMoETrace(TraceTemplate):
    def build_fi_trace_fn(self, fi_api):
        def generate(*, self, act_pack, weight_pack, **kwargs):
            template = _resolve_template(
                self=self, act_pack=act_pack, weight_pack=weight_pack
            )
            if template is None:
                return {}
            activations, weights = _pack_tensors(act_pack, weight_pack)
            if kwargs.get("name") is None:
                # Native views have many storage axes; the generic name would
                # exceed filesystem component limits. The prefix hashes all
                # shape, dtype, and configuration metadata.
                kwargs["name"] = template.name_prefix
            return template.build_fi_trace_fn(fi_api)(
                act_pack=tuple(t for _, t in activations),
                weight_pack=tuple(t for _, t in weights),
                **kwargs,
            )

        return generate


@functools.lru_cache(maxsize=32)
def _template(activation_shapes, weight_shapes, hidden_size, output_dtype, config):
    axes = {"num_tokens": Var(), "hidden_size": Const(value=hidden_size)}
    inputs = {}
    for param, specs in (
        ("act_pack", activation_shapes),
        ("weight_pack", weight_shapes),
    ):
        for index, (name, shape, dtype) in enumerate(specs):
            dimensions = []
            for dim, size in enumerate(shape):
                if (
                    param == "act_pack"
                    and dim == 0
                    and name
                    in (
                        "hidden_states_q",
                        "topk_ids",
                        "topk_weights",
                        "per_token_scale",
                    )
                ):
                    axis = "num_tokens"
                else:
                    axis = f"{name}_dim{dim}"
                    axes[axis] = Const(value=size)
                dimensions.append(axis)
            inputs[name] = Tensor(dimensions, param=param, tuple_idx=index, dtype=dtype)
    identity = hashlib.sha256(
        repr((activation_shapes, weight_shapes, config)).encode()
    ).hexdigest()[:16]
    return TraceTemplate(
        op_type="moe_layer",
        name_prefix=f"moe_layer_{identity}",
        axes=axes,
        inputs=inputs,
        outputs={"output": Tensor(["num_tokens", "hidden_size"], dtype=output_dtype)},
        description=f"Finalized MoELayer with precomputed routing. Native weight-view tensors are preserved. Config: {config}",
    )


def _resolve_template(*, self, act_pack, weight_pack):
    """Resolve a packed-call schema without reading tensor contents or launching."""
    config = self.config
    if not config.finalize.do_finalize or act_pack.topk_ids is None:
        return None
    activations, weights = _pack_tensors(act_pack, weight_pack)
    if not activations or not weights:
        return None
    hidden = act_pack.hidden_states_q.shape[1]
    if config.quant.activation.name in ("NVFP4", "MXFP4"):
        hidden *= 2
    output_dtype = {"BF16": "bfloat16", "FP16": "float16", "FP32": "float32"}.get(
        config.quant.output.name
    )
    if output_dtype is None:
        return None
    # Token count is dynamic in data/routing tensors. Scale storage can be
    # padded/swizzled, so its physical shape remains explicit in the schema.
    return _template(
        tuple(
            (name, tuple(t.shape), str(t.dtype).removeprefix("torch."))
            for name, t in activations
        ),
        tuple(
            (name, tuple(t.shape), str(t.dtype).removeprefix("torch."))
            for name, t in weights
        ),
        hidden,
        output_dtype,
        repr(config),
    )


# The logging dispatcher caches adapters by identity. Keep one adapter and a
# bounded schema cache, so tracing dynamic pack shapes cannot retain templates
# indefinitely in the decorator's dispatcher cache.
_ADAPTER = _PackedMoETrace(op_type="moe_layer", axes={}, inputs={}, outputs={})


def moe_layer_trace(**kwargs):
    return _ADAPTER
