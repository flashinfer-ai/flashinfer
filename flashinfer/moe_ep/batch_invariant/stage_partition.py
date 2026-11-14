# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""K-stage regrouping with unchanged native MMA atoms and physical padding.

A different stage size is a transport partition only when both FC1 and FC2
execute exactly the canonical padded native-instruction sequence. No zero
instruction is dropped, and the M/N/CTA primitive remains unchanged.
"""


def padded_length(length, stage):
    return ((length + stage - 1) // stage) * stage


def plan_component_terms(module, shape, stage_k, kwargs):
    kwargs = dict(kwargs)
    fp = kwargs.pop("fp", None)
    arch = kwargs["arch"]
    provider = kwargs.get("implementation", "standard")
    mma_args = dict(arch=arch, implementation=provider)
    if module.__name__.rsplit(".", 1)[-1] != "nvfp4_spec":
        mma_args["quant_kind"] = kwargs.get(
            "quant_kind",
            "mxfp8_e4m3" if module.__name__.endswith("mxfp8_spec") else "mxfp4_mxfp8",
        )
    cls = module.numerical_class(**mma_args)
    allowed = (
        (cls.stage_k, 2 * cls.stage_k) if provider == "standard" else (cls.stage_k,)
    )
    if type(stage_k) is not int or stage_k not in allowed:
        raise ValueError(
            "stage_k must be an admitted canonical/double standard K stage"
        )
    if any(
        padded_length(k, stage_k) != padded_length(k, cls.stage_k) for k in shape[:2]
    ):
        raise ValueError(
            "K stage changes the canonical padded MMA instruction sequence"
        )
    if fp is None:
        if kwargs.get("combine_dtype", "bf16") != "bf16":
            from .wire_spec import OrderedTermAlgebra

            fp = OrderedTermAlgebra()
        else:
            fp = module.OrderedTermAlgebra()
    terms = module.component_terms(*shape, **kwargs, fp=fp)
    for name, length, operand in (
        ("fc1_gate", shape[0], "input_w13.gate"),
        ("fc1_up", shape[0], "input_w13.up"),
        ("fc2", shape[1], "handoff_w2"),
    ):
        canonical = module.mma_fold(fp, length, operand, **mma_args)
        candidate = module.mma_fold(
            fp, length, operand, **mma_args, blocked=True, stage_k=stage_k
        )
        # FC2 may wrap the accumulator in alpha multiplication and BF16 CVT,
        # or retain FP32 for legacy MXFP8's quantized return wire.
        pending, replaced = [(terms[name], False)], {}
        while pending:
            value, visited = pending.pop()
            if value is canonical:
                replaced[value] = candidate
            elif value not in replaced:
                children = [arg for arg in value.args if hasattr(arg, "op")]
                if visited:
                    replaced[value] = fp.op(
                        value.op.removeprefix("op:"),
                        *(
                            replaced[arg] if hasattr(arg, "op") else arg
                            for arg in value.args
                        ),
                    )
                else:
                    pending.append((value, True))
                    pending.extend((child, False) for child in children)
        terms[name] = replaced[terms[name]]
    return terms
