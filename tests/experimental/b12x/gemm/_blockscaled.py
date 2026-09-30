"""Native preparation helpers for blockscaled numerical and replay tests."""
from contextlib import contextmanager

import torch

from b12x.gemm import blockscaled
from b12x.gemm._tuning import DenseGemmQuery
from b12x.preparation import PreparationSession, PreparedCall


@contextmanager
def prepared(source, weight, *, freeze=True, override=None, **options):
    query = blockscaled.query_from_call(source, weight, **options)
    plan = blockscaled.plan(query, override=override)
    values = source[0] if isinstance(source, tuple) else source

    def call(state):
        if isinstance(query, DenseGemmQuery):
            run = lambda: state.run(
                *source, *weight, options.get("alpha"), options.get("out"),
                options.get("workspace"), getattr(torch, query.output_dtype), None,
            )
        elif isinstance(query, blockscaled.BlockscaledQuery):
            from b12x.gemm.blockscaled._a16 import _weight_parts
            packed_values, scales, gain, _ = _weight_parts(weight)
            run = lambda: state.run(
                source, packed_values, scales, gain,
                activation_scale=options.get("activation_global_scale"),
                out=options.get("out"), workspace=options.get("workspace"),
            )
        elif query.call_kind == "serialized":
            run = lambda: state.run_serialized(
                *source, *weight, options.get("alpha"),
                ab_dtype=options["ab_dtype"], sf_dtype=options["sf_dtype"],
                c_dtype=query.output_dtype, sf_vec_size=options["sf_vec_size"],
                block_fp8=options.get("block_fp8", False), stream=None,
            )
        else:
            run = lambda: state.run_mxfp8(
                values, weight.weight.values, weight.weight.scale_mma,
                source_scale=source[1] if isinstance(source, tuple) else None,
                out_dtype=getattr(torch, query.output_dtype), stream=None,
            )
        return PreparedCall(run=run)

    with PreparationSession(device=values.device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="blockscaled", prepare_call=call),))
        if freeze:
            session.freeze()
        yield plan
