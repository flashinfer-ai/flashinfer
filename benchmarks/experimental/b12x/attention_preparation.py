"""Native preparation for attention benchmark inputs and caller-owned buffers."""
from dataclasses import replace

import torch

from b12x.attention import compressed_sparse_mla as mla
from b12x.attention import dsa_indexer
from b12x.preparation import PreparedCall


def prepare_mxfp4(session, plan, *, q, keys, slots, arguments):
    def prepare(state):
        trial = dict(arguments)
        (spec,) = state.layout.scratch_specs()
        trial["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
        for name in ("q_mxfp4", "q_scales", "output_indices", "output_scores",
                     "candidate_output", "candidate_output_lengths"):
            if trial.get(name) is not None:
                trial[name] = torch.empty_like(trial[name])
        state.quantize_query(q, q_mxfp4=trial["q_mxfp4"], q_scales=trial["q_scales"])
        state.write_index_keys(keys, index_k_cache=trial["index_k_cache"], slot_mapping=slots)
        binding = state.bind(**trial)
        return PreparedCall(run=lambda: state.run(binding), owners=(trial, binding))
    session.prepare((plan.request(name="mxfp4-indexer", prepare_call=prepare),))
    dsa_indexer.quantize_q_mxfp4(plan, q, q_mxfp4=arguments["q_mxfp4"], q_scales=arguments["q_scales"])


def prepare_compressed(session, caps, *, bind_args, run_args, config=None):
    from b12x.attention.compressed_sparse_mla._tuning import TUNING, _split_config

    q = bind_args["q"]
    invocation = mla.invocation_from_tensors(
        q=q, swa_k_cache=run_args["swa_k_cache"],
        indexed_k_cache=run_args.get("indexed_k_cache"),
        attn_sink=run_args.get("attn_sink"), out=run_args.get("out"),
        return_lse=run_args.get("return_lse", False),
        lse_scale=run_args.get("lse_scale", "base2"),
    )
    plan = mla.plan(caps, invocation=invocation)
    if config is None:
        config = TUNING.configure(plan.query, device=session.device.identity).default
        if caps.max_chunks_per_row is not None and not config.single_pass:
            split = _split_config(plan.query, caps.max_chunks_per_row)
            config = replace(config, max_chunks_per_row=split.num_chunks, split_chunk_size=split.chunk_size)
    plan = mla.plan(caps, invocation=invocation, override=config)
    def prepare(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
        trial = dict(run_args)
        if trial.get("out") is not None:
            trial["out"] = torch.empty_like(trial["out"])
        binding = state.bind_for_preparation(scratch=scratch, **bind_args)
        return PreparedCall(run=lambda: state.run(binding, **trial), owners=(scratch, binding, trial))
    session.prepare((plan.request(name="compressed-attention", prepare_call=prepare),))
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
    return plan, mla.bind(plan, scratch=scratch, **bind_args)
