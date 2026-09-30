"""Complete lazy inventory of native preparation contracts."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from .tuning import TuningContract


@dataclass(frozen=True, kw_only=True)
class KernelTuningRegistration:
    op_qualname: str
    contract_ref: str
    variant: str = "default"

    def load(self):
        module, separator, attribute = self.contract_ref.partition(":")
        if not separator or not attribute:
            raise ValueError("invalid tuning contract reference")
        contract = getattr(importlib.import_module(module), attribute)
        if not isinstance(contract, TuningContract):
            raise TypeError(f"{self.contract_ref} is not a TuningContract")
        return contract


TUNING_COMPONENTS = (
    KernelTuningRegistration(op_qualname="sequence.embedding", contract_ref="b12x.sequence.embedding._tuning:TUNING"),
    KernelTuningRegistration(op_qualname="attention.compressed_sparse_mla", contract_ref="b12x.attention.compressed_sparse_mla.cast:TUNING", variant="cast"),
    KernelTuningRegistration(op_qualname='attention.compressed_sparse_mla', contract_ref='b12x.attention.compressed_sparse_mla._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='attention.compressed_sparse_mla', contract_ref='b12x.attention.compressed_sparse_mla.rotary:TUNING', variant='rotate'),
    KernelTuningRegistration(op_qualname='attention.compressed_sparse_mla', contract_ref='b12x.attention.compressed_sparse_mla.weight_scale:TUNING', variant='index_weights'),
    KernelTuningRegistration(op_qualname='attention.compressed_sparse_mla', contract_ref='b12x.attention.compressed_sparse_mla.cache_writer:TUNING', variant='cache_write'),
    KernelTuningRegistration(op_qualname='attention.dense_mla', contract_ref='b12x.attention.dense_mla._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='attention.dsa_indexer', contract_ref='b12x.attention.dsa_indexer._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='attention.paged', contract_ref='b12x.attention.paged._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='attention.qsa', contract_ref='b12x.attention.qsa._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='attention.sparse_mla', contract_ref='b12x.attention.sparse_mla._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='attention.varlen', contract_ref='b12x.attention.varlen._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.bf16_vocab_projection', contract_ref='b12x.gemm.bf16_vocab_projection._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.block_fp8_linear', contract_ref='b12x.gemm.block_fp8_linear._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.wo_projection', contract_ref='b12x.gemm.wo_projection._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='moe.fused_moe', contract_ref='b12x.moe.fused_moe._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='moe.ep_moe', contract_ref='b12x.moe.ep_moe._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='norm.hyperconnection', contract_ref='b12x.norm.hyperconnection._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='norm.mhc', contract_ref='b12x.norm.mhc._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='quantization.nvfp4', contract_ref='b12x.quantization.nvfp4._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='sequence.gdn_decode', contract_ref='b12x.sequence.gdn_decode._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='sequence.mtp_feedback', contract_ref='b12x.sequence.mtp_feedback._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='sequence.ple', contract_ref='b12x.sequence.ple._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='sequence.ple_embedding', contract_ref='b12x.sequence.ple_embedding._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='sequence.ple_hash', contract_ref='b12x.sequence.ple_hash._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.blockscaled', contract_ref='b12x.gemm.blockscaled._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='sequence.gdn_prefill', contract_ref='b12x.sequence.gdn_prefill._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='sequence.kda_prefill', contract_ref='b12x.sequence.kda_prefill._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='comm.pcie', contract_ref='b12x.comm.pcie._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='comm.roce', contract_ref='b12x.comm.roce._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.bf16_gemv', contract_ref='b12x.gemm.bf16_gemv._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.bmm', contract_ref='b12x.gemm._bmm._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.mla_query_projection', contract_ref='b12x.gemm.mla_query_projection._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.mxfp8_linear', contract_ref='b12x.gemm.mxfp8_linear._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.mxfp8_linear', contract_ref='b12x.gemm.mxfp8_linear._tuning:FIXED_TUNING', variant='fixed'),
    KernelTuningRegistration(op_qualname='gemm.tensor_fp8_linear', contract_ref='b12x.gemm.tensor_fp8_linear._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.trellis_linear', contract_ref='b12x.gemm.trellis_linear._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.mm', contract_ref='b12x.gemm._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='gemm.blockscaled', contract_ref='b12x.gemm._tuning:TUNING', variant='native'),
    KernelTuningRegistration(op_qualname='gemm.blockscaled', contract_ref='b12x.gemm.blockscaled._tuning:FIXED_TUNING', variant='fixed'),
    KernelTuningRegistration(op_qualname='quantization.mxfp8', contract_ref='b12x.quantization.mxfp8._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='quantization.mxfp6', contract_ref='b12x.quantization.mxfp6._tuning:TUNING', variant='default'),
    KernelTuningRegistration(op_qualname='sequence.engram', contract_ref='b12x.sequence.engram._tuning:TUNING'),
    KernelTuningRegistration(op_qualname='attention.mla_compress', contract_ref='b12x.attention.mla_compress._tuning:TUNING'),
    KernelTuningRegistration(op_qualname='norm.vision', contract_ref='b12x.norm._vision_preparation:TUNING'),
    KernelTuningRegistration(op_qualname='moe.fused_moe', contract_ref='b12x.moe.fused_moe._tuning:ROUTE_TUNING', variant='route_topk'),
    KernelTuningRegistration(op_qualname='moe.fused_moe', contract_ref='b12x.moe.fused_moe._tuning:FC2_TUNING', variant='fc2'),
)
_TUNING_INDEX = {(item.op_qualname, item.variant): item for item in TUNING_COMPONENTS}
if len(_TUNING_INDEX) != len(TUNING_COMPONENTS):
    raise ValueError("tuning registrations must have unique API variants")


def list_tuning_components():
    return tuple(sorted(TUNING_COMPONENTS, key=lambda item: (item.op_qualname, item.variant)))


def get_tuning_contract(op_qualname, *, variant="default"):
    return _TUNING_INDEX[op_qualname, variant].load()
