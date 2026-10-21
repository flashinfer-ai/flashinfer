# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-native semantic validation for the GEMM node family (issue #704).

The frost_gemm family's ``EngineFamily.validator``: when a python engine is a
candidate for a GEMM graph, ``pygraph.validate()`` runs these rules instead of
eagerly lowering to C++ and asking the backend -- which couples a graph the
FROST GEMM engine fully serves to the installed backend's version (the MoE
grouped-matmul node needs cuDNN >= 9.15 forward / 9.22 backward *to validate*,
attributes the backend would never execute for a FROST-served graph). The
backend's own verdict is deferred to planning, exactly as for SDPA.

Only the version- and arch-agnostic subset of the classic checks lives here:
the C++ matmul node has no semantic pre-validation of its own (the backend
judges shapes at plan time), so this module carries the structural facts a
matmul must satisfy to mean anything -- and the MoE grouped-matmul node's
ATTRIBUTE_NOT_SET checks with their classic messages. Nothing about device
arch or backend version: those are support-surface answers each engine gives
at planning.

Error types match the classic surface: ``ValueError`` for ATTRIBUTE_NOT_SET /
INVALID_VALUE parity, ``cudnn.cudnnGraphNotSupportedError`` for a graph the
backend would decline as GRAPH_NOT_SUPPORTED.
"""

from __future__ import annotations

from typing import Any

from .graph_types import NodeType

_MOE_FWD_INPUTS = ("token", "weight", "first_token_offset")
_MOE_BWD_INPUTS = ("doutput", "token", "first_token_offset")

COVERED_NODE_TYPES = frozenset(
    {
        NodeType.MATMUL,
        NodeType.MOE_GROUPED_MATMUL,
        NodeType.MOE_GROUPED_MATMUL_BWD,
        NodeType.BLOCK_SCALE_DEQUANTIZE,
        NodeType.BLOCK_SCALE_QUANTIZE,
        NodeType.POINTWISE,
        NodeType.REDUCTION,
    }
)


def _not_supported(message: str) -> Exception:
    """Classic GRAPH_NOT_SUPPORTED parity: the error type callers catch to skip a config."""
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    return cudnn.cudnnGraphNotSupportedError(message)


def _dims(t: Any):
    d = t.get_dim() if t is not None else None
    return list(d) if d else None


def validate_graph(graph) -> bool:
    """Validate GEMM/MoE graphs and their fusion nodes in Python.
    Return False for uncovered nodes so they retain classic eager lowering."""
    nodes = list(graph._nodes)
    if not nodes or any(n.node_type not in COVERED_NODE_TYPES for n in nodes):
        return False
    for node in nodes:
        validate_node(node)
    return True


def validate_node(node) -> None:
    """Dispatch one GEMM-family node to its rules."""
    if node.node_type == NodeType.MATMUL:
        _validate_matmul(node)
    elif node.node_type == NodeType.MOE_GROUPED_MATMUL:
        _validate_required(node, "MoeGroupedMatmul", _MOE_FWD_INPUTS, ("OUT_0",))
        _validate_moe_offsets(node)
        mode = getattr(node.params.get("mode"), "name", None)
        if mode in ("GATHER", "SCATTER", "COMBINE"):
            _validate_required(node, "MoeGroupedMatmul", ("token_index",), ())
        if mode in ("SCATTER", "COMBINE"):
            _validate_required(node, "MoeGroupedMatmul", ("token_ks",), ())
        if mode == "COMBINE":
            _validate_required(node, "MoeGroupedMatmul", ("top_k_scores",), ())
            _validate_moe_combine(node)
        elif node.inputs.get("top_k_scores") is not None:
            raise ValueError("top_k_scores is only valid for MoE COMBINE")
    elif node.node_type == NodeType.MOE_GROUPED_MATMUL_BWD:
        _validate_required(node, "MoeGroupedMatmulBwd", _MOE_BWD_INPUTS, ("dweight",))
        _validate_moe_offsets(node)
    elif node.node_type == NodeType.BLOCK_SCALE_DEQUANTIZE:
        _validate_required(node, "BlockScaleDequantize", ("input", "descale"), ("OUT_0",))
        if not node.params.get("block_size"):
            raise ValueError("Block size not set")
        if not node.outputs["OUT_0"].get_is_virtual():
            raise ValueError("Output tensor of dequantize node should be virtual")
    elif node.node_type == NodeType.BLOCK_SCALE_QUANTIZE:
        _validate_required(node, "BlockScaleQuantize", ("input",), ("Y", "scale"))
        if not node.params.get("block_size"):
            raise ValueError("Block size not set")
    elif node.node_type == NodeType.REDUCTION:
        _validate_required(node, "Reduction", ("input",), ("OUT_0",))
        if not node.params.get("mode"):
            raise ValueError("Reduction mode not set")
    elif node.node_type == NodeType.POINTWISE:
        _validate_required(node, "Pointwise", ("IN_0",), ("OUT_0",))
        if not node.params.get("mode"):
            raise ValueError("Pointwise mode not set")
        # Node.infer_properties checks input broadcasting when inferring an
        # output. Also check it when callers supplied an explicit output shape.
        out = _dims(node.outputs["OUT_0"])
        for tensor in node.inputs.values():
            dim = _dims(tensor)
            if dim and out and (len(dim) > len(out) or any(x != 1 and x != y for x, y in zip(reversed(dim), reversed(out)))):
                raise _not_supported("Pointwise inputs do not broadcast to the output shape")


def _validate_moe_combine(node) -> None:
    """COMBINE changes the token extent; preserve that public graph contract."""
    token, weight = _dims(node.inputs["token"]), _dims(node.inputs["weight"])
    scores = _dims(node.inputs["top_k_scores"])
    top_k = node.params.get("top_k")
    if not token or not weight or len(token) != 3 or len(weight) != 3:
        raise ValueError("MoE COMBINE token and weight must have rank 3")
    if not isinstance(top_k, int) or not 1 <= top_k <= weight[0]:
        raise ValueError("MoE COMBINE requires 1 <= top_k <= num_experts")
    if not scores or len(scores) != 3 or scores[0] != 1 or scores[1] <= 0 or scores[2] != top_k:
        raise ValueError("MoE COMBINE top_k_scores must have shape [1, tokens, top_k]")
    for role in ("token_index", "token_ks"):
        if _dims(node.inputs[role]) != [1, token[1], 1]:
            raise ValueError(f"MoE COMBINE {role} must have shape [1, routed_rows, 1]")
    if _dims(node.outputs["OUT_0"]) != [1, scores[1], weight[2]]:
        raise ValueError("MoE COMBINE output must have shape [1, tokens, N] matching top_k_scores")


def _validate_matmul(node) -> None:
    """C = A @ B with batch broadcasting: both operands bound, same rank >= 2,
    the contraction extent agrees, batch extents equal or 1, and a user-declared
    C carries the (M, N) the operands imply."""
    a, b, c = node.inputs.get("A"), node.inputs.get("B"), node.outputs.get("C")
    if a is None or b is None:
        raise ValueError("Matmul inputs A and B must both be set.")
    if c is None:
        raise ValueError("Matmul output C not set.")
    ad, bd = _dims(a), _dims(b)
    if not ad or not bd:
        raise ValueError("Matmul inputs A and B must have their dims set.")
    if len(ad) != len(bd) or len(ad) < 2:
        raise _not_supported(f"Matmul requires A and B of equal rank >= 2; got {ad} and {bd}")
    if ad[-1] != bd[-2]:
        raise _not_supported(f"Matmul contraction mismatch: A's last dim {ad[-1]} != B's second-to-last dim {bd[-2]}")
    for i, (x, y) in enumerate(zip(ad[:-2], bd[:-2])):
        if x != y and x != 1 and y != 1:
            raise _not_supported(f"Matmul batch dim {i} not broadcastable: {x} vs {y}")
    cd = _dims(c)
    if cd:
        if len(cd) != len(ad):
            raise _not_supported(f"Matmul output rank {len(cd)} != operand rank {len(ad)}")
        if cd[-2] != ad[-2] or cd[-1] != bd[-1]:
            raise _not_supported(f"Matmul output dims {cd} do not match (M, N) = ({ad[-2]}, {bd[-1]})")


def _validate_required(node, label: str, inputs, outputs) -> None:
    """Classic ATTRIBUTE_NOT_SET parity for required node ports."""
    for port in inputs:
        if node.inputs.get(port) is None:
            raise ValueError(f"{label} input {port} not set.")
    for port in outputs:
        if node.outputs.get(port) is None:
            raise ValueError(f"{label} output {port} not set.")


def moe_offset_mode(offset_count: int, num_experts: int) -> bool:
    """Infer G starts vs G+1 boundaries, with G a positive multiple of E.

    E=1 always uses explicit boundaries. For E>1, the length modulo E
    distinguishes the two modes without adding an operation attribute.
    """
    if num_experts > 0 and offset_count > 0:
        if num_experts == 1:
            if offset_count >= 2:
                return True
        elif offset_count % num_experts == 0:
            return False
        if offset_count > 1 and (offset_count - 1) % num_experts == 0:
            return True
    raise ValueError("first_token_offset must contain G starts or G+1 boundaries, where G is a positive multiple of the expert count (E=1 requires G+1)")


def _validate_moe_offsets(node) -> None:
    offsets = _dims(node.inputs["first_token_offset"])
    expert_tensor = node.inputs["weight"] if node.node_type == NodeType.MOE_GROUPED_MATMUL else node.outputs["dweight"]
    experts = _dims(expert_tensor)
    if offsets and experts:
        moe_offset_mode(offsets[0], experts[0])
