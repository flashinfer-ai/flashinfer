# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural recogniser for the SDPA **epilogue-gate tail**.

The fused gate the Rubin d256 SDPA kernels carry (``O := O * sigmoid(G)``,
``TemplateParams.epilogue_gate``) is spelled on the graph as three nodes::

    O_v, ... = g.sdpa(q, k, v, ...)          # O_v VIRTUAL, dim + stride declared
    s        = g.sigmoid(input=G)            # G a graph INPUT, dims == O's
    O        = g.mul(a=O_v, b=s)             # the REAL output (set_output(True))

:func:`match_gate_tail` answers "is this node list exactly that shape?" and
nothing else -- it does not judge head dims, dtypes or arch (that is the engine
row's ``Capabilities.epilogue_gate*`` in ``cudnn.sdpa.fwd.engines.mismatch``),
and it does not validate the tensors (``cudnn._sdpa_validate`` does).  Two
callers share it so they cannot drift on what "the tail" means:

* ``cudnn.sdpa.graph_analyzer`` -- extracts the ``has_epilogue_gate`` facts and
  rebinds ``o_t`` to the mul output, so the fused kernel writes the graph's
  real O and the virtual ``O_v`` / ``s`` are never bound;
* ``cudnn._sdpa_validate.validate_graph`` -- python-native validation of the
  three-node graph, so ``pygraph.validate()`` does not fall back to the eager
  C++ lowering (whose ``pre_validate_node`` needs a rank-4 dim + stride on the
  sdpa node's O -- see the validator).

Import-light on purpose, like ``_sdpa_validate``: only the IR enum.  Never import
``cudnn.sdpa`` (torch / cutlass) or the compiled binding here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .graph_types import NodeType

# Forward SDPA flavors the tail may hang off.  Backward nodes have no O output
# to gate, so they are deliberately absent.
_SDPA_FWD_TYPES = (NodeType.SDPA, NodeType.SDPA_FP8, NodeType.SDPA_MXFP8)


@dataclass(frozen=True)
class GateTail:
    """The three nodes of a recognised gate tail plus the four tensors that matter.

    ``sdpa`` is the attention node; ``o_virtual`` its (virtual) O output;
    ``gate`` the graph input G that feeds the sigmoid; ``sig_out`` the (virtual)
    sigmoid output; ``o_final`` the mul output -- the graph's real O.
    """

    sdpa: Any
    o_virtual: Any
    gate: Any
    sig_out: Any
    o_final: Any


def _pointwise(node, mode: str, n_inputs: int) -> bool:
    """A POINTWISE node of ``mode`` with exactly ``n_inputs`` bound inputs and one output.

    ``params["mode"]`` is the canonical op name (``_pygraph._pointwise``); node
    NAMES are debug labels (duplicates legal) and never matched on.
    """
    if node.node_type is not NodeType.POINTWISE or node.params.get("mode") != mode:
        return False
    ins = [t for t in node.inputs.values() if t is not None]
    return len(ins) == n_inputs and node.outputs.get("OUT_0") is not None


def match_gate_tail(nodes) -> Optional[GateTail]:
    """The :class:`GateTail` when ``nodes`` is EXACTLY the three-node gate shape, else None.

    Matched by IDENTITY on the tensors (``is``), never by name or uid:

    * exactly three nodes: one forward SDPA (``SDPA`` / ``SDPA_FP8`` /
      ``SDPA_MXFP8``), one ``sigmoid`` pointwise with one input, one ``mul``
      pointwise with two inputs;
    * the mul's two inputs are the sdpa's ``O`` and the sigmoid's output, in
      either order;
    * the sigmoid's input G is a graph INPUT -- produced by none of the nodes
      (``sigmoid(O_v) * G`` is a different graph and is not matched);
    * the sdpa's ``O`` and the sigmoid output are virtual (intermediates the
      fused kernel never materialises); the mul output is not (it is the real
      O the kernel writes).

    Anything else -- a fourth node, a relu tail, a real (bound) ``O_v``, the
    sigmoid output marked as an output -- returns None, and the caller treats
    the graph as it always did (the analyzer declines a non-single-node graph;
    the validator falls back to the classic lowering).
    """
    nodes = list(nodes)
    if len(nodes) != 3:
        return None
    sdpa_nodes = [n for n in nodes if n.node_type in _SDPA_FWD_TYPES]
    sig_nodes = [n for n in nodes if _pointwise(n, "sigmoid", 1)]
    mul_nodes = [n for n in nodes if _pointwise(n, "mul", 2)]
    if len(sdpa_nodes) != 1 or len(sig_nodes) != 1 or len(mul_nodes) != 1:
        return None
    sdpa, sig, mul = sdpa_nodes[0], sig_nodes[0], mul_nodes[0]

    o_virtual = sdpa.outputs.get("O")
    gate = sig.inputs.get("IN_0")
    sig_out = sig.outputs.get("OUT_0")
    o_final = mul.outputs.get("OUT_0")
    if o_virtual is None or gate is None or sig_out is None or o_final is None:
        return None

    mul_ins = [mul.inputs.get("IN_0"), mul.inputs.get("IN_1")]
    if not ((mul_ins[0] is o_virtual and mul_ins[1] is sig_out) or (mul_ins[0] is sig_out and mul_ins[1] is o_virtual)):
        return None
    # G must be a graph input: produced by no node in the graph.
    if any(t is gate for n in nodes for t in n.outputs.values()):
        return None
    if not (getattr(o_virtual, "is_virtual", False) and getattr(sig_out, "is_virtual", False)):
        return None
    if getattr(o_final, "is_virtual", False):
        return None
    return GateTail(sdpa=sdpa, o_virtual=o_virtual, gate=gate, sig_out=sig_out, o_final=o_final)


__all__ = ["GateTail", "match_gate_tail"]
