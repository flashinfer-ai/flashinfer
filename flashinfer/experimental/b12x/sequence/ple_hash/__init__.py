"""Prime-hashed learned n-gram embedding IDs.

``compute_geometry`` calculates and validates host metadata separately from
execution; ``allocate_geometry`` explicitly allocates checkpoint-owned tensors.
Loaders register those exact tensors before checkpoint load. ``plan`` declares
geometry and capacity without device allocation, and ``PreparationSession``
produces the prepared plan required by ``bind``. Device checkpoint tensors passed
to a declaration require their host ``geometry`` metadata; preparation borrows
them without replacing their storage. ``run`` writes logical embedding IDs
while leaving committed history immutable.

Checkpoint multiplier indices denote token lag: the hash of an order-``n``
window is ``xor(token[t-i] * multipliers[i] for i in range(n))``. Predecessors
behind an EOS boundary are replaced by EOS before hashing. Chronological
history buffers retain oldest-to-current storage order.

Exact PyTorch oracles are available from
:mod:`b12x.sequence.ple_hash.reference` and are never runtime fallbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="ple_hash",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "PleHashConfig",
        "PleHashQuery",
        "Geometry",
        "GeometryTensors",
        "compute_geometry",
        "allocate_geometry",
        "invocation_from_tensors",
        "plan",
        "bind",
        "run",
        "is_supported",
    ),
    dtypes=("int64",),
    recipes=("packed_eos_bounded",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="3a437ab51680",
        paths=(
            "b12x/_lib/scratch.py",
            "b12x/_lib/scratch_layout.py",
        ),
    ),
    test_path="tests/experimental/b12x/sequence/test_ple.py",
    since="1.3.0",
    notes=(
        "Packed hashing uses fixed-capacity caller-owned scratch. The Triton "
        "implementation is a correctness reference and is not "
        "throughput-qualified."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        PleHashConfig,
        PleHashQuery,
        Geometry,
        GeometryTensors,
        compute_geometry,
        allocate_geometry,
        invocation_from_tensors,
        bind,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
