"""Independent persistent SM90 queue: every L1 ticket precedes L2."""
import cutlass.cute as cute
from cutlass import Uint32
from ..ptx.ld_st import atom_add_u32


@cute.jit
def claim_sm90_task(counter: cute.Pointer) -> Uint32:
    return atom_add_u32(counter, Uint32(1))
