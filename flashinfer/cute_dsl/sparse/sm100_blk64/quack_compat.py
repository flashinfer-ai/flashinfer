import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import dsl_user_op


def ensure_cute_core_typing_compat() -> None:
    """Expose moved CuTe typing classes under cute.core for quack annotations."""
    for name in ("ThrMma", "ThrCopy", "TiledMma", "TiledCopy", "CopyAtom"):
        if hasattr(cute, name) and not hasattr(cute.core, name):
            setattr(cute.core, name, getattr(cute, name))
    if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
        cute.make_fragment = cute.make_rmem_tensor


def ensure_intvalue_block_arg_compat() -> None:
    """Avoid cute.IntValue repr crashes for MLIR block arguments."""
    if getattr(cute.core.IntValue.get_typed_value, "_bsa_block_arg_compat", False):
        return

    @dsl_user_op
    def get_typed_value(self, *, loc=None, ip=None) -> ir.Value:
        if isinstance(self.type, ir.IntegerType):
            def_op = getattr(self.owner, "operation", None)
            if def_op is not None and def_op.name == "cute.get_scalars":
                return def_op.operands[0]

        assert not isinstance(self.type, cute.core._cute_ir.IntTupleType)

        res_ty, _ = cute.core._cute_ir.pack_int_tuple(self)
        return cute.core._cute_ir.MakeIntTupleOp(res_ty, [self], loc=loc, ip=ip).result

    get_typed_value._bsa_block_arg_compat = True
    cute.core.IntValue.get_typed_value = get_typed_value


ensure_cute_core_typing_compat()
ensure_intvalue_block_arg_compat()
