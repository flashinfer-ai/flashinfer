import os
import tvm
from tvm.script import ir as I, relax as R, tir as T


def IRModuleGen(dtype: str) -> tvm.IRModule:
    @I.ir_module
    class FlashInfer:
        @R.function
        def decode(
            q: R.Tensor(("num_heads", "head_dim"), dtype),
            k: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
            v: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
        ) -> R.Tensor(("seq_len", "num_heads", "head_dim"), dtype):
            num_heads = T.int64()
            head_dim = T.int64()
            with R.dataflow():
                o = R.call_dps_packed(
                    "fused_parallel_attention",
                    (q, k, v),
                    out_sinfo=R.Tensor((num_heads, head_dim), dtype),
                )
                R.output(o)
            return o

        @R.function
        def fused_rope_decode(
            q: R.Tensor(("num_heads", "head_dim"), dtype),
            k: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
            v: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
        ) -> R.Tuple(
            R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
            R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
        ):
            num_heads = T.int64()
            head_dim = T.int64()
            with R.dataflow():
                o = R.call_dps_packed(
                    "fused_rope_parallel_attention",
                    (q, k, v),
                    out_sinfo=R.Tensor((num_heads, head_dim), dtype),
                )
                R.output(o)
            return k, o

        @R.function
        def fused_updated_rope_decode(
            q: R.Tensor(("num_heads", "head_dim"), dtype),
            k: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
            v: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
        ) -> R.Tuple(
            R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
            R.Tensor(("seq_len", "num_heads", "head_dim"), dtype),
        ):
            num_heads = T.int64()
            head_dim = T.int64()
            with R.dataflow():
                o = R.call_dps_packed(
                    "fused_updated_rope_parallel_attention",
                    (q, k, v),
                    out_sinfo=R.Tensor((num_heads, head_dim), dtype),
                )
                R.output(o)
            return k, o

    ext_mod = tvm.runtime.load_static_library(
        os.path.dirname(os.path.realpath(__file__))
        + "/../../build/CMakeFiles/tvm_binding.dir/src/tvm_wrapper.cu.o",
        ["fused_updated_rope_parallel_attention"],
    )

    return FlashInfer.with_attr("external_mods", [ext_mod])
