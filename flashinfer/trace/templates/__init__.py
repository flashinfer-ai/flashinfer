# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Per-op TraceTemplate definitions for FlashInfer APIs.

How to add a new template
-------------------------
1. **Choose or create a file.**
   Group templates by op_type. Existing files:
   - ``norm.py``       — rmsnorm, fused_add_rmsnorm
   - ``sampling.py``   — top-k / top-p sampling
   - ``gemm.py``       — bf16 / fp8 GEMM
   - ``attention.py``  — gqa_paged, gqa_ragged, mla_paged, dsa_paged
   - ``gdn.py``        — gated delta-net decode
   - ``moe.py``        — mixture-of-experts
   Create a new file for a genuinely new op_type (e.g. ``conv.py``).

2. **Define the template.**  Example::

       from ..template import Const, Scalar, Tensor, TraceTemplate, Var

       my_op_trace = TraceTemplate(
           op_type="my_op",
           description="One-line description.",
           axes={
               "batch_size": Var(),           # runtime-variable
               "hidden_size": Const(),         # fixed by model config
           },
           inputs={
               # Key = JSON name = Python param name (override with param=)
               "x": Tensor(["batch_size", "hidden_size"]),
               "weight": Tensor(["hidden_size"]),
               "eps": Scalar("float32"),
           },
           outputs={
               "out": Tensor(["batch_size", "hidden_size"], dtype_from="x"),
           },
           tags=["status:verified"],
       )

   Key rules:
   - ``Var()``   → axis value is NOT baked into the generated name or JSON value.
   - ``Const()`` → axis value IS extracted from a tensor and written to JSON.
   - Axis values are extracted **automatically** from the first ``Tensor`` input
     whose ``dim_names`` list contains that axis name.
   - For tuple parameters (e.g. ``paged_kv_cache=(k, v)``), set
     ``param="paged_kv_cache"`` and ``tuple_idx=0`` / ``tuple_idx=1``.
   - For output dtype, prefer ``dtype_from="<input_param>"`` to copy from an
     input tensor, or set ``dtype="float32"`` for a fixed dtype.

3. **Attach to the API.**  In the API file::

       from .trace.templates.my_file import my_op_trace

       @flashinfer_api(trace=my_op_trace)
       def my_op(x, weight, eps=1e-6):
           ...

   The ``fi_api`` tag is derived automatically from
   ``func.__module__ + "." + func.__qualname__``.

4. **Test it.**  Add a test to ``tests/test_fi_trace.py``::

       def test_my_op_fi_trace():
           defn = flashinfer.my_module.my_op.fi_trace(x=x_tensor, weight=w_tensor)
           assert defn["op_type"] == "my_op"
           assert defn["axes"]["hidden_size"]["value"] == 4096
"""
