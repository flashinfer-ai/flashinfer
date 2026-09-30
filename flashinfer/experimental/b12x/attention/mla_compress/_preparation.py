"""Fixed native compilation and materialization for MLA compression."""
from __future__ import annotations

from b12x._lib.program_cache import program_cache
from ._tuning import MlaCompressQuery, TUNING
from b12x._lib.compile_pool import CompileJob
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._impl import Caps,_State


@program_cache(scope="preparation")
def compile_mla_compress(query_payload,device_ordinal):
 from ._cute import compile_compress
 q=MlaCompressQuery(**dict(query_payload))
 return compile_compress(q.ratio,q.max_tokens,q.max_requests,q.max_states,device_ordinal)

def make_plan(caps:Caps,*,invocation=FrozenMapping(),override=None):
 invocation=FrozenMapping(invocation)
 if invocation: raise ValueError("MLA compression has no invocation metadata")
 query=MlaCompressQuery(ratio=caps.ratio,max_tokens=caps.max_tokens,max_requests=caps.max_requests,max_states=caps.max_states,head_dim=caps.head_dim)
 def jobs(config,device):
  return (CompileJob.create("b12x.attention.mla_compress._preparation:compile_mla_compress",TUNING.encode_query(query),device.ordinal),)
 def memory(config,device):
  del config,device
  return MemoryRequirements()
 def materialize(selection,device):
  return _State(caps,compile_mla_compress(TUNING.encode_query(query),device.ordinal))
 return Plan(contract=TUNING,query=query,invocation=invocation,override=override,_compile_jobs=jobs,_memory_requirements=memory,_materialize=materialize,_device=caps.device)
