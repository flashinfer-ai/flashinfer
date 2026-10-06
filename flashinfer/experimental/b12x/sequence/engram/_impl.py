"""Prepared native Engram hashing and FP8 lookup bindings."""
from __future__ import annotations
import operator
from dataclasses import dataclass
import torch
from b12x._lib.scratch import ScratchBufferSpec,scratch_buffer_spec,scratch_tensor
from b12x._lib.scratch_layout import SCRATCH_ALIGN_BYTES,align_up,materialize_scratch_view
from b12x.sequence.ple_hash._contracts import _require_mutation_alias_contract, _require_tensor
from b12x.preparation import Plan
from b12x.preparation.types import require_prepared
from .geometry import Geometry

def _device(value):
 d=torch.device(value); return torch.device("cuda",torch.cuda.current_device()) if d.type=="cuda" and d.index is None else d
@dataclass(frozen=True,kw_only=True)
class Caps:
 device:torch.device|str; max_tokens:int; max_seqs:int; max_requests:int; vocab_size:int=129280; layer_id:int=1; tp_size:int=8; tp_rank:int=0
 def __post_init__(self):
  object.__setattr__(self,"device",_device(self.device))
  if self.device.type!="cuda" or any(type(getattr(self,n)) is not int or getattr(self,n)<=0 for n in("max_tokens","max_seqs","max_requests","vocab_size","tp_size")): raise ValueError("Engram requires positive CUDA capacities")
  if not 0<=self.tp_rank<self.tp_size or self.max_tokens>=2**31 or self.max_seqs>=2**31: raise ValueError("invalid Engram rank or int32 capacity")
@dataclass(frozen=True)
class _State:
 caps:Caps; geometry:Geometry|None; token_map:torch.Tensor|None; multipliers:torch.Tensor|None; primes:torch.Tensor|None; offsets:torch.Tensor|None; pad_id:int; table_rows:int; shard_start:int; shard_end:int; shard_rows:int; operation:str; compact_rows:bool; _scratch_specs:tuple[ScratchBufferSpec,...]; programs:tuple; resident_scales:bool=False
 def scratch_specs(self): return self._scratch_specs
 @property
 def weight_shape(self): return (self.shard_rows,256)
 @property
 def scale_shape(self): return (self.shard_rows,8)
 def run(self, binding, prepared):
  if self.operation != "hash": raise ValueError("lookup plan cannot run Engram hash")
  compress,requests,hashes=self.programs
  if prepared:
   compress[(prepared,1,1)](binding.token_ids,binding.token_mask,self.token_map,binding.num_tokens,binding.compressed,self.caps.vocab_size)
   requests[(prepared,1,1)](binding.query_start_loc,binding.num_seqs,binding.num_tokens,binding.request_ids,self.caps.max_tokens)
   hashes[(prepared,24,1)](binding.compressed,binding.query_start_loc,binding.request_slots,binding.committed_history,binding.num_tokens,binding.request_ids,self.multipliers,self.primes,self.offsets,binding.hash_ids,self.pad_id)
  return binding.hash_ids
 def run_lookup(self, binding, prepared, *, clear_tail):
  if self.operation != "lookup": raise ValueError("hash plan cannot run Engram lookup")
  if prepared: self.programs[0][(prepared,24,1)](binding.weight,binding.scale_bytes,binding.hash_ids,binding.num_tokens,binding.out,prepared)
  if clear_tail and prepared<self.caps.max_tokens: binding.out[prepared:].zero_()
  return binding.out
@dataclass(frozen=True)
class Binding:
 _state:_State; token_ids:torch.Tensor; token_mask:torch.Tensor; query_start_loc:torch.Tensor; request_slots:torch.Tensor; committed_history:torch.Tensor; num_seqs:torch.Tensor; num_tokens:torch.Tensor; hash_ids:torch.Tensor; compressed:torch.Tensor; request_ids:torch.Tensor; scratch:torch.Tensor; plan:Plan
@dataclass(frozen=True)
class LookupBinding:
 _state:_State; weight:torch.Tensor; scale_bytes:torch.Tensor; hash_ids:torch.Tensor; num_tokens:torch.Tensor; out:torch.Tensor; plan:Plan; disk_table:object|None=None

def _need(n,x,shape,dtype,device):
 _require_tensor(n,x,shape=shape,dtype=dtype,device=device)
def _bind_state(p: _State, *, scratch, token_ids, token_mask, query_start_loc,
                request_slots, committed_history, num_seqs, num_tokens, hash_ids,
                plan: Plan | None = None):
 c=p.caps
 tensors=(("token_ids",token_ids,(c.max_tokens,),torch.int64),("token_mask",token_mask,(c.max_tokens,),torch.bool),("query_start_loc",query_start_loc,(c.max_seqs+1,),torch.int32),("request_slots",request_slots,(c.max_seqs,),torch.int32),("committed_history",committed_history,(c.max_requests,3),torch.int64),("num_seqs",num_seqs,(1,),torch.int32),("num_tokens",num_tokens,(1,),torch.int32),("hash_ids",hash_ids,(c.max_tokens,24),torch.int64))
 for n,x,shape,dtype in tensors: _need(n,x,shape,dtype,c.device)
 storage=scratch_tensor(scratch,p.scratch_specs(),owner="Engram"); compressed,_=materialize_scratch_view(storage,offset_bytes=0,shape=(c.max_tokens,),dtype=torch.int64); req=align_up(c.max_tokens*8,SCRATCH_ALIGN_BYTES); request_ids,_=materialize_scratch_view(storage,offset_bytes=req,shape=(c.max_tokens,),dtype=torch.int32)
 _require_mutation_alias_contract(mutable=(("scratch",storage),("hash_ids",hash_ids)),read_only=tuple((name,value) for name,value,*_ in tensors if name!="hash_ids")+(("token_map",p.token_map),("multipliers",p.multipliers),("primes",p.primes),("offsets",p.offsets)))
 return Binding(p,token_ids,token_mask,query_start_loc,request_slots,committed_history,num_seqs,num_tokens,hash_ids,compressed,request_ids,storage,plan)
def bind(plan: Plan, **kwargs):
 p=require_prepared(plan,"sequence.engram")
 if not isinstance(p,_State) or p.operation!="hash": raise TypeError("plan does not own hash Engram")
 return _bind_state(p, plan=plan, **kwargs)

def _bind_lookup_state(p: _State, *, weight=None, scales=None, hash_ids, num_tokens, out,
                       disk_table=None, plan: Plan | None = None):
 c=p.caps
 if disk_table is not None:
  from ._disk import DiskTable
  if not isinstance(disk_table,DiskTable) or disk_table.state is not p: raise ValueError("disk table must own this prepared Engram state")
  if weight is not None or scales is not None: raise ValueError("disk lookup owns its staged weight and scale planes")
  if not p.compact_rows or p.resident_scales != disk_table.resident_scales: raise ValueError("disk layouts differ from preparation")
  disk_table.require_complete(); weight,scales=disk_table.weight,disk_table.scale_bytes; weight_shape,scale_shape=(c.max_tokens*24,256),(c.max_tokens*24,8)
  if disk_table.resident_scales: scale_shape=p.scale_shape
 else:
  if p.compact_rows: raise ValueError("compact lookup requires a disk table")
  if weight is None or scales is None: raise ValueError("resident lookup requires weight and scales")
  weight_shape,scale_shape=p.weight_shape,p.scale_shape
 if scales.dtype not in (torch.uint8,torch.float8_e8m0fnu): raise TypeError("scales must be E8M0 bytes or float8_e8m0fnu")
 scale=scales.view(torch.uint8)
 _need("weight",weight,weight_shape,torch.float8_e4m3fn,c.device); _need("scales",scale,scale_shape,torch.uint8,c.device); _need("hash_ids",hash_ids,(c.max_tokens,24),torch.int64,c.device); _need("num_tokens",num_tokens,(1,),torch.int32,c.device); _need("out",out,(c.max_tokens,6144),torch.bfloat16,c.device)
 _require_mutation_alias_contract(mutable=(("out",out),),read_only=(("weight",weight),("scales",scales),("hash_ids",hash_ids),("num_tokens",num_tokens)))
 if disk_table is not None: disk_table.freeze()
 return LookupBinding(p,weight,scale,hash_ids,num_tokens,out,plan,disk_table)
def bind_lookup(plan: Plan, **kwargs):
 p=require_prepared(plan,"sequence.engram")
 if not isinstance(p,_State) or p.operation!="lookup": raise TypeError("plan does not own lookup Engram")
 return _bind_lookup_state(p, plan=plan, **kwargs)
def run(binding,token_count=None):
 if not isinstance(binding,Binding): raise TypeError("hash run requires an Engram hash binding")
 prepared=binding.token_ids.shape[0] if token_count is None else operator.index(token_count)
 if not 0<=prepared<=binding.token_ids.shape[0]: raise ValueError("token count exceeds capacity")
 from ._kernels import hash_op
 hash_op(binding.plan.handle,binding.token_ids,binding.token_mask,binding.query_start_loc,binding.request_slots,binding.committed_history,binding.num_seqs,binding.num_tokens,binding.hash_ids,binding.compressed,binding.request_ids,prepared)
 return binding.hash_ids
def run_lookup(binding,token_count=None,*,clear_tail=True):
 if not isinstance(binding,LookupBinding): raise TypeError("lookup run requires an Engram lookup binding")
 prepared=binding.hash_ids.shape[0] if token_count is None else operator.index(token_count)
 if not 0<=prepared<=binding.hash_ids.shape[0]: raise ValueError("token count exceeds capacity")
 from ._kernels import lookup_op
 if binding.disk_table is None:
  lookup_op(binding.plan.handle,binding.weight,binding.scale_bytes,binding.hash_ids,binding.num_tokens,binding.out,prepared,clear_tail)
 else:
  table=binding.disk_table
  table._require_open()
  with table._cache.transaction():
   table._cache.read_rows(binding.hash_ids,prepared*24)
   lookup_op(binding.plan.handle,binding.weight,binding.scale_bytes,binding.hash_ids,binding.num_tokens,binding.out,prepared,clear_tail)
 return binding.out
_READ_POOL=None
def run_lookups(bindings,token_counts,*,clear_tail=True):
 """Run several lookups; disk tables read their rows concurrently.

 Each disk table owns an io_uring reader and its read releases the GIL, so the
 host reads for different tables overlap instead of running back to back.
 Stream order and results match calling run_lookup for each binding in turn.
 """
 global _READ_POOL
 bindings=tuple(bindings); counts=tuple(operator.index(n) for n in token_counts)
 if len(bindings)!=len(counts): raise ValueError("one token count per lookup binding")
 for b,n in zip(bindings,counts):
  if not isinstance(b,LookupBinding): raise TypeError("lookup run requires an Engram lookup binding")
  if not 0<=n<=b.hash_ids.shape[0]: raise ValueError("token count exceeds capacity")
 disk=[i for i,b in enumerate(bindings) if b.disk_table is not None]
 if len(disk)<2: return tuple(run_lookup(b,n,clear_tail=clear_tail) for b,n in zip(bindings,counts))
 from concurrent.futures import ThreadPoolExecutor
 from contextlib import ExitStack
 from ._kernels import lookup_op
 if _READ_POOL is None: _READ_POOL=ThreadPoolExecutor(max_workers=3,thread_name_prefix="b12x-engram-read")
 with ExitStack() as stack:
  caches={}
  for i in disk:
   bindings[i].disk_table._require_open()
   caches[i]=stack.enter_context(bindings[i].disk_table._cache.transaction())
   caches[i]._stage_ids(bindings[i].hash_ids,counts[i]*24)
  futures=[_READ_POOL.submit(caches[i]._read_staged,counts[i]*24) for i in disk[1:]]
  try: caches[disk[0]]._read_staged(counts[disk[0]]*24)
  finally: errors=[f.exception() for f in futures]
  for error in errors:
   if error is not None: raise error
  for b,n in zip(bindings,counts):
   lookup_op(b.plan.handle,b.weight,b.scale_bytes,b.hash_ids,b.num_tokens,b.out,n,clear_tail)
 return tuple(b.out for b in bindings)
