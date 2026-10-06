"""Prepared-only binding for native CSA1/CSA2 compression."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared
from b12x.sequence.ple_hash._contracts import _require_mutation_alias_contract


def _device(device):
    result=torch.device(device)
    return torch.device("cuda", torch.cuda.current_device()) if result.type=="cuda" and result.index is None else result

@dataclass(frozen=True, kw_only=True)
class Caps:
    device: torch.device|str
    max_tokens:int
    max_requests:int
    max_states:int
    ratio:int
    head_dim:int=512
    def __post_init__(self):
        object.__setattr__(self,"device",_device(self.device))
        if self.device.type!="cuda": raise ValueError("MLA compression requires CUDA")
        if self.ratio not in (1,2) or self.head_dim!=512: raise ValueError("CSA compression supports ratio 1/2 and head_dim 512")
        if min(self.max_tokens,self.max_requests,self.max_states)<=0: raise ValueError("compression capacities must be positive")
        if max(self.max_tokens,self.max_requests)>=2**31 or self.max_states>(2**63-1)//512: raise ValueError("compression capacity exceeds native index range")

@dataclass(frozen=True)
class _State:
    caps: Caps
    compiled: object

    def bind(self, **kwargs) -> "Binding":
        """Private binding hook used before a state is published."""
        return _bind_state(self, **kwargs)

    def run(self, binding: "Binding"):
        """Private launch hook used by mandatory preparation callbacks."""
        if binding._state is not self:
            raise ValueError("MLA compression binding belongs to another state")
        from ._cute import launch
        launch(binding)
        return binding.out, binding.emitted, binding.emitted_slots

@dataclass(frozen=True)
class Binding:
    _state:_State
    values:torch.Tensor; gates:torch.Tensor|None; weight:torch.Tensor
    query_start_loc:torch.Tensor; positions:torch.Tensor; state_ids:torch.Tensor
    destination_slots:torch.Tensor; live_counts:torch.Tensor
    pending_values:torch.Tensor|None; pending_gates:torch.Tensor|None; pending_position:torch.Tensor|None
    out:torch.Tensor; emitted:torch.Tensor; emitted_slots:torch.Tensor; _pointers:tuple
    plan:Plan|None

def _need(name,tensor,shape,dtype,device):
    if not isinstance(tensor,torch.Tensor): raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape)!=shape or tensor.dtype!=dtype or tensor.device!=device or not tensor.is_contiguous(): raise ValueError(f"invalid {name} tensor contract")

def _bind_state(state: _State, *, values, gates=None, weight, query_start_loc, positions,
                state_ids, destination_slots, live_counts, out, emitted, emitted_slots,
                pending_values=None, pending_gates=None, pending_position=None,
                plan: Plan | None = None):
    c=state.caps
    _need("values",values,(c.max_tokens,512),torch.float32 if c.ratio==2 else torch.bfloat16,c.device)
    _need("weight",weight,(512,),torch.float32,c.device); _need("query_start_loc",query_start_loc,(c.max_requests+1,),torch.int32,c.device)
    _need("positions",positions,(c.max_requests,),torch.int64,c.device); _need("state_ids",state_ids,(c.max_requests,),torch.int64,c.device)
    _need("destination_slots",destination_slots,(c.max_tokens,),torch.int64,c.device); _need("live_counts",live_counts,(2,),torch.int32,c.device)
    _need("out",out,(c.max_tokens,512),torch.bfloat16,c.device); _need("emitted",emitted,(c.max_tokens,),torch.bool,c.device); _need("emitted_slots",emitted_slots,(c.max_tokens,),torch.int64,c.device)
    if c.ratio==2:
        _need("gates",gates,(c.max_tokens,512),torch.float32,c.device); _need("pending_values",pending_values,(c.max_states,512),torch.float32,c.device); _need("pending_gates",pending_gates,(c.max_states,512),torch.float32,c.device); _need("pending_position",pending_position,(c.max_states,),torch.int64,c.device)
    else:
        if any(value is not None for value in (gates,pending_values,pending_gates,pending_position)): raise ValueError("ratio 1 does not consume gates or pending state")
        gates=pending_values=pending_gates=pending_position=None
    mutable=(("out",out),("emitted",emitted),("emitted_slots",emitted_slots))
    if c.ratio==2: mutable+=(("pending_values",pending_values),("pending_gates",pending_gates),("pending_position",pending_position))
    _require_mutation_alias_contract(
        mutable=mutable,
        read_only=(("values",values),("gates",gates if gates is not None else values),("weight",weight),("query_start_loc",query_start_loc),("positions",positions),("state_ids",state_ids),("destination_slots",destination_slots),("live_counts",live_counts)),
    )
    from ._cute import pointers
    tensors=(values,gates if gates is not None else values,weight,query_start_loc,positions,state_ids,destination_slots,live_counts,pending_values if pending_values is not None else values,pending_gates if pending_gates is not None else values,pending_position if pending_position is not None else state_ids,out,emitted,emitted_slots)
    return Binding(state,values,gates,weight,query_start_loc,positions,state_ids,destination_slots,live_counts,pending_values,pending_gates,pending_position,out,emitted,emitted_slots,pointers(tensors),plan)

def bind(plan: Plan, **kwargs):
    state=require_prepared(plan, "attention.mla_compress")
    if not isinstance(state,_State): raise TypeError("plan does not own MLA compression")
    return state.bind(plan=plan, **kwargs)

@torch.library.custom_op(
    "b12x::mla_compress_ratio1",
    mutates_args=("out", "emitted", "emitted_slots"),
)
def _run_ratio1(plan_handle: int, values: torch.Tensor, weight: torch.Tensor,
                query_start_loc: torch.Tensor, positions: torch.Tensor,
                state_ids: torch.Tensor, destination_slots: torch.Tensor,
                live_counts: torch.Tensor, out: torch.Tensor, emitted: torch.Tensor,
                emitted_slots: torch.Tensor) -> None:
    binding=bind(plan_from_handle(plan_handle), values=values, weight=weight,
                 query_start_loc=query_start_loc, positions=positions,
                 state_ids=state_ids, destination_slots=destination_slots,
                 live_counts=live_counts, out=out, emitted=emitted,
                 emitted_slots=emitted_slots)
    binding._state.run(binding)

@torch.library.register_fake("b12x::mla_compress_ratio1")
def _run_ratio1_fake(plan_handle: int, values: torch.Tensor,
                     weight: torch.Tensor, query_start_loc: torch.Tensor,
                     positions: torch.Tensor, state_ids: torch.Tensor,
                     destination_slots: torch.Tensor, live_counts: torch.Tensor,
                     out: torch.Tensor, emitted: torch.Tensor,
                     emitted_slots: torch.Tensor) -> None:
    return None

@torch.library.custom_op(
    "b12x::mla_compress_ratio2",
    mutates_args=("out", "emitted", "emitted_slots", "pending_values",
                  "pending_gates", "pending_position"),
)
def _run_ratio2(plan_handle: int, values: torch.Tensor, gates: torch.Tensor,
                weight: torch.Tensor, query_start_loc: torch.Tensor,
                positions: torch.Tensor, state_ids: torch.Tensor,
                destination_slots: torch.Tensor, live_counts: torch.Tensor,
                pending_values: torch.Tensor, pending_gates: torch.Tensor,
                pending_position: torch.Tensor, out: torch.Tensor,
                emitted: torch.Tensor, emitted_slots: torch.Tensor) -> None:
    binding=bind(plan_from_handle(plan_handle), values=values, gates=gates, weight=weight,
                 query_start_loc=query_start_loc, positions=positions,
                 state_ids=state_ids, destination_slots=destination_slots,
                 live_counts=live_counts, pending_values=pending_values,
                 pending_gates=pending_gates, pending_position=pending_position,
                 out=out, emitted=emitted, emitted_slots=emitted_slots)
    binding._state.run(binding)

@torch.library.register_fake("b12x::mla_compress_ratio2")
def _run_ratio2_fake(plan_handle: int, values: torch.Tensor,
                     gates: torch.Tensor, weight: torch.Tensor,
                     query_start_loc: torch.Tensor, positions: torch.Tensor,
                     state_ids: torch.Tensor, destination_slots: torch.Tensor,
                     live_counts: torch.Tensor, pending_values: torch.Tensor,
                     pending_gates: torch.Tensor, pending_position: torch.Tensor,
                     out: torch.Tensor, emitted: torch.Tensor,
                     emitted_slots: torch.Tensor) -> None:
    return None

def run(binding:Binding):
    if not isinstance(binding, Binding):
        raise TypeError("run requires an MLA compression Binding")
    plan=binding.plan
    if plan is None:
        raise TypeError("public MLA compression run requires a prepared Plan")
    # Validate the plan inside the opaque operator; its private members must
    # never be inspected by Dynamo while tracing this Python facade.
    if binding.gates is None:
        _run_ratio1(plan.handle, binding.values, binding.weight,
                    binding.query_start_loc, binding.positions, binding.state_ids,
                    binding.destination_slots, binding.live_counts, binding.out,
                    binding.emitted, binding.emitted_slots)
    else:
        if binding.gates is None or binding.pending_values is None or binding.pending_gates is None or binding.pending_position is None:
            raise ValueError("ratio 2 MLA compression binding requires gate and pending-state tensors")
        _run_ratio2(plan.handle, binding.values, binding.gates, binding.weight,
                    binding.query_start_loc, binding.positions, binding.state_ids,
                    binding.destination_slots, binding.live_counts,
                    binding.pending_values, binding.pending_gates,
                    binding.pending_position, binding.out, binding.emitted,
                    binding.emitted_slots)
    return binding.out,binding.emitted,binding.emitted_slots
