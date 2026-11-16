"""NCCL-owned symmetric arena, LSA peer mappings and GIN network transport."""
import os, ctypes as C, math, json
from pathlib import Path
import torch
import cutlass, cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64
from .gin_runtime import ROOT, select_side, validate_artifacts, device_bitcode
SIDE='none'
STATE=0;HANDLE=None;HOST=None;POOL=None;ARENA=None;CURSOR=0;CONTEXT=None
OFFSETS=();BUFFERS=[];ALLOCATOR=None
NET_ARENA=None;NET_CURSOR=0;NET_EXPECTED=[];NET_INDEX=0

def check(rc):
 if rc: raise RuntimeError(HOST.eq_error().decode())

def initialize(context, arena_bytes):
 global STATE,HANDLE,HOST,POOL,ARENA,CONTEXT,OFFSETS,SIDE,ALLOCATOR
 if CONTEXT is not None:raise RuntimeError('One NCCL memory session per process')
 CONTEXT=context;network=context.ep_transport_topology.cross_numa_peer_count>0
 SIDE='both' if network else 'none'
 rank=context.ep_rank;world=context.ep_size
 out=Path(os.environ.get('AUDIT_OUT',str(Path.home()/'.cache/megamoe/gin_logs'/str(os.getpid()))));out.mkdir(parents=True,exist_ok=True)
 hcas=os.environ.get('MEGAMOE_GIN_HCA_MAP','mlx5_3,mlx5_2,mlx5_1,mlx5_0,mlx5_7,mlx5_6,mlx5_5,mlx5_4').split(',')
 os.environ.update(NCCL_IB_HCA='='+hcas[context.local_rank]+':1',NCCL_GIN_TYPE='3',NCCL_GIN_ENABLE='1' if network else '0',NCCL_IB_MERGE_NICS='0',NCCL_GIN_GDAKI_NIC_HANDLER='2',NCCL_GIN_GDAKI_QP_DEPTH='1024',NCCL_P2P_DISABLE='0',NCCL_NVLS_ENABLE='0',NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,NET,REG',NCCL_DEBUG_FILE=str(out/f'gin_rank{rank}.log'))
 library=validate_artifacts();HOST=C.CDLL(str(ROOT/'gin_host.so'));V=C.c_void_p;I=C.c_int;U=C.c_ulonglong
 HOST.eq_set_library.argtypes=[C.c_char_p];HOST.eq_set_library.restype=I;check(HOST.eq_set_library(str(library).encode()))
 HOST.eq_error.restype=C.c_char_p;HOST.eq_version.argtypes=[C.POINTER(I)];HOST.eq_unique_id.argtypes=[V]
 HOST.eq_create.argtypes=[V,I,I,I,C.POINTER(V),C.POINTER(U),C.POINTER(I)];HOST.eq_register.argtypes=[V,U,U];HOST.eq_destroy.argtypes=[V];HOST.eq_peer.argtypes=[V,I,I,C.POINTER(U)]
 version=I();check(HOST.eq_version(C.byref(version)));assert version.value==23203
 import torch.distributed as dist
 uid=C.create_string_buffer(128)
 if rank==0:check(HOST.eq_unique_id(uid))
 x=[bytes(uid) if rank==0 else None];dist.broadcast_object_list(x,src=context.ep_global_ranks[0],group=context.ep_control_group);C.memmove(uid,x[0],128)
 handle=V();state=U();info=(I*6)();check(HOST.eq_create(uid,rank,world,int(network),C.byref(handle),C.byref(state),info))
 HANDLE=handle;STATE=state.value
 ALLOCATOR=torch.cuda.memory.CUDAPluggableAllocator(str(ROOT/'gin_host.so'),'eq_pool_allocate','eq_pool_free')
 POOL=torch.cuda.MemPool(ALLOCATOR.allocator())
 capacity=int(os.environ.get('MEGAMOE_SYMMETRIC_BYTES',str(arena_bytes)))
 if capacity<arena_bytes:raise ValueError('MEGAMOE_SYMMETRIC_BYTES is smaller than required P2P storage')
 with torch.cuda.use_mem_pool(POOL):ARENA=torch.empty((capacity,),dtype=torch.uint8,device='cuda')
 assert ARENA.data_ptr()%4096==0
 check(HOST.eq_register(HANDLE,ARENA.data_ptr(),ARENA.numel()))
 peers=[]
 for peer in range(world):
  address=U();check(HOST.eq_peer(HANDLE,0,peer,C.byref(address)))
  if not address.value and not (context.ep_transport_topology.force_ibgda_peer_mask & (1<<peer)):
   raise RuntimeError(f'Required local P2P peer {peer} is not LSA accessible')
  peers.append(address.value-ARENA.data_ptr() if address.value else 0)
 OFFSETS=tuple(peers)
 (out/f'nccl_memory_rank{rank}.json').write_text(json.dumps(dict(info=list(info),capacity=capacity,base=ARENA.data_ptr(),peer_offsets=OFFSETS,network=network,library=str(library))))

def arena_zeros(shape,dtype):
 global CURSOR
 if ARENA is None:raise RuntimeError('NCCL memory session is not initialized')
 n=math.prod(shape)*torch.empty((),dtype=dtype).element_size();size=(n+4095)//4096*4096
 start=CURSOR
 if start+size>ARENA.numel():raise MemoryError(f'NCCL symmetric arena exhausted: requested end={start+size}, capacity={ARENA.numel()}')
 CURSOR+=size
 span=ARENA[start:start+size];span.zero_();view=span[:n].view(dtype).reshape(shape)
 BUFFERS.append((ARENA,span,view));return view

def p2p_capacity(tokens, hidden, topk, experts, world, rank_local):
 # Inputs, BF16 top-k output, shared dispatch metadata, rank-combine flags.
 # Mirrors the public shared region layout; include per-region alignment slack.
 sizes=[tokens*hidden//2, tokens*hidden//16, tokens*topk*8,
        tokens*topk*4, tokens*topk*hidden*2]
 shared=experts*tokens*topk*4+experts*8+(experts//world)*8+34*world*4
 if rank_local:shared+=(2*world*tokens+world)*4
 sizes.append(shared+6*16)
 if rank_local:sizes.append((tokens+1)*world*4)
 total=sum((n+4095)//4096*4096 for n in sizes)
 return (total+2*1024**2-1)//(2*1024**2)*(2*1024**2)

def network_region_bytes(tester, local_ws_bytes, owner_slots, owner_block, owner_payload):
    """Mirror the existing 19 network allocations; preserve each logical shape."""
    t = tester
    world = t.world_size
    routes = t.problem.num_tokens_per_rank * t.problem.num_topk
    experts = t.problem.num_experts_per_rank
    blocks = (routes + owner_payload - 1) // owner_payload
    route_capacity = (blocks + experts) * owner_block
    count = owner_slots * world * owner_block * 8
    count_signal = owner_slots * world * 8
    route = owner_slots * world * route_capacity * 8
    route_signal = owner_slots * world * experts * 8
    dc = t._transport.dispatch_ring_channels
    ds = t._transport.dispatch_ring_slots
    chunk = t._kernel.token_comm.ibgda_dispatch_chunk_tokens
    act = t.my_activation[0].numel() * t.my_activation.element_size()
    scale = t.my_activation_sf[0].numel() * t.my_activation_sf.element_size()
    weight = t.my_topk_weights[0].numel() * t.my_topk_weights.element_size()
    dslot = ((chunk * (4 + act + scale + weight) + 8 + 15) // 16) * 16
    cc = t._transport.combine_ring_channels
    cs = t._transport.combine_ring_slots
    rows = t.kernel_config.combine_chunk_rows
    payload = t.problem.hidden * t.combine_output.element_size()
    cslot = ((rows * (4 + payload) + 8 + 15) // 16) * 16
    return [local_ws_bytes, count, count_signal, count_signal, count,
            route, route_signal, route_signal, route,
            world*dc*ds*dslot, world*dc*ds*dslot,
            world*dc*ds*128, world*dc*ds*128, world*128, world*128,
            world*cc*cs*cslot, world*cc*cs*cslot,
            world*cc*cs*128, world*cc*cs*128]

def prepare_network_arena(tester, local_ws_bytes, owner_slots, owner_block, owner_payload):
    global NET_ARENA, NET_CURSOR, NET_EXPECTED, NET_INDEX
    if NET_ARENA is not None:
        raise RuntimeError('Network arena already allocated')
    NET_EXPECTED = network_region_bytes(tester, local_ws_bytes, owner_slots, owner_block, owner_payload)
    needed = sum(((n + 4095) // 4096 * 4096) + 4096 for n in NET_EXPECTED)
    capacity = (needed + 2*1024**2 - 1) // (2*1024**2) * (2*1024**2)
    with torch.cuda.use_mem_pool(POOL):
        NET_ARENA = torch.empty((capacity,), dtype=torch.uint8, device='cuda')
    NET_CURSOR = 0
    NET_INDEX = 0
    check(HOST.eq_register(HANDLE, NET_ARENA.data_ptr(), capacity))

def sym_zeros(shape, dtype):
    global NET_CURSOR, NET_INDEX
    if NET_ARENA is None:
        raise RuntimeError('Network arena must be prepared before network buffers')
    n = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if NET_INDEX >= len(NET_EXPECTED) or n != NET_EXPECTED[NET_INDEX]:
        raise RuntimeError(f'Network layout differs from planned allocation {NET_INDEX}: {n}')
    size = (n + 4095) // 4096 * 4096
    start = NET_CURSOR
    if start + size + 4096 > NET_ARENA.numel():
        raise MemoryError('Network arena capacity exceeded')
    NET_CURSOR += size + 4096
    NET_INDEX += 1
    span = NET_ARENA[start:start+size]
    span.zero_()
    view = span[:n].view(dtype).reshape(shape)
    BUFFERS.append((NET_ARENA, span, view))
    return view

def peer_offsets(tensor,world):
 if len(OFFSETS)!=world:raise ValueError('EP world mismatch')
 if not ARENA.data_ptr()<=tensor.data_ptr()<ARENA.data_ptr()+CURSOR:raise ValueError('Tensor outside symmetric arena')
 return tensor.data_ptr(),OFFSETS

def configure(tester):
 select_side(tester.problem,tester.world_size) # validate public override, no split fallback
 if SIDE=='both' and NET_INDEX!=len(NET_EXPECTED):raise RuntimeError('Incomplete network allocation plan')
 tester.network_backend='gin' if SIDE=='both' else 'p2p'
 import torch.distributed as dist
 cursors=[None]*CONTEXT.ep_size;dist.all_gather_object(cursors,(CURSOR,NET_CURSOR,[v.numel()*v.element_size() for b,s,v in BUFFERS]),group=CONTEXT.ep_control_group)
 if not all(x==cursors[0] for x in cursors):raise RuntimeError(f'Non-symmetric allocation order: {cursors}')
 torch.cuda.synchronize();dist.barrier(group=CONTEXT.ep_control_group)
 out=Path(os.environ.get('AUDIT_OUT',str(Path.home()/'.cache/megamoe/gin_logs'/str(os.getpid()))));out.mkdir(parents=True,exist_ok=True)
 (out/f'gin_session_rank{tester.rank}.json').write_text(json.dumps(dict(side=SIDE,network_backend=tester.network_backend,arena_used=CURSOR,arena_capacity=ARENA.numel(),dependency='NCCL only',network_arena_capacity=NET_ARENA.numel() if NET_ARENA is not None else 0,network_arena_used=NET_CURSOR,windows=[dict(address=ARENA.data_ptr(),registered_bytes=ARENA.numel())]+([dict(address=NET_ARENA.data_ptr(),registered_bytes=NET_ARENA.numel())] if NET_ARENA is not None else []))))

def close():
 global HANDLE
 if HANDLE is not None:check(HOST.eq_destroy(HANDLE));HANDLE=None

def bind(obj,role):
 obj.gin_enabled=SIDE == 'both'
 obj.gin_state_ptr=STATE
 obj.gin_context_base={'dispatch_plan':0,'dispatch_tx':8,'dispatch_rx':16,'return_tx':24,'return_rx':32}[role]
 if obj.gin_enabled:assert STATE

put_ffi=cute.ffi(name='megamoe_gin_full_put',params_types=[Int64,Int32,Int32,Int32,Int64,Int64,Int64,Int32])
flush_ffi=cute.ffi(name='megamoe_gin_full_flush',params_types=[Int64,Int32,Int32])
lsa_put_ffi=cute.ffi(name='megamoe_lsa_put',params_types=[Int64,Int32,Int64,Int64,Int64])
fence_ffi=cute.ffi(name='megamoe_system_fence',params_types=[])
@cute.jit
def fence():
 fence_ffi()
@cute.jit
def put(obj,dst,src,peer,*,blocking:cutlass.Constexpr):
 if (Int32(obj.hybrid_force_ibgda_mask)&(Int32(1)<<peer))!=Int32(0):
  cute.arch.barrier(barrier_id=0,number_of_threads=128)
  if cute.arch.thread_idx()[0]==Int32(0):
   put_ffi(Int64(obj.gin_state_ptr),Int32(obj.gin_context_base)+cute.arch.block_idx()[0]%Int32(8),Int32(obj.gin_context_base//8*512)+cute.arch.block_idx()[0]+cute.arch.grid_dim()[0]*cute.arch.block_idx()[1],peer,dst.iterator.toint(),src.iterator.toint(),Int64(cute.size(src))*Int64(src.element_type.width//8),Int32(1 if blocking else 0))
  cute.arch.barrier(barrier_id=0,number_of_threads=128)
 else:
  lsa_put_ffi(Int64(obj.gin_state_ptr),peer,dst.iterator.toint(),src.iterator.toint(),Int64(cute.size(src))*Int64(src.element_type.width//8))
@cute.jit
def quiet(obj):
 if cutlass.const_expr(obj.gin_enabled):
  if cute.arch.thread_idx()[0]==Int32(0):flush_ffi(Int64(obj.gin_state_ptr),Int32(obj.gin_context_base)+cute.arch.block_idx()[0]%Int32(8),Int32(obj.gin_context_base//8*512)+cute.arch.block_idx()[0]+cute.arch.grid_dim()[0]*cute.arch.block_idx()[1])
 fence()
@cute.jit
def put_block(obj,dst,src,peer):put(obj,dst,src,peer,blocking=True)
@cute.jit
def put_nbi_block(obj,dst,src,peer):put(obj,dst,src,peer,blocking=False)

fused_ffi=cute.ffi(name='megamoe_gin_put_commit',params_types=[Int64,Int32,Int32,Int32,Int64,Int64,Int64,Int64,Int64])
@cute.jit
def put_fused(obj,dst,src,signal_dst,signal_src,peer):
 cute.arch.barrier(barrier_id=0,number_of_threads=128)
 if cute.arch.thread_idx()[0]==Int32(0):
  fused_ffi(Int64(obj.gin_state_ptr),Int32(obj.gin_context_base)+cute.arch.block_idx()[0]%Int32(8),Int32(obj.gin_context_base//8*512)+cute.arch.block_idx()[0]+cute.arch.grid_dim()[0]*cute.arch.block_idx()[1],peer,dst.iterator.toint(),src.iterator.toint(),Int64(cute.size(src))*Int64(src.element_type.width//8),signal_dst.iterator.toint(),signal_src.iterator.toint())
 cute.arch.barrier(barrier_id=0,number_of_threads=128)


from ..sm120_ptx_helpers import spin_wait_i32_ge_cv_raw, ldg_b32_cv_raw
from ..moe_utils import spin_wait
wait_signal_ffi=cute.ffi(name='megamoe_gin_wait_signal',params_types=[Int64,Int64,Int64,Int32])
read_signal_ffi=cute.ffi(name='megamoe_gin_read_signal',params_types=[Int64,Int64],return_type=Int64)

@cute.jit
def wait_commit32(obj,peer,addr,expected):
 if cutlass.const_expr(obj.gin_enabled):
  if (Int32(obj.hybrid_force_ibgda_mask)&(Int32(1)<<peer))!=Int32(0):
   wait_signal_ffi(Int64(obj.gin_state_ptr),Int64(addr),Int64(expected),Int32(32))
  else:spin_wait_i32_ge_cv_raw(addr,expected)
 else:spin_wait_i32_ge_cv_raw(addr,expected)

@cute.jit
def wait_commit64(obj,peer,ptr,expected):
 if cutlass.const_expr(obj.gin_enabled):
  if (Int32(obj.hybrid_force_ibgda_mask)&(Int32(1)<<peer))!=Int32(0):
   wait_signal_ffi(Int64(obj.gin_state_ptr),Int64(ptr.toint()),Int64(expected),Int32(64))
  else:spin_wait(ptr,lambda value:value>=Int64(expected),fail_sleep_cycles=500)
 else:spin_wait(ptr,lambda value:value>=Int64(expected),fail_sleep_cycles=500)

@cute.jit
def read_commit_chunk(obj,peer,addr):
 value=Int32(0)
 if cutlass.const_expr(obj.gin_enabled):
  if (Int32(obj.hybrid_force_ibgda_mask)&(Int32(1)<<peer))!=Int32(0):
   whole=read_signal_ffi(Int64(obj.gin_state_ptr),Int64(addr))
   value=Int32(whole>>Int64(32))
  else:value=ldg_b32_cv_raw(Int64(addr)+Int64(4))
 else:value=ldg_b32_cv_raw(Int64(addr)+Int64(4))
 return value
