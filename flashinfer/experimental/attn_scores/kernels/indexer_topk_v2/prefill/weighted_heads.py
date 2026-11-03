"""Four FP32 FMA chains, preserving the indexer head-reduction order."""
from ..packed import fma_packed_f32x2 as indexer_fma_packed_f32x2
from ..packed import add_packed_f32x2 as indexer_add_packed_f32x2
import cutlass
import cutlass.cute as cute
from .integer_relu_fast import integer_relu4, finite_integer_relu4, mixed_integer_relu4

@cute.jit
def weighted_heads(fragment, weights, slot, stage, heads:cutlass.Constexpr,
                   relu_trick:cutlass.Constexpr, prepared:cutlass.Constexpr):
    acc=cute.full((4,),0.0,dtype=cutlass.Float32)
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(heads//4):
        x=cute.make_tensor(fragment.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu_trick):
            positive=indexer_add_packed_f32x2(x,abs(x),rnd="rn")
        else:
            positive=cute.max(x,zero,propagate_nan=True)
        w=cute.make_rmem_tensor((4,),cutlass.Float32)
        for j in cutlass.range_constexpr(4):
            value=weights[(slot*heads+group*4+j,stage)]
            if cutlass.const_expr(relu_trick and not prepared):value=value*0.5
            w[j]=value
        acc=indexer_fma_packed_f32x2(positive,w.load(),acc,rnd="rn")
    return acc[0]+acc[1]+acc[2]+acc[3]

@cute.jit
def weighted_heads_cached(fragment, weights, cache, slot, local_q, stage,
                          heads:cutlass.Constexpr, cached:cutlass.Constexpr,
                          relu_trick:cutlass.Constexpr, prepared:cutlass.Constexpr):
    acc=cute.full((4,),0.0,dtype=cutlass.Float32)
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(heads//4):
        x=cute.make_tensor(fragment.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu_trick):positive=indexer_add_packed_f32x2(x,abs(x),rnd="rn")
        else:positive=cute.max(x,zero,propagate_nan=True)
        w=cute.make_rmem_tensor((4,),cutlass.Float32)
        for j in cutlass.range_constexpr(4):
            if cutlass.const_expr(group*4+j<cached):value=cache[local_q*cached+group*4+j]
            else:value=weights[(slot*heads+group*4+j,stage)]
            if cutlass.const_expr(relu_trick and not prepared):value=value*0.5
            w[j]=value
        acc=indexer_fma_packed_f32x2(positive,w.load(),acc,rnd="rn")
    return acc[0]+acc[1]+acc[2]+acc[3]

@cute.jit
def weighted_heads_partial(fragment,weights,slot,stage,offset:cutlass.Constexpr,width:cutlass.Constexpr,heads:cutlass.Constexpr,acc,relu:cutlass.Constexpr):
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(width//4):
        x=cute.make_tensor(fragment.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu):positive=indexer_add_packed_f32x2(x,abs(x),rnd="rn")
        else:positive=cute.max(x,zero,propagate_nan=True)
        w=cute.make_rmem_tensor((4,),cutlass.Float32)
        for j in cutlass.range_constexpr(4):
            value=weights[(slot*heads+offset+group*4+j,stage)]
            if cutlass.const_expr(relu):value=value*0.5
            w[j]=value
        acc=indexer_fma_packed_f32x2(positive,w.load(),acc,rnd="rn")
    return acc

@cute.jit
def weighted_partial_cached(fragment,weights,cache,slot,local_q,stage,offset:cutlass.Constexpr,width:cutlass.Constexpr,cached:cutlass.Constexpr,acc,relu:cutlass.Constexpr,prepared:cutlass.Constexpr,integer_period:cutlass.Constexpr=0,integer_width:cutlass.Constexpr=4):
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(width//4):
        x=cute.make_tensor(fragment.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu and integer_period>0 and (offset+group*4)%integer_period==0):
            positive=integer_relu4(x)
        elif cutlass.const_expr(relu):positive=indexer_add_packed_f32x2(x,abs(x),rnd="rn")
        else:positive=cute.max(x,zero,propagate_nan=True)
        w=cute.make_rmem_tensor((4,),cutlass.Float32)
        for j in cutlass.range_constexpr(4):
            if cutlass.const_expr(offset+group*4+j<cached):value=cache[local_q*cached+offset+group*4+j]
            else:value=weights[(slot*32+offset+group*4+j,stage)]
            if cutlass.const_expr(relu and not prepared):value=value*0.5
            if cutlass.const_expr(relu and integer_period>0 and (offset+group*4)%integer_period==0):value=value*2.0
            w[j]=value
        acc=indexer_fma_packed_f32x2(positive,w.load(),acc,rnd="rn")
    return acc

@cute.jit
def weighted_pair_partial_cached(fragment0,fragment1,cache,local_q,
        offset:cutlass.Constexpr,width:cutlass.Constexpr,acc0,acc1,
        relu:cutlass.Constexpr,prepared:cutlass.Constexpr,integer_period:cutlass.Constexpr=0,integer_width:cutlass.Constexpr=4):
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(width//4):
        x0=cute.make_tensor(fragment0.iterator+group*4,cute.make_layout(4)).load()
        x1=cute.make_tensor(fragment1.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu and integer_period>0 and (offset+group*4)%integer_period==0):
            positive0=mixed_integer_relu4(x0,True) if cutlass.const_expr(integer_width==2) else finite_integer_relu4(x0)
            positive1=mixed_integer_relu4(x1,True) if cutlass.const_expr(integer_width==2) else finite_integer_relu4(x1)
        elif cutlass.const_expr(relu):
            positive0=indexer_add_packed_f32x2(x0,abs(x0),rnd="rn")
            positive1=indexer_add_packed_f32x2(x1,abs(x1),rnd="rn")
        else:
            positive0=cute.max(x0,zero,propagate_nan=True)
            positive1=cute.max(x1,zero,propagate_nan=True)
        w0=cute.make_rmem_tensor((4,),cutlass.Float32)
        w1=cute.make_rmem_tensor((4,),cutlass.Float32)
        for j in cutlass.range_constexpr(4):
            v0=cache[local_q*32+offset+group*4+j]
            v1=cache[(local_q+1)*32+offset+group*4+j]
            if cutlass.const_expr(relu and not prepared):
                v0=v0*0.5;v1=v1*0.5
            w0[j]=v0;w1[j]=v1
        acc0=indexer_fma_packed_f32x2(positive0,w0.load(),acc0,rnd="rn")
        acc1=indexer_fma_packed_f32x2(positive1,w1.load(),acc1,rnd="rn")
    return acc0,acc1
