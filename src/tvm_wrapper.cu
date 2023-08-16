#include <thrust/device_vector.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <flashinfer.cuh>

int _fused_parallel_attention(DLTensor *q, DLTensor *k, DLTensor *v, DLTensor *o, flashinfer::RotaryMode rm)
{
    assert(q->ndim == 2);
    size_t num_heads = q->shape[0];
    size_t head_dim = q->shape[1];
    assert(k->ndim == 3);
    size_t seq_len = k->shape[0];
    assert(k->shape[1] == num_heads);
    assert(k->shape[2] == head_dim);
    assert(v->ndim == 3);
    assert(v->shape[0] == seq_len);
    assert(v->shape[1] == num_heads);
    assert(v->shape[2] == head_dim);
    assert(o->ndim == 2);
    assert(o->shape[0] == num_heads);
    assert(o->shape[1] == head_dim);

    thrust::device_vector<float> m_global(num_heads * head_dim);
    thrust::device_vector<float> d_global(num_heads * head_dim);
    thrust::device_vector<int> mutex(num_heads * head_dim);
    thrust::fill(m_global.begin(), m_global.end(), -INFINITY);
    thrust::fill(d_global.begin(), d_global.end(), 0.f);
    thrust::fill(mutex.begin(), mutex.end(), 0);

    if (tvm::runtime::DLDataType2String(q->dtype) == "float16")
    {
        flashinfer::SingleDecodeWithKVCache(
            (half *)q->data, (half *)k->data, (half *)v->data, (half *)o->data, thrust::raw_pointer_cast(m_global.data()), thrust::raw_pointer_cast(d_global.data()),
            thrust::raw_pointer_cast(mutex.data()), num_heads, seq_len, head_dim, rm);
    }
    else if (tvm::runtime::DLDataType2String(q->dtype) == "float32")
    {
        flashinfer::SingleDecodeWithKVCache(
            (float *)q->data, (float *)k->data, (float *)v->data, (float *)o->data, thrust::raw_pointer_cast(m_global.data()), thrust::raw_pointer_cast(d_global.data()),
            thrust::raw_pointer_cast(mutex.data()), num_heads, seq_len, head_dim, rm);
    }

    return 0;
}

#ifdef __cplusplus
extern "C"
{
#endif
    TVM_DLL int32_t fused_parallel_attention(TVMValue *args, int *type_code, int num_args, TVMValue *out_value, int *out_type_code)
    {
        DLTensor *arg0 = (DLTensor *)(((TVMValue *)args)[0].v_handle);
        DLTensor *arg1 = (DLTensor *)(((TVMValue *)args)[1].v_handle);
        DLTensor *arg2 = (DLTensor *)(((TVMValue *)args)[2].v_handle);
        DLTensor *ret3 = (DLTensor *)(((TVMValue *)args)[3].v_handle);
        _fused_parallel_attention(arg0, arg1, arg2, ret3, flashinfer::RotaryMode::kNone);
        return 0;
    }
    TVM_DLL int32_t fused_rope_parallel_attention(TVMValue *args, int *type_code, int num_args, TVMValue *out_value, int *out_type_code)
    {
        DLTensor *arg0 = (DLTensor *)(((TVMValue *)args)[0].v_handle);
        DLTensor *arg1 = (DLTensor *)(((TVMValue *)args)[1].v_handle);
        DLTensor *arg2 = (DLTensor *)(((TVMValue *)args)[2].v_handle);
        DLTensor *ret3 = (DLTensor *)(((TVMValue *)args)[3].v_handle);
        _fused_parallel_attention(arg0, arg1, arg2, ret3, flashinfer::RotaryMode::kApplyRotary);
        return 0;
    }
    TVM_DLL int32_t fused_updated_rope_parallel_attention(TVMValue *args, int *type_code, int num_args, TVMValue *out_value, int *out_type_code)
    {
        DLTensor *arg0 = (DLTensor *)(((TVMValue *)args)[0].v_handle);
        DLTensor *arg1 = (DLTensor *)(((TVMValue *)args)[1].v_handle);
        DLTensor *arg2 = (DLTensor *)(((TVMValue *)args)[2].v_handle);
        DLTensor *ret3 = (DLTensor *)(((TVMValue *)args)[3].v_handle);
        _fused_parallel_attention(arg0, arg1, arg2, ret3, flashinfer::RotaryMode::kApplyRotaryUpdateLastK);
        return 0;
    }
#ifdef __cplusplus
}
#endif