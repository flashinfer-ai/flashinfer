template <typename T, typename TmaParamsT>
__global__ void __launch_bounds__(T::NUM_THREADS, 1)
    dense_swa_wgmma_roles_kernel(const typename T::InputT* q, const typename T::InputT* kv,
                                 const int* block_table, const int* cache_seqlens,
                                 typename T::InputT* out, float* lse, int num_splits,
                                 int64_t out_stride_split, int batch_size, int s_q, int num_heads,
                                 int max_pages, int window_size, float softmax_scale,
                                 int64_t q_stride_b, int64_t q_stride_s, int64_t q_stride_h,
                                 int64_t kv_stride_token, int64_t block_table_stride_b,
                                 int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
                                 __grid_constant__ const TmaParamsT tma_params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  attention_body<T>(q, kv, block_table, cache_seqlens, out, lse, num_splits, out_stride_split,
                    batch_size, s_q, num_heads, max_pages, window_size, softmax_scale, q_stride_b,
                    q_stride_s, q_stride_h, kv_stride_token, block_table_stride_b, out_stride_b,
                    out_stride_s, out_stride_h, tma_params);
#endif
}

template <typename T, typename TmaParamsT>
__global__ void __launch_bounds__(T::NUM_THREADS, 1)
    dense_swa_direct_kernel(const typename T::InputT* q, const typename T::InputT* kv,
                            const int* block_table, const int* cache_seqlens,
                            typename T::InputT* out, int batch_size, int s_q, int num_heads,
                            int max_pages, int window_size, float softmax_scale, int64_t q_stride_b,
                            int64_t q_stride_s, int64_t q_stride_h, int64_t kv_stride_token,
                            int64_t block_table_stride_b, int64_t out_stride_b,
                            int64_t out_stride_s, int64_t out_stride_h,
                            __grid_constant__ const TmaParamsT tma_params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  attention_body<T>(q, kv, block_table, cache_seqlens, out, nullptr, 1, int64_t{0}, batch_size, s_q,
                    num_heads, max_pages, window_size, softmax_scale, q_stride_b, q_stride_s,
                    q_stride_h, kv_stride_token, block_table_stride_b, out_stride_b, out_stride_s,
                    out_stride_h, tma_params);
#endif
}

template <class T, class TMA>
auto tactic_kernel_pointer() {
  if constexpr (T::Config::Backend == mla_swa::Reduce::Direct)
    return &dense_swa_direct_kernel<T, TMA>;
  else
    return &dense_swa_wgmma_roles_kernel<T, TMA>;
}
