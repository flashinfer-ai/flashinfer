#include "flash.h"
#include "static_switch.h"

// Ref from FlashAttention3-official Repo
// https://github.com/Dao-AILab/flash-attention/blob/bdf733be55f0b323a8cf7cc6745a81c3f43cd7f0/hopper/flash_api.cpp

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      const size_t b, const size_t seqlen_q, const size_t seqlen_k,
                      const size_t seqlen_q_rounded, const size_t seqlen_k_rounded, const size_t h,
                      const size_t h_k, const size_t d, const size_t d_rounded,
                      // device pointers
                      const at::Tensor q, const at::Tensor k, const at::Tensor v, at::Tensor out,
                      void* cu_seqlens_q_d, void* cu_seqlens_k_d, void* seqused_k, void* p_d,
                      void* softmax_lse_d, float p_dropout, float softmax_scale,
                      int window_size_left, int window_size_right,
                      bool seqlenq_ngroups_swapped = false, bool unpadded_lse = false) {
  // Reset the parameters
  params = {};

  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    if (seqlenq_ngroups_swapped) {
      params.q_batch_stride *= seqlen_q;
      params.o_batch_stride *= seqlen_q;
    }
  }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int*>(seqused_k);

  TORCH_CHECK(bool(params.cu_seqlens_q) == bool(params.cu_seqlens_k),
              "cu_seqlens_q and cu_seqlens_k must be both null or non-null");

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

  // Set the different scale values.
  constexpr float log2e = 1.44269504088896340736f;
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * log2e;
  __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
  __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
  params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  TORCH_CHECK(p_dropout < 1.f);
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
#endif

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
              "This flash attention build does not support local attention.");
#endif

  params.is_seqlens_k_cumulative = true;

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  TORCH_CHECK(d == d_rounded,
              "This flash attention build does not support headdim not being a multiple of 32.");
#endif

  params.unpadded_lse = unpadded_lse;
}

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream, bool force_split_kernel = false) {
  // HEADDIM_SWITCH(params.d, [&] {
  //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
  // });
  if (!params.is_e4m3) {
    if (params.is_bf16) {
      if (params.d == 64) {
        run_mha_fwd_<cutlass::bfloat16_t, 64>(params, stream);
      } else if (params.d == 128) {
        run_mha_fwd_<cutlass::bfloat16_t, 128>(params, stream);
      } else {
        run_mha_fwd_<cutlass::bfloat16_t, 256>(params, stream);
      }
    } else {
      if (params.d == 64) {
        run_mha_fwd_<cutlass::half_t, 64>(params, stream);
      } else if (params.d == 128) {
        run_mha_fwd_<cutlass::half_t, 128>(params, stream);
      } else {
        run_mha_fwd_<cutlass::half_t, 256>(params, stream);
      }
    }
  } else {
    if (params.d == 64) {
      run_mha_fwd_<cutlass::float_e4m3_t, 64>(params, stream);
    } else if (params.d == 128) {
      run_mha_fwd_<cutlass::float_e4m3_t, 128>(params, stream);
    } else if (params.d == 256) {
      run_mha_fwd_<cutlass::float_e4m3_t, 256>(params, stream);
    }
  }
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor& q,                    // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k,              // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v,              // batch_size x seqlen_k x num_heads_k x head_size
    c10::optional<at::Tensor>& out_,  // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale,
    c10::optional<at::Tensor>& descale_q_,  // 1
    c10::optional<at::Tensor>& descale_k_,  // 1
    c10::optional<at::Tensor>& descale_v_,  // 1
    bool is_causal) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  TORCH_CHECK(is_sm90, "FlashAttention only supports Hopper GPUs or newer.");

  auto q_dtype = q.dtype();
  // TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
  //             "FlashAttention only support fp16 and bf16 data type for now");
  // TODO: will add e4m3 later
  // TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kFloat8_e4m3fn,
  //             "FlashAttention only support fp16 and bf16 data type");
  //             "FlashAttention only support fp16 and fp8 (e4m3) data type for now");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size_og <= 256,
              "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(num_heads % num_heads_k == 0,
              "Number of heads in key/value must divide number of heads in query");

  TORCH_CHECK(head_size_og == 64 || head_size_og == 128 || head_size_og == 256,
              "Only support head size 64, 128, and 256 for now");

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

  at::Tensor q_padded, k_padded, v_padded;
  if (head_size_og % 8 != 0) {
    q_padded = torch::nn::functional::pad(
        q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    k_padded = torch::nn::functional::pad(
        k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    v_padded = torch::nn::functional::pad(
        v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  } else {
    q_padded = q;
    k_padded = k;
    v_padded = v;
  }

  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    // TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    TORCH_CHECK(q_dtype == at::ScalarType::Float8_e4m3fn ? (out.dtype() == at::kHalf)
                                                         : (out.dtype() == q_dtype),
                "Output must have the same dtype as input dtype if dtype is "
                "not fp8, or fp16 for fp8 input.");
    CHECK_DEVICE(out);
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
    if (head_size_og % 8 != 0) {
      out = torch::empty_like(q_padded);
    }
  } else {
    if (q_dtype == at::ScalarType::Float8_e4m3fn)
      out = torch::empty_like(q_padded, at::kHalf);
    else
      out = torch::empty_like(q_padded);
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  auto opts = q.options();

  auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
  at::Tensor p;

  Flash_fwd_params params;
  set_params_fprop(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded,
                   num_heads, num_heads_k, head_size, head_size_rounded, q_padded, k_padded,
                   v_padded, out,
                   /*cu_seqlens_q_d=*/nullptr,
                   /*cu_seqlens_k_d=*/nullptr,
                   /*seqused_k=*/nullptr, nullptr, softmax_lse.data_ptr(),
                   /*p_dropout=*/0.f, softmax_scale,
                   /*window_size_left=*/-1,
                   /*window_size_right=*/is_causal ? 0 : -1);

  auto tile_count_semaphore = is_causal ? torch::zeros({1}, opts.dtype(torch::kInt32))
                                        : torch::empty({1}, opts.dtype(torch::kInt32));
  params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

  if (q_dtype == at::ScalarType::Float8_e4m3fn) {
    at::Tensor descale_q, descale_k, descale_v;
    if (descale_q_.has_value() && descale_k_.has_value() && descale_k_.has_value()) {
      descale_q = descale_q_.value();
      descale_k = descale_k_.value();
      descale_v = descale_v_.value();
      CHECK_DEVICE(descale_q);
      CHECK_DEVICE(descale_k);
      CHECK_DEVICE(descale_v);
      CHECK_SHAPE(descale_q, 1);
      CHECK_SHAPE(descale_k, 1);
      CHECK_SHAPE(descale_v, 1);
    } else {
      descale_q = torch::ones({1}, opts.dtype(at::kFloat));
      descale_k = torch::ones({1}, opts.dtype(at::kFloat));
      descale_v = torch::ones({1}, opts.dtype(at::kFloat));
    }
    params.descale_q_ptr = descale_q.data_ptr<float>();
    params.descale_k_ptr = descale_k.data_ptr<float>();
    params.descale_v_ptr = descale_v.data_ptr<float>();
  } else {
    params.descale_q_ptr = nullptr;
    params.descale_k_ptr = nullptr;
    params.descale_v_ptr = nullptr;
  }

  if (seqlen_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }

  at::Tensor out_padded = out;
  if (head_size_og % 8 != 0) {
    out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    if (out_.has_value()) {
      out_.value().copy_(out);
    }
  }

  return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p};
}