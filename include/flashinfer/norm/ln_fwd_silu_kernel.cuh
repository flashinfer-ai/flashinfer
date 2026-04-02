#pragma once
// Fused RmsNorm+SiLU forward kernel (SM100 optimized).
//
// IMPORTANT: Include ln_silu_headers.cuh and the config .inc BEFORE this file.
// The config must define Ktraits, DESIRED_OCCUPANCY, and all constexpr flags.

constexpr int mxfp8_block_size = 32;
constexpr int nvfp4_block_size = 16;

template <typename _Traits>
struct LnFwdShared {
  using Traits = _Traits;
  static constexpr int32_t SMEM_STATS_ELEMENTS =
      ((Traits::Stats::SMEM_BYTES > 0) ? Traits::Stats::SMEM_BYTES : 1);
  static constexpr int32_t SMEM_BAR_ELEMENTS =
      ((Traits::USE_CLUSTER && Traits::CTAS_PER_ROW > 1)
           ? (Traits::WARPS_M + 1 + Traits::WARPS_M * Traits::CTAS_PER_ROW)
           : 1);
  static constexpr int32_t SMEM_MXFP8_ELEMENTS =
      ((isBlockScale_1D2X2X && !useBlockScaleColwiseKernel)
           ? ((mxfp8_block_size * Traits::NUM_ELTS + (Traits::NUM_ELTS - 1)) *
              (mxfp8_block_size + 1))
           : 1);
  static constexpr int32_t GAMMA_ELEMENTS =
      ((Traits::hasGamma && Traits::USE_GAMMA_SMEM)
           ? (Traits::BATCH_SIZE * Traits::LDGS * Traits::THREADS_PER_ROW * Traits::NUM_ELTS)
           : 1);
  static constexpr int32_t BETA_ELEMENTS =
      ((Traits::hasBeta && Traits::USE_GAMMA_SMEM)
           ? (Traits::BATCH_SIZE * Traits::LDGS * Traits::THREADS_PER_ROW * Traits::NUM_ELTS)
           : 1);

  __align__(16) char smem_stats[SMEM_STATS_ELEMENTS];
  __align__(16) uint64_t smem_bar[SMEM_BAR_ELEMENTS];
  __align__(16) typename Traits::weight_t smem_gamma[GAMMA_ELEMENTS];
  __align__(16) typename Traits::weight_t smem_beta[BETA_ELEMENTS];
  __align__(16) float smem_mxfp8[SMEM_MXFP8_ELEMENTS];
};

__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA, DESIRED_OCCUPANCY) void ln_fwd_kernel(
    PersistentLnFwdParams params,
    reduced_divisor
        divisor) {  // divisor is div_batch if it is batch-first case, else it is div_seqLen
  enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
  enum { WARPS_N = Ktraits::WARPS_N };
  enum { WARPS_M = Ktraits::WARPS_M };
  enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
  enum { VEC_COLS_PER_LDG = Ktraits::VEC_COLS_PER_LDG };
  enum { VEC_COLS = Ktraits::VEC_COLS };
  enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
  enum { LDGS = Ktraits::LDGS };
  enum { NUM_ELTS = Ktraits::NUM_ELTS };
  enum { THREADS_PER_WARP = Ktraits::THREADS_PER_WARP };
  enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };
  enum { COLS = Ktraits::COLS };
  enum { COLS_PER_LDG = Ktraits::VEC_COLS_PER_LDG * Ktraits::NUM_ELTS };
  enum { COLS_PER_LDG_PER_CTA = COLS_PER_LDG / Ktraits::CTAS_PER_ROW };
  enum { VEC_COLS_PER_LDG_PER_CTA = VEC_COLS_PER_LDG / Ktraits::CTAS_PER_ROW };
  enum { USE_GAMMA_SMEM = Ktraits::USE_GAMMA_SMEM };
  enum { BATCH_SIZE = Ktraits::BATCH_SIZE };
  enum { isAdaLN = Ktraits::isAdaLN };
  enum { isBatchFirst = Ktraits::isBatchFirst };

  using output_t = typename Ktraits::output_t;
  using weight_t = typename Ktraits::weight_t;
  using index_t = typename Ktraits::index_t;
  using compute_t = typename Ktraits::compute_t;
  using norm_output_t = typename Ktraits::norm_output_t;
  using Ivec = typename Ktraits::Ivec;
  using Ovec = typename Ktraits::Ovec;
  using Wvec = typename Ktraits::Wvec;
  using Cvec = typename Ktraits::Cvec;
  using NormOvec = typename Ktraits::NormOvec;

  using Stats = typename Ktraits::Stats;
  using stats_t = typename Stats::stats_t;

#ifdef USE_STATIC_SMEM_VALUE
  __shared__ __align__(16) char smem_base_[USE_STATIC_SMEM_VALUE];
#else
  extern __shared__ char smem_base_[];
#endif

  LnFwdShared<Ktraits>* shared = reinterpret_cast<LnFwdShared<Ktraits>*>(smem_base_);

  uint64_t* smemBar = shared->smem_bar;
#if LN_USE_CLUSTER
  if (CTAS_PER_ROW > 1) {
#if (__CUDA_ARCH__ >= 900) && (CUDART_VERSION >= 12080)
    // Init the empty bars for each warp
    if (threadIdx.x < WARPS_M) {
      cuda::ptx::mbarrier_init(&smemBar[threadIdx.x], CTAS_PER_ROW * WARPS_N * THREADS_PER_WARP);
    }
    // Init the full bar (shared by the CTA)
    if (threadIdx.x == 0) {
      cuda::ptx::mbarrier_init(&smemBar[WARPS_M], 1);
      cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
    }
    cuda::ptx::barrier_cluster_arrive(cuda::ptx::sem_relaxed);
    cuda::ptx::barrier_cluster_wait();
#else
    static_assert(true, "Cluster enabled on host side but not available on device");
#endif  // (__CUDA_ARCH__ >= 900) && (CUDART_VERSION >= 12080)
  }
#endif  // LN_USE_CLUSTER

  const index_t tidx = threadIdx.x;
  const index_t bidn = blockIdx.x % CTAS_PER_ROW;
  const index_t bidm = blockIdx.x / CTAS_PER_ROW;
  const index_t lane = tidx % THREADS_PER_WARP;
  const index_t warp = tidx / THREADS_PER_WARP;
  const index_t warp_m = warp / WARPS_N;
  const index_t warp_n = warp % WARPS_N;

  const index_t r = bidm * ROWS_PER_CTA + warp_m;

  const index_t col_in_tile = warp_n * THREADS_PER_WARP + lane;
  const index_t c = bidn * THREADS_PER_ROW + col_in_tile;

  Stats stats(params, bidm, bidn, warp_m, warp_n, tidx, lane, shared->smem_stats, smemBar);

  // Unused when USE_GAMMA_SMEM is true and will be optimized out
  [[maybe_unused]] Wvec gamma_regs[BATCH_SIZE][LDGS];
  [[maybe_unused]] Wvec beta_regs[BATCH_SIZE][LDGS];
  weight_t *gamma_smem = nullptr, *beta_smem = nullptr;
  if constexpr (USE_GAMMA_SMEM) {
    static constexpr int32_t SMEM_BYTES_GAMMA = THREADS_PER_ROW * BATCH_SIZE * LDGS * sizeof(Wvec);
    if constexpr (Ktraits::hasGamma) {
      gamma_smem = shared->smem_gamma;
    }
    if constexpr (Ktraits::hasBeta) {
      beta_smem = shared->smem_beta;
    }
  }

  // If we are mxfp8 output type, we need shared memory for amax calculations across warps
  // 2d1x1x (not yet implemented) requires 1 (1 for a 32x32 block)
  // 1d1x1x requires 0 (since it reduces over a row which can be done with warp reduce)
  // 1d2x2x requires 32x(32+1)xNUM_ELTS, more details as follows:
  constexpr int block_scale_size = isFP4Out ? nvfp4_block_size : mxfp8_block_size;
  BlockScaleRowHelper<Cvec, Ovec, block_scale_size> rowwise_scale_helper{};
  BlockScaleColHelper<Cvec, Ovec, mxfp8_block_size> colwise_scale_helper{shared->smem_mxfp8};
  compute_t* mu_ptr = static_cast<compute_t*>(params.mu);
  compute_t* rs_ptr = static_cast<compute_t*>(params.rs);

  constexpr compute_t rn = 1.f / compute_t(Ktraits::COLS);

  index_t idx = c;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n");
#endif

  // Load gamma and beta into shared memory or registers
#pragma unroll
  for (int b = 0; b < BATCH_SIZE; b++) {
// CL-14115: The unroll factor 128 for LDGS was chosen based on the compilation/perf results for
// APEX LN_fwd engines
#pragma unroll 128
    for (int it = 0; it < LDGS; it++) {
      if constexpr (USE_GAMMA_SMEM) {
        if (warp_m == 0) {
          const index_t cur_gamma_smem_base_idx = (b * LDGS + it) * THREADS_PER_ROW * NUM_ELTS +
                                                  warp_n * THREADS_PER_WARP * NUM_ELTS + lane;
          if constexpr (Ktraits::hasGamma) {
            Wvec cur_gamma_vec;
            cur_gamma_vec.load_from(params.gamma, idx);
#pragma unroll
            for (int jt = 0; jt < NUM_ELTS; jt++) {
              const index_t cur_gamma_smem_idx = cur_gamma_smem_base_idx + jt * THREADS_PER_WARP;
              gamma_smem[cur_gamma_smem_idx] = cur_gamma_vec.data.elt[jt];
            }
          }
          if constexpr (Ktraits::hasBeta) {
            Wvec cur_beta_vec;
            cur_beta_vec.load_from(params.beta, idx);
#pragma unroll
            for (int jt = 0; jt < NUM_ELTS; jt++) {
              const index_t cur_beta_smem_idx = cur_gamma_smem_base_idx + jt * THREADS_PER_WARP;
              beta_smem[cur_beta_smem_idx] = cur_beta_vec.data.elt[jt];
            }
          }
        }
      } else if constexpr (!GAMMA_ON_DEMAND) {
        if constexpr (Ktraits::hasGamma) {
          gamma_regs[b][it].load_from(params.gamma, idx);
        }
        if constexpr (Ktraits::hasBeta) {
          beta_regs[b][it].load_from(params.beta, idx);
        }
      }
      idx += VEC_COLS_PER_LDG;
    }
  }

  if constexpr (USE_GAMMA_SMEM) {
    __syncthreads();
  }

  // Initialize scale and bias for FP8 output
  compute_t scale = 1.f;
  if constexpr (isFP8Out) {
    scale = __ldg(params.scale);
  }
  compute_t amax = 0;

  index_t remaining_rows = params.rows - bidm * ROWS_PER_CTA;
  int row_increment_step = params.ctas_per_col * ROWS_PER_CTA;
  int batch_idx = 0, remainder = 0;
  int batch_increment_step = 0, step_remainder = 0;
  if constexpr (isAdaLN) {
    if constexpr (isBatchFirst) {
      divisor.divmod(r, batch_idx, remainder);  //  row =  batch_idx * seqLen + remainder;
                                                //  (remainder < seqLen)  batch_idx = r/seqLen
      divisor.divmod(row_increment_step, batch_increment_step,
                     step_remainder);  // row_increment_step =
                                       // batch_increment_step * seqLen +
                                       // step_remainder (remainder <
                                       // seqLen)
    } else {
      batch_idx = divisor.mod(r);  // batch_idx = row % BATCH_SIZE;
      batch_increment_step = divisor.mod(
          row_increment_step);  // batch_increment_step = row_increment_step % BATCH_SIZE;
    }
  }

  for (int row = r; row < params.rows;
       row += row_increment_step, batch_idx += batch_increment_step, remainder += step_remainder) {
    index_t idx = static_cast<index_t>(row) * VEC_COLS + c;

    // Load x and convert to compute type per row per thread
    compute_t xf[LDGS * NUM_ELTS];
#pragma unroll 128
    for (int it = 0; it < LDGS; it++) {
      Ivec x_it{};
      x_it.load_from(params.x, idx);
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t x_ij = compute_t(x_it.data.elt[jt]);
        xf[it * NUM_ELTS + jt] = x_ij;
      }
      idx += VEC_COLS_PER_LDG;
    }

    // Compute mean and variance per row per thread
    // How many rows current CTA will handle for this iteration
    int rows_per_cta = remaining_rows >= ROWS_PER_CTA ? ROWS_PER_CTA : remaining_rows;
    stats_t s = stats.compute<Ktraits::isRMSNorm, LDGS, NUM_ELTS>(xf, rn, rows_per_cta);
    remaining_rows -= params.ctas_per_col * ROWS_PER_CTA;  // for next iteration
    compute_t mu = Get<0>::of<stats_t, compute_t>(s);
    compute_t m2 = Get<1>::of<stats_t, compute_t>(s);
    if constexpr (!Ktraits::isRMSNorm) {
      if (bidn == 0 && warp_n == 0 && lane == 0) {
        mu_ptr[row] = mu;
      }
    }
    compute_t rs = rsqrtf(rn * m2 + params.epsilon);

    if (bidn == 0 && warp_n == 0 && lane == 0) {
      rs_ptr[row] = rs;
    }

    idx = row * VEC_COLS + c;

    if constexpr (isAdaLN) {
      if constexpr (isBatchFirst) {
        if (remainder >= params.seqLen) {
          batch_idx += 1;
          remainder -= params.seqLen;
        }
      } else {
        if (batch_idx >= BATCH_SIZE) {
          batch_idx -= BATCH_SIZE;
        }
      }
    }
    index_t gamma_idx = c + (batch_idx * LDGS) * VEC_COLS_PER_LDG;
#pragma unroll 128
    for (int it = 0; it < LDGS; it++) {
      Cvec z_math;
      [[maybe_unused]] Wvec g_wt;
      [[maybe_unused]] Wvec b_wt;
      if constexpr (GAMMA_ON_DEMAND && Ktraits::hasGamma && !USE_GAMMA_SMEM) {
        g_wt.load_from(params.gamma, gamma_idx);

        if constexpr (Ktraits::hasBeta) {
          b_wt.load_from(params.beta, gamma_idx);
        }
      }
#pragma unroll
      // Compute output per ldg per row per thread
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t y_ij = rs * (xf[it * NUM_ELTS + jt] - mu);

        if constexpr (Ktraits::hasGamma) {
          weight_t g_ij_wt{};
          const int32_t cur_gamma_smem_base_idx =
              (batch_idx * LDGS + it) * THREADS_PER_ROW * NUM_ELTS +
              warp_n * THREADS_PER_WARP * NUM_ELTS + lane;
          const int32_t cur_gamma_smem_idx = cur_gamma_smem_base_idx + jt * THREADS_PER_WARP;
          if constexpr (USE_GAMMA_SMEM) {
            g_ij_wt = gamma_smem[cur_gamma_smem_idx];
          } else if constexpr (GAMMA_ON_DEMAND) {
            g_ij_wt = g_wt.data.elt[jt];
          } else {
            g_ij_wt = gamma_regs[batch_idx][it].data.elt[jt];
          }
          compute_t g_ij = static_cast<compute_t>(g_ij_wt);
          if constexpr (isZeroCenteredGamma) {
            if constexpr (isZeroCenteredGammaCastBeforeAdd) {
              g_ij = static_cast<compute_t>(g_ij_wt) + static_cast<compute_t>(1.f);
            } else {
              g_ij = static_cast<compute_t>(g_ij_wt + static_cast<weight_t>(1.f));
            }
          }
          if constexpr (Ktraits::hasBeta) {
            compute_t b_ij{};
            const int32_t cur_beta_smem_idx = cur_gamma_smem_base_idx + jt * THREADS_PER_WARP;
            if constexpr (USE_GAMMA_SMEM) {
              b_ij = beta_smem[cur_beta_smem_idx];
            } else if constexpr (GAMMA_ON_DEMAND) {
              b_ij = static_cast<compute_t>(b_wt.data.elt[jt]);
            } else {
              b_ij = beta_regs[batch_idx][it].data.elt[jt];
            }
            y_ij = g_ij * y_ij + b_ij;
          } else {
            y_ij = g_ij * y_ij;
          }
        }

        // SiLU activation: y = y * sigmoid(y) = y / (1 + exp(-y))
        // Applied after norm + gamma [+ beta], before FP8/block-scale quantization.
        y_ij = __fdividef(y_ij, 1.0f + __expf(-y_ij));

        if constexpr (isFP8Out) {
          if (hasAmax) {
            __builtin_assume(amax >= 0);
            amax = fmaxf(amax, fabsf(y_ij));
          }
          y_ij *= scale;
        }
        z_math.data.elt[jt] = y_ij;
      }  // NUM_ELTS

      if constexpr (isBlockScaleOut) {
        static_assert(!isBlockScaleOut ||
                      (Ktraits::COLS % mxfp8_block_size == 0));  // ensure cols divisable by 32

        index_t sf_row_idx = idx / mxfp8_block_size;
        [[maybe_unused]] NormOvec z_intermediate;
        if constexpr (std::is_same<compute_t, norm_output_t>::value) {
          rowwise_scale_helper.blockQuantizeStore(z_math, params.scale_row, sf_row_idx, params.z,
                                                  idx);
        } else {
          z_math.to(z_intermediate);
          rowwise_scale_helper.blockQuantizeStore(z_intermediate, params.scale_row, sf_row_idx,
                                                  params.z, idx);
        }
        if constexpr (isBlockScale_1D2X2X) {
          if constexpr (useBlockScaleColwiseKernel) {
            // Store the temporary z_math values in workspace and launch a separate kernel to
            // compute the colwise scaling results
            if constexpr (std::is_same<compute_t, norm_output_t>::value) {
              z_math.store_to(params.z_math, idx);
            } else {
              z_intermediate.store_to(params.z_math, idx);
            }
          } else {
            if constexpr (std::is_same<compute_t, norm_output_t>::value) {
              colwise_scale_helper.initTile(z_math, THREADS_PER_ROW * WARPS_M);
              //                            static_assert(!std::is_same<compute_t,
              //                            norm_output_t>::value);
            } else {
              colwise_scale_helper.initTile(z_intermediate, THREADS_PER_ROW * WARPS_M);
            }
            index_t sf_col_row_idx = 0;
            index_t sf_col_col_idx = 0;
            index_t sf_col_row_width = 0;
            index_t z_col_idx = 0;
            index_t z_row_offset = row - row % mxfp8_block_size;
            if constexpr (!isBlockScale_1D2X2X_Transpose) {
              sf_col_row_idx = row / mxfp8_block_size;
              sf_col_col_idx = it * VEC_COLS_PER_LDG + bidn * VEC_COLS_PER_LDG_PER_CTA + warp;
              sf_col_row_width = VEC_COLS;
              z_col_idx = (z_row_offset + lane) * VEC_COLS + sf_col_col_idx;
            } else {
              constexpr index_t group_size = mxfp8_block_size / NUM_ELTS;
              sf_col_row_idx = it * COLS_PER_LDG + bidn * COLS_PER_LDG_PER_CTA + warp * NUM_ELTS;
              sf_col_col_idx = row / mxfp8_block_size;
              sf_col_row_width = params.rows / mxfp8_block_size;
              z_col_idx = (sf_col_row_idx + lane / group_size) * params.rows / NUM_ELTS +
                          z_row_offset / NUM_ELTS + (lane % group_size);
            }
            colwise_scale_helper.blockQuantizeStore<isBlockScale_1D2X2X_Transpose>(
                params.scale_col, sf_col_row_idx, sf_col_col_idx, sf_col_row_width, params.z_col,
                z_col_idx, THREADS_PER_ROW * WARPS_M);
          }
        }
      } else {
        Ovec z;
        z_math.to(z);
        z.store_to(params.z, idx);
      }
      idx += VEC_COLS_PER_LDG;
      gamma_idx += VEC_COLS_PER_LDG;
    }  // LDGS
  }  // grid stride loop

  // Write scale_inv before launch_dependents - consumer needs it to dequantize FP8 output
  if constexpr (isFP8Out) {
    if (hasScaleInv && blockIdx.x == 0 && threadIdx.x == 0) {
      *reinterpret_cast<compute_t*>(params.scale_inv) = __fdividef(1.f, scale);
    }
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n");
#endif

  // amax can be after launch_dependents - only needed for delayed scaling (next iteration)
  if constexpr (isFP8Out) {
    if constexpr (hasAmax) {
      amax = reduce_max<WARPS_M * WARPS_N>(amax, warp, threadIdx.x);
      if (threadIdx.x == 0) {
        atomicMaxFloat(reinterpret_cast<compute_t*>(params.amax), amax);
      }
    }
  }
}
