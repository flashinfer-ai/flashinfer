template <typename InputT_, typename Config_>
struct Traits {
  using Config = Config_;
  using InputT = InputT_;
  static constexpr int BLOCK_M = Config::M;
  static constexpr int BLOCK_N = Config::Page;
  static constexpr int HEAD_DIM_K = Config::Dq;
  static constexpr int HEAD_DIM_V = Config::Dv;
  static constexpr int TILE_K = Config::Slice;
  static constexpr int NUM_K_TILES = HEAD_DIM_K / TILE_K;
  static constexpr int NUM_V_TILES = HEAD_DIM_V / TILE_K;
  static constexpr int LOOKAHEAD_K = Config::LookAhead;
  static constexpr int AUX_GUARD_SLICES = 0;
  static constexpr int NUM_PV_WARPGROUPS = Config::PvWG;
  static constexpr int NUM_SOFTMAX_WARPGROUPS = 0 ? 1 : 0;
  static constexpr int NUM_THREADS = (1 + NUM_SOFTMAX_WARPGROUPS + NUM_PV_WARPGROUPS + 0) * 128;
  static constexpr int QK_WG_REGISTERS =
      NUM_PV_WARPGROUPS == 2 ? 128 : (NUM_PV_WARPGROUPS == 3 ? 96 : 64);
  static constexpr int PV_WG_REGISTERS =
      NUM_PV_WARPGROUPS == 2 ? 192 : (NUM_PV_WARPGROUPS == 3 ? 136 : 112);
  static_assert(HEAD_DIM_K % TILE_K == 0);
  static_assert(LOOKAHEAD_K == 0 || LOOKAHEAD_K == 2 || LOOKAHEAD_K == 3 || LOOKAHEAD_K == 4 ||
                LOOKAHEAD_K == 5 || LOOKAHEAD_K == 6);
  static_assert(AUX_GUARD_SLICES == 0 || AUX_GUARD_SLICES == 1);
  static_assert(BLOCK_M == 8 || BLOCK_M == 16 || BLOCK_M == 32);
  static_assert(HEAD_DIM_V % TILE_K == 0);
  static_assert(std::is_same_v<InputT, cutlass::bfloat16_t>,
                "The first target implementation is BF16 only");
  using TiledMMAQK = decltype(make_tiled_mma(
      GMMA::ss_op_selector<InputT, InputT, float, Shape<Int<BLOCK_N>, Int<BLOCK_M>, Int<TILE_K>>,
                           GMMA::Major::K, GMMA::Major::K>(),
      Layout<Shape<_1, _1, _1>>{}));
  using TiledMMAPV = decltype(make_tiled_mma(
      GMMA::ss_op_selector<InputT, InputT, float, Shape<Int<TILE_K>, Int<BLOCK_M>, Int<BLOCK_N>>,
                           GMMA::Major::MN, GMMA::Major::K>(),
      Layout<Shape<_1, _1, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<InputT>{},
                                             Shape<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}));
  using SmemLayoutQSlice = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<InputT>{},
                                                  Shape<Int<BLOCK_M>, Int<TILE_K>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<InputT>{},
                                              Shape<Int<BLOCK_N>, Int<HEAD_DIM_K>>{}));
  using SmemLayoutKVSlice = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<InputT>{},
                                                   Shape<Int<BLOCK_N>, Int<TILE_K>>{}));
  using SmemLayoutP = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<InputT>{},
                                             Shape<Int<BLOCK_M>, Int<BLOCK_N>>{}));
  using SmemLayoutScore = decltype(make_layout(Shape<Int<BLOCK_M>, Int<BLOCK_N>>{}, GenRowMajor{}));
  using SmemLayoutV = decltype(composition(
      SmemLayoutKV{}, make_layout(Shape<Int<HEAD_DIM_V>, Int<BLOCK_N>>{}, GenRowMajor{})));
  using SmemLayoutVSlice = decltype(composition(
      SmemLayoutKVSlice{}, make_layout(Shape<Int<TILE_K>, Int<BLOCK_N>>{}, GenRowMajor{})));
  using ScoreFragmentLayout =
      decltype(layout(partition_fragment_C(TiledMMAQK{}, Shape<Int<BLOCK_N>, Int<BLOCK_M>>{})));
  using OutputSliceFragmentLayout =
      decltype(layout(partition_fragment_C(TiledMMAPV{}, Shape<Int<TILE_K>, Int<BLOCK_M>>{})));
  static constexpr int SCORE_FLOATS_PER_THREAD = cosize_v<ScoreFragmentLayout>;
  static constexpr int OUTPUT_FLOATS_PER_THREAD_PER_SLICE = cosize_v<OutputSliceFragmentLayout>;
  static constexpr int OUTPUT_SLICES_PER_PV_WG =
      (NUM_V_TILES + NUM_PV_WARPGROUPS - 1) / NUM_PV_WARPGROUPS;
  static constexpr int OUTPUT_FLOATS_PER_PV_THREAD =
      OUTPUT_FLOATS_PER_THREAD_PER_SLICE * OUTPUT_SLICES_PER_PV_WG;
  using SharedMemoryPlan = PhysicalStorage<Traits<InputT_, Config_>, Config::StorageKind>;

  static constexpr int SHARED_BYTES = sizeof(SharedMemoryPlan);
  static_assert(SHARED_BYTES < 227 * 1024,
                "M32/N64 single-buffer design must fit Hopper shared memory");
};
