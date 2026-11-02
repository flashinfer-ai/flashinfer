template <class T, int Kind>
struct PhysicalStorage;

template <class T>
struct PhysicalStorage<T, 0> {
  using InputT = typename T::InputT;
  using SmemLayoutQ = typename T::SmemLayoutQ;
  using SmemLayoutKV = typename T::SmemLayoutKV;
  using SmemLayoutKVSlice = typename T::SmemLayoutKVSlice;
  using SmemLayoutP = typename T::SmemLayoutP;
  static constexpr int BLOCK_M = T::BLOCK_M, BLOCK_N = T::BLOCK_N;
  static constexpr int NUM_K_TILES = T::NUM_K_TILES, NUM_V_TILES = T::NUM_V_TILES;

  cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<InputT, cosize_v<SmemLayoutKV>> kv;
  cute::array_aligned<InputT, cosize_v<SmemLayoutP> * 1> p;
  cute::array_aligned<float, BLOCK_M> row_max;
  cute::array_aligned<float, BLOCK_M> row_sum;
  cute::array_aligned<float, BLOCK_M * 1> row_scale;
  cute::array_aligned<float, 4 * BLOCK_M> reduction;
  int physical_page;
  TMABarrier q_barrier;
  TMABarrier q_slice_barriers[NUM_K_TILES];
  TMABarrier kv_barriers[NUM_K_TILES];
  TMABarrier kv_free_barriers[NUM_V_TILES];
};

template <class T>
struct PhysicalStorage<T, 1> {
  using InputT = typename T::InputT;
  using SmemLayoutQ = typename T::SmemLayoutQ;
  using SmemLayoutKV = typename T::SmemLayoutKV;
  using SmemLayoutKVSlice = typename T::SmemLayoutKVSlice;
  using SmemLayoutP = typename T::SmemLayoutP;
  static constexpr int BLOCK_M = T::BLOCK_M, BLOCK_N = T::BLOCK_N;
  static constexpr int NUM_K_TILES = T::NUM_K_TILES, NUM_V_TILES = T::NUM_V_TILES;

  cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<InputT, cosize_v<SmemLayoutKV>> kv;
  cute::array_aligned<InputT, cosize_v<SmemLayoutKVSlice>> aux_kv[4 + 0];
  cute::array_aligned<InputT, cosize_v<SmemLayoutP> * 1> p;
  cute::array_aligned<float, BLOCK_M> row_max;
  cute::array_aligned<float, BLOCK_M> row_sum;
  cute::array_aligned<float, BLOCK_M * 1> row_scale;
  cute::array_aligned<float, 4 * BLOCK_M> reduction;
  cute::array_aligned<float, BLOCK_M * BLOCK_N * 1> softmax_score;
  int physical_page;
  TMABarrier q_barrier;
  TMABarrier pv_done_barrier;
  TMABarrier kv_barriers[NUM_K_TILES];
  TMABarrier aux_kv_barriers[4 + 0];
  TMABarrier kv_free_barriers[NUM_V_TILES];
  TMABarrier softmax_score_ready;
  TMABarrier softmax_done;
};

template <class T>
struct PhysicalStorage<T, 2> {
  using InputT = typename T::InputT;
  using SmemLayoutQ = typename T::SmemLayoutQ;
  using SmemLayoutKV = typename T::SmemLayoutKV;
  using SmemLayoutKVSlice = typename T::SmemLayoutKVSlice;
  using SmemLayoutP = typename T::SmemLayoutP;
  static constexpr int BLOCK_M = T::BLOCK_M, BLOCK_N = T::BLOCK_N;
  static constexpr int NUM_K_TILES = T::NUM_K_TILES, NUM_V_TILES = T::NUM_V_TILES;

  cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<InputT, cosize_v<SmemLayoutKV>> kv;
  cute::array_aligned<InputT, cosize_v<SmemLayoutP> * 1> p;
  cute::array_aligned<float, BLOCK_M> row_max;
  cute::array_aligned<float, BLOCK_M> row_sum;
  cute::array_aligned<float, BLOCK_M * 1> row_scale;
  cute::array_aligned<float, 4 * BLOCK_M> reduction;
  cute::array_aligned<float, BLOCK_M * BLOCK_N * 1> softmax_score;
  int physical_page;
  TMABarrier q_barrier;
  TMABarrier q_slice_barriers[NUM_K_TILES];
  TMABarrier kv_barriers[NUM_K_TILES];
  TMABarrier kv_free_barriers[NUM_V_TILES];
};

template <class T>
struct PhysicalStorage<T, 3> {
  using InputT = typename T::InputT;
  using SmemLayoutQ = typename T::SmemLayoutQ;
  using SmemLayoutKV = typename T::SmemLayoutKV;
  using SmemLayoutKVSlice = typename T::SmemLayoutKVSlice;
  using SmemLayoutP = typename T::SmemLayoutP;
  static constexpr int BLOCK_M = T::BLOCK_M, BLOCK_N = T::BLOCK_N;
  static constexpr int NUM_K_TILES = T::NUM_K_TILES, NUM_V_TILES = T::NUM_V_TILES;

  cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<InputT, cosize_v<SmemLayoutKV>> kv;
  cute::array_aligned<InputT, cosize_v<SmemLayoutP> * 1> p;
  cute::array_aligned<float, BLOCK_M> row_max;
  cute::array_aligned<float, BLOCK_M> row_sum;
  cute::array_aligned<float, BLOCK_M * 1> row_scale;
  cute::array_aligned<float, 4 * BLOCK_M> reduction;
  cute::array_aligned<float, BLOCK_M * BLOCK_N * 1> softmax_score;
  int physical_page;
  TMABarrier q_barrier;
  cutlass::arch::ClusterBarrier p1_ready[4];
  cutlass::arch::ClusterBarrier p1_ack[4];
  TMABarrier q_slice_barriers[NUM_K_TILES];
  TMABarrier kv_barriers[NUM_K_TILES];
  TMABarrier kv_free_barriers[NUM_V_TILES];
};
