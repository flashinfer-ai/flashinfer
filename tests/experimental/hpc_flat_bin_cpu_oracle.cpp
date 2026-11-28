// Diagnostic harness using the unmodified HPC-Ops CPU reference function.
// Tencent/hpc-ops 1cd332980ed46bd0172091c1c35d55338fcae47a.
// Original function copyright: hpc-ops authors; see source repository license.
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
// MIT license: include/flashinfer/attention/HPC_OPS_LICENSE.txt.
// Host-only mirror of the original 48-byte task record.
namespace hpc::attention::decode::dynamic {
struct alignas(16) TaskScheduleInfo {
  int ihead_kv, ibatch, ichunk, iseq_start;
  int num_seqkv, num_seqkvcache, num_tile_kv, num_tile_full;
  int is_casual_chunk;
  int pad[3];
};
}  // namespace hpc::attention::decode::dynamic
namespace hpc {
namespace attention {
namespace decode {
std::pair<std::vector<dynamic::TaskScheduleInfo>, std::vector<int>>
assign_attention_decode_task_sync(const int* num_seq_kvcache, int num_total_ctas, int num_batch,
                                  int num_head_kv, int num_seq_q, int tilen, bool new_kv_included,
                                  int min_process_len) {
  std::vector<int> num_seqkvs(num_batch);
  std::vector<int> num_tiles(num_batch);

  int total_tiles_per_head = 0;
  for (int ibatch = 0; ibatch < num_batch; ibatch++) {
    // Skip batches whose KV cache is empty
    int num_seqkv =
        (new_kv_included ? num_seq_kvcache[ibatch] : num_seq_kvcache[ibatch] + num_seq_q);
    num_seqkvs[ibatch] = num_seqkv;
    num_tiles[ibatch] = (num_seqkv + tilen - 1) / tilen;
    total_tiles_per_head += num_tiles[ibatch];
  }

  int total_tiles_all_heads = total_tiles_per_head * num_head_kv;

  int num_tile_per_cta = std::max((total_tiles_all_heads + num_total_ctas - 1) / num_total_ctas,
                                  min_process_len / tilen);

  std::vector<dynamic::TaskScheduleInfo> tasks(num_total_ctas * (num_tile_per_cta + 1));

  // num_chunks has one entry per (ihead_kv, ibatch). The trailing slot
  // (num_chunks[num_head_kv * num_batch]) is repurposed to carry
  // num_tile_per_cta+1 for the CPU entry wrapper
  std::vector<int> num_chunks(num_head_kv * num_batch + 1, 0);
  std::vector<int> start_tiles(num_batch * num_head_kv, 0);
  std::vector<int> chunks_in_progress(num_batch * num_head_kv, 0);
  std::vector<int> num_tiles_left(num_batch * num_head_kv, 0);
  for (int h = 0; h < num_head_kv; h++) {
    for (int b = 0; b < num_batch; b++) {
      num_tiles_left[h * num_batch + b] = num_tiles[b];
    }
  }

  int ihead_kv = 0;
  int ibatch = 0;

  int last_cta = 0;
  int last_task = 0;

  for (int icta = 0; icta < num_total_ctas; icta++) {
    int bucket = num_tile_per_cta;
    int itask = 0;
    while (bucket > 0 && ihead_kv < num_head_kv) {
      int idx = ihead_kv * num_batch + ibatch;
      int num_tile = num_tiles_left[idx];

      // Skip (ihead_kv, ibatch) with no tiles
      if (num_tile <= 0) {
        ibatch++;
        if (ibatch >= num_batch) {
          ibatch = 0;
          ihead_kv++;
          if (ihead_kv >= num_head_kv) {
            break;
          }
        }
        continue;
      }

      int add_tiles = std::min(num_tile, bucket);
      int num_seqkv = num_seqkvs[ibatch];

      if (chunks_in_progress[idx] == num_total_ctas - 1) {
        add_tiles = num_tile;
      }

      dynamic::TaskScheduleInfo task_info;
      task_info.ihead_kv = ihead_kv;
      task_info.ibatch = ibatch;
      task_info.ichunk = chunks_in_progress[idx];
      task_info.iseq_start = start_tiles[idx] * tilen;
      task_info.num_seqkv = std::min(add_tiles * tilen, num_seqkv - task_info.iseq_start);
      task_info.num_seqkvcache = task_info.num_seqkv;
      task_info.num_tile_kv = (task_info.num_seqkv + tilen - 1) / tilen;
      task_info.num_tile_full = task_info.num_seqkvcache / tilen;
      task_info.is_casual_chunk = 0;

      tasks[icta * (num_tile_per_cta + 1) + itask] = task_info;

      itask++;
      chunks_in_progress[idx]++;
      start_tiles[idx] += add_tiles;
      num_tiles_left[idx] -= add_tiles;
      bucket -= add_tiles;

      if (num_tiles_left[idx] <= 0) {
        // last chunk
        auto& cur = tasks[icta * (num_tile_per_cta + 1) + itask - 1];
        cur.is_casual_chunk = 1;
        cur.num_seqkvcache -= num_seq_q;
        cur.num_tile_full = std::max(cur.num_seqkvcache / tilen, 0);
        num_chunks[ihead_kv * num_batch + ibatch] = chunks_in_progress[idx];

        // When raw_seqkvcache < 0, some Q tokens overflow into the previous
        // chunk. Mark prev as causal and reduce its num_seqkvcache accordingly.
        if (cur.num_seqkvcache < 0) {
          auto& prev = tasks[last_cta * (num_tile_per_cta + 1) + last_task];
          prev.is_casual_chunk = 1;
          prev.num_seqkvcache += cur.num_seqkvcache;
          prev.num_tile_full = std::max(prev.num_seqkvcache / tilen, 0);
        }

        // advance (ihead_kv, ibatch)
        ibatch++;
        if (ibatch >= num_batch) {
          ibatch = 0;
          ihead_kv++;
        }
      }
      last_task = itask - 1;
    }

    last_cta = icta;

    tasks[icta * (num_tile_per_cta + 1) + itask].ihead_kv = -1;
    tasks[icta * (num_tile_per_cta + 1) + itask].ibatch = -1;
    // Fill unused tail slots with -1 terminators too — consistent with CUDA.
    for (int slot = itask + 1; slot <= num_tile_per_cta; slot++) {
      tasks[icta * (num_tile_per_cta + 1) + slot].ihead_kv = -1;
      tasks[icta * (num_tile_per_cta + 1) + slot].ibatch = -1;
    }
  }

  num_chunks[num_head_kv * num_batch] = num_tile_per_cta + 1;

  return std::make_pair(tasks, num_chunks);
}
}  // namespace decode
}  // namespace attention
}  // namespace hpc
int main() {
  int B, H, P, minimum;
  while (std::cin >> B >> H >> P >> minimum) {
    std::vector<int> lengths(B);
    for (auto& n : lengths) std::cin >> n;
    auto result = hpc::attention::decode::assign_attention_decode_task_sync(lengths.data(), P, B, H,
                                                                            1, 128, true, minimum);
    int stride = result.second.back();
    std::cout << "{\"capacity\":" << stride - 1 << ",\"tasks\":[";
    bool first = true;
    for (int c = 0; c < P; ++c) {
      for (int j = 0; j < stride; ++j) {
        auto t = result.first[c * stride + j];
        if (t.ihead_kv < 0 || t.ibatch < 0) break;
        if (!first) std::cout << ",";
        first = false;
        std::cout << "[" << c << "," << t.ibatch << "," << t.ihead_kv << "," << t.iseq_start << ","
                  << t.iseq_start + t.num_seqkv << "," << t.ichunk << "]";
      }
    }
    std::cout << "],\"records\":[";
    first = true;
    for (const auto& t : result.first) {
      if (t.ihead_kv < 0 || t.ibatch < 0) continue;
      if (!first) std::cout << ",";
      first = false;
      std::cout << "[" << t.ihead_kv << "," << t.ibatch << "," << t.ichunk << "," << t.iseq_start
                << "," << t.num_seqkv << "," << t.num_seqkvcache << "," << t.num_tile_kv << ","
                << t.num_tile_full << "," << t.is_casual_chunk << "]";
    }
    std::cout << "]}\n";
  }
}
