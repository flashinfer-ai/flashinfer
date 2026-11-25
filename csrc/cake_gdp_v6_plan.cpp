/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The PLAN CHOOSER of the Cake GDP Blackwell prefill (flashinfer/cake_gdp_backend/cake_jit.py) in native code: the
// launch of one batch and the int32 words of its item table, for the eager call's host time.  It is a line-by-line
// port of the Python chooser, which stays the reference (``prefill_v6_plan_reference``): the same integer model, the
// same candidates in the same order, the same placement (the bulk LPT of ``lpt_rounds`` / ``lpt_counts``), the same
// robustness rule and tie breaks, so both give the same plan bit for bit (tests/cpu/gdp checks it on random batches).
// Every constant comes from the caller (the Python constants of the split rules and of the chooser), so the rules
// keep one text.  No device code, no CUDA call: a pure host function of its arguments.

#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace {

// A per-call bump arena (thread local): every container of the chooser allocates from it and nothing is freed until
// the next call resets it, which keeps one block (the last, largest) unless the call grew it past 4 MiB.
class Arena {
 public:
  void* Get(size_t n, size_t align) {
    size_t at = (used_ + align - 1) & ~(align - 1);
    if (cur_ == nullptr || at + n > cap_) {
      NewBlock(n + align);
      at = 0;
    }
    used_ = at + n;
    return cur_ + at;
  }
  void Reset() {
    used_ = 0;
    if (blocks_.empty()) return;
    if (total_ > (size_t{4} << 20)) {
      blocks_.clear();
      cur_ = nullptr;
      cap_ = total_ = 0;
      return;
    }
    std::unique_ptr<char[]> last = std::move(blocks_.back());
    blocks_.clear();
    blocks_.push_back(std::move(last));
    cur_ = blocks_.back().get();
    total_ = cap_;
  }

 private:
  void NewBlock(size_t need) {
    size_t size = std::max<size_t>(need, cap_ ? 2 * cap_ : size_t{64} << 10);
    blocks_.emplace_back(new char[size]);
    cur_ = blocks_.back().get();
    cap_ = size;
    used_ = 0;
    total_ += size;
  }
  std::vector<std::unique_ptr<char[]>> blocks_;
  char* cur_ = nullptr;
  size_t used_ = 0, cap_ = 0, total_ = 0;
};

thread_local Arena tls_arena;

template <class T>
struct ArenaAlloc {
  using value_type = T;
  ArenaAlloc() = default;
  template <class U>
  ArenaAlloc(const ArenaAlloc<U>&) {}  // NOLINT(google-explicit-constructor)
  T* allocate(size_t n) { return static_cast<T*>(tls_arena.Get(n * sizeof(T), alignof(T))); }
  void deallocate(T*, size_t) {}
  template <class U>
  bool operator==(const ArenaAlloc<U>&) const { return true; }
  template <class U>
  bool operator!=(const ArenaAlloc<U>&) const { return false; }
};
template <class T>
using Vec = std::vector<T, ArenaAlloc<T>>;
template <class K, class V, class C = std::less<K>>
using Map = std::map<K, V, C, ArenaAlloc<std::pair<const K, V>>>;


using i64 = int64_t;
using Row = std::array<i64, 4>;  // (seq, head, cut, span)

// Python floor division and ceiling division (``-(-a // b)``)
i64 FloorDiv(i64 a, i64 b) {
  i64 q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}
i64 CeilDiv(i64 a, i64 b) { return -FloorDiv(-a, b); }

// The parameter words (``_v6_native_params`` in cake_jit.py; one layout, checked by its length).
enum Param : int {
  kChunk,
  kWarmup,
  kItem,
  kDecode,
  kFixed,
  kBlockChunk,
  kBlockItem,
  kBlockDecode,
  kBlockFixed,
  kMargin,
  kRobust,
  kWarmupCap,
  kMinChunks,
  kSplitMax,
  kCostChunk,
  kCostItem,
  kBalMinSlot,
  kBalSlots,
  kInfItemCost,
  kInfWarmupCost,
  kSpanMinRecord,
  kNumSpecs,
  kSpecs,  // then per spec: floor (-1: exact), V8-BAL flag
};

struct Params {
  Vec<i64> w;
  i64 operator[](int i) const { return w[i]; }
  i64 num_specs() const { return w[kNumSpecs]; }
  i64 spec_floor(i64 s) const { return w[kSpecs + 2 * s]; }
  bool spec_bal(i64 s) const { return w[kSpecs + 2 * s + 1] != 0; }
};

struct Decision {
  i64 p;     // 0: V8-BAL (span = slot); 1: no cut; else the uniform cut count
  i64 span;  // the span, or the slot
  bool operator==(const Decision& o) const { return p == o.p && span == o.span; }
  bool operator<(const Decision& o) const { return p < o.p || (p == o.p && span < o.span); }
};

struct Batch {
  i64 H, n, sm, total, arch_slots;
  Vec<i64> lens, chunks;
  Map<i64, Vec<i64>> by_len;  // length -> sequences, ascending
  Map<i64, i64> chunk_of;             // length -> chunks
  Vec<i64> distinct;               // the distinct chunk counts, ascending
  bool empty = false;
};

i64 SplitChunks(i64 len, i64 n) { return 2 * CeilDiv(len * n, 128); }
i64 SplitAlign(i64 n) { return n % 2 == 0 ? n : 2 * n; }
i64 SplitSpan(i64 chunks, i64 p, i64 n) {
  i64 a = SplitAlign(n);
  return a * CeilDiv(CeilDiv(chunks, p), a);
}

struct ScanEntry {
  i64 span, p_eff, cost;
};

Vec<ScanEntry> CountScan(const Params& P, i64 chunks, i64 H, i64 n, i64 sm, i64 span_min) {
  Vec<ScanEntry> out;
  for (i64 p = 1; p <= P[kSplitMax]; ++p) {
    i64 span = p == 1 ? chunks : SplitSpan(chunks, p, n);
    if (p > 1 && span < span_min) break;
    i64 p_eff = CeilDiv(chunks, span);
    out.push_back({span, p_eff,
                   CeilDiv(H * p_eff, sm) * ((span + 1) * P[kCostChunk] + P[kCostItem])});
  }
  return out;
}

Decision ScanBest(const Vec<ScanEntry>& scan, i64 span_min) {
  const ScanEntry* best = nullptr;
  for (size_t at = 0; at < scan.size(); ++at) {
    if (at && scan[at].span < span_min) break;
    if (best == nullptr || scan[at].cost < best->cost) best = &scan[at];
  }
  return {best->p_eff, best->span};
}

// ``_v6_decisions``: the cut of a sequence of ``chunks`` chunks under every split spec.
Vec<Decision> Decisions(const Params& P, const Batch& bt, i64 chunks) {
  i64 S = P.num_specs();
  if (chunks < P[kMinChunks]) return Vec<Decision>(S, Decision{1, chunks > 0 ? chunks : 0});
  i64 record = P[kSpanMinRecord], least = record;
  for (i64 s = 0; s < S; ++s)
    if (P.spec_floor(s) >= 0) least = std::min(least, P.spec_floor(s));
  auto scan = CountScan(P, chunks, bt.H, bt.n, bt.sm, least);
  i64 slot = 0, slots = P[kBalSlots];
  if (slots > 0) {
    i64 align = SplitAlign(bt.n);
    i64 padded = align * CeilDiv(chunks, align);
    slot = align * CeilDiv(CeilDiv(bt.H * padded, slots), align);
    if (!(P[kBalMinSlot] <= slot && slot < padded && padded <= (P[kSplitMax] - 1) * slot)) slot = 0;
  }
  Vec<ScanEntry> bal_scan;
  if (slot) bal_scan = bt.sm == slots ? scan : CountScan(P, chunks, bt.H, bt.n, slots, least);
  Vec<Decision> out;
  for (i64 s = 0; s < S; ++s) {
    i64 floor = P.spec_floor(s);
    if (floor < 0) {
      out.push_back({1, chunks});
      continue;
    }
    i64 span_min = std::min(floor, record);
    if (P.spec_bal(s) && slot && 2 * slots <= bt.H * ScanBest(bal_scan, span_min).p) {
      out.push_back({0, slot});
      continue;
    }
    out.push_back(ScanBest(scan, span_min));
  }
  return out;
}

// ``_v6_cut_shape``: the rows of a sequence (index 0) cut by ``d``, per row its chunk count and kind (0: cut 0,
// 1: cut > 0, -1: a pad row).
struct CutShape {
  Vec<Row> rows;
  Vec<i64> ks, kinds;
  Vec<i64> costs[2];  // calibrated / record row costs, filled on first use
};

CutShape MakeCutShape(const Params& P, i64 chunks, Decision d, i64 H, i64 n) {
  CutShape cs;
  if (d.p == 0) {
    i64 slot = d.span, align = SplitAlign(n);
    i64 padded = align * CeilDiv(chunks, align), top = H * padded;
    i64 slots = P[kBalSlots];
    Vec<std::pair<Row, i64>> first, second;
    first.reserve(slots);
    second.reserve(slots);
    Vec<std::pair<Row, i64>> pieces;  // reused: a slot spans at most two heads (slot < padded)
    for (i64 k = 0; k < slots; ++k) {
      i64 lo = k * slot, hi = std::min((k + 1) * slot, top);
      pieces.clear();
      while (lo < hi) {
        i64 head = lo / padded, cut = lo % padded;
        i64 end = std::min(hi, (head + 1) * padded);
        if (cut < chunks) pieces.push_back({Row{0, head, cut, -slot}, std::min(end - head * padded, chunks) - cut});
        lo = end;
      }
      std::pair<Row, i64> pad{Row{0, std::min(k * slot / padded, H - 1), chunks, -slot}, 0};
      if (pieces.size() == 2 && pieces[1].first[2] == 0) std::swap(pieces[0], pieces[1]);
      first.push_back(pieces.empty() ? pad : pieces[0]);
      second.push_back(pieces.size() > 1 ? pieces[1] : pad);
    }
    cs.rows.reserve(2 * slots);
    cs.ks.reserve(2 * slots);
    cs.kinds.reserve(2 * slots);
    for (auto* half : {&first, &second})
      for (auto& [row, k] : *half) {
        cs.rows.push_back(row);
        cs.ks.push_back(k);
        cs.kinds.push_back(row[2] >= chunks ? -1 : (row[2] ? 1 : 0));
      }
    return cs;
  }
  cs.rows.reserve(d.p * H);
  cs.ks.reserve(d.p * H);
  cs.kinds.reserve(d.p * H);
  for (i64 j = 0; j < d.p; ++j)
    for (i64 head = 0; head < H; ++head) {
      cs.rows.push_back(Row{0, head, j * d.span, d.span});
      cs.ks.push_back(std::max<i64>(std::min(d.span, chunks - j * d.span), 0));
      cs.kinds.push_back(j ? 1 : 0);
    }
  return cs;
}

struct Model {
  const Params& P;
  const Batch& bt;
  Map<std::pair<i64, Decision>, CutShape> shapes;

  const CutShape& Shape(i64 chunks, Decision d) {
    auto key = std::make_pair(chunks, d);
    auto it = shapes.find(key);
    if (it == shapes.end()) it = shapes.emplace(key, MakeCutShape(P, chunks, d, bt.H, bt.n)).first;
    return it->second;
  }
  // ``_v6_cut_costs``: calibrated (or, ``record``, the record) costs of the rows of a cut
  const Vec<i64>& CutCosts(i64 chunks, Decision d, bool record) {
    CutShape& cs = const_cast<CutShape&>(Shape(chunks, d));
    Vec<i64>& out = cs.costs[record ? 1 : 0];
    if (out.size() == cs.ks.size() && !cs.ks.empty()) return out;
    i64 a = record ? 1 : P[kChunk], item = record ? P[kInfItemCost] : P[kItem];
    i64 warm = record ? P[kInfWarmupCost] : P[kWarmup];
    out.resize(cs.ks.size());
    for (size_t i = 0; i < out.size(); ++i)
      out[i] = cs.kinds[i] < 0 ? 0 : cs.ks[i] * a + item + (cs.kinds[i] ? warm : 0);
    return out;
  }
  // ``_v6_cut_stats``: (rows, total, largest) calibrated cost of the rows of a cut sequence
  std::array<i64, 3> CutStats(i64 chunks, Decision d) {
    if (d.p == 0) {
      const auto& c = CutCosts(chunks, d, false);
      return {static_cast<i64>(c.size()), std::accumulate(c.begin(), c.end(), i64{0}),
              *std::max_element(c.begin(), c.end())};
    }
    i64 sum = 0, top = 0;
    for (i64 j = 0; j < d.p; ++j) {
      i64 e = std::max<i64>(std::min(d.span, chunks - j * d.span), 0) * P[kChunk] + P[kItem] + (j ? P[kWarmup] : 0);
      sum += e;
      top = j ? std::max(top, e) : e;
    }
    return {bt.H * d.p, bt.H * sum, top};
  }
};

// ``lpt_rounds``
struct Rounds {
  Vec<i64> head, phase, tail;
  i64 full = 0;
};

i64 ArgMinLoad(const Vec<i64>& loads) {
  i64 best = 0;
  for (i64 b = 1; b < static_cast<i64>(loads.size()); ++b)
    if (loads[b] < loads[best]) best = b;
  return best;
}

bool IdleShortcut(const Vec<i64>& loads, i64 k, Vec<i64>* idle) {
  i64 G = loads.size();
  if (k > G) return false;
  i64 low = *std::min_element(loads.begin(), loads.end());
  idle->clear();
  for (i64 b = 0; b < G; ++b)
    if (loads[b] == low) idle->push_back(b);
  if (static_cast<i64>(idle->size()) < k) return false;
  idle->resize(k);
  return true;
}

Rounds LptRounds(const Vec<i64>& loads, i64 cost, i64 k) {
  Rounds r;
  i64 G = loads.size();
  if (k <= 0) return r;
  if (cost <= 0) {
    r.head.assign(k, ArgMinLoad(loads));
    return r;
  }
  Vec<i64> idle;
  if (IdleShortcut(loads, k, &idle)) {
    r.head = idle;
    return r;
  }
  Vec<i64> q(G);
  for (i64 b = 0; b < G; ++b) q[b] = FloorDiv(loads[b], cost);
  Vec<i64> wake(G);
  std::iota(wake.begin(), wake.end(), 0);
  std::stable_sort(wake.begin(), wake.end(), [&](i64 a, i64 b) { return q[a] < q[b]; });
  Vec<i64> keys, order;
  i64 at = 0, placed = 0, rnd = q[wake[0]];
  while (at < G) {
    Vec<i64> woke;
    while (at < G && q[wake[at]] <= rnd) {
      i64 b = wake[at];
      woke.push_back((loads[b] - q[b] * cost) * G + b);
      ++at;
    }
    if (!woke.empty()) {
      std::sort(woke.begin(), woke.end());
      size_t mid = keys.size();
      keys.insert(keys.end(), woke.begin(), woke.end());
      std::inplace_merge(keys.begin(), keys.begin() + mid, keys.end());
      order.resize(keys.size());
      for (size_t i = 0; i < keys.size(); ++i) order[i] = keys[i] % G;
    }
    if (at == G) break;
    i64 len = order.size();
    i64 rounds = std::min(q[wake[at]] - rnd, (k - placed) / len);
    for (i64 i = 0; i < rounds; ++i) r.head.insert(r.head.end(), order.begin(), order.end());
    placed += rounds * len;
    rnd += rounds;
    if (rnd < q[wake[at]]) {
      r.head.insert(r.head.end(), order.begin(), order.begin() + (k - placed));
      return r;
    }
  }
  i64 rest = (k - placed) % G;
  r.full = (k - placed) / G;
  r.phase = order;
  r.tail.assign(order.begin(), order.begin() + rest);
  return r;
}

void LptCounts(const Vec<i64>& loads, i64 cost, i64 k, Vec<i64>& counts, Vec<i64>& idle) {
  i64 G = loads.size();
  counts.assign(G, 0);
  if (k <= 0) return;
  if (cost <= 0) {
    counts[ArgMinLoad(loads)] = k;
    return;
  }
  if (IdleShortcut(loads, k, &idle)) {
    for (i64 b : idle) counts[b] = 1;
    return;
  }
  Vec<i64> q(G);
  for (i64 b = 0; b < G; ++b) q[b] = FloorDiv(loads[b], cost);
  Vec<i64> ordered(q);
  std::sort(ordered.begin(), ordered.end());
  i64 total = 0, t = 0;
  for (i64 i = 1; i <= G; ++i) {
    total += ordered[i - 1];
    t = FloorDiv(k + total, i);
    if (i == G || t < ordered[i]) break;
  }
  i64 sum = 0;
  for (i64 b = 0; b < G; ++b) {
    counts[b] = q[b] < t ? t - q[b] : 0;
    sum += counts[b];
  }
  i64 rest = k - sum;
  if (rest) {
    Vec<i64> keys;
    for (i64 b = 0; b < G; ++b)
      if (q[b] <= t) keys.push_back((loads[b] - q[b] * cost) * G + b);
    std::sort(keys.begin(), keys.end());
    for (i64 i = 0; i < rest; ++i) counts[keys[i] % G] += 1;
  }
}

// A CTA's rows in the table: its long cut-0 rows, its long later-cut rows, its short rows (``inference_lists``), each
// in placement order.  Rows are collected as (3 * CTA + part, row) and laid out by a stable counting sort.
using Placed = Vec<std::pair<i64, Row>>;

void LptPlace(Vec<i64>& loads, i64 cost, const Vec<Row>& rows, Placed& placed) {
  i64 G = loads.size();
  Rounds r = LptRounds(loads, cost, rows.size());
  size_t at = 0;
  for (; at < r.head.size() && at < rows.size(); ++at) {
    placed.push_back({3 * r.head[at] + 2, rows[at]});
    loads[r.head[at]] += cost;
  }
  at = r.head.size();
  if (r.full) {
    size_t end = at + r.full * G;
    for (size_t j = 0; j < r.phase.size(); ++j) {
      i64 b = r.phase[j];
      for (size_t i = at + j; i < end; i += G) placed.push_back({3 * b + 2, rows[i]});
      loads[b] += r.full * cost;
    }
    at = end;
  }
  for (size_t i = 0; i < r.tail.size() && at + i < rows.size(); ++i) {
    placed.push_back({3 * r.tail[i] + 2, rows[at + i]});
    loads[r.tail[i]] += cost;
  }
}

// The plan rows of ``placed`` grouped by CTA (stable), and the CTAs' row offsets.
void LayOut(const Placed& placed, i64 G, Vec<i64>& count, Vec<Row>* rows, Vec<i64>* offsets) {
  count.assign(3 * G + 1, 0);
  for (const auto& pr : placed) ++count[pr.first + 1];
  for (i64 b = 0; b < 3 * G; ++b) count[b + 1] += count[b];
  offsets->resize(G + 1);
  for (i64 b = 0; b <= G; ++b) (*offsets)[b] = count[3 * b];
  rows->resize(placed.size());
  for (const auto& pr : placed) (*rows)[count[pr.first]++] = pr.second;
}

// A launch: the rows in table order, the build, the grid, the CTAs' row offsets (split build).
struct Plan {
  Vec<Row> rows;
  bool split = false;
  i64 grid = 1;
  Vec<i64> offsets;
};

// ``_V6Split``
struct SplitCand {
  i64 name;
  bool record;
  Map<i64, Decision> dec;
  Vec<i64> longs;
  // placement cost -> [(sequences, calibrated cost)], iterated by placement cost descending
  Map<i64, Vec<std::pair<Vec<i64>, i64>>, std::greater<i64>> shorts;
  i64 items = 0, grid = 1, bound = 0;
};

// ``_V6Block``
struct BlockCand {
  i64 name;
  bool balanced;
  Vec<i64> order;
  Vec<std::pair<i64, i64>> runs;  // (rows, cost) in row order
  i64 items = 0, grid = 1, bound = 0;
  bool ok = true;
  // balanced placement
  Vec<i64> cap, assign;
  i64 least = 0;
};

i64 BlockRowCost(const Params& P, i64 tokens, i64 n) {
  if (tokens == 1) return P[kBlockDecode];
  return n * CeilDiv(tokens, 64) * P[kBlockChunk] + P[kBlockItem];
}

i64 SplitRowCost(const Params& P, const Row& row, i64 chunks, i64 tokens, i64 n) {
  i64 head = row[1], cut = row[2], span = row[3];
  if (span == 0) return P[kItem];
  i64 end;
  if (span < 0) {
    if (cut >= chunks) return 0;
    i64 align = SplitAlign(n), padded = align * CeilDiv(chunks, align);
    i64 at = head * padded + cut;
    end = std::min((FloorDiv(at, -span) + 1) * -span - head * padded, chunks);
  } else {
    end = std::min(cut + span, chunks);
  }
  if (tokens == 1) return P[kDecode];
  return std::max<i64>(end - cut, 0) * P[kChunk] + P[kItem] + (cut > 0 ? P[kWarmup] : 0);
}

struct Chooser {
  const Params& P;
  const Batch& bt;
  Model model;
  // scratch buffers reused across candidates
  Vec<i64> cloads_, ploads_, counts_, idle_, loads_, count_, costs_;
  Vec<Row> rows_;
  Placed placed_;

  Chooser(const Params& p, const Batch& b) : P(p), bt(b), model{p, b, {}} {}

  SplitCand MakeSplit(i64 name, const Map<i64, Decision>& dec, bool record) {
    SplitCand c;
    c.name = name;
    c.record = record;
    c.dec = dec;
    i64 rows = 0, total = 0, top = 0;
    for (const auto& [length, seqs] : bt.by_len) {
      i64 chunks = bt.chunk_of.at(length);
      Decision d = dec.at(chunks);
      i64 count = seqs.size();
      if (d.p != 1) {
        c.longs.insert(c.longs.end(), seqs.begin(), seqs.end());
        auto st = model.CutStats(chunks, d);
        rows += st[0] * count;
        total += st[1] * count;
        top = std::max(top, st[2]);
        continue;
      }
      i64 cc = length == 0 ? P[kItem] : length == 1 ? P[kDecode] : chunks * P[kChunk] + P[kItem];
      i64 pc = record ? chunks + P[kInfItemCost] : cc;
      c.shorts[pc].push_back({seqs, cc});
      rows += bt.H * count;
      total += bt.H * count * cc;
      top = std::max(top, cc);
    }
    std::sort(c.longs.begin(), c.longs.end());
    c.items = rows;
    c.grid = std::max<i64>(1, std::min(bt.sm, rows));
    c.bound = P[kFixed] + std::max(top, CeilDiv(total, c.grid));
    return c;
  }

  // the loads of a cut's rows dealt round-robin over G CTAs from CTA ``shift`` on
  void AddCutLoads(Vec<i64>& loads, i64 chunks, Decision d, i64 shift, bool record) {
    const auto& costs = model.CutCosts(chunks, d, record);
    i64 G = loads.size(), b = shift % G;
    for (size_t i = 0; i < costs.size(); ++i) {
      loads[b] += costs[i];
      if (++b == G) b = 0;
    }
  }

  i64 Cost(const SplitCand& c) {
    i64 G = c.grid, H = bt.H;
    Vec<i64>& cloads = cloads_;
    Vec<i64>& ploads = ploads_;
    cloads.assign(G, 0);
    ploads.assign(G, 0);
    i64 at = 0;
    for (i64 q : c.longs) {
      i64 chunks = bt.chunks[q];
      Decision d = c.dec.at(chunks);
      i64 shift = at % G;
      AddCutLoads(cloads, chunks, d, shift, false);
      if (c.record) AddCutLoads(ploads, chunks, d, shift, true);
      at += d.p == 0 ? static_cast<i64>(model.Shape(chunks, d).ks.size()) : bt.H * d.p;
    }
    if (!c.record) ploads.assign(cloads.begin(), cloads.end());
    for (const auto& [pc, groups] : c.shorts) {
      i64 seqs = 0;
      bool one = true;
      for (const auto& g : groups) {
        seqs += g.first.size();
        one = one && g.second == groups[0].second;
      }
      i64 k = H * seqs;
      if (one) {
        i64 cc = groups[0].second;
        Vec<i64>& counts = counts_;
        LptCounts(ploads, pc, k, counts, idle_);
        for (i64 b = 0; b < G; ++b) {
          if (c.record) {
            ploads[b] += counts[b] * pc;
            cloads[b] += counts[b] * cc;
          } else {
            cloads[b] += counts[b] * cc;
          }
        }
        if (!c.record) ploads.assign(cloads.begin(), cloads.end());
        continue;
      }
      Vec<std::pair<i64, i64>> per_seq;
      for (const auto& g : groups)
        for (i64 q : g.first) per_seq.push_back({q, g.second});
      std::sort(per_seq.begin(), per_seq.end());
      Rounds r = LptRounds(ploads, pc, k);
      Vec<i64> slots(r.head);
      for (i64 f = 0; f < r.full; ++f) slots.insert(slots.end(), r.phase.begin(), r.phase.end());
      slots.insert(slots.end(), r.tail.begin(), r.tail.end());
      size_t i = 0;
      for (const auto& [q, cc] : per_seq)
        for (i64 h = 0; h < H && i < slots.size(); ++h, ++i) {
          cloads[slots[i]] += cc;
          ploads[slots[i]] += pc;
        }
    }
    return P[kFixed] + *std::max_element(cloads.begin(), cloads.end());
  }

  Plan Build(const SplitCand& c) {
    i64 G = c.grid, H = bt.H;
    i64 count = 0;
    for (i64 q : c.longs) {
      i64 chunks = bt.chunks[q];
      count += static_cast<i64>(model.Shape(chunks, c.dec.at(chunks)).rows.size());
    }
    // ``inference_lists``: the long rows dealt round-robin, each CTA's cut-0 rows first when a CTA holds several
    Placed& placed = placed_;
    placed.clear();
    Vec<i64>& loads = loads_;
    loads.assign(G, 0);
    i64 i = 0;
    for (i64 q : c.longs) {
      i64 chunks = bt.chunks[q];
      Decision d = c.dec.at(chunks);
      const CutShape& cs = model.Shape(chunks, d);
      const auto& costs = model.CutCosts(chunks, d, c.record);
      for (size_t j = 0; j < cs.rows.size(); ++j, ++i) {
        const Row& r = cs.rows[j];
        i64 b = count <= G ? i : i % G;
        placed.push_back({3 * b + (count > G && r[2] != 0 ? 1 : 0), Row{q, r[1], r[2], r[3]}});
        loads[b] += costs[j];
      }
    }
    Vec<Row>& rows = rows_;
    for (const auto& [pc, groups] : c.shorts) {
      Vec<i64> seqs;
      for (const auto& g : groups) seqs.insert(seqs.end(), g.first.begin(), g.first.end());
      if (groups.size() > 1) std::sort(seqs.begin(), seqs.end());
      rows.clear();
      rows.reserve(seqs.size() * H);
      for (i64 q : seqs)
        for (i64 h = 0; h < H; ++h) rows.push_back(Row{q, h, 0, bt.chunks[q]});
      LptPlace(loads, pc, rows, placed);
    }
    Plan plan;
    plan.split = true;
    plan.grid = G;
    LayOut(placed, G, count_, &plan.rows, &plan.offsets);
    return plan;
  }

  BlockCand MakeBlock(i64 name, bool balanced) {
    BlockCand c;
    c.name = name;
    c.balanced = balanced;
    Map<i64, Vec<std::pair<i64, i64>>, std::greater<i64>> by_blocks;
    for (const auto& [length, seqs] : bt.by_len)
      if (length > 0)
        for (i64 q : seqs) by_blocks[CeilDiv(length, 64)].push_back({q, length});
    i64 total = 0, top = 0;
    for (auto& [blocks, group] : by_blocks) {
      std::sort(group.begin(), group.end());
      for (const auto& [q, length] : group) {
        c.order.push_back(q);
        i64 cost = BlockRowCost(P, length, bt.n);
        if (!c.runs.empty() && c.runs.back().second == cost)
          c.runs.back().first += bt.H;
        else
          c.runs.push_back({bt.H, cost});
        total += bt.H * cost;
        top = std::max(top, cost);
      }
    }
    c.items = static_cast<i64>(c.order.size()) * bt.H;
    c.grid = std::max<i64>(1, std::min(bt.sm, c.items));
    c.bound = P[kBlockFixed] + std::max(top, CeilDiv(total, c.grid));
    if (balanced) {
      i64 least = 0, big = 0;
      bool first = true;
      for (const auto& rc : c.runs) {
        least = first ? rc.second : std::min(least, rc.second);
        first = false;
      }
      for (const auto& rc : c.runs)
        if (rc.second > least) big += rc.first;
      c.ok = 0 < big && big <= 4 * c.grid;
    }
    return c;
  }

  i64 Cost(BlockCand& c) {
    i64 G = c.grid;
    if (!c.balanced) {
      Vec<i64> loads(G, 0);
      i64 pos = 0;
      for (const auto& [rows, cost] : c.runs)
        for (i64 i = 0; i < rows; ++i, ++pos) loads[pos % G] += cost;
      return P[kBlockFixed] + *std::max_element(loads.begin(), loads.end());
    }
    i64 R = c.items;
    i64 least = c.runs[0].second;
    for (const auto& rc : c.runs) least = std::min(least, rc.second);
    c.cap.assign(G, 0);
    for (i64 b = 0; b < G; ++b) c.cap[b] = R / G + (b < R % G ? 1 : 0);
    Vec<i64> used(G, 0), loads(G, 0);
    using Key = std::pair<i64, i64>;
    std::priority_queue<Key, Vec<Key>, std::greater<Key>> heap;
    for (i64 b = 0; b < G; ++b) heap.push({c.cap[b] * least, b});
    Vec<std::pair<i64, i64>> big;
    for (const auto& rc : c.runs)
      if (rc.second > least) big.push_back(rc);
    std::stable_sort(big.begin(), big.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    c.assign.clear();
    for (const auto& [rows, cost] : big)
      for (i64 i = 0; i < rows; ++i) {
        auto [key, b] = heap.top();
        heap.pop();
        used[b] += 1;
        loads[b] += cost;
        c.assign.push_back(b);
        if (used[b] < c.cap[b]) heap.push({loads[b] + (c.cap[b] - used[b]) * least, b});
      }
    i64 top = 0;
    for (i64 b = 0; b < G; ++b) top = std::max(top, loads[b] + (c.cap[b] - used[b]) * least);
    c.least = least;
    return P[kBlockFixed] + top;
  }

  Plan Build(BlockCand& c) {
    i64 G = c.grid, H = bt.H;
    Vec<Row> rows;
    rows.reserve(c.items);
    for (i64 q : c.order)
      for (i64 h = 0; h < H; ++h) rows.push_back(Row{q, h, 0, CeilDiv(bt.lens[q], 64)});
    Plan plan;
    plan.split = false;
    plan.grid = G;
    if (!c.balanced) {
      plan.rows = std::move(rows);
      return plan;
    }
    if (c.assign.empty()) Cost(c);
    Vec<i64> cost_of(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) cost_of[i] = BlockRowCost(P, bt.lens[rows[i][0]], bt.n);
    Vec<i64> big;
    for (size_t i = 0; i < rows.size(); ++i)
      if (cost_of[i] > c.least) big.push_back(i);
    std::stable_sort(big.begin(), big.end(), [&](i64 a, i64 b) { return cost_of[a] > cost_of[b]; });
    // per CTA: its big rows in assignment order, then the next cheapest rows up to its capacity; the table takes
    // the CTAs' r-th rows for r = 0, 1, ...
    Vec<i64> start(G + 1, 0), fill(G, 0);
    for (i64 b = 0; b < G; ++b) start[b + 1] = start[b] + c.cap[b];
    Vec<Row> flat(start[G]);
    for (size_t i = 0; i < big.size(); ++i) {
      i64 b = c.assign[i];
      flat[start[b] + fill[b]++] = rows[big[i]];
    }
    size_t rest = 0;
    i64 rounds = 0;
    for (i64 b = 0; b < G; ++b) {
      while (fill[b] < c.cap[b]) {
        while (cost_of[rest] > c.least) ++rest;
        flat[start[b] + fill[b]++] = rows[rest++];
      }
      rounds = std::max(rounds, c.cap[b]);
    }
    plan.rows.reserve(flat.size());
    for (i64 r = 0; r < rounds; ++r)
      for (i64 b = 0; b < G; ++b)
        if (r < c.cap[b]) plan.rows.push_back(flat[start[b] + r]);
    return plan;
  }

  // ``_v6_scenario_time`` over a plan prepared once: per row of a cut (sequence, head) its cut, the next cut of
  // that (sequence, head) (else the chunk count) and the range of its ascending cuts in ``cuts``; any other row
  // its fixed cost.
  struct RowInfo {
    i64 fixed, cut, next, chunks;
    i64 lo, hi;  // its (sequence, head)'s cuts: cuts[lo, hi); lo < 0: the fixed cost
  };
  struct Prepared {
    bool split = false;
    i64 grid = 1, block_time = 0;
    const Vec<i64>* offsets = nullptr;
    Vec<RowInfo> rows;
    Vec<i64> cuts;
  };

  void Prepare(const Plan& plan, Prepared* out, Vec<i64>* needs) {
    out->split = plan.split;
    out->grid = plan.grid;
    if (!plan.split) {
      Vec<i64>& loads = loads_;
      loads.assign(plan.grid, 0);
      for (size_t i = 0; i < plan.rows.size(); ++i)
        loads[i % plan.grid] += BlockRowCost(P, bt.lens[plan.rows[i][0]], bt.n);
      out->block_time = P[kBlockFixed] + *std::max_element(loads.begin(), loads.end());
      return;
    }
    out->offsets = &plan.offsets;
    size_t R = plan.rows.size();
    out->rows.assign(R, RowInfo{0, 0, 0, 0, -1, -1});
    out->cuts.clear();
    // the rows of every (sequence, head) that are neither a state copy nor a pad, ordered by (sequence, head, cut)
    Vec<std::array<i64, 3>>& keyed = keyed_;
    keyed.clear();
    keyed.reserve(R);
    for (size_t i = 0; i < R; ++i) {
      const Row& r = plan.rows[i];
      if (r[3] != 0 && r[2] < bt.chunks[r[0]]) keyed.push_back({r[0] * bt.H + r[1], r[2], static_cast<i64>(i)});
    }
    // ordered by (sequence * H + head, cut, row): a counting sort by (sequence, head) then an insertion sort of
    // each group by (cut, row) when the groups are few, else one sort
    i64 groups = static_cast<i64>(bt.lens.size()) * bt.H;
    if (groups <= 4 * static_cast<i64>(keyed.size()) + 1024) {
      Vec<i64>& start = packed_;
      start.assign(groups + 1, 0);
      for (const auto& k : keyed) ++start[k[0] + 1];
      for (i64 g = 0; g < groups; ++g) start[g + 1] += start[g];
      Vec<std::array<i64, 3>>& sorted = sorted_;
      sorted.resize(keyed.size());
      for (const auto& k : keyed) sorted[start[k[0]]++] = k;
      for (size_t a = 0, b; a < sorted.size(); a = b) {
        for (b = a + 1; b < sorted.size() && sorted[b][0] == sorted[a][0];) ++b;
        for (size_t k = a + 1; k < b; ++k)
          for (size_t j = k; j > a && (sorted[j][1] < sorted[j - 1][1] ||
                                       (sorted[j][1] == sorted[j - 1][1] && sorted[j][2] < sorted[j - 1][2]));
               --j)
            std::swap(sorted[j], sorted[j - 1]);
      }
      keyed.swap(sorted);
    } else {
      std::sort(keyed.begin(), keyed.end());
    }
    for (size_t a = 0, b; a < keyed.size(); a = b) {
      for (b = a + 1; b < keyed.size() && keyed[b][0] == keyed[a][0];) ++b;
      if (b - a < 2) continue;
      i64 lo = out->cuts.size(), hi = lo + static_cast<i64>(b - a);
      for (size_t k = a; k < b; ++k) {
        out->cuts.push_back(keyed[k][1]);
        if (keyed[k][1] > 0 && keyed[k][1] <= P[kWarmupCap]) needs->push_back(keyed[k][1]);
      }
      i64 chunks = bt.chunks[plan.rows[keyed[a][2]][0]];
      for (size_t k = a; k < b; ++k)
        out->rows[keyed[k][2]] = RowInfo{0, keyed[k][1], k + 1 < b ? keyed[k + 1][1] : chunks, chunks, lo, hi};
    }
    for (size_t i = 0; i < R; ++i)
      if (out->rows[i].lo < 0) {
        const Row& r = plan.rows[i];
        out->rows[i].fixed = SplitRowCost(P, r, bt.chunks[r[0]], bt.lens[r[0]], bt.n);
      }
  }

  i64 ScenarioTime(const Prepared& pl, i64 W) {
    if (!pl.split) return pl.block_time;
    bool live = W >= 0 && W <= P[kWarmupCap];
    i64 warm = live ? std::max(P[kWarmup], W * P[kChunk]) : 0;
    i64 top = 0;
    const Vec<i64>& off = *pl.offsets;
    const i64* cuts = pl.cuts.data();
    for (i64 b = 0; b < pl.grid; ++b) {
      i64 load = 0;
      for (i64 i = off[b]; i < off[b + 1]; ++i) {
        const RowInfo& r = pl.rows[i];
        if (r.lo < 0) {
          load += r.fixed;
          continue;
        }
        if (r.cut && !(live && r.cut >= W)) {
          load += P[kItem];  // an unsaturated cut: its scan
          continue;
        }
        i64 end = r.chunks;
        if (live) {
          if (r.cut) {
            end = r.next;
          } else {
            const i64* nx = std::upper_bound(cuts + r.lo, cuts + r.hi, W - 1);
            if (nx != cuts + r.hi) end = *nx;
          }
        }
        load += (end - r.cut) * P[kChunk] + P[kItem] + (r.cut ? warm : 0);
      }
      top = std::max(top, load);
    }
    return P[kFixed] + top;
  }

  // ``ref`` prepared once (with its cuts in ``ref_needs``); its scenario times memoized by W in ``ref_times``
  bool Robust(const Plan& cand, const Prepared& ref, const Vec<i64>& ref_needs,
              Vec<std::pair<i64, i64>>* ref_times) {
    Vec<i64> needs(ref_needs);
    Prepare(cand, &cand_, &needs);
    std::sort(needs.begin(), needs.end());
    needs.erase(std::unique(needs.begin(), needs.end()), needs.end());
    needs.push_back(-1);  // no cut saturates
    for (i64 W : needs) {
      i64 theirs = -1;
      for (const auto& [w, t] : *ref_times)
        if (w == W) theirs = t;
      if (theirs < 0) {
        theirs = ScenarioTime(ref, W);
        ref_times->push_back({W, theirs});
      }
      if (100 * ScenarioTime(cand_, W) > (100 + P[kRobust]) * theirs) return false;
    }
    return true;
  }
  Prepared cand_;
  Vec<std::array<i64, 3>> keyed_, sorted_;
  Vec<i64> packed_;
};

// The candidate names in V6_PLAN_CANDIDATES order: the split specs, then main, block, blockbal.
struct Choice {
  Plan plan;
  i64 name;
};

Choice Choose(const Params& P, const Batch& bt) {
  Chooser ch(P, bt);
  i64 S = P.num_specs();
  i64 kMain = S, kBlock = S + 1, kBlockBal = S + 2;
  Vec<Vec<Decision>> per;
  for (i64 c : bt.distinct) per.push_back(Decisions(P, bt, c));
  // distinct split candidates, first name wins
  Vec<SplitCand> splits;
  Vec<Vec<Decision>> keys;
  i64 f48 = -1;  // the split spec of the record decisions (floor 48 with V8-BAL)
  for (i64 s = 0; s < S; ++s)
    if (P.spec_floor(s) == 48 && P.spec_bal(s)) {
      f48 = s;
      break;
    }
  TVM_FFI_ICHECK(f48 >= 0) << "the split specs hold no f48";
  Map<i64, Decision> record;
  for (i64 s = 0; s < S; ++s) {
    Vec<Decision> key;
    for (const auto& d : per) key.push_back(d[s]);
    auto seen = std::find(keys.begin(), keys.end(), key);
    Map<i64, Decision> dec;
    for (size_t i = 0; i < bt.distinct.size(); ++i) dec[bt.distinct[i]] = key[i];
    if (s == f48) record = dec;
    if (seen != keys.end()) continue;
    keys.push_back(key);
    splits.push_back(ch.MakeSplit(s, dec, false));
  }
  bool record_split = bt.empty;
  for (const auto& [c, d] : record) record_split = record_split || d.p != 1;
  // candidates in order: splits, main (split of record) or none, block, blockbal
  struct Cand {
    int kind;  // 0 split, 1 block
    size_t at;
  };
  Vec<Cand> cands;
  for (size_t i = 0; i < splits.size(); ++i) cands.push_back({0, i});
  if (record_split) {
    splits.push_back(ch.MakeSplit(kMain, record, true));
    cands.push_back({0, splits.size() - 1});
  }
  Vec<BlockCand> blocks;
  if (!bt.empty) {
    blocks.push_back(ch.MakeBlock(kBlock, false));
    BlockCand bal = ch.MakeBlock(kBlockBal, true);
    bal.order = blocks[0].order;
    bal.runs = blocks[0].runs;
    if (bal.ok) blocks.push_back(bal);
    for (size_t i = 0; i < blocks.size(); ++i) cands.push_back({1, i});
  }
  // the plan of record: main, else the block candidate
  size_t ref_at = 0;
  for (size_t i = 0; i < cands.size(); ++i) {
    bool is_ref = record_split ? (cands[i].kind == 0 && splits[cands[i].at].name == kMain)
                               : (cands[i].kind == 1 && blocks[cands[i].at].name == kBlock);
    if (is_ref) ref_at = i;
  }
  auto bound = [&](const Cand& c) {
    return c.kind == 0 ? splits[c.at].bound : blocks[c.at].bound;
  };
  auto cost = [&](const Cand& c) {
    return c.kind == 0 ? ch.Cost(splits[c.at]) : ch.Cost(blocks[c.at]);
  };
  auto build = [&](const Cand& c) {
    return c.kind == 0 ? ch.Build(splits[c.at]) : ch.Build(blocks[c.at]);
  };
  auto name = [&](const Cand& c) { return c.kind == 0 ? splits[c.at].name : blocks[c.at].name; };
  i64 limit = (100 - P[kMargin]) * cost(cands[ref_at]);
  Vec<std::pair<i64, size_t>> fast;
  for (size_t i = 0; i < cands.size(); ++i) {
    if (i == ref_at || 100 * bound(cands[i]) >= limit) continue;
    i64 c = cost(cands[i]);
    if (100 * c < limit) fast.push_back({c, i});
  }
  std::sort(fast.begin(), fast.end());
  Plan ref_plan;
  bool have_ref = false;
  Chooser::Prepared ref_prep;
  Vec<i64> ref_needs;
  Vec<std::pair<i64, i64>> ref_times;
  for (const auto& [c, i] : fast) {
    Plan plan = build(cands[i]);
    if (!have_ref) {
      ref_plan = build(cands[ref_at]);
      ch.Prepare(ref_plan, &ref_prep, &ref_needs);
      have_ref = true;
    }
    if (ch.Robust(plan, ref_prep, ref_needs, &ref_times)) return {std::move(plan), name(cands[i])};
  }
  if (!have_ref) ref_plan = build(cands[ref_at]);
  return {std::move(ref_plan), name(cands[ref_at])};
}

Vec<i64> ReadInts(const tvm::ffi::Bytes& b, int64_t width, const char* what) {
  TVM_FFI_ICHECK(width == 4 || width == 8) << what << ": element width 4 or 8, got " << width;
  TVM_FFI_ICHECK(b.size() % width == 0) << what << ": " << b.size() << " bytes is not a multiple of " << width;
  Vec<i64> out(b.size() / width);
  for (size_t i = 0; i < out.size(); ++i) {
    if (width == 8) {
      std::memcpy(&out[i], b.data() + 8 * i, 8);
    } else {
      int32_t v;
      std::memcpy(&v, b.data() + 4 * i, 4);
      out[i] = v;
    }
  }
  return out;
}

// The launch of one batch: ``offsets`` the ``N + 1`` host offsets (``width``-byte integers), ``params`` the int64
// parameter words, ``flags`` bit 0/1: the split build reads CTA offsets / the planned offsets (csr / plan_guard), bit
// 2/3 the same for the block build.  Returns int32 words: ``(candidate, split, grid, rows, words)``, then the
// ``words`` int32 words of the program's ``items`` formal (``v6_item_words``).
tvm::ffi::Bytes V6Launch(tvm::ffi::Bytes offsets, int64_t width, int64_t num_heads, int64_t num_householder,
                         int64_t sm_count, int64_t total_tokens, tvm::ffi::Bytes params, int64_t flags) {
  tls_arena.Reset();
  Params P{ReadInts(params, 8, "params")};
  TVM_FFI_ICHECK(static_cast<i64>(P.w.size()) >= kSpecs && P.w.size() == static_cast<size_t>(kSpecs + 2 * P[kNumSpecs]))
      << "cake_gdp_v6_plan: " << P.w.size() << " parameter words do not match the layout";
  Vec<i64> cu = ReadInts(offsets, width, "offsets");
  TVM_FFI_ICHECK(cu.size() >= 2) << "cake_gdp_v6_plan: at least two offsets";
  TVM_FFI_ICHECK(sm_count > 0 && num_heads > 0 && num_householder > 0)
      << "cake_gdp_v6_plan: positive heads, householder steps and SM count";
  Batch bt;
  bt.H = num_heads;
  bt.n = num_householder;
  bt.sm = sm_count;
  bt.total = total_tokens;
  i64 N = cu.size() - 1, total = total_tokens;
  i64 start = std::min(std::max<i64>(cu[0], 0), total);
  for (i64 q = 0; q < N; ++q) {
    i64 raw = cu[q + 1];
    i64 end = std::min(std::max(raw, start), total);
    i64 len = end - start;
    bt.lens.push_back(len);
    bt.by_len[len].push_back(q);
    start = std::min(std::max<i64>(raw, 0), total);
  }
  for (const auto& [len, seqs] : bt.by_len) {
    i64 c = SplitChunks(len, bt.n);
    bt.chunk_of[len] = c;
    bt.distinct.push_back(c);
  }
  std::sort(bt.distinct.begin(), bt.distinct.end());
  bt.distinct.erase(std::unique(bt.distinct.begin(), bt.distinct.end()), bt.distinct.end());
  for (i64 len : bt.lens) bt.chunks.push_back(bt.chunk_of[len]);
  bt.empty = bt.by_len.count(0) > 0;
  Choice choice = Choose(P, bt);
  const Plan& plan = choice.plan;
  TVM_FFI_ICHECK(plan.grid >= 1 && plan.grid <= sm_count)
      << "cake_gdp_v6_plan: grid " << plan.grid << " outside 1.." << sm_count;
  bool csr = flags & (plan.split ? 1 : 4), guard = flags & (plan.split ? 2 : 8);
  Vec<int32_t> words;
  words.reserve(5 + 4 * std::max<size_t>(plan.rows.size(), 1) + (csr ? plan.offsets.size() : 0) +
                (guard ? cu.size() + 1 : 0));
  words.resize(5);
  if (plan.rows.empty()) words.insert(words.end(), 4, 0);
  for (const Row& r : plan.rows)
    for (i64 v : r) words.push_back(static_cast<int32_t>(v));
  if (csr) {
    TVM_FFI_ICHECK(!plan.offsets.empty()) << "cake_gdp_v6_plan: the csr build runs a plan with row offsets";
    for (i64 o : plan.offsets) words.push_back(static_cast<int32_t>(o));
  }
  if (guard) {
    words.push_back(static_cast<int32_t>(N));
    for (i64 c : cu) words.push_back(static_cast<int32_t>(c));
  }
  words[0] = static_cast<int32_t>(choice.name);
  words[1] = plan.split ? 1 : 0;
  words[2] = static_cast<int32_t>(plan.grid);
  words[3] = static_cast<int32_t>(plan.rows.size());
  words[4] = static_cast<int32_t>(words.size() - 5);
  return tvm::ffi::Bytes(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(int32_t));
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(v6_launch, V6Launch);
