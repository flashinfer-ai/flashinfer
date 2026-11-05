#pragma once
#include <type_traits>

namespace mla_swa {
enum class Reduce { Direct, GlobalFP32, DSM };
enum class Mailbox { None, RowPacked, NaturalFragment };
enum class PartialForm { Normalized, Raw };
enum class Protocol { RegisterSoftmax, HelperLookahead, AllWGSoftmax };
enum class Producer { QKWarpStreams, LookaheadOwner, BaseOwner };
enum class Arena { None, DeadKV, DeadQ };
enum class ReduceSync { None, Cluster, PerWGReadyAck };

template <int M_, int S_>
struct ShapePolicy {
  static constexpr int M = M_, S = S_, Page = 64, Dq = 1088, Dv = 1024, Slice = 64,
                       VSliceCount = 16;
};
template <int PV_, int QKPV_, bool Interleave_>
struct RolePolicy {
  static constexpr int PvWG = PV_, QkPV = QKPV_, Threads = (PV_ + 1) * 128;
  static constexpr bool Interleave = Interleave_;
};
template <int A, int B, bool Zero = false>
struct QKPair {
  static constexpr int first = A, second = B;
  static constexpr bool zero = Zero;
};
template <int A, bool Zero = false>
struct QKSingle {
  static constexpr int first = A, second = -1;
  static constexpr bool zero = Zero;
};
template <class... Ops>
struct QKSequence {
  static constexpr unsigned coverage =
      ((unsigned(1) << Ops::first | (Ops::second < 0 ? 0 : unsigned(1) << Ops::second)) | ...);
  static constexpr int tiles = ((Ops::second < 0 ? 1 : 2) + ...);
  static_assert(coverage == (1u << 17) - 1 && tiles == 17, "QK must cover 17 slices exactly once");
};
using M8QK = QKSequence<QKPair<16, 0, true>, QKPair<1, 6>, QKPair<11, 2>, QKPair<7, 12>,
                        QKPair<3, 8>, QKPair<13, 4>, QKPair<9, 14>, QKPair<5, 10>, QKSingle<15>>;
using M32QK = QKSequence<QKSingle<16, true>, QKPair<0, 5>, QKPair<10, 1>, QKPair<6, 11>,
                         QKPair<2, 7>, QKPair<12, 3>, QKPair<8, 13>, QKPair<4, 9>, QKPair<14, 15>>;
template <int K>
struct LookaheadQK {
  static constexpr int lookahead = K;
};
template <int M, int K>
struct SchedulePolicy {
  static constexpr int LookAhead = K;
  static constexpr Protocol Mainloop = M == 8    ? Protocol::RegisterSoftmax
                                       : M == 16 ? Protocol::HelperLookahead
                                                 : Protocol::AllWGSoftmax;
  static constexpr Producer TmaProducer = M == 8    ? Producer::QKWarpStreams
                                          : M == 16 ? Producer::LookaheadOwner
                                                    : Producer::BaseOwner;
  static constexpr bool StoreP0First = M == 16;
  using QKProgram =
      std::conditional_t<M == 8, M8QK, std::conditional_t<M == 32, M32QK, LookaheadQK<K>>>;
};
template <Reduce B, Mailbox L, PartialForm F, int M>
struct ReducePolicy {
  static constexpr Reduce Backend = B;
  static constexpr Mailbox Payload = L;
  static constexpr PartialForm PartialFormat = F;
  static constexpr Arena MailboxArena = B != Reduce::DSM ? Arena::None
                                        : M == 8         ? Arena::DeadKV
                                                         : Arena::DeadQ;
  static constexpr ReduceSync Synchronization = B != Reduce::DSM ? ReduceSync::None
                                                : M == 32        ? ReduceSync::PerWGReadyAck
                                                                 : ReduceSync::Cluster;
  static constexpr int VectorWords = B != Reduce::DSM ? 0 : M == 8 ? 2 : 4;
};
template <class Shape, class Roles, class Schedule, class Merge>
struct AttentionTactic : Shape, Roles, Schedule, Merge {
  using ShapeTraits = Shape;
  using RoleTraits = Roles;
  using ScheduleTraits = Schedule;
  using ReduceTraits = Merge;
  static constexpr bool DenseRowMailbox = Shape::M == 8 && Merge::Payload == Mailbox::RowPacked &&
                                          (Shape::VSliceCount - Roles::QkPV) % Roles::PvWG == 0;
  static constexpr int StorageKind = Shape::M == 8                   ? 0
                                     : Shape::M == 16                ? 1
                                     : Merge::Backend == Reduce::DSM ? 3
                                                                     : 2;
  static_assert((Merge::Backend == Reduce::Direct) == (Shape::S == 1));
  static_assert(
      (Shape::M == 8 && Roles::PvWG == 3 && Roles::QkPV == 1 && !Roles::Interleave &&
       Schedule::LookAhead == 0 &&
       ((Shape::S == 3 && Merge::Backend == Reduce::DSM && Merge::Payload == Mailbox::RowPacked &&
         Merge::PartialFormat == PartialForm::Raw) ||
        ((Shape::S == 5 || Shape::S == 9) && Merge::Backend == Reduce::GlobalFP32 &&
         Merge::Payload == Mailbox::None && Merge::PartialFormat == PartialForm::Normalized))) ||
          (Shape::M == 16 && Shape::S == 2 && Roles::PvWG == 4 && Roles::QkPV == 0 &&
           Roles::Interleave && Schedule::LookAhead == 4 && Merge::Backend == Reduce::DSM &&
           Merge::Payload == Mailbox::NaturalFragment &&
           Merge::PartialFormat == PartialForm::Raw) ||
          (Shape::M == 32 && Roles::PvWG == 3 && Roles::QkPV == 0 && !Roles::Interleave &&
           Schedule::LookAhead == 0 &&
           ((Shape::S == 1 && Merge::Backend == Reduce::Direct &&
             Merge::Payload == Mailbox::None) ||
            (Shape::S == 2 && Merge::Backend == Reduce::DSM &&
             Merge::Payload == Mailbox::NaturalFragment &&
             Merge::PartialFormat == PartialForm::Raw))),
      "Only the six frozen tactics are supported; not an arbitrary parameter product");
};
template <int M, int S, int PV, int QKPV, bool Interleave, int K, Reduce B, Mailbox L,
          PartialForm F = PartialForm::Normalized>
using KernelConfig = AttentionTactic<ShapePolicy<M, S>, RolePolicy<PV, QKPV, Interleave>,
                                     SchedulePolicy<M, K>, ReducePolicy<B, L, F, M>>;

template <class C>
struct PVOwnership {
  static constexpr int Remaining = C::VSliceCount - C::QkPV;
  static constexpr int MaxSlices = (Remaining + C::PvWG - 1) / C::PvWG;
  __host__ __device__ static constexpr int begin(int group) {
    return C::QkPV + (C::Interleave ? group : Remaining * group / C::PvWG);
  }
  __host__ __device__ static constexpr int end(int group) {
    return C::Interleave ? C::VSliceCount : C::QkPV + Remaining * (group + 1) / C::PvWG;
  }
  static constexpr int Stride = C::Interleave ? C::PvWG : 1;
  __host__ __device__ static constexpr int slice(int group, int local) {
    return begin(group) + local * Stride;
  }
};
template <class C>
struct RowOwnership {
  __host__ __device__ static constexpr int begin(int rank) { return C::M * rank / C::S; }
  __host__ __device__ static constexpr int end(int rank) { return C::M * (rank + 1) / C::S; }
};
template <class C>
struct EventTraits {
  static constexpr int Threads = C::Threads, WGThreads = 128, PVThreads = C::PvWG * 128;
  static constexpr int ScoreReady = C::M == 8 ? -1 : 8;
  static constexpr int PReady = C::M == 32 ? 9 : -1;
  static constexpr int PFree = C::M == 32 ? 10 : -1;
  static constexpr int HelperDone = C::M == 16 ? 9 : -1;
  // Named barrier numbers above are documentation contracts; the exact
  // instructions remain in the validated protocol primitives.
};
using M8S9 = KernelConfig<8, 9, 3, 1, false, 0, Reduce::GlobalFP32, Mailbox::None>;
using M8S5 = KernelConfig<8, 5, 3, 1, false, 0, Reduce::GlobalFP32, Mailbox::None>;
using M8S3 = KernelConfig<8, 3, 3, 1, false, 0, Reduce::DSM, Mailbox::RowPacked, PartialForm::Raw>;
using M16S2 =
    KernelConfig<16, 2, 4, 0, true, 4, Reduce::DSM, Mailbox::NaturalFragment, PartialForm::Raw>;
using M32S2 =
    KernelConfig<32, 2, 3, 0, false, 0, Reduce::DSM, Mailbox::NaturalFragment, PartialForm::Raw>;
using M32S1 = KernelConfig<32, 1, 3, 0, false, 0, Reduce::Direct, Mailbox::None>;
}  // namespace mla_swa
