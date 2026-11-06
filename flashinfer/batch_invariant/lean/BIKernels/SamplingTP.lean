import BIKernels.Sampling
/-!
# Tensor-parallel partials of `lse.row` (`sampling/shards.py`; `README.md`, Sampling: the tensor-parallel protocol)

The row trees of the class are defined on the global vocabulary: digits `[kb, 5]`, the block tree (bits 0 .. kb-1) of
every lane under the lane tree (bits kb .. kb+4). A rank holding the slot-aligned shard `[lo, hi)` sees, for lane `l`,
the contiguous block range `[a, b) = laneRange lo hi l`; its partials are the **maximal in-shard sub-trees** of the
lane's block tree (`isPiece`: in shard, parent not in shard), i.e. the dyadic decomposition of `[a, b)`. This file proves

* `mergeNode_eq_subtree`: rebuilding a block tree from any family of *given* nodes that carry the tree's own sub-trees,
  combining the children of the nodes not given and dropping absent children, yields the tree -- provided every
  present leaf lies under a given node (`h0`);
* `tiling_cover` / `h0_of_ranges`: for shards that tile `[0, N)` on slot boundaries (`tiling`), every present block of
  every lane lies under a piece of some rank (the climb from the block to its maximal in-shard ancestor);
* **`tpMax_eq_specMax` / `tpSum_eq_specSum`**: the merge of the ranks' pieces, followed by the lane tree, IS the
  spec's maximum / sum term (term identity, not only `Equiv`: no commutativity assumption is needed);
* **`tp_protocol`**: the two rounds -- round 2's partials are the sum tree's pieces over `ex2((x - m) c)` with the
  MERGED `m` -- give the spec's `(m, s)` (`specSumOfMax`), hence the spec's outputs (`tpOutputs_eq_specOutputs`);
* `piece_via_units`: a piece computed by the kernel across units (the engine's reg / warp / cluster factors of the
  block digit: a tree over an aligned group of chunk roots) is the sub-tree (`treeC_split`).

What the kernel does beyond this model is plumbing: the slot folds are `canon.init_state` (the model's `slotMax` /
`slotSum`), the pieces are stored at the slots of `ShardLayout` (the generated cases decide the production layouts
against the Python tables), the merge kernel executes `merge_ops` (the recursion `mergeNode` linearised per lane class,
checked node for node by the CPU suite).
-/
namespace BIK
namespace SamplingTP
open Sampling

/-! ## Bits -/

/-- The `n`-bit little-endian representation of `v` (the inverse of `bitsToNat` below `2^n`). -/
def natToBits : Nat → Nat → List Bool
  | 0, _ => []
  | n+1, v => (v % 2 == 1) :: natToBits n (v / 2)

theorem natToBits_length : ∀ n v, (natToBits n v).length = n
  | 0, _ => rfl
  | n+1, v => congrArg (· + 1) (natToBits_length n (v / 2))

theorem natToBits_double (n i : Nat) : natToBits (n+1) (2*i) = false :: natToBits n i := by
  have h1 : 2 * i % 2 = 0 := Nat.mul_mod_right 2 i
  have h2 : 2 * i / 2 = i := Nat.mul_div_cancel_left i (by decide)
  rw [natToBits, h1, h2]
  rfl

theorem natToBits_double_succ (n i : Nat) : natToBits (n+1) (2*i+1) = true :: natToBits n i := by
  have h1 : (2 * i + 1) % 2 = 1 := by omega
  have h2 : (2 * i + 1) / 2 = i := by omega
  rw [natToBits, h1, h2]
  rfl

theorem bitsToNat_append : ∀ (u v : List Bool), bitsToNat (u ++ v) = bitsToNat u + 2 ^ u.length * bitsToNat v
  | [], v => by simp [bitsToNat]
  | b :: u, v => by
      rw [List.cons_append]
      simp only [bitsToNat, List.length_cons, Nat.pow_succ]
      rw [bitsToNat_append u v, Nat.mul_right_comm, Nat.mul_comm (2 ^ u.length * bitsToNat v) 2]
      omega

theorem bitsToNat_natToBits : ∀ n v, v < 2 ^ n → bitsToNat (natToBits n v) = v
  | 0, v, h => by
      simp only [natToBits, bitsToNat]
      omega
  | n+1, v, h => by
      simp only [natToBits, bitsToNat]
      have ih := bitsToNat_natToBits n (v / 2) (by rw [Nat.pow_succ] at h; omega)
      rw [ih]
      rcases Nat.mod_two_eq_zero_or_one v with h2 | h2 <;> simp [h2] <;> omega

theorem bitsToNat_lt_pow : ∀ l : List Bool, bitsToNat l < 2 ^ l.length
  | [] => by simp [bitsToNat]
  | b :: bs => by
      simp only [bitsToNat, List.length_cons, Nat.pow_succ]
      have := bitsToNat_lt_pow bs
      cases b <;> simp <;> omega

theorem takeN_append_left' (u v : List Bool) (n : Nat) (h : u.length = n) : takeN n (u ++ v) = u := by
  subst h
  exact takeN_append_left_aux u v
where
  takeN_append_left_aux : ∀ (u v : List Bool), takeN u.length (u ++ v) = u
    | [], _ => rfl
    | b :: u, v => by
        show b :: takeN u.length (u ++ v) = b :: u
        rw [takeN_append_left_aux u v]

theorem dropN_append_left' (u v : List Bool) (n : Nat) (h : u.length = n) : dropN n (u ++ v) = v := by
  subst h
  exact dropN_append_left_aux u v
where
  dropN_append_left_aux : ∀ (u v : List Bool), dropN u.length (u ++ v) = v
    | [], _ => rfl
    | _ :: u, v => dropN_append_left_aux u v

/-- Extensionality of `treeC` on the leaves it reaches (bit lists of the tree's length). -/
theorem treeC_ext_len (c : Comb) : ∀ (n : Nat) {f g : List Bool → Option Term},
    (∀ bs, bs.length = n → f bs = g bs) → treeC c n f = treeC c n g
  | 0, f, g, h => h [] rfl
  | n+1, f, g, h => by
      simp only [treeC]
      rw [treeC_ext_len c n (fun bs hb => h (bs ++ [false]) (by simp [hb])),
          treeC_ext_len c n (fun bs hb => h (bs ++ [true]) (by simp [hb]))]

/-! ## Sub-trees of one lane's block tree -/

/-- The sub-tree of the block tree at level `L`, index `i` (blocks `[i 2^L, (i+1) 2^L)`), over the lane's block leaf
`leafL` (block bits of length `kb`). -/
def subtree (c : Comb) (kb : Nat) (leafL : List Bool → Option Term) (L i : Nat) : Option Term :=
  treeC c L (fun lo => leafL (lo ++ natToBits (kb - L) i))

theorem subtree_zero (c : Comb) (kb : Nat) (leafL : List Bool → Option Term) (i : Nat) :
    subtree c kb leafL 0 i = leafL (natToBits kb i) := by
  simp only [subtree, treeC, Nat.sub_zero, List.nil_append]

theorem subtree_succ (c : Comb) (kb : Nat) (leafL : List Bool → Option Term) (L i : Nat) (h : L + 1 ≤ kb) :
    subtree c kb leafL (L+1) i = ocomb c (subtree c kb leafL L (2*i)) (subtree c kb leafL L (2*i+1)) := by
  simp only [subtree, treeC]
  have e : kb - L = (kb - (L + 1)) + 1 := by omega
  rw [e, natToBits_double, natToBits_double_succ]
  congr 1 <;> apply treeC_ext <;> intro bs <;> simp [List.append_assoc]

theorem subtree_top (c : Comb) (kb : Nat) (leafL : List Bool → Option Term) :
    subtree c kb leafL kb 0 = treeC c kb leafL := by
  simp only [subtree, Nat.sub_self, natToBits]
  apply treeC_ext
  intro lo
  rw [List.append_nil]

/-- **A piece through the unit hierarchy.** The kernel forms a piece at level `L` as the tree over the aligned group of
its `2^(L-g)` sub-pieces at level `g` (chunk roots after the warp / cluster exchange, `candidate_nodes_cta` /
`candidate_nodes_cluster`): that is the piece (`treeC_split`). -/
theorem piece_via_units (c : Comb) (kb : Nat) (leafL : List Bool → Option Term) (g L i : Nat) (h : g ≤ L) :
    treeC c (L - g) (fun hi => treeC c g (fun lo => leafL (lo ++ hi ++ natToBits (kb - L) i))) = subtree c kb leafL L i := by
  unfold subtree
  have e : L = g + (L - g) := by omega
  have hs := treeC_split c g (L - g) (fun lo => leafL (lo ++ natToBits (kb - L) i))
  rw [← e] at hs
  exact hs.symm

/-! ## The merge of given nodes -/

/-- Rebuild the block tree from the nodes marked `isGiven` with values `val` (a rank's pieces at their slots): a given
node is taken as is, any other node is the combine of its children, absent children dropped
(`merge_partials.block_node`, `merge_ops`). -/
def mergeNode (c : Comb) (isGiven : Nat → Nat → Bool) (val : Nat → Nat → Option Term) : Nat → Nat → Option Term
  | 0, i => if isGiven 0 i then val 0 i else none
  | L+1, i =>
    if isGiven (L+1) i then val (L+1) i
    else ocomb c (mergeNode c isGiven val L (2*i)) (mergeNode c isGiven val L (2*i+1))

/-- A proper ancestor of `(L, i)` is given. -/
def Covered (isGiven : Nat → Nat → Bool) (L i : Nat) : Prop :=
  ∃ d, 0 < d ∧ isGiven (L + d) (i / 2 ^ d) = true

theorem div_pow_succ (i d : Nat) : i / 2 ^ (d+1) = i / 2 / 2 ^ d := by
  rw [Nat.div_div_eq_div_mul, Nat.pow_succ, Nat.mul_comm]

theorem not_covered_child (isGiven : Nat → Nat → Bool) (L i j : Nat) (hj : j / 2 = i)
    (hn : isGiven (L+1) i = false) (hc : ¬ Covered isGiven (L+1) i) : ¬ Covered isGiven L j := by
  intro ⟨d, hd, hv⟩
  cases d with
  | zero => exact absurd hd (Nat.lt_irrefl 0)
  | succ d' =>
      rw [div_pow_succ, hj] at hv
      cases d' with
      | zero =>
          simp only [Nat.pow_zero, Nat.div_one] at hv
          rw [hn] at hv
          exact Bool.noConfusion hv
      | succ d'' =>
          apply hc
          refine ⟨d'' + 1, Nat.succ_pos _, ?_⟩
          have e : L + 1 + (d'' + 1) = L + (d'' + 1 + 1) := by omega
          rw [e]
          exact hv

/-- **Merging given sub-trees rebuilds the tree.** If every given node carries the sub-tree at its position and every
uncovered, ungiven leaf is absent, then at every uncovered node of the tree the merge equals the sub-tree (the
recursion only reaches uncovered nodes). -/
theorem mergeNode_eq_subtree (c : Comb) (kb : Nat) (leafL : List Bool → Option Term) (isGiven : Nat → Nat → Bool)
    (val : Nat → Nat → Option Term)
    (hg : ∀ L i, isGiven L i = true → val L i = subtree c kb leafL L i)
    (h0 : ∀ i, i < 2 ^ kb → ¬ Covered isGiven 0 i → isGiven 0 i = false → leafL (natToBits kb i) = none) :
    ∀ L, L ≤ kb → ∀ i, i < 2 ^ (kb - L) → ¬ Covered isGiven L i → mergeNode c isGiven val L i = subtree c kb leafL L i
  | 0, _, i, hi, hc => by
      simp only [mergeNode]
      cases hgi : isGiven 0 i with
      | true => rw [if_pos rfl]; exact hg 0 i hgi
      | false =>
          rw [if_neg (by simp), subtree_zero]
          exact (h0 i (by simpa using hi) hc hgi).symm
  | L+1, hL, i, hi, hc => by
      simp only [mergeNode]
      cases hgi : isGiven (L+1) i with
      | true => rw [if_pos rfl]; exact hg (L+1) i hgi
      | false =>
          rw [if_neg (by simp), subtree_succ c kb leafL L i hL]
          have hi' : 2 * i + 1 < 2 ^ (kb - L) := by
            have e : kb - L = (kb - (L + 1)) + 1 := by omega
            rw [e, Nat.pow_succ]
            omega
          rw [mergeNode_eq_subtree c kb leafL isGiven val hg h0 L (Nat.le_of_succ_le hL) (2*i) (by omega)
                (not_covered_child isGiven L i (2*i) (by omega) hgi hc),
              mergeNode_eq_subtree c kb leafL isGiven val hg h0 L (Nat.le_of_succ_le hL) (2*i+1) hi'
                (not_covered_child isGiven L i (2*i+1) (by omega) hgi hc)]

/-! ## The shards' pieces -/

/-- Node `(L, i)` lies inside the lane's block range `[a, b)`. -/
def inShard (a b L i : Nat) : Bool := decide (a ≤ i * 2 ^ L) && decide ((i + 1) * 2 ^ L ≤ b)

/-- A maximal in-shard node: in shard, and its parent is not (or it is the root). -/
def isPiece (kb a b L i : Nat) : Bool := inShard a b L i && (decide (L = kb) || !inShard a b (L+1) (i / 2))

/-- Some rank's piece. -/
def anyPiece (kb : Nat) (ranges : List (Nat × Nat)) (L i : Nat) : Bool := ranges.any (fun r => isPiece kb r.1 r.2 L i)

/-- The merge of the ranks' pieces of one lane (the pieces carry the sub-tree values the kernel stores). -/
def tpMerge (c : Comb) (kb : Nat) (leafL : List Bool → Option Term) (ranges : List (Nat × Nat)) : Option Term :=
  mergeNode c (anyPiece kb ranges) (subtree c kb leafL) kb 0

/-- The climb: an in-shard node has a maximal in-shard ancestor (itself included) within the tree. -/
theorem climb (kb a b : Nat) : ∀ n L i, kb - L = n → L ≤ kb → inShard a b L i = true →
    ∃ d, L + d ≤ kb ∧ isPiece kb a b (L + d) (i / 2 ^ d) = true
  | 0, L, i, hn, hL, hin => by
      have hLk : L = kb := by omega
      subst hLk
      refine ⟨0, by omega, ?_⟩
      simp only [isPiece, Nat.add_zero, Nat.pow_zero, Nat.div_one, hin, decide_true, Bool.true_or, Bool.and_true]
  | n+1, L, i, hn, hL, hin => by
      cases hp : inShard a b (L+1) (i / 2) with
      | true =>
          obtain ⟨d, hd, hpc⟩ := climb kb a b n (L+1) (i / 2) (by omega) (by omega) hp
          refine ⟨d + 1, by omega, ?_⟩
          have e1 : L + (d + 1) = L + 1 + d := by omega
          rw [e1, div_pow_succ]
          exact hpc
      | false =>
          refine ⟨0, by omega, ?_⟩
          simp only [isPiece, Nat.add_zero, Nat.pow_zero, Nat.div_one, hin, hp, Bool.not_false, Bool.or_true, Bool.and_true]

/-- **The cover.** If every present block of the lane lies in some rank's range, then an uncovered, ungiven leaf is
absent: a present block in a range climbs to a maximal in-shard ancestor, which is a piece -- the block itself (then
it is given) or a proper ancestor (then the block is covered). -/
theorem h0_of_ranges (kb : Nat) (leafL : List Bool → Option Term) (ranges : List (Nat × Nat)) (i : Nat)
    (hpres : leafL (natToBits kb i) ≠ none → ∃ r ∈ ranges, r.1 ≤ i ∧ i < r.2) :
    ¬ Covered (anyPiece kb ranges) 0 i → anyPiece kb ranges 0 i = false → leafL (natToBits kb i) = none := by
  intro hc hn
  by_cases hleaf : leafL (natToBits kb i) = none
  · exact hleaf
  obtain ⟨r, hr, ha, hb⟩ := hpres hleaf
  have hin : inShard r.1 r.2 0 i = true := by
    simp only [inShard, Nat.pow_zero, Nat.mul_one, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨ha, hb⟩
  obtain ⟨d, hd, hpc⟩ := climb kb r.1 r.2 (kb - 0) 0 i rfl (Nat.zero_le _) hin
  have hany : anyPiece kb ranges (0 + d) (i / 2 ^ d) = true :=
    List.any_eq_true.mpr ⟨r, hr, hpc⟩
  cases d with
  | zero =>
      simp only [Nat.add_zero, Nat.pow_zero, Nat.div_one] at hany
      rw [hany] at hn
      exact Bool.noConfusion hn
  | succ d' => exact absurd ⟨d' + 1, Nat.succ_pos _, hany⟩ hc

/-- Above the root nothing is in shard when the lane's range ends within the tree (`b ≤ 2^kb`). -/
theorem root_not_covered (kb : Nat) (ranges : List (Nat × Nat)) (hb : ∀ r ∈ ranges, r.2 ≤ 2 ^ kb) :
    ¬ Covered (anyPiece kb ranges) kb 0 := by
  intro ⟨d, hd, hv⟩
  obtain ⟨r, hr, hp⟩ := List.any_eq_true.mp hv
  simp only [isPiece, Bool.and_eq_true] at hp
  have hin := hp.1
  simp only [inShard, Nat.zero_div, Nat.zero_mul, Nat.zero_add, Nat.one_mul, Bool.and_eq_true, decide_eq_true_eq] at hin
  have h2 : 2 ^ (kb + 1) ≤ 2 ^ (kb + d) := Nat.pow_le_pow_right (by decide) (by omega)
  rw [Nat.pow_succ] at h2
  have := hb r hr
  have hpos := Nat.two_pow_pos kb
  omega

/-! ## The shards of a vocabulary: lane ranges and the tiling -/

/-- Lane `lane`'s block range of the shard `[lo, hi)` (`shards.lane_block_range`; a partial last slot counts). -/
def laneRange (lo hi lane : Nat) : Nat × Nat := ((lo / 8 + 31 - lane) / 32, ((hi + 7) / 8 + 31 - lane) / 32)

def laneRanges (bounds : List (Nat × Nat)) (lane : Nat) : List (Nat × Nat) := bounds.map (fun p => laneRange p.1 p.2 lane)

/-- The shards tile `[pos, N)` contiguously on slot boundaries, in rank order (the last may end at a partial slot). -/
def tiling : List (Nat × Nat) → Nat → Nat → Bool
  | [], pos, N => pos == N
  | p :: rest, pos, N => (p.1 == pos) && decide (p.1 ≤ p.2) && (p.1 % 8 == 0) && (p.2 % 8 == 0 || p.2 == N) && tiling rest p.2 N

theorem tiling_cover : ∀ (bounds : List (Nat × Nat)) (pos N lane i : Nat), tiling bounds pos N = true → lane < 32 → pos % 8 = 0 →
    (pos / 8 + 31 - lane) / 32 ≤ i → (i * 32 + lane) * 8 < N → ∃ r ∈ laneRanges bounds lane, r.1 ≤ i ∧ i < r.2
  | [], pos, N, lane, i, ht, hl, hp, hi, hN => by
      simp only [tiling, beq_iff_eq] at ht
      subst ht
      exfalso
      omega
  | p :: rest, pos, N, lane, i, ht, hl, hp, hi, hN => by
      simp only [tiling, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.or_eq_true] at ht
      obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := ht
      by_cases hb : i < ((p.2 + 7) / 8 + 31 - lane) / 32
      · refine ⟨laneRange p.1 p.2 lane, List.mem_map.mpr ⟨p, List.mem_cons_self _ _, rfl⟩, ?_, hb⟩
        show (p.1 / 8 + 31 - lane) / 32 ≤ i
        rw [h1]
        exact hi
      · cases h4 with
        | inl h8 =>
            obtain ⟨r, hr, hr1, hr2⟩ := tiling_cover rest p.2 N lane i h5 hl h8 (by omega) hN
            exact ⟨r, List.mem_cons_of_mem _ hr, hr1, hr2⟩
        | inr hN' =>
            exfalso
            subst hN'
            omega

theorem laneRange_le (lo hi lane N kb : Nat) (hhi : hi ≤ N) (hN : N ≤ 2 ^ kb * 256) (hl : lane < 32) :
    (laneRange lo hi lane).2 ≤ 2 ^ kb := by
  show ((hi + 7) / 8 + 31 - lane) / 32 ≤ 2 ^ kb
  omega

theorem tiling_hi_le : ∀ (bounds : List (Nat × Nat)) (pos N : Nat), tiling bounds pos N = true → ∀ p ∈ bounds, p.2 ≤ N
  | [], _, _, _, _, h => absurd h (List.not_mem_nil _)
  | q :: rest, pos, N, ht, p, hp => by
      simp only [tiling, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.or_eq_true] at ht
      obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := ht
      cases List.mem_cons.mp hp with
      | inl e =>
          subst e
          cases rest with
          | nil => simp only [tiling, beq_iff_eq] at h5; omega
          | cons r rest' =>
              have := tiling_hi_le (r :: rest') p.2 N h5 r (List.mem_cons_self _ _)
              simp only [tiling, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h5
              omega
      | inr hm => exact tiling_hi_le rest q.2 N h5 p hm

/-! ## The class's trees -/

/-- The filter of a slot's present elements is non-empty exactly when the slot starts inside the row. -/
theorem slot_filter_present (N s : Nat) : ((List.range 8).filter (fun j => decide (s * 8 + j < N))) ≠ [] → s * 8 < N := by
  intro h
  cases hf : (List.range 8).filter (fun j => decide (s * 8 + j < N)) with
  | nil => exact absurd hf h
  | cons j js =>
      have hj : j ∈ (List.range 8).filter (fun j => decide (s * 8 + j < N)) := by rw [hf]; exact List.mem_cons_self _ _
      have := (List.mem_filter.mp hj).2
      simp only [decide_eq_true_eq] at this
      omega

theorem slotMax_present (N : Nat) (row : Nat → Term) (s : Nat) : Sampling.slotMax N row s ≠ none → s * 8 < N := by
  intro h
  apply slot_filter_present N s
  intro hf
  apply h
  unfold Sampling.slotMax
  simp only [hf]

theorem slotSum_present (N : Nat) (row : Nat → Term) (nm c : Term) (s : Nat) : Sampling.slotSum N row nm c s ≠ none → s * 8 < N := by
  intro h
  apply slot_filter_present N s
  intro hf
  apply h
  unfold Sampling.slotSum
  simp only [hf]

/-- Lane `v`'s block leaf of the maximum (block bits `u`, then the lane bits). -/
def laneLeafMax (N kb : Nat) (row : Nat → Term) (v : List Bool) : List Bool → Option Term := fun u => maxLeaf N kb row (u ++ v)
def laneLeafSum (N kb : Nat) (row : Nat → Term) (nm c : Term) (v : List Bool) : List Bool → Option Term := fun u => sumLeaf N kb row nm c (u ++ v)

theorem laneLeafMax_present (N kb : Nat) (row : Nat → Term) (v : List Bool) (i : Nat) (hi : i < 2 ^ kb) :
    laneLeafMax N kb row v (natToBits kb i) ≠ none → (i * 32 + bitsToNat v) * 8 < N := by
  intro h
  simp only [laneLeafMax, maxLeaf] at h
  have hl := natToBits_length kb i
  rw [takeN_append_left' _ _ _ hl, dropN_append_left' _ _ _ hl, bitsToNat_natToBits kb i hi] at h
  exact slotMax_present N row _ h

theorem laneLeafSum_present (N kb : Nat) (row : Nat → Term) (nm c : Term) (v : List Bool) (i : Nat) (hi : i < 2 ^ kb) :
    laneLeafSum N kb row nm c v (natToBits kb i) ≠ none → (i * 32 + bitsToNat v) * 8 < N := by
  intro h
  simp only [laneLeafSum, sumLeaf] at h
  have hl := natToBits_length kb i
  rw [takeN_append_left' _ _ _ hl, dropN_append_left' _ _ _ hl, bitsToNat_natToBits kb i hi] at h
  exact slotSum_present N row nm c _ h

/-- The merged maximum: every lane's block root rebuilt from the ranks' pieces, then the lane tree. -/
def tpMax (N kb : Nat) (row : Nat → Term) (bounds : List (Nat × Nat)) : Option Term :=
  treeC .fmaxc 5 (fun v => tpMerge .fmaxc kb (laneLeafMax N kb row v) (laneRanges bounds (bitsToNat v)))

/-- The merged sum (round 2) over the given negated maximum and scale. -/
def tpSum (N kb : Nat) (row : Nat → Term) (nm c : Term) (bounds : List (Nat × Nat)) : Option Term :=
  treeC .add 5 (fun v => tpMerge .add kb (laneLeafSum N kb row nm c v) (laneRanges bounds (bitsToNat v)))

theorem spec_lanes (c : Comb) (kb : Nat) (leaf : List Bool → Option Term) :
    canonicalDigitsC c [kb, 5] leaf = treeC c 5 (fun v => treeC c kb (fun u => leaf (u ++ v))) := by
  simp only [canonicalDigitsC, shrinkC]
  apply treeC_ext; intro v; apply treeC_ext; intro u
  rw [List.append_nil]

/-- One lane: the merge of the pieces equals the lane's block root. -/
theorem tpMerge_eq_root (c : Comb) (N kb : Nat) (leafL : List Bool → Option Term) (bounds : List (Nat × Nat)) (lane : Nat)
    (ht : tiling bounds 0 N = true) (hN : N ≤ 2 ^ kb * 256) (hl : lane < 32)
    (hpres : ∀ i, i < 2 ^ kb → leafL (natToBits kb i) ≠ none → (i * 32 + lane) * 8 < N) :
    tpMerge c kb leafL (laneRanges bounds lane) = treeC c kb leafL := by
  unfold tpMerge
  rw [← subtree_top c kb leafL]
  apply mergeNode_eq_subtree c kb leafL (anyPiece kb (laneRanges bounds lane)) (subtree c kb leafL) (fun _ _ _ => rfl)
  · intro i hi hc hn
    apply h0_of_ranges kb leafL (laneRanges bounds lane) i _ hc hn
    intro hp
    exact tiling_cover bounds 0 N lane i ht hl (by decide) (by omega) (hpres i hi hp)
  · exact Nat.le_refl kb
  · simp
  · apply root_not_covered
    intro r hr
    obtain ⟨p, hp, e⟩ := List.mem_map.mp hr
    rw [← e]
    exact laneRange_le p.1 p.2 lane N kb (tiling_hi_le bounds 0 N ht p hp) hN hl

/-- **The merged maximum is the spec's maximum** (term identity). -/
theorem tpMax_eq_specMax (N kb : Nat) (row : Nat → Term) (bounds : List (Nat × Nat))
    (ht : tiling bounds 0 N = true) (hN : N ≤ 2 ^ kb * 256) :
    tpMax N kb row bounds = specMax N kb row := by
  unfold tpMax specMax
  rw [spec_lanes]
  apply treeC_ext_len
  intro v hv
  have hl : bitsToNat v < 32 := by
    have := bitsToNat_lt_pow v
    rw [hv] at this
    exact this
  exact tpMerge_eq_root .fmaxc N kb (laneLeafMax N kb row v) bounds (bitsToNat v) ht hN hl
    (fun i hi hp => laneLeafMax_present N kb row v i hi hp)

/-- **The merged sum is the spec's sum** for any negated maximum and scale (term identity). -/
theorem tpSum_eq_specSum (N kb : Nat) (row : Nat → Term) (nm c : Term) (bounds : List (Nat × Nat))
    (ht : tiling bounds 0 N = true) (hN : N ≤ 2 ^ kb * 256) :
    tpSum N kb row nm c bounds = specSum N kb row nm c := by
  unfold tpSum specSum
  rw [spec_lanes]
  apply treeC_ext_len
  intro v hv
  have hl : bitsToNat v < 32 := by
    have := bitsToNat_lt_pow v
    rw [hv] at this
    exact this
  exact tpMerge_eq_root .add N kb (laneLeafSum N kb row nm c v) bounds (bitsToNat v) ht hN hl
    (fun i hi hp => laneLeafSum_present N kb row nm c v i hi hp)

/-! ## The two-round protocol -/

/-- Round 2 over the merged maximum of round 1. -/
def tpSumOfMax (N kb : Nat) (row : Nat → Term) (T : Option Term) (bounds : List (Nat × Nat)) : Option Term :=
  match tpMax N kb row bounds with
  | none => none
  | some m => tpSum N kb row (.neg m) (scale T) bounds

/-- The outputs every rank builds from the merged statistics (`log_probs_sharded`, `log_softmax_sharded`, `softmax_sharded`). -/
def tpOutputs (fmt : String) (N kb : Nat) (row : Nat → Term) (T : Option Term) (bounds : List (Nat × Nat)) : Option Outputs :=
  match tpMax N kb row bounds with
  | none => none
  | some m =>
    match tpSum N kb row (.neg m) (scale T) bounds with
    | none => none
    | some s => some (outputsOf fmt row m s (scale T))

/-- **The protocol computes the spec's statistics**: the merged sum over the merged maximum is the spec's sum over the
spec's maximum. -/
theorem tp_protocol (N kb : Nat) (row : Nat → Term) (T : Option Term) (bounds : List (Nat × Nat))
    (ht : tiling bounds 0 N = true) (hN : N ≤ 2 ^ kb * 256) :
    tpSumOfMax N kb row T bounds = specSumOfMax N kb row T := by
  unfold tpSumOfMax specSumOfMax
  rw [tpMax_eq_specMax N kb row bounds ht hN]
  cases specMax N kb row with
  | none => rfl
  | some m => exact tpSum_eq_specSum N kb row (.neg m) (scale T) bounds ht hN

/-- **Every output of every rank is the spec's output** (term identity, hence bitwise equal under every interpretation). -/
theorem tpOutputs_eq_specOutputs (fmt : String) (N kb : Nat) (row : Nat → Term) (T : Option Term) (bounds : List (Nat × Nat))
    (ht : tiling bounds 0 N = true) (hN : N ≤ 2 ^ kb * 256) :
    tpOutputs fmt N kb row T bounds = specOutputs fmt N kb row T := by
  unfold tpOutputs specOutputs
  rw [tpMax_eq_specMax N kb row bounds ht hN]
  cases specMax N kb row with
  | none => rfl
  | some m =>
      simp only [tpSum_eq_specSum N kb row (.neg m) (scale T) bounds ht hN]
      cases specSum N kb row (.neg m) (scale T) <;> rfl

/-! ## The layout as data (decided against the Python tables by the generated cases) -/

/-- The pieces of a lane's range, by level then index (`ShardLayout.pieces` sorted by `(level, idx)`). -/
def piecesOf (kb a b : Nat) : List (Nat × Nat) :=
  (List.range (kb + 1)).flatMap (fun L => ((List.range (2 ^ (kb - L))).filter (fun i => isPiece kb a b L i)).map (fun i => (L, i)))

/-- Every rank's every lane's pieces. -/
def layoutTable (kb : Nat) (bounds : List (Nat × Nat)) : List (List (List (Nat × Nat))) :=
  bounds.map (fun p => (List.range 32).map (fun l => piecesOf kb (laneRange p.1 p.2 l).1 (laneRange p.1 p.2 l).2))

end SamplingTP
end BIK
