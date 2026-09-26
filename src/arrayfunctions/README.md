`src/arrayfunctions/` — Sorting and duplicate removal {#mftl_mod_arrayfunctions}
=====================================================

**Public header:** `include/mftl_arrayfunctions.h` · **Layer:** 0 ·
**Module group:** @ref mftl_arrayfunctions

A stable least-significant-digit (LSD) radix sort for 32-bit signed and 64-bit
unsigned keys, with key-only and key/value (structure-of-arrays) variants, plus
inline helpers for duplicate removal and binary search. `radixsort.c` derives
from Cameron Hart's zlib-licensed radix sort (notice preserved in the file).

| Caller | Key | Purpose |
|---|---|---|
| `MFTLObjectReorderMorton` | 63-bit Morton code; min vertex index | space-filling-curve renumbering of the pools |
| `MFTL_GenerateLBVH` | 63-bit Morton code | the sorted order *is* the LBVH leaf order |
| `src/communication/redistribute.c` | vertex index | de-duplicate a destination's vertex list |
| `src/topology_change/` | body id | canonical body sets of a cluster |

**Stability is a correctness requirement.** Primitives with equal Morton codes
must come out in a deterministic order, or two runs build different trees; and
`MFTLUniquePairs` keeps the *first* value of each run of equal keys, which after
a stable sort is a rank-independent choice.

---

## Algorithm {#arr_algorithm}

An LSD radix sort performs $\lceil b/r \rceil$ stable counting-sort passes over
$n$ keys of $b$ bits, $r$ bits (one *digit*) at a time:

$$
  T(n) = O\!\left(\frac{b}{r}\,(n + 2^{r})\right), \qquad r = 8 .
$$

Each pass computes, per digit value $j$, the exclusive prefix sum
$o_j = \sum_{i<j} c_i$ of the digit histogram $c$ and scatters every key to
position $o_{d(k)}$, incrementing that offset. Scanning the input in order makes
each pass stable, so after the pass on digit $p$ the keys are sorted by digits
$0\ldots p$.

Three refinements, none of which changes the result:

1. **One counting pass for all digits.** `init_histograms32/64` read every key
   once and update all $b/r$ histograms (4 KiB / 8 KiB, resident in L1). The
   same pass tests whether the input is already sorted, in which case no
   scatter runs.
2. **Constant digits are skipped.** If the bucket of the first key's digit holds
   all $n$ keys, the digit is constant and the pass is the identity. Global ids
   are dense in $[0, n)$, so for fewer than 65 536 entities the two top bytes of
   a 32-bit id are constant and half the passes vanish. An odd number of
   executed passes ends in the scratch buffer and is copied back.
3. **Signed order for free.** For the most significant digit of an `int` key the
   prefix sum starts at bucket 128 and wraps:
   $o_{128} = 0,\ o_{129} = c_{128},\ \ldots,\ o_0 = \sum_{j=128}^{255} c_j$.
   Negative keys (top byte $\ge$ 0x80 in two's complement) therefore precede
   non-negative ones without any key being modified.

### Small inputs {#arr_cutoff}

At $n \le 32$ a stable insertion sort replaces the radix passes: the
$n$-independent costs (histogram clearing, 256-entry prefix sums per pass,
scratch allocation) exceed the $O(n^2/4)$ moves of an L1-resident insertion
sort. Measured (ns/call, Apple M-series, `-O3`):

| n | radix, random | insertion, random | radix, descending | insertion, descending |
|---|---|---|---|---|
| 16 | 577 | **78** | 277 | **75** |
| 32 | 749 | **496** | 323 | **287** |
| 48 | **859** | 866 | **369** | 639 |
| 64 | **973** | 1310 | **415** | 1299 |

32 is the largest size at which insertion sort wins on both random and
descending (its worst case) input. Redistribution sorts one small buffer per
neighbour rank, so the small path is the common one there.

---

## Interface notes {#arr_interface}

- `MFTLRadixSort(keys, temp, n)` and `MFTLSortPairs(keys, vals, n)` dispatch on
  the key type with C11 `_Generic`; under C99 they fall back to the 32-bit
  variants so that 64-bit keys fail to compile instead of mis-sorting.
- The key type is the ordering contract: `int` sorts signed, `uint64_t` sorts
  unsigned.
- `MFTLUniqueIntArray` / `MFTLUniquePairs` compact sorted input in one scan;
  `MFTLSortedIndexOf` is the binary search behind every sparse id-keyed
  directory in the library (no hash maps); `MFTLGetUniqueIntegers` is an
  $O(n\,u)$ first-seen-order collector for tiny sets such as destination ranks.
- Two compile-time checks guard the 32-bit path, which reinterprets `int` as
  `uint32_t` (permitted by C11 §6.5p7): `int` is 32 bits and two's complement.
