`src/arrayfunctions/` — Sorting and duplicate removal {#mftl_mod_arrayfunctions}
=====================================================

**Public header:** `include/mftl_arrayfunctions.h` · **Layer:** 0 ·
**Module group:** @ref mftl_arrayfunctions

The only algorithm in MFTL with no mesh in it. One translation unit,
`radixsort.c`, providing a **stable LSD radix sort** for keys and for key/value
pairs, plus two duplicate-removal helpers that live inline in the header.

Derived from Cameron Hart's public-domain-style radix sort, specialised for MFTL:
unified 32/64-bit key width, SoA key/value variants, signed 32-bit ordering,
constant-digit pass skipping, and a small-array cutoff.

---

## Who uses it, and why it has to be stable

| Caller | Key | Why sorted |
|---|---|---|
| `MFTLObjectReorderMorton` | 63-bit Morton code | walk the Z-curve |
| `MFTL_GenerateLBVH` | 63-bit Morton code | the radix split *is* the sorted order |
| `src/communication/redistribute.c` | vertex index | de-duplicate a destination's vertex list |
| `src/remeshing/` | global id | resolve arriving ids to slots |

**Stability is a correctness requirement, not a nicety.** Two primitives with
equal Morton codes must come out in a deterministic order, or two ranks that
compute the same partition produce different trees — and the distributed contact
test compares gathered *sets* across process counts. It is also what makes
`MFTLUniquePairs` keep the *first* value of each run, which is the value whose
entry came first in the caller's array: a rank-independent choice.

---

## The algorithm

An LSD radix sort makes $\lceil b/r \rceil$ stable counting-sort passes over $n$
keys of $b$ bits, $r$ bits at a time:

$$
  T(n) = O\!\left(\frac{b}{r}\,(n + 2^{r})\right)
$$

With $r = 8$ one pass needs a $2^8$-entry histogram, i.e. 1 KiB. All passes
together are 4 KiB (32-bit keys) or 8 KiB (64-bit) — both comfortably inside L1D,
which is what makes the implementation's central trick worth doing.

### One counting pass for every digit

`init_histograms32` / `init_histograms64` stream the keys **once** and update all
$b/r$ histograms per key. The alternative — one counting pass per digit — reads
the key array $b/r$ times for counting alone. Since every histogram update hits
L1, the fused form costs the same instructions and $b/r$ times less traffic.

```
   for each key k:                 for pass p:            <-- NOT this
       hist[0][byte0(k)]++             for each key k:
       hist[1][byte1(k)]++                 hist[p][byte_p(k)]++
       ...
```

### Constant digits are skipped

`prefix_sum_histograms` converts each histogram from counts to exclusive prefix
sums, and while doing so checks whether the bucket belonging to `first_key`'s
digit already holds all $n$ elements. If it does, every key has the same digit
there and the pass is the identity — so it is skipped.

This is not a corner case in this library. **Global IDs are dense in $[0, n)$**
(see @ref mftl_page_invariants), so for a mesh of under 65 536 entities the top
two bytes of a 32-bit id are always zero and half the passes vanish.

### Signed 32-bit ordering is free

The most significant pass processes the sign byte. Running its prefix sum
starting at bucket 128 and wrapping to bucket 0 places the negative keys first:

$$
  \text{offset}[128] = 0,\quad \text{offset}[129] = c_{128},\ \ldots,\ 
  \text{offset}[0] = \sum_{j=128}^{255} c_j,\ \ldots
$$

No key is ever modified, so no un-flipping pass is needed at the end and the
sort works on the caller's array in its own representation.

### Below 32 elements, insertion sort wins

The fixed costs of a radix sort — clearing the histograms, the 256-entry prefix
sums per pass, and (in the wrapper functions) a `malloc`/`free` pair — are all
independent of $n$. Below some size they outweigh the $O(n^2/4)$ moves of an
array that never leaves L1.

Measured (ns/call, Apple M-series, `-O3`, `MFTLSortIds` / `MFTLSortPairs64`):

| n | radix, random | insertion, random | radix, descending | insertion, descending |
|---|---|---|---|---|
| 16 | 577 | **78** | 277 | **75** |
| 32 | 749 | **496** | 323 | **287** |
| 48 | **859** | 866 | **369** | 639 |
| 64 | **973** | 1310 | **415** | 1299 |

`#define INSERTION_CUTOFF 32` is where insertion sort still wins on **both**
input patterns. A larger cutoff looks better on random data (64-bit keys, 8
passes, still 1.6× at $n = 64$) and turns into a 1.4–3× loss on descending input,
where insertion sort hits its $O(n^2)$ worst case *and* the radix path gets
faster because a narrow key range skips passes. One conservative cutoff for both
key widths keeps that trap closed.

The redistribution sorts one small buffer per neighbour rank, so this is not a
corner case there either — it is the common case.

---

## Files

| File | Contents |
|---|---|
| `radixsort.c` | insertion-sort fallbacks, histogram construction, prefix sums, the four scatter passes, and the five public entry points |

The two duplicate-removal helpers (`MFTLUniqueIntArray`, `MFTLUniquePairs`) and
the small-set collector `MFTLGetUniqueIntegers` are `static inline` in the header
— they are three-line loops that callers want inlined into their own.

---

## Type dispatch

C11 `_Generic` picks the 32- or 64-bit variant at compile time:

```c
MFTLRadixSort(keys, temp, n);   /* int* -> ...32,  uint64_t* -> ...64 */
MFTLSortPairs(keys, vals, n);
```

Under C99 the macros expand to the 32-bit variants unconditionally, so passing
64-bit keys is a **compile error** rather than a silent mis-sort. Code that needs
64-bit sorting under C99 calls `MFTLRadixSort64` / `MFTLSortPairs64` directly.

---

## Two assumptions, checked at compile time

```c
_Static_assert(sizeof(int) == sizeof(uint32_t), "...");
_Static_assert((-1 & 3) == 3,                   "...");  /* two's complement */
```

The 32-bit paths reinterpret `int` as `uint32_t` for digit extraction, and the
pair sorts move `int` values through a `uint32_t` shuffle. Accessing an `int`
object through an `unsigned int` lvalue is explicitly allowed by C11 6.5p7, so
this is not a strict-aliasing violation — but the *width* and the *representation*
are worth checking here rather than discovering on an exotic target.
