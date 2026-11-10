/* glibc's nonnull declarations, for a build on a C library that has none.
 *
 * glibc declares the pointer arguments of memcpy, qsort and friends nonnull, so
 * memcpy(dst, NULL, 0) -- a harmless zero-length copy from an array a rank never
 * allocated -- is undefined behaviour that UBSan reports on Linux. macOS declares
 * none of them, so the same call is silent there, and the first anyone hears of
 * it is a Linux CI run. Force-included into every C file (-include, see
 * tools/check_glibc_nonnull.sh), this header redeclares them as glibc does, and
 * UBSan's nonnull-attribute check then fires on the Mac at the same call sites.
 *
 * Only the functions whose arguments a zero length makes otherwise legal to pass
 * as NULL are here; glibc marks more. On glibc itself the header does nothing.
 */
#ifndef MFTL_TOOLS_GLIBC_NONNULL_H
#define MFTL_TOOLS_GLIBC_NONNULL_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef __GLIBC__
/* macOS's fortified headers define these three as macros over __builtin___*_chk,
   which would bypass the declarations below. */
#undef memcpy
#undef memmove
#undef memset

extern void *memcpy(void *, const void *, size_t) __attribute__((nonnull(1, 2)));
extern void *memmove(void *, const void *, size_t) __attribute__((nonnull(1, 2)));
extern void *memset(void *, int, size_t) __attribute__((nonnull(1)));
extern int memcmp(const void *, const void *, size_t) __attribute__((nonnull(1, 2)));
extern void *memchr(const void *, int, size_t) __attribute__((nonnull(1)));
extern void qsort(void *, size_t, size_t, int (*)(const void *, const void *)) __attribute__((nonnull(1, 4)));
extern void *bsearch(const void *, const void *, size_t, size_t, int (*)(const void *, const void *)) __attribute__((nonnull(1, 2, 5)));
#endif

#endif
