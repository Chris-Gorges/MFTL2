#!/usr/bin/env bash
# The library may export only MFTL* symbols, each exactly once.
#
# A helper that lost its `static` becomes a global symbol that can collide with
# an application's own, and two translation units defining the same name link
# by accident until the day they don't. Both show up here and nowhere else.
#
# Usage: tools/check_symbols.sh [path/to/libMFTL.a]      (default build/lib/libMFTL.a)
#
# Run it on a build WITHOUT link-time optimisation: GCC's slim LTO objects carry
# no symbol table that plain `nm` can read (configure with -DMFTL_ENABLE_LTO=OFF,
# or set NM=gcc-nm).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
lib="${1:-$root/build/lib/libMFTL.a}"
NM="${NM:-nm}"

if [ ! -f "$lib" ]; then
  echo "no library at $lib -- build first" >&2
  exit 2
fi

# Defined, external text and data symbols. Mach-O prefixes C names with '_'.
syms="$("$NM" -g "$lib" 2>/dev/null | awk 'NF >= 3 && $2 ~ /^[TDBS]$/ {print $3}')"
if [ "$(uname)" = "Darwin" ]; then
  syms="$(printf '%s\n' "$syms" | sed 's/^_//')"
fi

if [ -z "$syms" ]; then
  echo "no symbols read from $lib (LTO archive? see the header of this script)" >&2
  exit 2
fi

status=0
foreign="$(printf '%s\n' "$syms" | grep -v '^MFTL' || true)"
if [ -n "$foreign" ]; then
  echo "exported symbols without the MFTL prefix:"
  printf '  %s\n' $foreign
  status=1
fi
dups="$(printf '%s\n' "$syms" | sort | uniq -d)"
if [ -n "$dups" ]; then
  echo "symbols defined more than once:"
  printf '  %s\n' $dups
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "$(printf '%s\n' "$syms" | wc -l | tr -d ' ') exported symbols, all MFTL*, no duplicates"
fi
exit "$status"
