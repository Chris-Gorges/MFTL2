#!/usr/bin/env bash
# Checks that the lines a change touches follow .clang-format, or fixes them.
#
# Only CHANGED lines are judged, the way LLVM checks its own pull requests: the
# existing code is hand-formatted in places clang-format would make worse (the
# statement arguments of the MFTL_FOREACH_* macros), so a whole-tree rewrite
# would cost a noisy commit and buy nothing. New and edited code has to conform.
#
# The clang-format version is pinned (tools/requirements-format.txt) because two
# releases format the same lines differently:
#
#   pip install -r tools/requirements-format.txt
#   tools/check_format.sh                 # lines changed since origin/main, working tree included
#   tools/check_format.sh <base>          # ... since <base>
#   tools/check_format.sh <base> <head>   # between two commits (CI)
#   tools/check_format.sh --fix [<base>]  # reformat those lines in place
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

want="$(sed -n 's/^clang-format==//p' tools/requirements-format.txt)"
have="$("$CLANG_FORMAT" --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)"
if [ "$have" != "$want" ]; then
  echo "clang-format $have found, $want pinned: pip install -r tools/requirements-format.txt" >&2
  exit 2
fi
if ! git clang-format -h > /dev/null 2>&1; then
  echo "git clang-format not found (it ships with the pinned pip package)" >&2
  exit 2
fi

fix=0
if [ "${1:-}" = "--fix" ]; then
  fix=1
  shift
fi
base="${1:-origin/main}"
head="${2:-}"

if [ "$fix" -eq 1 ]; then
  git clang-format --binary "$CLANG_FORMAT" "$base"
  exit 0
fi

# Exit status 1 and a diff when a changed line does not conform.
if git clang-format --binary "$CLANG_FORMAT" --diff "$base" ${head:+"$head"}; then
  echo "every changed line since $base matches .clang-format"
else
  status=$?
  echo
  echo "The lines above do not match .clang-format; run tools/check_format.sh --fix $base" >&2
  exit "$status"
fi
