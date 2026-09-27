#!/usr/bin/env bash
# Stage 16A (plans/16 §2.1): twin_gid is a state code (gid, boundary code -2-tag, or UNKNOWN).
# Raw comparisons of twin_gid are only allowed inside the predicate header, the twin-state
# module and the transport layer (packed codes); everything else must use MFTLHeIsLocal /
# IsRemote / IsBoundary / BoundaryTag / TwinGid / TwinCode / EdgeId (mftl_core.h: documentation
# of the encoding only). Exit 1 on a violation.
root=${1:-$(cd "$(dirname "$0")/.." && pwd)}
allow='include/mftl_core.h|include/mftl_topology.h|src/core/twin_states.c|src/parallel/redistribute.c|src/parallel/ghost_layers.c'
hits=$(cd "$root" && grep -rnE 'twin_gid[[:space:]]*(==|!=|<=|>=|<[^-<]|>[^>])|(==|!=|<=|>=)[[:space:]]*[A-Za-z_]+([.]|->|\[[^]]*\])*[A-Za-z_]*twin_gid' src include | grep -vE "^($allow):" | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|/?\*)')
if [ -n "$hits" ]; then
  echo "raw twin_gid comparisons outside the predicates:"; echo "$hits"; exit 1
fi
echo "twin_gid reader audit: OK"
