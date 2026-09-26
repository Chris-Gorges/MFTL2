/*
 * Stage 16A — library-internal helpers shared by the builders, the communication layer and
 * src/general/twin_states.c. Not part of the public API.
 */
#ifndef MFTL_TWIN_INTERNAL_H
#define MFTL_TWIN_INTERNAL_H

#include "mftl_core.h"

/* Tags n directed boundary half-edges ab[i] = {origin gid, target gid} with the index of their
   loop, loops ordered by their smallest vertex gid. MFTL_ERROR if the records do not form
   disjoint simple loops (branching / open chain). */
int MFTLInternalTagBoundaryLoops(int n, const int (*ab)[2], int *tag_out);

/* Number of duplicate directed live half-edges (same origin and target vertex gid). */
int MFTLInternalCountDirectedDuplicates(const MFTL_Object *obj, int verbose);

/* GLOBAL discovery mode, lookup only (collective). */
int MFTLInternalGlobalLookup(MFTL_Object *obj, int n, const int *want, const int *code, int *holder, int *match, int *conflict);

/* Number of REMOTE half-edges this rank would request in a ghost-completion round (G1/G2). */
int MFTLInternalCountGhostRequests(const MFTL_Object *obj);

/* Quiescent-point validation hook (full collective in Debug / MFTL_VALIDATE, local otherwise). */
int MFTLInternalQuiescentCheck(MFTL_Object *obj, int flags);

/* Cartesian box of a rank (communication.c). */
void MFTLInternalRankBox(const MFTL_Domain *dom, int rank, double bmin[3], double bmax[3]);

/* GID -> local index remap of ghosts appended after v_mark / he_mark (communication.c). */
void MFTLInternalRemapAppendedGhostIndices(MFTL_Object *obj, int v_mark, int he_mark);

#endif
