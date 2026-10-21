/** @file ensemble_fields.c
 * @brief Collective registration of optional mesh fields.
 */
#include "mftl_ensemble.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_memory.h"

int MFTLEnsembleFindField(const MFTL_Ensemble *ens, const char *name)
{
  if (!ens || !name) return MFTL_ERROR;
  for (int f = 0; f < ens->fields.n; ++f)
    if (strcmp(name, ens->fields.entries[f].name) == 0) return f;
  return MFTL_ERROR;
}

int MFTLEnsembleRegisterField(MFTL_Ensemble *ens, const char *name, MFTL_FieldEntity entity, int ncomp, MFTL_FieldMode mode)
{
  if (!ens) return MFTL_ERROR;
  MFTL_FieldRegistry *r = &ens->fields;
  const size_t len = name ? strlen(name) : 0;
  int valid = !r->locked && ens->n_bodies_global == 0 && len > 0 && len < INT_MAX && ncomp > 0 &&
              (entity == MFTL_FIELD_VERTEX || entity == MFTL_FIELD_TRIANGLE) &&
              (mode == MFTL_FIELD_RECOMPUTED || mode == MFTL_FIELD_INTERPOLATED || mode == MFTL_FIELD_CONSERVATIVE_DENSITY) &&
              !(mode == MFTL_FIELD_CONSERVATIVE_DENSITY && entity != MFTL_FIELD_TRIANGLE) && MFTLEnsembleFindField(ens, name) == MFTL_ERROR && r->n < INT_MAX;
  if (valid && mode != MFTL_FIELD_RECOMPUTED) valid = ncomp <= INT_MAX - r->stride[entity];
  int all_valid;
  MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, ens->domain.comm_world);
  if (!all_valid) return MFTL_ERROR;

  /* Compare exact descriptors, not pointer identities or a probabilistic hash.
   * All ranks have entered before any variable-sized broadcast is posted. */
  int root[5] = {(int) len, (int) entity, ncomp, (int) mode, r->n};
  MPI_Bcast(root, 5, MPI_INT, 0, ens->domain.comm_world);
  char *agreed_name = MFTLMallocOrDie((size_t) root[0] + 1, "field registration name");
  if (ens->domain.rank == 0) memcpy(agreed_name, name, len + 1);
  MPI_Bcast(agreed_name, root[0] + 1, MPI_CHAR, 0, ens->domain.comm_world);
  valid = root[0] == (int) len && root[1] == (int) entity && root[2] == ncomp && root[3] == (int) mode && root[4] == r->n && strcmp(agreed_name, name) == 0;
  MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, ens->domain.comm_world);
  if (!all_valid)
  {
    free(agreed_name);
    return MFTL_ERROR;
  }

  r->entries = MFTLReallocOrDie(r->entries, ((size_t) r->n + 1) * sizeof(*r->entries), "field descriptors");
  r->entries[r->n] = (MFTL_FieldDescriptor) {agreed_name, entity, mode, ncomp};
  if (mode != MFTL_FIELD_RECOMPUTED) r->stride[entity] += ncomp;
  return r->n++;
}
