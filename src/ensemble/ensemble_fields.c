/** @file ensemble_fields.c
 * @brief Collective registration of optional mesh fields.
 */
#include "mftl_ensemble.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_communication.h"
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

  /* Exact descriptors, not pointer identities or a probabilistic hash: the
   * local verdict rides the first agreement, which also fixes the name length,
   * so every rank packs the name into the same number of words for the second. */
  int status = valid ? 0 : 1;
  const uint64_t desc[5] = {(uint64_t) len, (uint64_t) entity, (uint64_t) ncomp, (uint64_t) mode, (uint64_t) r->n};
  if (MFTLAgreeWords(ens->domain.comm_world, desc, 5, &status) >= 0 || status != 0) return MFTL_ERROR;
  const int n_words = (int) (len / sizeof(uint64_t)) + 1;  // with the terminator, zero-padded
  uint64_t *const words = (uint64_t *) MFTLCallocOrDie((size_t) n_words, sizeof(uint64_t), "field registration name");
  memcpy(words, name, len);
  const int differs = MFTLAgreeWords(ens->domain.comm_world, words, n_words, MFTL_NULL);
  free(words);
  if (differs >= 0) return MFTL_ERROR;

  char *const agreed_name = (char *) MFTLMallocOrDie(len + 1, "field registration name");
  memcpy(agreed_name, name, len + 1);

  r->entries = MFTLReallocOrDie(r->entries, ((size_t) r->n + 1) * sizeof(*r->entries), "field descriptors");
  r->entries[r->n] = (MFTL_FieldDescriptor) {agreed_name, entity, mode, ncomp};
  if (mode != MFTL_FIELD_RECOMPUTED) r->stride[entity] += ncomp;
  return r->n++;
}
