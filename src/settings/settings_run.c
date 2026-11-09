/**
 * @file settings_run.c
 * @author Christian Gorges
 * @date 2026-09-29
 * @ingroup mftl_settings
 *
 * @brief The collective half of the settings module: loading on rank 0 and
 *        broadcasting, the process grid, the ensemble, and resolving the
 *        values that depend on the mesh.
 *
 * All of the module's MPI is here. The rule it follows is the one every other
 * collective in the library follows: a decision that changes which collectives
 * run is taken from data every rank holds identically. Loading gets that by
 * broadcasting the TEXT and parsing it everywhere -- the same bytes through the
 * same parser give the same struct and the same verdict, so no rank can take an
 * error path the others do not. Anything a caller sets by hand afterwards is
 * checked once, in MFTLSettingsResolve() and MFTLSettingsCreateEnsemble(), by
 * one reduction over a hash per key.
 */

#include "mftl_settings.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"

/** Capacity of the problem report a collective call prints on rank 0. */
#define REPORT_CHARS 8192

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Broadcasting text
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Broadcasts rank 0's text and parses it on every rank.
 *
 * @param s      The settings.
 * @param text   Rank 0's text (ignored elsewhere), or NULL on rank 0 to make
 *               every rank fail -- rank 0 has then already said why.
 * @param origin Name for messages.
 * @param comm   Communicator.
 * @return The common verdict.
 */
static int broadcast_and_parse(MFTL_Settings *s, const char *text, const char *origin, MPI_Comm comm)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  long long len = -1;
  if (rank == 0 && text) len = (long long) strlen(text);
  MPI_Bcast(&len, 1, MPI_LONG_LONG, 0, comm);
  if (len < 0) return MFTL_ERROR;
  if (len > INT_MAX - 1)
  {
    if (rank == 0) fprintf(stderr, "[MFTL] %s: settings text too large (%lld bytes)\n", origin, len);
    return MFTL_ERROR;
  }

  char *buf = (char *) malloc((size_t) len + 1);
  if (!buf) abort();
  if (rank == 0) memcpy(buf, text, (size_t) len);
  MPI_Bcast(buf, (int) len, MPI_CHAR, 0, comm);
  buf[len] = '\0';

  char *message = (char *) malloc(REPORT_CHARS);
  if (!message) abort();
  const int rc = MFTLSettingsParse(s, buf, origin, message, REPORT_CHARS);
  if (rc != MFTL_SUCCESS && rank == 0) fprintf(stderr, "[MFTL] invalid settings:\n%s", message);

  free(message);
  free(buf);
  return rc;
}

int MFTLSettingsLoad(MFTL_Settings *s, const char *path, MPI_Comm comm)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  char *text = MFTL_NULL;
  if (rank == 0)
  {
    FILE *f = path ? fopen(path, "rb") : MFTL_NULL;
    if (!f)
      fprintf(stderr, "[MFTL] MFTLSettingsLoad: cannot open settings file %s\n", path ? path : "(null)");
    else
    {
      size_t cap = 1 << 14, len = 0;
      text = (char *) malloc(cap);
      if (!text) abort();
      size_t got;
      while ((got = fread(text + len, 1, cap - len - 1, f)) > 0)
      {
        len += got;
        if (cap - len - 1 == 0)
        {
          cap *= 2;
          char *grown = (char *) realloc(text, cap);
          if (!grown) abort();
          text = grown;
        }
      }
      const bool failed = ferror(f) != 0;
      fclose(f);
      text[len] = '\0';
      if (failed || strlen(text) != len)
      {
        fprintf(stderr, "[MFTL] MFTLSettingsLoad: %s: %s\n", path, failed ? "read error" : "the file contains a NUL byte; is it a text file?");
        free(text);
        text = MFTL_NULL;
      }
    }
  }

  /* Every rank needs the verdict before it may touch s, so even a missing file
   * goes through the broadcast: rank 0 sends -1 and every rank returns. */
  if (!s)
  {
    free(text);
    text = MFTL_NULL;
  }
  const int rc = broadcast_and_parse(s, text, path ? path : "settings", comm);
  free(text);
  return rc;
}

bool MFTLSettingsIsOverride(const char *arg)
{
  if (!arg) return MFTL_FALSE;
  while (*arg == '-') ++arg;
  const char *eq = strchr(arg, '=');
  if (!eq || eq == arg) return MFTL_FALSE;

  bool dot = MFTL_FALSE;
  for (const char *c = arg; c < eq; ++c)
  {
    if (*c == '.')
      dot = MFTL_TRUE;
    else if (!((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') || (*c >= '0' && *c <= '9') || *c == '_'))
      return MFTL_FALSE;
  }
  return dot;
}

int MFTLSettingsLoadArgs(MFTL_Settings *s, int argc, char **argv, MPI_Comm comm)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  char *text = MFTL_NULL;
  if (rank == 0 && s)
  {
    size_t cap = 1, len = 0;
    for (int i = 1; i < argc; ++i)
      if (MFTLSettingsIsOverride(argv[i])) cap += strlen(argv[i]) + 4;
    text = (char *) malloc(cap);
    if (!text) abort();
    text[0] = '\0';

    /* One argument per line, "key = value", so a problem is reported against
     * the argument that caused it -- and a value cannot smuggle in a second
     * assignment, because an argument cannot contain the line break. */
    for (int i = 1; i < argc; ++i)
    {
      if (!MFTLSettingsIsOverride(argv[i])) continue;
      const char *a = argv[i];
      while (*a == '-') ++a;
      const char *eq = strchr(a, '=');
      if (strpbrk(eq + 1, "\r\n")) continue; /* cannot happen from a shell; refuse rather than split */
      /* A line is strlen(a) + 3 characters and cap reserved strlen(argv[i]) + 4
       * for it, so it always fits; say so rather than let a truncation push
       * len past cap, where cap - len would wrap around. */
      const int w = snprintf(text + len, cap - len, "%.*s = %s\n", (int) (eq - a), a, eq + 1);
      if (w < 0 || (size_t) w >= cap - len) abort();
      len += (size_t) w;
    }
  }

  const int rc = broadcast_and_parse(s, text, "command line", comm);
  free(text);
  return rc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Agreement
 * ═══════════════════════════════════════════════════════════════════════════ */

/** FNV-1a, 64 bit. Only equality is asked of it. */
static uint64_t fnv1a(const char *s)
{
  uint64_t h = 1469598103934665603ull;
  for (; *s; ++s)
  {
    h ^= (unsigned char) *s;
    h *= 1099511628211ull;
  }
  return h;
}

/**
 * @brief Do all ranks hold the same settings? One reduction, same verdict
 *        everywhere, and the first differing key named on rank 0.
 *
 * Each key is hashed from its TEXT form, which is exact (reals are written to
 * round-trip). The array of hashes is reduced with MIN together with the
 * bitwise complements -- the MIN of ~h is the complement of the MAX of h -- so
 * one call yields both extremes, and equal extremes mean every rank agrees.
 *
 * @param s      The settings.
 * @param comm   Communicator.
 * @param caller Name for the message.
 * @return #MFTL_SUCCESS or #MFTL_ERROR, identical on every rank.
 */
static int settings_agree(const MFTL_Settings *s, MPI_Comm comm, const char *caller)
{
  int nproc = 1, rank = 0;
  MPI_Comm_size(comm, &nproc);
  MPI_Comm_rank(comm, &rank);
  if (nproc == 1) return MFTL_SUCCESS;

  const int n = MFTLSettingsKeyCount();
  uint64_t *h = (uint64_t *) malloc(2 * (size_t) n * sizeof(uint64_t));
  if (!h) abort();
  for (int i = 0; i < n; ++i)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(i);
    char key[160], value[2 * MFTL_SETTINGS_STRING_MAX + 16];
    snprintf(key, sizeof key, "%s.%s", k->section, k->name);
    if (MFTLSettingsGet(s, key, value, sizeof value) != MFTL_SUCCESS) value[0] = '\0';
    h[i] = fnv1a(value);
    h[n + i] = ~h[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, h, 2 * n, MPI_UINT64_T, MPI_MIN, comm);

  int differs = -1;
  for (int i = 0; i < n && differs < 0; ++i)
    if (h[i] != ~h[n + i]) differs = i;
  free(h);

  if (differs < 0) return MFTL_SUCCESS;
  if (rank == 0)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(differs);
    fprintf(stderr,
            "[MFTL] %s: the ranks hold different settings (first difference: %s.%s). Load them with MFTLSettingsLoad(), or change them "
            "identically on every rank.\n",
            caller, k->section, k->name);
  }
  return MFTL_ERROR;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The process grid and the ensemble
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLSettingsChooseCuts(const MFTL_Settings *s, int nproc, int cuts[3])
{
  if (!s || !cuts || nproc < 1) return MFTL_ERROR;
  const int *fixed = s->domain.cuts;
  double len[3];
  long long fixed_product = 1;
  for (int k = 0; k < 3; ++k)
  {
    if (fixed[k] < 0) return MFTL_ERROR;
    if (fixed[k] > 0) fixed_product *= fixed[k];
    len[k] = s->domain.global_max[k] - s->domain.global_min[k];
    if (!(len[k] > 0.0)) return MFTL_ERROR;
  }
  if (fixed_product > nproc || nproc % fixed_product != 0) return MFTL_ERROR;

  bool found = MFTL_FALSE;
  double best_min = 0.0, best_max = 0.0;
  int best[3] = {0, 0, 0};

  /* Every factorisation c0 * c1 * c2 = nproc honouring the fixed entries. The
   * divisor loops cost O(sigma(nproc)), a few times nproc at worst -- once per
   * run, and nothing at the rank counts a job actually has. */
  for (int c0 = 1; c0 <= nproc; ++c0)
  {
    if (nproc % c0 != 0 || (fixed[0] > 0 && c0 != fixed[0])) continue;
    const int rest = nproc / c0;
    for (int c1 = 1; c1 <= rest; ++c1)
    {
      if (rest % c1 != 0 || (fixed[1] > 0 && c1 != fixed[1])) continue;
      const int c2 = rest / c1;
      if (fixed[2] > 0 && c2 != fixed[2]) continue;

      const double h0 = len[0] / c0, h1 = len[1] / c1, h2 = len[2] / c2;
      const double hmin = MFTL_MIN(h0, MFTL_MIN(h1, h2));
      const double hmax = MFTL_MAX(h0, MFTL_MAX(h1, h2));

      /* Widest narrowest box, then narrowest widest box, then more cuts on the
       * lower axis. The loops visit c0 ascending, so ">=" on the last criterion
       * keeps the later, larger c0. */
      bool better = !found || hmin > best_min || (hmin == best_min && hmax < best_max);
      if (found && hmin == best_min && hmax == best_max) better = (c0 > best[0]) || (c0 == best[0] && c1 > best[1]);
      if (better)
      {
        found = MFTL_TRUE;
        best_min = hmin;
        best_max = hmax;
        best[0] = c0;
        best[1] = c1;
        best[2] = c2;
      }
    }
  }
  if (!found) return MFTL_ERROR;
  cuts[0] = best[0];
  cuts[1] = best[1];
  cuts[2] = best[2];
  return MFTL_SUCCESS;
}

MFTL_Ensemble *MFTLSettingsCreateEnsemble(const MFTL_Settings *s, MPI_Comm comm)
{
  int rank = 0, nproc = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);
  if (!s) return MFTL_NULL;

  if (settings_agree(s, comm, "MFTLSettingsCreateEnsemble") != MFTL_SUCCESS) return MFTL_NULL;

  char *message = (char *) malloc(REPORT_CHARS);
  if (!message) abort();
  const int ok = MFTLSettingsCheck(s, message, REPORT_CHARS);
  if (ok != MFTL_SUCCESS && rank == 0) fprintf(stderr, "[MFTL] MFTLSettingsCreateEnsemble: invalid settings:\n%s", message);
  free(message);
  if (ok != MFTL_SUCCESS) return MFTL_NULL;

  int cuts[3];
  if (MFTLSettingsChooseCuts(s, nproc, cuts) != MFTL_SUCCESS)
  {
    if (rank == 0)
      fprintf(stderr, "[MFTL] MFTLSettingsCreateEnsemble: domain.cuts = [%d, %d, %d] cannot be completed to %d ranks.\n", s->domain.cuts[0],
              s->domain.cuts[1], s->domain.cuts[2], nproc);
    return MFTL_NULL;
  }

  /* Before the ensemble exists, so the first redistribution -- the scatter of
   * generated bodies, where a shrink slack matters most -- already runs under
   * the configured policy. */
  MFTLRedistributeSetSkipUnchangedBodies(s->redistribute.skip_unchanged_bodies);
  MFTLRedistributeSetShrinkSlack(s->redistribute.shrink_slack);

  double lo[3], hi[3];
  bool periodic[3];
  for (int k = 0; k < 3; ++k)
  {
    lo[k] = s->domain.global_min[k];
    hi[k] = s->domain.global_max[k];
    periodic[k] = s->domain.periodic[k];
  }
  return MFTLEnsembleCreate(comm, cuts, lo, hi, periodic, 8);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Resolving what depends on the mesh
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLSettingsResolve(MFTL_Settings *s, MFTL_Ensemble *ens)
{
  if (!s || !ens) return MFTL_ERROR;
  const MPI_Comm comm = ens->domain.comm_world;
  const int rank = ens->domain.rank;

  if (settings_agree(s, comm, "MFTLSettingsResolve") != MFTL_SUCCESS) return MFTL_ERROR;

  MFTL_RemeshParameters *p = &s->remesh.params;
  MFTL_TopologyOptions *t = &s->topology.options;
  const bool derive_min = (p->min_edge_length == 0.0);
  const bool derive_max = (p->max_edge_length == 0.0);
  const bool curvature = (p->sizing == MFTL_REMESH_SIZING_CURVATURE);
  const bool band_needs_length = derive_min || derive_max || (curvature && (p->curvature_tolerance == 0.0 || p->min_target_length == 0.0));
  const bool topology_needs_length = (t->dx == 0.0 || t->delta_breakup == 0.0 || t->delta_coalescence == 0.0);

  /* Every branch below reads only values that are identical on every rank --
   * the settings (just checked) and a reduced mean -- so every rank takes it. */
  if (p->target_length == 0.0 && (band_needs_length || topology_needs_length))
  {
    const double mean = MFTLEnsembleMeanEdgeLength(ens);
    if (mean > 0.0) p->target_length = mean;
  }

  const double L = p->target_length;
  if (L > 0.0)
  {
    /* The envelope [min_edge_length, max_edge_length] per sizing mode -- the
     * shortest and longest edge the pass can want, which is what the frozen
     * band and the parameter check read (MFTL_RemeshParameters::sizing). */
    switch (p->sizing)
    {
      case MFTL_REMESH_SIZING_CURVATURE:
        if (p->min_target_length == 0.0) p->min_target_length = s->remesh.min_target_factor * L;
        if (p->curvature_tolerance == 0.0) p->curvature_tolerance = s->remesh.curvature_tolerance_factor * L;
        if (derive_min) p->min_edge_length = MFTL_REMESH_COLLAPSE_RATIO * p->min_target_length;
        if (derive_max) p->max_edge_length = MFTL_REMESH_SPLIT_RATIO * L;
        break;
      case MFTL_REMESH_SIZING_UNIFORM:
      default:
        if (derive_min) p->min_edge_length = (4.0 / 5.0) * L;
        if (derive_max) p->max_edge_length = (4.0 / 3.0) * L;
        break;
    }
    if (t->dx == 0.0) t->dx = s->topology.dx_factor * L;
    if (t->delta_breakup == 0.0) t->delta_breakup = s->topology.delta_breakup_factor * L;
    if (t->delta_coalescence == 0.0) t->delta_coalescence = s->topology.delta_coalescence_factor * L;
  }
  else
  {
    /* Nothing to measure. That is only an error if something scheduled needs
     * the length; otherwise the zeros stay, and a remesh attempted with them
     * is refused loudly by the remesher's own band check. */
    const bool topology_runs = s->topology.every > 0;
    const bool remesh_runs = s->remesh.every > 0 || (topology_runs && s->topology.remesh_after);
    if ((band_needs_length && remesh_runs) || (topology_needs_length && topology_runs))
    {
      if (rank == 0)
        fprintf(stderr,
                "[MFTL] MFTLSettingsResolve: remesh.target_length is 0 (derive it from the mesh), but the ensemble has no edges to measure. "
                "Generate the bodies first, or set it.\n");
      return MFTL_ERROR;
    }
  }

  /* `type` is informational past MFTLInitRemeshParams(); keep it truthful. */
  p->type = (derive_min && derive_max) ? MFTL_REMESH_ISOTROPIC : MFTL_REMESH_CUSTOM;
  MFTLRemeshParamsFinalise(p);
  s->topology.options.remesh = s->topology.remesh_after ? p : MFTL_NULL;

  if (p->max_edge_length > 0.0 && !(p->min_edge_length <= p->max_edge_length))
  {
    if (rank == 0)
      fprintf(stderr, "[MFTL] MFTLSettingsResolve: the edge-length band [%g, %g] is empty; set remesh.min_edge_length and max_edge_length consistently.\n",
              p->min_edge_length, p->max_edge_length);
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}
