/**
 * @file mftl_settings.c
 * @author Christian Gorges
 * @date 2026-09-29
 *
 * @brief The settings tool: write the annotated template, check a file before a
 *        job is submitted, list the keys.
 *
 *     mftl_settings template > run.toml      every key, documented, at its default
 *     mftl_settings check run.toml [--np N] [section.key=value ...]
 *     mftl_settings keys                     one line per key
 *
 * `check` runs exactly the code a simulation runs -- MFTLSettingsLoad(), then
 * MFTLSettingsLoadArgs() -- so a file it accepts is a file the run accepts, and
 * every problem is reported with its line at once rather than one per queued
 * job. It prints what the file changes against the defaults and, with `--np`,
 * the process grid the run will get. It needs no mpirun.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

static void usage(FILE *out)
{
  fprintf(out,
          "usage: mftl_settings template                      write every key, documented, at its default\n"
          "       mftl_settings check FILE [--np N] [section.key=value ...]\n"
          "                                                   validate FILE (and overrides) as a run would;\n"
          "                                                   print what it changes and the grid at N ranks\n"
          "       mftl_settings keys                          one line per key: name, type, default, range\n");
}

static const char *type_name(MFTL_SettingType t)
{
  switch (t)
  {
    case MFTL_SETTING_INT: return "int";
    case MFTL_SETTING_REAL: return "real";
    case MFTL_SETTING_BOOL: return "bool";
    case MFTL_SETTING_KEYWORD: return "keyword";
    case MFTL_SETTING_STRING: return "string";
    case MFTL_SETTING_INT3: return "int[3]";
    case MFTL_SETTING_REAL3: return "real[3]";
    case MFTL_SETTING_BOOL3: return "bool[3]";
  }
  return "?";
}

static int list_keys(void)
{
  for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(i);
    char key[160];
    snprintf(key, sizeof key, "%s.%s", k->section, k->name);
    printf("%-40s %-8s %-24s %s%s\n", key, type_name(k->type), k->default_text, k->range, k->advanced ? "  (advanced)" : "");
  }
  return 0;
}

static int check(int argc, char **argv)
{
  const char *path = argv[2];
  int np = 0;
  for (int i = 3; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--np") && i + 1 < argc)
      np = atoi(argv[++i]);
    else if (!MFTLSettingsIsOverride(argv[i]))
    {
      fprintf(stderr, "mftl_settings: `%s` is neither --np N nor a section.key=value override\n", argv[i]);
      return 2;
    }
  }

  MFTL_Settings cfg;
  MFTLSettingsDefaults(&cfg);
  if (MFTLSettingsLoad(&cfg, path, MPI_COMM_WORLD) != MFTL_SUCCESS) return 1;
  if (MFTLSettingsLoadArgs(&cfg, argc, argv, MPI_COMM_WORLD) != MFTL_SUCCESS) return 1;

  printf("# %s is valid. What it changes against the defaults:\n", path);
  MFTLSettingsWrite(&cfg, stdout, MFTL_SETTINGS_WRITE_CHANGED);

  if (np > 0)
  {
    int cuts[3];
    if (MFTLSettingsChooseCuts(&cfg, np, cuts) != MFTL_SUCCESS)
    {
      fprintf(stderr, "\nmftl_settings: domain.cuts = [%d, %d, %d] cannot be completed to %d ranks\n", cfg.domain.cuts[0], cfg.domain.cuts[1],
              cfg.domain.cuts[2], np);
      return 1;
    }
    double h[3];
    for (int k = 0; k < 3; ++k) h[k] = (cfg.domain.global_max[k] - cfg.domain.global_min[k]) / cuts[k];
    printf("\n# at %d ranks: grid %d x %d x %d, rank boxes %.4g x %.4g x %.4g\n", np, cuts[0], cuts[1], cuts[2], h[0], h[1], h[2]);
  }

  /* The values a run derives from its mesh cannot be known here; say which. */
  const MFTL_RemeshParameters *p = &cfg.remesh.params;
  if (p->target_length == 0.0)
    printf("# remesh.target_length = 0: the run measures it from the mesh (MFTLSettingsResolve), and derives the lengths that depend on it.\n");
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rc = 0;

  if (argc >= 2 && !strcmp(argv[1], "template"))
  {
    MFTL_Settings d;
    MFTLSettingsDefaults(&d);
    rc = (MFTLSettingsWrite(&d, stdout, MFTL_SETTINGS_WRITE_DOCUMENTED) == MFTL_SUCCESS) ? 0 : 1;
  }
  else if (argc >= 3 && !strcmp(argv[1], "check"))
    rc = check(argc, argv);
  else if (argc >= 2 && !strcmp(argv[1], "keys"))
    rc = list_keys();
  else
  {
    usage((argc >= 2 && !strcmp(argv[1], "--help")) ? stdout : stderr);
    rc = (argc >= 2 && !strcmp(argv[1], "--help")) ? 0 : 2;
  }

  MPI_Finalize();
  return rc;
}
