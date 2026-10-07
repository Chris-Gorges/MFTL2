/*
 * @Author: Christian Gorges
 * @Date: 2026-09-29
 *
 * @brief The settings module: the key table, the file format, and the
 *        collective loading, grid choice and resolution.
 *
 * Runs at np = 1 and np = 2. Most of the module is rank-local text handling and
 * is the same at any rank count; np = 2 is what makes the three collective
 * promises testable at all -- that only rank 0 reads the file and every rank
 * still ends up with the same struct and the same verdict, that ranks holding
 * DIFFERENT settings are caught rather than left to deadlock later, and that a
 * derived length is the same number everywhere.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT A SETTINGS PARSER MUST NOT DO
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Its characteristic failure is silence: a misspelled key, a value in the wrong
 * section or a number outside its meaningful range that is quietly ignored or
 * clamped, so the run proceeds on a default nobody chose. Every such case here
 * is asserted to be an ERROR, and the typo cases to name the key that was meant.
 * The second failure is a half-applied file, which is why a failing text is
 * asserted to leave the struct exactly as it was.
 *
 * The table is also checked against the code it describes: every key's declared
 * type against the size of the member it names, every default against its own
 * range, and every module-block default against the module's own defaults, so
 * that the manual generated from the table cannot drift from the library.
 *
 * EVERY assertion goes through agreed(): TEST_ASSERT returns immediately, so an
 * assertion failing on some ranks only would leave the rest blocked in the next
 * collective -- a hang instead of a clean failure.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static int g_rank = 0;
static int g_nproc = 1;

/* ─────────────────────────────────────────────────────────────────────────────
 * Helpers
 * ─────────────────────────────────────────────────────────────────────────── */

static void full_name(const MFTL_SettingInfo *k, char *out, size_t n) { snprintf(out, n, "%s.%s", k->section, k->name); }

/** Every key's text form equal between two settings. */
static bool same_settings(const MFTL_Settings *a, const MFTL_Settings *b)
{
  for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
  {
    char key[160], va[600], vb[600];
    full_name(MFTLSettingsKey(i), key, sizeof key);
    if (MFTLSettingsGet(a, key, va, sizeof va) != MFTL_SUCCESS) return false;
    if (MFTLSettingsGet(b, key, vb, sizeof vb) != MFTL_SUCCESS) return false;
    if (strcmp(va, vb) != 0) return false;
  }
  return true;
}

/** Parses @p text over the defaults; the message lands in @p msg. */
static int parse_over_defaults(MFTL_Settings *s, const char *text, char *msg, int n)
{
  MFTLSettingsDefaults(s);
  return MFTLSettingsParse(s, text, "t.toml", msg, n);
}

/** Writes @p s through a temporary stream and returns the text (free it). */
static char *write_to_string(const MFTL_Settings *s, unsigned flags)
{
  FILE *f = tmpfile();
  if (!f) return NULL;
  if (MFTLSettingsWrite(s, f, flags) != MFTL_SUCCESS)
  {
    fclose(f);
    return NULL;
  }
  const long len = ftell(f);
  rewind(f);
  char *text = (char *) malloc((size_t) len + 1);
  const size_t got = fread(text, 1, (size_t) len, f);
  text[got] = '\0';
  fclose(f);
  return text;
}

static bool is_module_block_key(const MFTL_SettingInfo *k)
{
  const struct
  {
    size_t off, size;
  } blocks[] = {
      {offsetof(MFTL_Settings, remesh.params), sizeof(MFTL_RemeshParameters)},
      {offsetof(MFTL_Settings, topology.options), sizeof(MFTL_TopologyOptions)},
      {offsetof(MFTL_Settings, contact.params), sizeof(MFTL_BVHBuildParams)},
      {offsetof(MFTL_Settings, coupling.params), sizeof(MFTL_CouplerParams)},
      {offsetof(MFTL_Settings, output.vtk), sizeof(MFTL_VtkOptions)},
  };
  for (size_t b = 0; b < sizeof blocks / sizeof blocks[0]; ++b)
    if (k->offset >= blocks[b].off && k->offset < blocks[b].off + blocks[b].size) return true;
  return false;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * The table
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Every key's type agrees with the member it names, and the table is
 *        ordered, unique and complete in the ways the writer and the manual rely
 *        on.
 *
 * A key declared REAL over an `int` member would read and write eight bytes
 * into four; nothing but this comparison can see it before it corrupts the
 * neighbouring field.
 */
static int test_every_key_matches_its_member(void)
{
  const int n = MFTLSettingsKeyCount();
  TEST_ASSERT(n > 40);
  TEST_ASSERT(MFTLSettingsKey(-1) == NULL && MFTLSettingsKey(n) == NULL);

  int section = 0;
  for (int i = 0; i < n; ++i)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(i);
    size_t want = 0;
    switch (k->type)
    {
      case MFTL_SETTING_INT: want = sizeof(int); break;
      case MFTL_SETTING_REAL: want = sizeof(double); break;
      case MFTL_SETTING_BOOL: want = sizeof(bool); break;
      case MFTL_SETTING_KEYWORD: want = sizeof(int); break;
      case MFTL_SETTING_STRING: want = k->size; break;
      case MFTL_SETTING_INT3: want = 3 * sizeof(int); break;
      case MFTL_SETTING_REAL3: want = 3 * sizeof(double); break;
      case MFTL_SETTING_BOOL3: want = 3 * sizeof(bool); break;
    }
    if (k->size != want) printf("\n  %s.%s: type needs %zu bytes, member has %zu", k->section, k->name, want, k->size);
    TEST_ASSERT(k->size == want);
    TEST_ASSERT(k->offset + k->size <= sizeof(MFTL_Settings));
    TEST_ASSERT(k->summary && strlen(k->summary) > 10);
    TEST_ASSERT((k->type == MFTL_SETTING_KEYWORD) == (k->choices != NULL));
    TEST_ASSERT(k->type != MFTL_SETTING_STRING || k->size >= 2);

    /* Keys are grouped by section, in section order: the writer and the manual
     * walk them that way. */
    while (section < MFTLSettingsSectionCount() && strcmp(MFTLSettingsSection(section)->name, k->section) != 0) ++section;
    TEST_ASSERT(section < MFTLSettingsSectionCount());

    char key[160];
    full_name(k, key, sizeof key);
    TEST_ASSERT(MFTLSettingsFindKey(key) == k);
    for (const char *c = k->name; *c; ++c) TEST_ASSERT((*c >= 'a' && *c <= 'z') || (*c >= '0' && *c <= '9') || *c == '_');
  }

  /* Every section has at least one key. */
  for (int s = 0; s < MFTLSettingsSectionCount(); ++s)
  {
    bool any = false;
    for (int i = 0; i < n; ++i) any = any || strcmp(MFTLSettingsKey(i)->section, MFTLSettingsSection(s)->name) == 0;
    TEST_ASSERT(any);
  }

  /* No two keys share a byte. */
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
    {
      const MFTL_SettingInfo *a = MFTLSettingsKey(i), *b = MFTLSettingsKey(j);
      TEST_ASSERT(a->offset + a->size <= b->offset || b->offset + b->size <= a->offset);
    }

  TEST_ASSERT(MFTLSettingsFindKey("remesh") == NULL);
  TEST_ASSERT(MFTLSettingsFindKey("remesh.nope") == NULL);
  TEST_ASSERT(MFTLSettingsFindKey(NULL) == NULL);
  return 0;
}

/** @brief Every table default parses, lies in its range, and is what Defaults gives. */
static int test_every_default_is_admissible(void)
{
  MFTL_Settings d;
  MFTLSettingsDefaults(&d);
  char msg[4096];
  TEST_ASSERT(MFTLSettingsCheck(&d, msg, sizeof msg) == MFTL_SUCCESS);

  for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(i);
    char key[160];
    full_name(k, key, sizeof key);
    MFTL_Settings s;
    MFTLSettingsDefaults(&s);
    const int rc = MFTLSettingsSet(&s, key, k->default_text, msg, sizeof msg);
    if (rc != MFTL_SUCCESS) printf("\n  default of %s: %s", key, msg);
    TEST_ASSERT(rc == MFTL_SUCCESS);
    TEST_ASSERT(same_settings(&s, &d));
  }
  return 0;
}

/**
 * @brief The defaults of the module blocks are the modules' own, except the one
 *        difference the documentation states.
 *
 * The table carries each default as text for the manual; this is what keeps
 * that text from drifting when a module changes its default.
 */
static int test_defaults_follow_the_modules(void)
{
  MFTL_Settings d, m;
  MFTLSettingsDefaults(&d);
  MFTLSettingsDefaults(&m);
  MFTLInitRemeshParams(&m.remesh.params, MFTL_REMESH_ISOTROPIC, 0.0, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  m.topology.options = MFTLTopologyDefaultOptions();
  m.contact.params = MFTL_BVHDefaultParams();
  MFTLCouplerDefaultParams(&m.coupling.params);
  m.output.vtk = MFTLVtkDefaults();

  int n_block_keys = 0, n_different = 0;
  for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(i);
    if (!is_module_block_key(k)) continue;
    ++n_block_keys;
    char key[160], vd[600], vm[600];
    full_name(k, key, sizeof key);
    TEST_ASSERT(MFTLSettingsGet(&d, key, vd, sizeof vd) == MFTL_SUCCESS);
    TEST_ASSERT(MFTLSettingsGet(&m, key, vm, sizeof vm) == MFTL_SUCCESS);
    if (strcmp(vd, vm) == 0) continue;
    ++n_different;
    if (strcmp(key, "remesh.parallel_phase_budget") != 0) printf("\n  %s: table %s, module %s", key, vd, vm);
    TEST_ASSERT(strcmp(key, "remesh.parallel_phase_budget") == 0);
  }
  TEST_ASSERT(n_block_keys > 30);
  TEST_ASSERT(n_different == 1); /* the documented difference is real */
  TEST_ASSERT(d.remesh.params.parallel_phase_budget == 4 && m.remesh.params.parallel_phase_budget == 32);

  /* The fields no key covers come from the modules untouched. */
  TEST_ASSERT(d.remesh.params.type == MFTL_REMESH_ISOTROPIC);
  TEST_ASSERT(d.contact.params.pair_filter == NULL);
  TEST_ASSERT(d.topology.options.remesh == &d.remesh.params);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * The format
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Writing and reading back is the identity, bit for bit, in both
 *        forms -- which is what makes a run's printed configuration a way to
 *        reproduce it.
 */
static int test_round_trip_is_exact(void)
{
  MFTL_Settings s, back;
  MFTLSettingsDefaults(&s);
  s.remesh.params.target_length = 0.1 + 0.2; /* 0.30000000000000004 */
  s.remesh.params.min_edge_length = 1.0 / 3.0;
  s.remesh.params.max_edge_length = 2.0 / 3.0;
  s.domain.global_max[0] = 1e300;
  s.domain.global_min[2] = -5e-324; /* the smallest subnormal */
  s.domain.cuts[0] = 3;
  s.domain.periodic[1] = true;
  s.contact.params.proximity_factor = 1.0 / 7.0;
  s.remesh.params.dispersed_balance_slack = -HUGE_VAL;
  s.remesh.params.parallel_strategy = MFTL_REMESH_PARALLEL_DISPERSED;
  snprintf(s.output.prefix, sizeof s.output.prefix, "out/run \"a\" \\ # not a comment\t!");
  char msg[4096];
  TEST_ASSERT(MFTLSettingsCheck(&s, msg, sizeof msg) == MFTL_SUCCESS);

  const unsigned forms[3] = {0u, MFTL_SETTINGS_WRITE_DOCUMENTED, MFTL_SETTINGS_WRITE_CHANGED};
  for (int f = 0; f < 3; ++f)
  {
    char *text = write_to_string(&s, forms[f]);
    TEST_ASSERT(text != NULL);
    MFTLSettingsDefaults(&back);
    const int rc = MFTLSettingsParse(&back, text, "written", msg, sizeof msg);
    if (rc != MFTL_SUCCESS) printf("\n%s\n%s", msg, text);
    free(text);
    TEST_ASSERT(rc == MFTL_SUCCESS);
    TEST_ASSERT(same_settings(&s, &back));
    TEST_ASSERT(memcmp(&s.remesh.params.target_length, &back.remesh.params.target_length, sizeof(double)) == 0);
    TEST_ASSERT(memcmp(s.domain.global_min, back.domain.global_min, sizeof s.domain.global_min) == 0);
    TEST_ASSERT(strcmp(s.output.prefix, back.output.prefix) == 0);
  }
  return 0;
}

/** @brief The syntax TOML allows, and the leniencies a command line needs. */
static int test_the_accepted_syntax(void)
{
  MFTL_Settings s;
  char msg[4096];

  const char *text =
      "\xEF\xBB\xBF# a comment\r\n"
      "remesh.every = 5   # a dotted key before any section\r\n"
      "\n"
      "  [ domain ]  \n"
      "cuts = [2, 1, 1,]          # trailing comma\n"
      "periodic = true            # one value for all three\n"
      "global_max = 2.0, 3, 4e0   # no brackets\n"
      "[remesh]\n"
      "parallel_strategy = dispersed   # bare keyword\n"
      "smooth_mode = 'vertex'          # literal string\n"
      "target_length = 1e-3\n"
      "[output]\n"
      "prefix = \"a#b\"   # a hash inside a string is not a comment\n"
      "[remesh]\n"
      "dispersed_balance_slack = -inf   # an unbounded key takes infinity\n";
  TEST_ASSERT(parse_over_defaults(&s, text, msg, sizeof msg) == MFTL_SUCCESS);
  TEST_ASSERT(s.remesh.every == 5);
  TEST_ASSERT(s.domain.cuts[0] == 2 && s.domain.cuts[1] == 1 && s.domain.cuts[2] == 1);
  TEST_ASSERT(s.domain.periodic[0] && s.domain.periodic[1] && s.domain.periodic[2]);
  TEST_ASSERT(s.domain.global_max[0] == 2.0 && s.domain.global_max[1] == 3.0 && s.domain.global_max[2] == 4.0);
  TEST_ASSERT(s.remesh.params.parallel_strategy == MFTL_REMESH_PARALLEL_DISPERSED);
  TEST_ASSERT(s.remesh.params.smooth_mode == MFTL_SMOOTH_VERTEX_BASED);
  TEST_ASSERT(s.remesh.params.target_length == 1e-3);
  TEST_ASSERT(strcmp(s.output.prefix, "a#b") == 0);
  TEST_ASSERT(isinf(s.remesh.params.dispersed_balance_slack) && s.remesh.params.dispersed_balance_slack < 0.0);

  /* Layering: a second text overrides only what it names. */
  TEST_ASSERT(MFTLSettingsParse(&s, "[remesh]\nevery = 9\n", "cli", msg, sizeof msg) == MFTL_SUCCESS);
  TEST_ASSERT(s.remesh.every == 9 && s.remesh.params.target_length == 1e-3);

  /* The squared band follows the bounds on every parse, so a bound set after
   * MFTLSettingsResolve() cannot leave a stale square behind. */
  TEST_ASSERT(MFTLSettingsSet(&s, "remesh.max_edge_length", "0.5", msg, sizeof msg) == MFTL_SUCCESS);
  TEST_ASSERT(s.remesh.params.max_edge_length_sq == 0.25);

  /* An empty text changes nothing and succeeds. */
  MFTL_Settings t = s;
  TEST_ASSERT(MFTLSettingsParse(&t, "", "empty", msg, sizeof msg) == MFTL_SUCCESS);
  TEST_ASSERT(same_settings(&s, &t));
  return 0;
}

/** @brief Silence is the failure: every one of these must be an error. */
static int test_mistakes_are_errors(void)
{
  MFTL_Settings s;
  char msg[4096];

  static const struct
  {
    const char *text;
    const char *must_mention;
  } cases[] = {
      {"[remesh]\ntarget_lenght = 1\n", "did you mean remesh.target_length"},
      {"[remsh]\nevery = 1\n", "did you mean [remesh]"},
      {"[topology]\nparallel_phase_budget = 3\n", "did you mean remesh.parallel_phase_budget"},
      {"target_length = 1\n", "remesh.target_length"},
      {"[remesh]\nremesh.every = 2\n", "write `every = ...`"},
      {"[remesh]\ntopology.every = 2\n", "move it under its own section"},
      {"[remesh]\niterations = 2.5\n", "expected an integer"},
      {"[remesh]\nsmooth_relaxation = 1.5\n", "outside [0, 1]"},
      {"[remesh]\ntarget_length = -1\n", "outside [0, inf)"},
      {"[remesh]\ntarget_length = nan\n", "expected a number"},
      {"[remesh]\ntarget_length = 0x1p3\n", "expected a number"},
      {"[remesh]\nrenumber_global_ids = yes\n", "expected true or false"},
      {"[remesh]\nparallel_strategy = dispersd\n", "did you mean \"dispersed\""},
      {"[output]\nlayout = 3\n", "is not one of"},
      {"[domain]\ncuts = [2, 2]\n", "expected three values"},
      {"[domain]\ncuts = [2, -1, 1]\n", "outside [0, inf)"},
      {"[remesh]\nevery = 1\nevery = 2\n", "set twice (first on line 2)"},
      {"[remesh]\nevery\n", "expected `key = value`"},
      {"[remesh]\nevery =\n", "has no value"},
      {"[[remesh]]\n", "arrays of tables"},
      {"[remesh\n", "malformed section header"},
      {"[output]\nlayout = \"single_file\"\nencoding = \"ascii\"\n", "needs output.encoding"},
      {"[domain]\nglobal_max = [1, 0, 1]\n", "the domain is empty along y"},
      {"[remesh]\nmin_edge_length = 2\nmax_edge_length = 1\n", "exceeds remesh.max_edge_length"},
      {"[redistribute]\nshrink_slack = 0.5\n", "at least 1"},
      {"[sphere]\nrefinements = 11\n", "outside [0, 10]"},
      {"[sphere]\nradius = 0\n", "outside (0, inf)"},
  };

  for (size_t c = 0; c < sizeof cases / sizeof cases[0]; ++c)
  {
    const int rc = parse_over_defaults(&s, cases[c].text, msg, sizeof msg);
    const bool mentions = strstr(msg, cases[c].must_mention) != NULL;
    if (rc != MFTL_ERROR || !mentions) printf("\n  case %zu: rc %d, message: %s", c, rc, msg);
    TEST_ASSERT(rc == MFTL_ERROR);
    TEST_ASSERT(mentions);
    TEST_ASSERT(strstr(msg, "t.toml:") == msg); /* every problem names the file */
  }

  /* A string longer than the member is refused, not truncated. */
  char big[400];
  memset(big, 'x', sizeof big - 1);
  big[sizeof big - 1] = '\0';
  MFTLSettingsDefaults(&s);
  TEST_ASSERT(MFTLSettingsSet(&s, "output.prefix", big, msg, sizeof msg) == MFTL_ERROR);
  TEST_ASSERT(strstr(msg, "longer than") != NULL);

  /* Every problem is reported, not just the first. */
  TEST_ASSERT(parse_over_defaults(&s, "[remesh]\nevry = 1\nitertions = 2\n[nope]\n", msg, sizeof msg) == MFTL_ERROR);
  int lines = 0;
  for (const char *p = msg; *p; ++p) lines += (*p == '\n');
  TEST_ASSERT(lines == 3);
  return 0;
}

/** @brief A text with any mistake in it changes nothing at all. */
static int test_a_failed_parse_changes_nothing(void)
{
  MFTL_Settings s, before;
  char msg[4096];
  MFTLSettingsDefaults(&s);
  TEST_ASSERT(MFTLSettingsSet(&s, "remesh.every", "7", msg, sizeof msg) == MFTL_SUCCESS);
  before = s;

  /* The valid lines come first, so a line-at-a-time parser would have applied them. */
  TEST_ASSERT(MFTLSettingsParse(&s, "[remesh]\nevery = 3\niterations = 9\n[domain]\ncuts = 1\nperiodic = maybe\n", "t", msg, sizeof msg) == MFTL_ERROR);
  TEST_ASSERT(same_settings(&s, &before));

  /* The same for a combination that is only wrong as a whole. */
  TEST_ASSERT(MFTLSettingsParse(&s, "[output]\nlayout = single_file\nencoding = ascii\n", "t", msg, sizeof msg) == MFTL_ERROR);
  TEST_ASSERT(same_settings(&s, &before));
  return 0;
}

/** @brief The key/value entry point a solver with its own input format uses. */
static int test_set_and_get(void)
{
  MFTL_Settings s;
  char msg[1024], v[64];
  MFTLSettingsDefaults(&s);

  TEST_ASSERT(MFTLSettingsSet(&s, "remesh.parallel_strategy", "spanning", msg, sizeof msg) == MFTL_SUCCESS);
  TEST_ASSERT(s.remesh.params.parallel_strategy == MFTL_REMESH_PARALLEL_SPANNING);
  TEST_ASSERT(MFTLSettingsGet(&s, "remesh.parallel_strategy", v, sizeof v) == MFTL_SUCCESS);
  TEST_ASSERT(strcmp(v, "\"spanning\"") == 0);

  TEST_ASSERT(MFTLSettingsSet(&s, "domain.cuts", "[1, 2, 3]", msg, sizeof msg) == MFTL_SUCCESS);
  TEST_ASSERT(MFTLSettingsGet(&s, "domain.cuts", v, sizeof v) == MFTL_SUCCESS);
  TEST_ASSERT(strcmp(v, "[1, 2, 3]") == 0);

  TEST_ASSERT(MFTLSettingsGet(&s, "remesh.every", v, sizeof v) == MFTL_SUCCESS && strcmp(v, "1") == 0);
  TEST_ASSERT(MFTLSettingsGet(&s, "remesh.target_length", v, sizeof v) == MFTL_SUCCESS && strcmp(v, "0.0") == 0);
  TEST_ASSERT(MFTLSettingsGet(&s, "contact.proximity_factor", v, sizeof v) == MFTL_SUCCESS && strcmp(v, "0.001") == 0);

  /* A value cannot smuggle in a second assignment. */
  TEST_ASSERT(MFTLSettingsSet(&s, "remesh.every", "3\nremesh.iterations = 99", msg, sizeof msg) == MFTL_ERROR);
  TEST_ASSERT(s.remesh.every == 1 && s.remesh.params.iterations == 3);

  TEST_ASSERT(MFTLSettingsSet(&s, "every", "3", msg, sizeof msg) == MFTL_ERROR);
  TEST_ASSERT(MFTLSettingsSet(&s, "remesh.evry", "3", msg, sizeof msg) == MFTL_ERROR);
  TEST_ASSERT(strstr(msg, "remesh.every") != NULL);
  TEST_ASSERT(MFTLSettingsGet(&s, "remesh.evry", v, sizeof v) == MFTL_ERROR);
  TEST_ASSERT(MFTLSettingsGet(&s, "domain.global_max", v, 4) == MFTL_ERROR); /* buffer too small */
  return 0;
}

/** @brief The annotated template names every key, and the changed form only the changes. */
static int test_the_written_forms(void)
{
  MFTL_Settings d;
  char msg[1024];
  MFTLSettingsDefaults(&d);

  char *tpl = write_to_string(&d, MFTL_SETTINGS_WRITE_DOCUMENTED);
  TEST_ASSERT(tpl != NULL);
  for (int s = 0; s < MFTLSettingsSectionCount(); ++s)
  {
    char head[80];
    snprintf(head, sizeof head, "\n[%s]\n", MFTLSettingsSection(s)->name);
    TEST_ASSERT(strstr(tpl, head) != NULL);
  }
  for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
  {
    char line[200];
    snprintf(line, sizeof line, "\n%s ", MFTLSettingsKey(i)->name);
    TEST_ASSERT(strstr(tpl, line) != NULL);
  }
  TEST_ASSERT(strstr(tpl, "\"dispersed\"") != NULL); /* keyword choices are listed */
  free(tpl);

  MFTL_Settings s = d;
  TEST_ASSERT(MFTLSettingsSet(&s, "topology.every", "4", msg, sizeof msg) == MFTL_SUCCESS);
  TEST_ASSERT(MFTLSettingsSet(&s, "output.encoding", "ascii", msg, sizeof msg) == MFTL_SUCCESS);
  char *changed = write_to_string(&s, MFTL_SETTINGS_WRITE_CHANGED);
  TEST_ASSERT(changed != NULL);
  int assignments = 0;
  for (const char *p = changed; (p = strstr(p, " = ")) != NULL; ++p) ++assignments;
  TEST_ASSERT(assignments == 2);
  TEST_ASSERT(strstr(changed, "[remesh]") == NULL);
  free(changed);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * The process grid
 * ─────────────────────────────────────────────────────────────────────────── */

static bool cuts_are(const MFTL_Settings *s, int nproc, int a, int b, int c)
{
  int cuts[3] = {-1, -1, -1};
  if (MFTLSettingsChooseCuts(s, nproc, cuts) != MFTL_SUCCESS) return false;
  if (cuts[0] != a || cuts[1] != b || cuts[2] != c) printf("\n  np %d: got %d x %d x %d, expected %d x %d x %d", nproc, cuts[0], cuts[1], cuts[2], a, b, c);
  return cuts[0] == a && cuts[1] == b && cuts[2] == c;
}

/**
 * @brief The grid keeps the narrowest rank box as wide as the rank count
 *        allows, and honours fixed entries.
 *
 * On a cube this reproduces the grids the examples and tests use by hand, which
 * is the check that the rule is the one people were already applying.
 */
static int test_the_grid_protects_the_narrowest_box(void)
{
  MFTL_Settings s;
  MFTLSettingsDefaults(&s);
  TEST_ASSERT(cuts_are(&s, 1, 1, 1, 1));
  TEST_ASSERT(cuts_are(&s, 2, 2, 1, 1));
  TEST_ASSERT(cuts_are(&s, 4, 2, 2, 1));
  TEST_ASSERT(cuts_are(&s, 6, 3, 2, 1));
  TEST_ASSERT(cuts_are(&s, 8, 2, 2, 2));
  TEST_ASSERT(cuts_are(&s, 12, 3, 2, 2));
  TEST_ASSERT(cuts_are(&s, 27, 3, 3, 3));
  TEST_ASSERT(cuts_are(&s, 7, 7, 1, 1));

  /* The table examples/leveque_remeshing_parallel.c spells out by hand. */
  TEST_ASSERT(cuts_are(&s, 3, 3, 1, 1));
  TEST_ASSERT(cuts_are(&s, 16, 4, 2, 2));
  TEST_ASSERT(cuts_are(&s, 24, 4, 3, 2));
  TEST_ASSERT(cuts_are(&s, 32, 4, 4, 2));

  /* A long channel is cut along its length. */
  s.domain.global_max[0] = 24.0;
  s.domain.global_max[1] = 4.0;
  s.domain.global_max[2] = 4.0;
  TEST_ASSERT(cuts_are(&s, 4, 4, 1, 1));
  TEST_ASSERT(cuts_are(&s, 24, 12, 2, 1)); /* ties (6,2,2) on both widths; more cuts on x wins */

  /* Fixed entries are kept; the rest is chosen. */
  MFTLSettingsDefaults(&s);
  s.domain.cuts[2] = 1;
  TEST_ASSERT(cuts_are(&s, 8, 4, 2, 1));
  s.domain.cuts[0] = 3;
  int cuts[3];
  TEST_ASSERT(MFTLSettingsChooseCuts(&s, 4, cuts) == MFTL_ERROR); /* 3 does not divide 4 */
  TEST_ASSERT(cuts_are(&s, 6, 3, 2, 1));
  s.domain.cuts[1] = 2;
  TEST_ASSERT(MFTLSettingsChooseCuts(&s, 12, cuts) == MFTL_ERROR); /* fully fixed and wrong */
  TEST_ASSERT(cuts_are(&s, 6, 3, 2, 1));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Collective
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief A file only rank 0 can read ends up identical on every rank. */
static int test_load_reads_on_rank_zero_only(void)
{
  const char *path = "test_settings_load.toml";
  if (g_rank == 0)
  {
    FILE *f = fopen(path, "w");
    if (f)
    {
      fprintf(f, "[remesh]\nevery = 3\nparallel_strategy = \"dispersed\"\n[output]\nprefix = \"from_rank_zero\"\n");
      fclose(f);
    }
  }
  MFTL_Settings s;
  MFTLSettingsDefaults(&s);
  TEST_ASSERT(agreed(MFTLSettingsLoad(&s, path, MPI_COMM_WORLD) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(s.remesh.every == 3 && s.remesh.params.parallel_strategy == MFTL_REMESH_PARALLEL_DISPERSED, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(strcmp(s.output.prefix, "from_rank_zero") == 0, MPI_COMM_WORLD));

  /* A broken file fails on EVERY rank, and changes nothing anywhere. */
  if (g_rank == 0)
  {
    FILE *f = fopen(path, "w");
    if (f)
    {
      fprintf(f, "[remesh]\nevery = 5\nnot_a_key = 1\n");
      fclose(f);
    }
  }
  if (g_rank == 0) printf("\n  (an invalid-settings report is expected here)\n  ");
  TEST_ASSERT(agreed(MFTLSettingsLoad(&s, path, MPI_COMM_WORLD) == MFTL_ERROR, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(s.remesh.every == 3, MPI_COMM_WORLD));

  /* So does a missing one. */
  if (g_rank == 0) remove(path);
  if (g_rank == 0) printf("\n  (a cannot-open report is expected here)\n  ");
  TEST_ASSERT(agreed(MFTLSettingsLoad(&s, path, MPI_COMM_WORLD) == MFTL_ERROR, MPI_COMM_WORLD));
  return 0;
}

/** @brief `section.key=value` arguments override; everything else is the application's. */
static int test_command_line_overrides(void)
{
  TEST_ASSERT(MFTLSettingsIsOverride("remesh.every=3"));
  TEST_ASSERT(MFTLSettingsIsOverride("--remesh.every=3"));
  TEST_ASSERT(!MFTLSettingsIsOverride("--steps=3"));
  TEST_ASSERT(!MFTLSettingsIsOverride("run.toml"));
  TEST_ASSERT(!MFTLSettingsIsOverride("=3"));
  TEST_ASSERT(!MFTLSettingsIsOverride("a b.c=3"));
  TEST_ASSERT(!MFTLSettingsIsOverride(NULL));

  char *argv[] = {"prog", "run.toml", "--steps", "12", "remesh.every=7", "--output.prefix=out/run_a", "domain.cuts=0,0,1", "domain.periodic=true", NULL};
  const int argc = (int) (sizeof argv / sizeof argv[0]) - 1;

  MFTL_Settings s;
  MFTLSettingsDefaults(&s);
  TEST_ASSERT(agreed(MFTLSettingsLoadArgs(&s, argc, argv, MPI_COMM_WORLD) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(s.remesh.every == 7, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(strcmp(s.output.prefix, "out/run_a") == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(s.domain.cuts[2] == 1 && s.domain.periodic[1], MPI_COMM_WORLD));

  char *bad[] = {"prog", "remesh.evry=3", NULL};
  if (g_rank == 0) printf("\n  (an invalid-settings report is expected here)\n  ");
  TEST_ASSERT(agreed(MFTLSettingsLoadArgs(&s, 2, bad, MPI_COMM_WORLD) == MFTL_ERROR, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(s.remesh.every == 7, MPI_COMM_WORLD));
  return 0;
}

/** @brief The ensemble is the one `[domain]` describes, under the `[redistribute]` policy. */
static int test_the_ensemble_is_the_one_described(void)
{
  const bool skip0 = MFTLRedistributeGetSkipUnchangedBodies();
  const double slack0 = MFTLRedistributeGetShrinkSlack();

  MFTL_Settings s;
  char msg[1024];
  MFTLSettingsDefaults(&s);
  TEST_ASSERT(MFTLSettingsParse(&s,
                                "[domain]\nglobal_min = -1\nglobal_max = [1, 2, 3]\nperiodic = [true, false, true]\n"
                                "[redistribute]\nskip_unchanged_bodies = false\nshrink_slack = 4\n",
                                "t", msg, sizeof msg) == MFTL_SUCCESS);

  MFTL_Ensemble *ens = MFTLSettingsCreateEnsemble(&s, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(ens != NULL, MPI_COMM_WORLD));

  int cuts[3];
  TEST_ASSERT(MFTLSettingsChooseCuts(&s, g_nproc, cuts) == MFTL_SUCCESS);
  bool ok = true;
  for (int k = 0; k < 3; ++k)
  {
    ok = ok && ens->domain.cuts[k] == cuts[k];
    ok = ok && ens->domain.global_min[k] == -1.0 && ens->domain.global_max[k] == (double) (k + 1);
    ok = ok && ens->domain.is_periodic[k] == (k != 1);
  }
  ok = ok && !MFTLRedistributeGetSkipUnchangedBodies() && MFTLRedistributeGetShrinkSlack() == 4.0;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);

  MFTLRedistributeSetSkipUnchangedBodies(skip0);
  MFTLRedistributeSetShrinkSlack(slack0);

  /* Fixed cuts that cannot fit the job refuse on every rank. */
  s.domain.cuts[0] = g_nproc + 1;
  if (g_rank == 0) printf("\n  (a cannot-be-completed report is expected here)\n  ");
  ens = MFTLSettingsCreateEnsemble(&s, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(ens == NULL, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief The lengths left at 0 are derived from the mesh, identically on every
 *        rank, and the resolved block drives a real distributed remesh.
 */
static int test_resolve_derives_the_lengths(void)
{
  MFTL_Settings s;
  char msg[1024];
  MFTLSettingsDefaults(&s);
  TEST_ASSERT(MFTLSettingsParse(&s, "[sphere]\nrefinements = 2\nradius = 0.3\n[topology]\ndelta_coalescence = 0.05\n", "t", msg, sizeof msg) ==
              MFTL_SUCCESS);

  MFTL_Ensemble *ens = MFTLSettingsCreateEnsemble(&s, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(ens != NULL, MPI_COMM_WORLD));
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  MFTLGenerateSphere(obj, s.sphere.center, s.sphere.radius, s.sphere.refinements);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  const double L = MFTLEnsembleMeanEdgeLength(ens);
  TEST_ASSERT(agreed(MFTLSettingsResolve(&s, ens) == MFTL_SUCCESS, MPI_COMM_WORLD));

  const MFTL_RemeshParameters *p = &s.remesh.params;
  bool ok = L > 0.0 && p->target_length == L;
  ok = ok && p->min_edge_length == (4.0 / 5.0) * L && p->max_edge_length == (4.0 / 3.0) * L;
  ok = ok && p->min_edge_length_sq == p->min_edge_length * p->min_edge_length && p->max_edge_length_sq == p->max_edge_length * p->max_edge_length;
  ok = ok && s.topology.options.dx == L && s.topology.options.delta_breakup == 1.5 * L;
  ok = ok && s.topology.options.delta_coalescence == 0.05; /* given, so kept */
  ok = ok && s.topology.options.remesh == p;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* Bit-identical on every rank. */
  double mine[2] = {p->target_length, -p->target_length};
  MPI_Allreduce(MPI_IN_PLACE, mine, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(mine[0] == -mine[1], MPI_COMM_WORLD));

  /* Idempotent. */
  MFTL_Settings again = s;
  TEST_ASSERT(agreed(MFTLSettingsResolve(&again, ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(same_settings(&s, &again), MPI_COMM_WORLD));

  /* The resolved block is a valid remeshing configuration. */
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTL_RemeshStats st;
  memset(&st, 0, sizeof st);
  TEST_ASSERT(agreed(MFTLRemeshEnsemble(ens, &s.remesh.params, &st) == MFTL_SUCCESS, MPI_COMM_WORLD));
  bool sound = true;
  for (int o = 0; o < ens->n_local_objects; ++o) sound = sound && inv_partition_is_sound(ens->local_objects[o], g_rank);
  TEST_ASSERT(agreed(sound, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief Nothing to measure is an error exactly when something scheduled needs the length. */
static int test_resolve_on_an_empty_ensemble(void)
{
  MFTL_Settings s;
  MFTLSettingsDefaults(&s);
  MFTL_Ensemble *ens = MFTLSettingsCreateEnsemble(&s, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(ens != NULL, MPI_COMM_WORLD));

  if (g_rank == 0) printf("\n  (a no-edges report is expected here)\n  ");
  TEST_ASSERT(agreed(MFTLSettingsResolve(&s, ens) == MFTL_ERROR, MPI_COMM_WORLD)); /* remesh.every = 1 */

  s.remesh.every = 0;
  TEST_ASSERT(agreed(MFTLSettingsResolve(&s, ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(s.remesh.params.target_length == 0.0, MPI_COMM_WORLD));

  /* Explicit lengths need no mesh at all. */
  s.remesh.every = 1;
  s.remesh.params.target_length = 0.1;
  TEST_ASSERT(agreed(MFTLSettingsResolve(&s, ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(s.remesh.params.max_edge_length - 0.4 / 3.0) < 1e-15, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Ranks holding different settings are caught, on every rank, before
 *        they can disagree about a collective.
 */
static int test_disagreeing_ranks_are_caught(void)
{
  MFTL_Settings s;
  MFTLSettingsDefaults(&s);
  s.remesh.every = 0;
  MFTL_Ensemble *ens = MFTLSettingsCreateEnsemble(&s, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(ens != NULL, MPI_COMM_WORLD));

  MFTL_Settings mine = s;
  mine.remesh.params.iterations = 3 + g_rank;
  if (g_rank == 0 && g_nproc > 1) printf("\n  (two different-settings reports are expected here)\n  ");
  const int rc = MFTLSettingsResolve(&mine, ens);
  TEST_ASSERT(agreed(rc == ((g_nproc > 1) ? MFTL_ERROR : MFTL_SUCCESS), MPI_COMM_WORLD));

  MFTL_Ensemble *other = MFTLSettingsCreateEnsemble(&mine, MPI_COMM_WORLD);
  TEST_ASSERT(agreed((other == NULL) == (g_nproc > 1), MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&other);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== MFTL settings tests (%d rank%s) ===\n", g_nproc, g_nproc == 1 ? "" : "s");

  RUN_TEST(test_every_key_matches_its_member);
  RUN_TEST(test_every_default_is_admissible);
  RUN_TEST(test_defaults_follow_the_modules);
  RUN_TEST(test_round_trip_is_exact);
  RUN_TEST(test_the_accepted_syntax);
  RUN_TEST(test_mistakes_are_errors);
  RUN_TEST(test_a_failed_parse_changes_nothing);
  RUN_TEST(test_set_and_get);
  RUN_TEST(test_the_written_forms);
  RUN_TEST(test_the_grid_protects_the_narrowest_box);
  RUN_TEST(test_load_reads_on_rank_zero_only);
  RUN_TEST(test_command_line_overrides);
  RUN_TEST(test_the_ensemble_is_the_one_described);
  RUN_TEST(test_resolve_derives_the_lengths);
  RUN_TEST(test_resolve_on_an_empty_ensemble);
  RUN_TEST(test_disagreeing_ranks_are_caught);

  if (g_rank == 0) TEST_REPORT();

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
