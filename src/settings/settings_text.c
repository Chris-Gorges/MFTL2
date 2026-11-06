/**
 * @file settings_text.c
 * @author Christian Gorges
 * @date 2026-09-29
 * @ingroup mftl_settings
 *
 * @brief Settings values and their text form: the defaults, the parser, the
 *        writer and the checks of combinations. Rank-local; no MPI.
 *
 * Everything here is driven by the key table in settings_keys.c. The parser
 * never special-cases a key: a key's type decides how its value is read, its
 * range and choices decide whether the value is admissible, and its offset
 * decides where it goes. The only per-key knowledge in this file is in
 * MFTLSettingsCheck(), which is about COMBINATIONS no single key can express.
 *
 * Two properties are load-bearing:
 *
 * - **All or nothing.** A text is applied to a copy and committed only if every
 *   line and the final combination check pass, so a caller never holds a
 *   half-applied file.
 * - **Round trip.** MFTLSettingsGet() writes reals in the shortest form that
 *   reads back to the same double, so MFTLSettingsWrite() followed by
 *   MFTLSettingsParse() reproduces the struct bit for bit -- which is what makes
 *   the printed configuration of a run a way to reproduce it.
 */

#include "mftl_settings.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"

/** Longest line of settings text; a longer one is an error rather than a silent truncation. */
#define LINE_MAX_CHARS 4096

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Diagnostics
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Accumulates problems as lines of `origin:line: message`. */
typedef struct report
{
  char *buf;  ///< Caller's buffer, or NULL to count only.
  int cap;  ///< Capacity of `buf`.
  int len;  ///< Characters written so far.
  int count;  ///< Problems reported, including those that did not fit.
} report;

static void report_init(report *r, char *buf, int cap)
{
  r->buf = (buf && cap > 0) ? buf : MFTL_NULL;
  r->cap = r->buf ? cap : 0;
  r->len = 0;
  r->count = 0;
  if (r->buf) r->buf[0] = '\0';
}

/** Appends one problem; @p line 0 omits the line number. */
static void report_add(report *r, const char *origin, int line, const char *fmt, ...)
{
  r->count++;
  if (!r->buf) return;

  char text[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(text, sizeof text, fmt, ap);
  va_end(ap);

  const int room = r->cap - r->len;
  if (room <= 1) return;
  const int n = (line > 0) ? snprintf(r->buf + r->len, (size_t) room, "%s:%d: %s\n", origin, line, text)
                           : snprintf(r->buf + r->len, (size_t) room, "%s: %s\n", origin, text);
  if (n > 0) r->len += (n < room) ? n : room - 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Small helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static int key_index(const MFTL_SettingInfo *k) { return (int) (k - MFTLSettingsKey(0)); }

static void *value_ptr(MFTL_Settings *s, const MFTL_SettingInfo *k) { return (char *) s + k->offset; }

static const void *value_cptr(const MFTL_Settings *s, const MFTL_SettingInfo *k) { return (const char *) s + k->offset; }

static char *trim(char *p)
{
  while (*p == ' ' || *p == '\t') ++p;
  char *end = p + strlen(p);
  while (end > p && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) --end;
  *end = '\0';
  return p;
}

/** Levenshtein distance; strings longer than 127 bytes are simply "far". */
static int edit_distance(const char *a, const char *b)
{
  const size_t la = strlen(a), lb = strlen(b);
  if (la > 127 || lb > 127) return INT_MAX / 2;
  int prev[128], cur[128];
  for (size_t j = 0; j <= lb; ++j) prev[j] = (int) j;
  for (size_t i = 1; i <= la; ++i)
  {
    cur[0] = (int) i;
    for (size_t j = 1; j <= lb; ++j)
    {
      const int sub = prev[j - 1] + (a[i - 1] != b[j - 1]);
      const int del = prev[j] + 1;
      const int ins = cur[j - 1] + 1;
      cur[j] = MFTL_MIN(sub, MFTL_MIN(del, ins));
    }
    memcpy(prev, cur, (lb + 1) * sizeof(int));
  }
  return prev[lb];
}

/** Close enough to be a typo of each other. */
static bool is_near(int distance, const char *typed) { return distance <= 2 || distance <= (int) strlen(typed) / 4; }

/** The key a mistyped @p full probably meant: the same name in another section, else the nearest key. */
static bool suggest_key(const char *full, char *out, size_t n)
{
  const char *dot = strchr(full, '.');
  const char *name = dot ? dot + 1 : full;

  for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(i);
    if (strcmp(k->name, name) == 0)
    {
      snprintf(out, n, "%s.%s", k->section, k->name);
      return true;
    }
  }

  int best = INT_MAX;
  for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(i);
    char cand[160];
    snprintf(cand, sizeof cand, "%s.%s", k->section, k->name);
    const int d = edit_distance(dot ? full : name, dot ? cand : k->name);
    if (d < best)
    {
      best = d;
      snprintf(out, n, "%s", cand);
    }
  }
  return is_near(best, full);
}

static const MFTL_SettingSection *find_section(const char *name)
{
  for (int i = 0; i < MFTLSettingsSectionCount(); ++i)
  {
    const MFTL_SettingSection *sec = MFTLSettingsSection(i);
    if (strcmp(sec->name, name) == 0) return sec;
  }
  return MFTL_NULL;
}

static const char *suggest_section(const char *name)
{
  int best = INT_MAX;
  const char *hit = MFTL_NULL;
  for (int i = 0; i < MFTLSettingsSectionCount(); ++i)
  {
    const int d = edit_distance(name, MFTLSettingsSection(i)->name);
    if (d < best)
    {
      best = d;
      hit = MFTLSettingsSection(i)->name;
    }
  }
  return is_near(best, name) ? hit : MFTL_NULL;
}

/** `"a", "b" or "c"` */
static void list_choices(const MFTL_SettingChoice *c, char *out, size_t n)
{
  out[0] = '\0';
  int count = 0;
  while (c[count].name) ++count;
  size_t len = 0;
  for (int i = 0; i < count && len < n; ++i)
  {
    const char *sep = (i == 0) ? "" : (i == count - 1) ? " or " : ", ";
    const int w = snprintf(out + len, n - len, "%s\"%s\"", sep, c[i].name);
    if (w < 0) break;
    len += (size_t) w;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Ranges
 *
 * Written the way the manual shows them -- "[0, 1]", "(0, inf)", "[1, 64]" --
 * and read here, so the documented interval and the enforced one are the same
 * string. The tests parse every range in the table.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief A parsed interval; `bounded == false` admits everything. */
typedef struct range_t
{
  bool bounded;  ///< False for an empty range string.
  bool lo_open;  ///< `(` rather than `[`.
  bool hi_open;  ///< `)` rather than `]`.
  double lo;  ///< Lower bound.
  double hi;  ///< Upper bound.
} range_t;

static bool parse_bound(const char **p, double *v)
{
  const char *s = *p;
  while (*s == ' ') ++s;
  if (strncmp(s, "inf", 3) == 0 || strncmp(s, "+inf", 4) == 0 || strncmp(s, "-inf", 4) == 0)
  {
    *v = (*s == '-') ? -HUGE_VAL : HUGE_VAL;
    *p = s + ((*s == 'i') ? 3 : 4);
    return true;
  }
  char *end = MFTL_NULL;
  *v = strtod(s, &end);
  if (end == s) return false;
  *p = end;
  return true;
}

static bool parse_range(const char *text, range_t *r)
{
  memset(r, 0, sizeof *r);
  if (!text || !*text) return true;

  const char *p = text;
  if (*p != '[' && *p != '(') return false;
  r->lo_open = (*p == '(');
  ++p;
  if (!parse_bound(&p, &r->lo)) return false;
  while (*p == ' ') ++p;
  if (*p != ',') return false;
  ++p;
  if (!parse_bound(&p, &r->hi)) return false;
  while (*p == ' ') ++p;
  if (*p != ']' && *p != ')') return false;
  r->hi_open = (*p == ')');
  ++p;
  if (*p != '\0' || !(r->lo <= r->hi)) return false;
  r->bounded = true;
  return true;
}

static bool range_contains(const range_t *r, double v)
{
  if (!r->bounded) return true;
  if (r->lo_open ? !(v > r->lo) : !(v >= r->lo)) return false;
  if (r->hi_open ? !(v < r->hi) : !(v <= r->hi)) return false;
  return true;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Reading one value
 * ═══════════════════════════════════════════════════════════════════════════ */

static bool read_int(const char *t, int *v)
{
  if (!*t) return false;
  errno = 0;
  char *end = MFTL_NULL;
  const long x = strtol(t, &end, 10);
  if (end == t || *end != '\0' || errno == ERANGE || x < INT_MIN || x > INT_MAX) return false;
  *v = (int) x;
  return true;
}

/** A decimal real or `[+-]inf`. `nan` and hexadecimal floats are refused. */
static bool read_real(const char *t, double *v)
{
  if (!strcmp(t, "inf") || !strcmp(t, "+inf"))
  {
    *v = HUGE_VAL;
    return true;
  }
  if (!strcmp(t, "-inf"))
  {
    *v = -HUGE_VAL;
    return true;
  }
  if (!*t || strpbrk(t, "xXnN")) return false;
  errno = 0;
  char *end = MFTL_NULL;
  const double x = strtod(t, &end);
  if (end == t || *end != '\0') return false;
  if (errno == ERANGE && fabs(x) == HUGE_VAL) return false;
  *v = x;
  return true;
}

static bool read_bool(const char *t, bool *v)
{
  if (!strcmp(t, "true"))
  {
    *v = MFTL_TRUE;
    return true;
  }
  if (!strcmp(t, "false"))
  {
    *v = MFTL_FALSE;
    return true;
  }
  return false;
}

/**
 * @brief Reads a TOML basic ("..."), literal ('...') or bare string.
 * @return 1 on success, 0 if malformed, -1 if longer than @p n - 1 bytes.
 */
static int read_string(const char *t, char *out, size_t n)
{
  size_t len = 0;
  const char q = *t;

  if (q != '"' && q != '\'')
  {
    const size_t l = strlen(t);
    if (l + 1 > n) return -1;
    memcpy(out, t, l + 1);
    return 1;
  }

  const char *p = t + 1;
  for (; *p && *p != q; ++p)
  {
    char c = *p;
    if (q == '"' && c == '\\')
    {
      ++p;
      switch (*p)
      {
        case 'n': c = '\n'; break;
        case 't': c = '\t'; break;
        case 'r': c = '\r'; break;
        case '"': c = '"'; break;
        case '\\': c = '\\'; break;
        case 'u':
        {
          unsigned code = 0;
          for (int i = 1; i <= 4; ++i)
          {
            const char h = p[i];
            const int d = (h >= '0' && h <= '9') ? h - '0' : (h >= 'a' && h <= 'f') ? h - 'a' + 10 : (h >= 'A' && h <= 'F') ? h - 'A' + 10 : -1;
            if (d < 0) return 0;
            code = 16u * code + (unsigned) d;
          }
          if (code == 0 || code > 0x7f) return 0; /* ASCII only */
          c = (char) code;
          p += 4;
          break;
        }
        default: return 0;
      }
    }
    if (len + 1 >= n) return -1;
    out[len++] = c;
  }
  if (*p != q || p[1] != '\0') return 0;
  out[len] = '\0';
  return 1;
}

/** Splits `[a, b, c]` or `a, b, c` in place into trimmed elements. @return the count, or -1 if malformed. */
static int split_list(char *t, char *elem[], int max_elem)
{
  size_t len = strlen(t);
  if (t[0] == '[')
  {
    if (len < 2 || t[len - 1] != ']') return -1;
    t[len - 1] = '\0';
    ++t;
  }
  int n = 0;
  char *p = t;
  while (true)
  {
    char *comma = strchr(p, ',');
    if (comma) *comma = '\0';
    char *e = trim(p);
    if (*e == '\0')
    {
      /* A trailing comma is legal TOML; an empty element anywhere else is not. */
      if (!comma && n > 0) break;
      return -1;
    }
    if (n == max_elem) return -1;
    elem[n++] = e;
    if (!comma) break;
    p = comma + 1;
  }
  return n;
}

/** @brief Reads one scalar component of a numeric or boolean key into @p out, checking the range. */
static bool read_component(const MFTL_SettingInfo *k, const range_t *r, const char *t, void *out, char *why, size_t n)
{
  switch (k->type)
  {
    case MFTL_SETTING_INT:
    case MFTL_SETTING_INT3:
    {
      int v = 0;
      if (!read_int(t, &v))
      {
        snprintf(why, n, "expected an integer, got `%s`", t);
        return false;
      }
      if (!range_contains(r, (double) v))
      {
        snprintf(why, n, "%d is outside %s", v, k->range);
        return false;
      }
      *(int *) out = v;
      return true;
    }
    case MFTL_SETTING_REAL:
    case MFTL_SETTING_REAL3:
    {
      double v = 0.0;
      if (!read_real(t, &v))
      {
        snprintf(why, n, "expected a number, got `%s`", t);
        return false;
      }
      if (!range_contains(r, v))
      {
        snprintf(why, n, "%s is outside %s", t, k->range);
        return false;
      }
      *(double *) out = v;
      return true;
    }
    case MFTL_SETTING_BOOL:
    case MFTL_SETTING_BOOL3:
    {
      bool v = MFTL_FALSE;
      if (!read_bool(t, &v))
      {
        snprintf(why, n, "expected true or false, got `%s`", t);
        return false;
      }
      *(bool *) out = v;
      return true;
    }
    default: return false;
  }
}

/**
 * @brief Parses @p text as the value of @p k and stores it into @p s.
 *
 * Nothing is stored unless the whole value is valid.
 *
 * @param      k    The key.
 * @param      s    The settings written to.
 * @param      text The value in file syntax, comment-free and trimmed.
 * @param[out] why  The reason, on failure.
 * @param      n    Capacity of @p why.
 * @return `true` if the value was stored.
 */
static bool apply_value(const MFTL_SettingInfo *k, MFTL_Settings *s, const char *text, char *why, size_t n)
{
  range_t r;
  if (!parse_range(k->range, &r))
  {
    snprintf(why, n, "the key table has a malformed range \"%s\"", k->range);
    return false;
  }
  void *dst = value_ptr(s, k);

  switch (k->type)
  {
    case MFTL_SETTING_INT:
    case MFTL_SETTING_REAL:
    case MFTL_SETTING_BOOL:
    {
      union
      {
        int i;
        double d;
        bool b;
      } v;
      if (!read_component(k, &r, text, &v, why, n)) return false;
      if (k->type == MFTL_SETTING_INT) *(int *) dst = v.i;
      if (k->type == MFTL_SETTING_REAL) *(double *) dst = v.d;
      if (k->type == MFTL_SETTING_BOOL) *(bool *) dst = v.b;
      return true;
    }

    case MFTL_SETTING_INT3:
    case MFTL_SETTING_REAL3:
    case MFTL_SETTING_BOOL3:
    {
      char copy[LINE_MAX_CHARS];
      snprintf(copy, sizeof copy, "%s", text);
      char *elem[4];
      const int count = split_list(copy, elem, 4);
      if (count != 1 && count != 3)
      {
        snprintf(why, n, "expected three values or one for all three, e.g. %s = %s", k->name,
                 (k->type == MFTL_SETTING_INT3) ? "[2, 2, 1]" : (k->type == MFTL_SETTING_REAL3) ? "[0.0, 0.0, 1.5]" : "[true, false, false]");
        return false;
      }
      union
      {
        int i[3];
        double d[3];
        bool b[3];
      } v;
      for (int c = 0; c < 3; ++c)
      {
        const char *t = elem[(count == 3) ? c : 0];
        void *out = (k->type == MFTL_SETTING_INT3) ? (void *) &v.i[c] : (k->type == MFTL_SETTING_REAL3) ? (void *) &v.d[c] : (void *) &v.b[c];
        if (!read_component(k, &r, t, out, why, n)) return false;
      }
      if (k->type == MFTL_SETTING_INT3) memcpy(dst, v.i, sizeof v.i);
      if (k->type == MFTL_SETTING_REAL3) memcpy(dst, v.d, sizeof v.d);
      if (k->type == MFTL_SETTING_BOOL3) memcpy(dst, v.b, sizeof v.b);
      return true;
    }

    case MFTL_SETTING_KEYWORD:
    {
      char name[64];
      if (read_string(text, name, sizeof name) != 1)
      {
        snprintf(why, n, "expected a keyword, got `%s`", text);
        return false;
      }
      int best = INT_MAX;
      const char *near = MFTL_NULL;
      for (const MFTL_SettingChoice *c = k->choices; c && c->name; ++c)
      {
        if (strcmp(c->name, name) == 0)
        {
          *(int *) dst = c->value;
          return true;
        }
        const int d = edit_distance(name, c->name);
        if (d < best)
        {
          best = d;
          near = c->name;
        }
      }
      char all[256];
      list_choices(k->choices, all, sizeof all);
      if (near && is_near(best, name))
        snprintf(why, n, "\"%s\" is not one of %s -- did you mean \"%s\"?", name, all, near);
      else
        snprintf(why, n, "\"%s\" is not one of %s", name, all);
      return false;
    }

    case MFTL_SETTING_STRING:
    {
      char buf[MFTL_SETTINGS_STRING_MAX];
      const size_t cap = MFTL_MIN(k->size, sizeof buf);
      const int rc = read_string(text, buf, cap);
      if (rc == -1)
      {
        snprintf(why, n, "longer than %d bytes", (int) cap - 1);
        return false;
      }
      if (rc != 1)
      {
        snprintf(why, n, "malformed string `%s`", text);
        return false;
      }
      memcpy(dst, buf, strlen(buf) + 1);
      return true;
    }
  }
  snprintf(why, n, "the key table has an unknown type");
  return false;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Writing one value
 * ═══════════════════════════════════════════════════════════════════════════ */

/** The shortest decimal that reads back to @p v, and always recognisable as a real. */
static void format_real(double v, char *out, size_t n)
{
  if (isinf(v))
  {
    snprintf(out, n, "%s", (v > 0) ? "inf" : "-inf");
    return;
  }
  if (isnan(v))
  {
    snprintf(out, n, "nan");
    return;
  }
  for (int p = 1; p <= 17; ++p)
  {
    snprintf(out, n, "%.*g", p, v);
    if (strtod(out, MFTL_NULL) == v) break;
  }
  if (!strpbrk(out, ".eE"))
  {
    const size_t l = strlen(out);
    if (l + 3 <= n) memcpy(out + l, ".0", 3);
  }
}

static void format_string(const char *s, char *out, size_t n)
{
  size_t len = 0;
  if (n < 3)
  {
    if (n) out[0] = '\0';
    return;
  }
  out[len++] = '"';
  for (const char *p = s; *p && len + 8 < n; ++p)
  {
    const unsigned char c = (unsigned char) *p;
    if (c == '"' || c == '\\')
    {
      out[len++] = '\\';
      out[len++] = (char) c;
    }
    else if (c == '\n')
      len += (size_t) snprintf(out + len, n - len, "\\n");
    else if (c == '\t')
      len += (size_t) snprintf(out + len, n - len, "\\t");
    else if (c < 0x20)
      len += (size_t) snprintf(out + len, n - len, "\\u%04x", c);
    else
      out[len++] = (char) c;
  }
  out[len++] = '"';
  out[len] = '\0';
}

/** @return the length written, or -1 when @p n is too small. */
static int format_value(const MFTL_SettingInfo *k, const MFTL_Settings *s, char *out, size_t n)
{
  char tmp[3 * 40 + 16];
  const void *src = value_cptr(s, k);

  switch (k->type)
  {
    case MFTL_SETTING_INT: snprintf(tmp, sizeof tmp, "%d", *(const int *) src); break;
    case MFTL_SETTING_REAL: format_real(*(const double *) src, tmp, sizeof tmp); break;
    case MFTL_SETTING_BOOL: snprintf(tmp, sizeof tmp, "%s", *(const bool *) src ? "true" : "false"); break;
    case MFTL_SETTING_INT3:
    {
      const int *v = (const int *) src;
      snprintf(tmp, sizeof tmp, "[%d, %d, %d]", v[0], v[1], v[2]);
      break;
    }
    case MFTL_SETTING_REAL3:
    {
      const double *v = (const double *) src;
      char a[40], b[40], c[40];
      format_real(v[0], a, sizeof a);
      format_real(v[1], b, sizeof b);
      format_real(v[2], c, sizeof c);
      snprintf(tmp, sizeof tmp, "[%s, %s, %s]", a, b, c);
      break;
    }
    case MFTL_SETTING_BOOL3:
    {
      const bool *v = (const bool *) src;
      snprintf(tmp, sizeof tmp, "[%s, %s, %s]", v[0] ? "true" : "false", v[1] ? "true" : "false", v[2] ? "true" : "false");
      break;
    }
    case MFTL_SETTING_KEYWORD:
    {
      const int v = *(const int *) src;
      /* A value outside the list is written bare, so that reading it back fails loudly. */
      snprintf(tmp, sizeof tmp, "%d", v);
      for (const MFTL_SettingChoice *c = k->choices; c && c->name; ++c)
      {
        if (c->value == v)
        {
          snprintf(tmp, sizeof tmp, "\"%s\"", c->name);
          break;
        }
      }
      break;
    }
    case MFTL_SETTING_STRING:
    {
      char big[2 * MFTL_SETTINGS_STRING_MAX + 16];
      format_string((const char *) src, big, sizeof big);
      const size_t l = strlen(big);
      if (l + 1 > n) return -1;
      memcpy(out, big, l + 1);
      return (int) l;
    }
  }

  const size_t l = strlen(tmp);
  if (l + 1 > n) return -1;
  memcpy(out, tmp, l + 1);
  return (int) l;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Defaults
 * ═══════════════════════════════════════════════════════════════════════════ */

static void link_topology_remesh(MFTL_Settings *s) { s->topology.options.remesh = s->topology.remesh_after ? &s->remesh.params : MFTL_NULL; }

void MFTLSettingsDefaults(MFTL_Settings *s)
{
  if (!s) return;
  memset(s, 0, sizeof *s);

  /* Module blocks start from their module's defaults, so the fields no key
   * covers (the band's squares, the remesh type, the pair filter) are exactly
   * what the module would give; the table then sets every key. */
  MFTLInitRemeshParams(&s->remesh.params, MFTL_REMESH_ISOTROPIC, 0.0, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  s->topology.options = MFTLTopologyDefaultOptions();
  s->volume.params = MFTLVolumeCorrectionDefaultParameters();
  s->undulation.params = MFTLUndulationDefaultParameters();
  s->contact.params = MFTL_BVHDefaultParams();
  MFTLCouplerDefaultParams(&s->coupling.params);
  s->output.vtk = MFTLVtkDefaults();

  for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(i);
    char why[256];
    const bool ok = apply_value(k, s, k->default_text, why, sizeof why);
    MFTL_ASSERT(ok && "every table default must parse (test_every_default_is_admissible)");
    (void) ok;
  }

  MFTLRemeshParamsFinalise(&s->remesh.params);
  link_topology_remesh(s);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. Checks of combinations
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Every key's CURRENT value against its range and choices -- the struct may have been edited in C. */
static void check_keys(const MFTL_Settings *s, report *rep, const char *origin)
{
  for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
  {
    const MFTL_SettingInfo *k = MFTLSettingsKey(i);
    range_t r;
    parse_range(k->range, &r);
    const void *src = value_cptr(s, k);
    const int dims = (k->type == MFTL_SETTING_INT3 || k->type == MFTL_SETTING_REAL3 || k->type == MFTL_SETTING_BOOL3) ? 3 : 1;

    for (int c = 0; c < dims; ++c)
    {
      bool ok = MFTL_TRUE;
      double v = 0.0;
      if (k->type == MFTL_SETTING_INT || k->type == MFTL_SETTING_INT3)
        v = (double) ((const int *) src)[c];
      else if (k->type == MFTL_SETTING_REAL || k->type == MFTL_SETTING_REAL3)
      {
        v = ((const double *) src)[c];
        if (isnan(v)) ok = MFTL_FALSE;
      }
      else
        continue;
      if (!ok || !range_contains(&r, v))
      {
        char text[80];
        format_value(k, s, text, sizeof text);
        report_add(rep, origin, 0, "%s.%s = %s is outside %s", k->section, k->name, text, k->range);
        break;
      }
    }

    if (k->type == MFTL_SETTING_KEYWORD)
    {
      const int v = *(const int *) src;
      bool found = MFTL_FALSE;
      for (const MFTL_SettingChoice *c = k->choices; c && c->name; ++c) found = found || (c->value == v);
      if (!found) report_add(rep, origin, 0, "%s.%s holds %d, which is not one of its keywords", k->section, k->name, v);
    }
    if (k->type == MFTL_SETTING_STRING && !memchr(src, '\0', k->size))
      report_add(rep, origin, 0, "%s.%s is not NUL-terminated within %d bytes", k->section, k->name, (int) k->size);
  }
}

static void check_combinations(const MFTL_Settings *s, report *rep, const char *origin)
{
  static const char axis[3] = {'x', 'y', 'z'};
  for (int k = 0; k < 3; ++k)
  {
    if (!(s->domain.global_max[k] > s->domain.global_min[k]))
      report_add(rep, origin, 0, "the domain is empty along %c: domain.global_max[%d] = %g is not greater than domain.global_min[%d] = %g", axis[k], k,
                 s->domain.global_max[k], k, s->domain.global_min[k]);
  }

  if (s->output.vtk.layout == MFTL_IO_SINGLE_FILE && s->output.vtk.encoding != MFTL_VTK_BINARY)
    report_add(rep, origin, 0, "output.layout = \"single_file\" needs output.encoding = \"binary\"");

  const MFTL_RemeshParameters *p = &s->remesh.params;
  if (p->min_edge_length > 0.0 && p->max_edge_length > 0.0 && !(p->min_edge_length <= p->max_edge_length))
    report_add(rep, origin, 0, "remesh.min_edge_length = %g exceeds remesh.max_edge_length = %g", p->min_edge_length, p->max_edge_length);

  const double slack = s->redistribute.shrink_slack;
  if (slack != 0.0 && slack < 1.0)
    report_add(rep, origin, 0, "redistribute.shrink_slack = %g: use 0 to disable, otherwise at least 1 (4 is the thrash-free choice)", slack);
}

int MFTLSettingsCheck(const MFTL_Settings *s, char *message, int message_size)
{
  report rep;
  report_init(&rep, message, message_size);
  if (!s)
  {
    report_add(&rep, "MFTLSettingsCheck", 0, "no settings");
    return MFTL_ERROR;
  }
  check_keys(s, &rep, "settings");
  check_combinations(s, &rep, "settings");
  return (rep.count == 0) ? MFTL_SUCCESS : MFTL_ERROR;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 8. The parser
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Where the parser is: the current section and what has been set. */
typedef struct parser
{
  MFTL_Settings *work;  ///< The copy being written.
  report *rep;  ///< Problems.
  const char *origin;  ///< Name for messages.
  char section[64];  ///< Current `[section]`, empty at the top level.
  bool section_ok;  ///< False after an unknown header: its keys are skipped, not reported again.
  int *seen;  ///< Per key: the line it was set on in this text, or 0.
} parser;

/** Cuts a `#` comment off @p line, respecting quoted strings. */
static void strip_comment(char *line)
{
  char quote = 0;
  for (char *p = line; *p; ++p)
  {
    if (quote)
    {
      if (quote == '"' && *p == '\\' && p[1])
        ++p;
      else if (*p == quote)
        quote = 0;
    }
    else if (*p == '"' || *p == '\'')
      quote = *p;
    else if (*p == '#')
    {
      *p = '\0';
      return;
    }
  }
}

static bool is_key_char(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.'; }

static void parse_header(parser *ps, char *line, int line_no)
{
  if (line[1] == '[')
  {
    report_add(ps->rep, ps->origin, line_no, "arrays of tables ([[...]]) are not part of the settings format");
    ps->section_ok = MFTL_FALSE;
    return;
  }
  char *close = strchr(line, ']');
  if (!close || *trim(close + 1) != '\0')
  {
    report_add(ps->rep, ps->origin, line_no, "malformed section header `%s`; expected `[name]`", line);
    ps->section_ok = MFTL_FALSE;
    return;
  }
  *close = '\0';
  const char *name = trim(line + 1);
  snprintf(ps->section, sizeof ps->section, "%s", name);

  if (find_section(name))
  {
    ps->section_ok = MFTL_TRUE;
    return;
  }
  ps->section_ok = MFTL_FALSE;
  const char *near = suggest_section(name);
  if (near)
    report_add(ps->rep, ps->origin, line_no, "unknown section [%s] -- did you mean [%s]?", name, near);
  else
    report_add(ps->rep, ps->origin, line_no, "unknown section [%s]", name);
}

static void parse_assignment(parser *ps, char *line, int line_no)
{
  char *eq = strchr(line, '=');
  if (!eq)
  {
    report_add(ps->rep, ps->origin, line_no, "expected `key = value`, `[section]` or a comment, got `%s`", line);
    return;
  }
  *eq = '\0';
  const char *key = trim(line);
  const char *value = trim(eq + 1);

  if (!*key)
  {
    report_add(ps->rep, ps->origin, line_no, "missing key before `=`");
    return;
  }
  for (const char *c = key; *c; ++c)
  {
    if (!is_key_char(*c))
    {
      report_add(ps->rep, ps->origin, line_no, "`%s` is not a key: keys are bare words like remesh.target_length", key);
      return;
    }
  }
  if (!ps->section_ok) return; /* the header was reported; its keys would only repeat it */

  char full[192];
  if (ps->section[0])
  {
    const size_t sl = strlen(ps->section);
    if (strncmp(key, ps->section, sl) == 0 && key[sl] == '.')
    {
      report_add(ps->rep, ps->origin, line_no, "inside [%s], write `%s = ...`, not `%s = ...`", ps->section, key + sl + 1, key);
      return;
    }
    if (strchr(key, '.') && MFTLSettingsFindKey(key))
    {
      /* TOML reads a dotted key inside a section as relative to it. */
      report_add(ps->rep, ps->origin, line_no, "`%s` inside [%s] would mean %s.%s; move it under its own section header, or above the first one", key,
                 ps->section, ps->section, key);
      return;
    }
    snprintf(full, sizeof full, "%s.%s", ps->section, key);
  }
  else
  {
    if (!strchr(key, '.'))
    {
      char near[192];
      if (suggest_key(key, near, sizeof near))
        report_add(ps->rep, ps->origin, line_no, "`%s` is outside any section -- write it under its [section], e.g. `%s`", key, near);
      else
        report_add(ps->rep, ps->origin, line_no, "`%s` is outside any section and is not a known key", key);
      return;
    }
    snprintf(full, sizeof full, "%s", key);
  }

  const MFTL_SettingInfo *k = MFTLSettingsFindKey(full);
  if (!k)
  {
    char near[192];
    if (suggest_key(full, near, sizeof near))
      report_add(ps->rep, ps->origin, line_no, "unknown key %s -- did you mean %s?", full, near);
    else
      report_add(ps->rep, ps->origin, line_no, "unknown key %s", full);
    return;
  }

  const int idx = key_index(k);
  if (ps->seen[idx])
  {
    report_add(ps->rep, ps->origin, line_no, "%s is set twice (first on line %d)", full, ps->seen[idx]);
    return;
  }
  ps->seen[idx] = line_no;

  if (!*value)
  {
    report_add(ps->rep, ps->origin, line_no, "%s has no value", full);
    return;
  }
  char why[512];
  if (!apply_value(k, ps->work, value, why, sizeof why)) report_add(ps->rep, ps->origin, line_no, "%s: %s", full, why);
}

int MFTLSettingsParse(MFTL_Settings *s, const char *text, const char *origin, char *message, int message_size)
{
  report rep;
  report_init(&rep, message, message_size);
  if (!origin) origin = "settings";
  if (!s || !text)
  {
    report_add(&rep, origin, 0, "no settings or no text");
    return MFTL_ERROR;
  }

  MFTL_Settings work = *s;
  parser ps;
  memset(&ps, 0, sizeof ps);
  ps.work = &work;
  ps.rep = &rep;
  ps.origin = origin;
  ps.section_ok = MFTL_TRUE;
  ps.seen = (int *) calloc((size_t) MFTLSettingsKeyCount(), sizeof(int));
  if (!ps.seen) abort();

  const char *p = text;
  if ((unsigned char) p[0] == 0xEF && (unsigned char) p[1] == 0xBB && (unsigned char) p[2] == 0xBF) p += 3; /* UTF-8 BOM */

  int line_no = 0;
  while (*p)
  {
    ++line_no;
    const char *eol = strchr(p, '\n');
    const size_t len = eol ? (size_t) (eol - p) : strlen(p);
    if (len >= LINE_MAX_CHARS)
      report_add(&rep, origin, line_no, "line longer than %d characters", LINE_MAX_CHARS - 1);
    else
    {
      char buf[LINE_MAX_CHARS];
      memcpy(buf, p, len);
      buf[len] = '\0';
      strip_comment(buf);
      char *line = trim(buf);
      if (line[0] == '[')
        parse_header(&ps, line, line_no);
      else if (line[0])
        parse_assignment(&ps, line, line_no);
    }
    p = eol ? eol + 1 : p + len;
  }
  free(ps.seen);

  if (rep.count == 0)
  {
    MFTLRemeshParamsFinalise(&work.remesh.params);
    check_keys(&work, &rep, origin);
    check_combinations(&work, &rep, origin);
  }
  if (rep.count != 0) return MFTL_ERROR;

  *s = work;
  link_topology_remesh(s);
  return MFTL_SUCCESS;
}

int MFTLSettingsSet(MFTL_Settings *s, const char *key, const char *value, char *message, int message_size)
{
  report rep;
  report_init(&rep, message, message_size);
  if (!s || !key || !value)
  {
    report_add(&rep, "MFTLSettingsSet", 0, "NULL argument");
    return MFTL_ERROR;
  }
  if (strpbrk(value, "\r\n") || strpbrk(key, "\r\n="))
  {
    report_add(&rep, "MFTLSettingsSet", 0, "%s: a key or value may not contain a line break or, in the key, `=`", key);
    return MFTL_ERROR;
  }
  if (!strchr(key, '.'))
  {
    report_add(&rep, "MFTLSettingsSet", 0, "`%s` is not a full key; write section.name", key);
    return MFTL_ERROR;
  }

  char line[LINE_MAX_CHARS];
  const int n = snprintf(line, sizeof line, "%s = %s", key, value);
  if (n < 0 || n >= (int) sizeof line)
  {
    report_add(&rep, "MFTLSettingsSet", 0, "%s: value too long", key);
    return MFTL_ERROR;
  }
  return MFTLSettingsParse(s, line, "MFTLSettingsSet", message, message_size);
}

int MFTLSettingsGet(const MFTL_Settings *s, const char *key, char *value, int value_size)
{
  if (!s || !key || !value || value_size <= 0) return MFTL_ERROR;
  const MFTL_SettingInfo *k = MFTLSettingsFindKey(key);
  if (!k) return MFTL_ERROR;
  return (format_value(k, s, value, (size_t) value_size) < 0) ? MFTL_ERROR : MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 9. The writer
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Writes @p text as `# ` comment lines no wider than about 88 columns. */
static void write_wrapped(FILE *out, const char *text)
{
  const int width = 86;
  const char *p = text;
  while (*p)
  {
    while (*p == ' ') ++p;
    if (!*p) break;
    int take = (int) strlen(p);
    if (take > width)
    {
      take = width;
      while (take > 0 && p[take] != ' ') --take;
      if (take == 0) take = width;
    }
    fprintf(out, "# %.*s\n", take, p);
    p += take;
  }
}

static const char *type_phrase(MFTL_SettingType t)
{
  switch (t)
  {
    case MFTL_SETTING_INT: return "integer";
    case MFTL_SETTING_REAL: return "real";
    case MFTL_SETTING_BOOL: return "true or false";
    case MFTL_SETTING_KEYWORD: return "keyword";
    case MFTL_SETTING_STRING: return "string";
    case MFTL_SETTING_INT3: return "three integers";
    case MFTL_SETTING_REAL3: return "three reals";
    case MFTL_SETTING_BOOL3: return "three of true or false";
  }
  return "value";
}

static void write_key_doc(FILE *out, const MFTL_SettingInfo *k)
{
  fprintf(out, "\n");
  write_wrapped(out, k->summary);

  char line[512];
  int len = snprintf(line, sizeof line, "%s", type_phrase(k->type));
  if (k->range[0]) len += snprintf(line + len, sizeof line - (size_t) len, " in %s", k->range);
  if (k->unit[0]) len += snprintf(line + len, sizeof line - (size_t) len, ", %s", k->unit);
  if (k->advanced) len += snprintf(line + len, sizeof line - (size_t) len, ", advanced");
  snprintf(line + len, sizeof line - (size_t) len, "; default %s", k->default_text);
  fprintf(out, "# (%s)\n", line);

  if (k->type == MFTL_SETTING_KEYWORD)
  {
    int w = 0;
    for (const MFTL_SettingChoice *c = k->choices; c && c->name; ++c) w = MFTL_MAX(w, (int) strlen(c->name));
    for (const MFTL_SettingChoice *c = k->choices; c && c->name; ++c) fprintf(out, "#   \"%s\"%*s  %s\n", c->name, w - (int) strlen(c->name), "", c->summary);
  }
}

int MFTLSettingsWrite(const MFTL_Settings *s, FILE *out, unsigned flags)
{
  if (!s || !out) return MFTL_ERROR;
  const bool documented = (flags & MFTL_SETTINGS_WRITE_DOCUMENTED) != 0;
  const bool changed_only = (flags & MFTL_SETTINGS_WRITE_CHANGED) != 0;

  MFTL_Settings defaults;
  if (changed_only) MFTLSettingsDefaults(&defaults);

  if (documented)
  {
    fprintf(out, "# MFTL settings.\n");
    fprintf(out, "#\n");
    fprintf(out, "# A subset of TOML: [section] headers, key = value, # comments; three-vectors as\n");
    fprintf(out, "# [x, y, z] (one value sets all three). Keys left out keep their default; unknown\n");
    fprintf(out, "# keys, wrong types and values out of range are errors. Load with\n");
    fprintf(out, "# MFTLSettingsLoad(), override on the command line with section.key=value.\n");
  }
  else
    fprintf(out, "# MFTL settings%s\n", changed_only ? " (keys that differ from the defaults)" : "");

  for (int si = 0; si < MFTLSettingsSectionCount(); ++si)
  {
    const MFTL_SettingSection *sec = MFTLSettingsSection(si);

    /* Which keys, and how wide the widest printed name is. */
    int width = 0, n_printed = 0;
    char value[2 * MFTL_SETTINGS_STRING_MAX + 16], other[2 * MFTL_SETTINGS_STRING_MAX + 16];
    for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
    {
      const MFTL_SettingInfo *k = MFTLSettingsKey(i);
      if (strcmp(k->section, sec->name) != 0) continue;
      if (changed_only)
      {
        format_value(k, s, value, sizeof value);
        format_value(k, &defaults, other, sizeof other);
        if (strcmp(value, other) == 0) continue;
      }
      width = MFTL_MAX(width, (int) strlen(k->name));
      ++n_printed;
    }
    if (n_printed == 0) continue;

    fprintf(out, "\n");
    if (documented)
    {
      fprintf(out, "# %s\n", "=============================================================================");
      fprintf(out, "# %s\n", sec->title);
      write_wrapped(out, sec->summary);
      fprintf(out, "# %s\n", "=============================================================================");
    }
    fprintf(out, "[%s]\n", sec->name);

    for (int i = 0; i < MFTLSettingsKeyCount(); ++i)
    {
      const MFTL_SettingInfo *k = MFTLSettingsKey(i);
      if (strcmp(k->section, sec->name) != 0) continue;
      format_value(k, s, value, sizeof value);
      if (changed_only)
      {
        format_value(k, &defaults, other, sizeof other);
        if (strcmp(value, other) == 0) continue;
      }
      if (documented) write_key_doc(out, k);
      fprintf(out, "%-*s = %s\n", width, k->name, value);
    }
  }
  fflush(out);
  return ferror(out) ? MFTL_ERROR : MFTL_SUCCESS;
}
