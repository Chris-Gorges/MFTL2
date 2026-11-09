/**
 * @file mftl_settings.h
 * @author Christian Gorges
 * @date 2026-09-29
 * @ingroup mftl_settings
 *
 * @brief Run configuration: every runtime option of the library in one struct,
 *        filled from a settings file and the command line.
 *
 * ## WHAT IT IS
 *
 * #MFTL_Settings holds one section per subsystem, and each section EMBEDS the
 * parameter block its module already takes, so a call site reads
 *
 * @code
 * MFTLRemeshEnsemble(ens, &cfg.remesh.params, &stats);
 * MFTLEnsembleApplyTopologyChanges(ens, &cfg.topology.options, &tstats);
 * MFTLWriteVtu(ens, name, &cfg.output.vtk);
 * @endcode
 *
 * No module reads the settings on its own. A function takes exactly the block
 * it needs, as before, so a module stays testable in isolation and nothing
 * depends on hidden global state; the struct only puts every block in one
 * place, with one set of defaults, one file format and one validation.
 *
 * ## THE FILE
 *
 * A subset of TOML: `[section]` headers, `key = value` lines, `#` comments,
 * numbers, `true`/`false`, quoted strings, and `[x, y, z]` for the three-vectors
 * (a single value is broadcast to all three). A keyword may be written with or
 * without quotes. Python reads the same file with `tomllib`.
 *
 * @verbatim
   [domain]
   global_min = [0.0, 0.0, 0.0]
   global_max = [1.0, 1.0, 1.0]
   periodic   = false

   [remesh]
   every             = 1
   target_length     = 0          # 0: the mesh's own mean edge length
   parallel_strategy = "auto"
   @endverbatim
 *
 * Every key is declared once, in `src/settings/settings_keys.c`, with its type,
 * default, admissible range and a one-line description; parsing, validation,
 * MFTLSettingsWrite() and the manual's settings reference are all derived from
 * that table. Unknown keys, wrong types, values outside their range and keys
 * given twice are ERRORS with the file name, the line and, for a typo, the key
 * that was probably meant -- a misspelled key silently left at its default is
 * the failure a settings file must not have.
 *
 * ## THE CALL SEQUENCE
 *
 * @code
 * MFTL_Settings cfg;
 * MFTLSettingsDefaults(&cfg);
 * if (MFTLSettingsLoad(&cfg, "run.toml", MPI_COMM_WORLD) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
 * if (MFTLSettingsLoadArgs(&cfg, argc, argv, MPI_COMM_WORLD) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
 *
 * MFTL_Ensemble *ens = MFTLSettingsCreateEnsemble(&cfg, MPI_COMM_WORLD);
 * ... create and generate the bodies (cfg.sphere) ...
 * MFTLSettingsResolve(&cfg, ens);            // "auto" lengths, derived bounds, agreement
 * if (rank == 0) MFTLSettingsWrite(&cfg, stdout, 0);   // the exact configuration of this run
 * @endcode
 *
 * ## COLLECTIVE STATE
 *
 * Several options change the number or length of messages, so the struct must
 * be identical on every rank. MFTLSettingsLoad() and MFTLSettingsLoadArgs()
 * guarantee it -- rank 0 reads, broadcasts the TEXT, and every rank parses the
 * same bytes -- and MFTLSettingsResolve() verifies it with one reduction, naming
 * the first key that differs. A caller that changes a field afterwards must do
 * so identically on every rank.
 */

#ifndef MFTL_SETTINGS_H
#define MFTL_SETTINGS_H

#include <stddef.h>
#include <stdio.h>

#include <mpi.h>

#include "mftl_bvh.h"
#include "mftl_correction.h"
#include "mftl_coupling.h"
#include "mftl_geometry.h"
#include "mftl_io.h"
#include "mftl_remeshing.h"
#include "mftl_topology_change.h"
#include "mftl_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** @brief Capacity of a string setting, terminating NUL included. */
#define MFTL_SETTINGS_STRING_MAX 256

// -----------------------------------------------------------------------------
// 1. THE SETTINGS
// -----------------------------------------------------------------------------

/**
 * @struct MFTL_DomainSettings
 * @brief The domain box and its decomposition over the ranks; read by
 *        MFTLSettingsCreateEnsemble().
 */
typedef struct MFTL_DomainSettings
{
  /**
   * Ranks per axis. A zero entry is chosen by MFTLSettingsChooseCuts(): the
   * factorisation of the rank count that makes the rank boxes widest on their
   * narrowest axis, which is the width the distributed remesher's band condition
   * \f$h > 4\delta\f$ is taken against. Non-zero entries are kept.
   */
  int cuts[3];
  double global_min[3];  ///< Lower corner of the domain box.
  double global_max[3];  ///< Upper corner; greater than `global_min` on every axis.
  bool periodic[3];  ///< Periodicity per axis.
} MFTL_DomainSettings;

/**
 * @struct MFTL_SphereSettings
 * @brief An initial icosphere, for drivers that start from one; see
 *        MFTLGenerateSphere(). The library never reads it on its own.
 */
typedef struct MFTL_SphereSettings
{
  int refinements;  ///< Subdivision levels: \f$20\cdot4^r\f$ triangles, built whole on rank 0.
  double radius;  ///< Sphere radius.
  double center[3];  ///< Sphere centre.
} MFTL_SphereSettings;

/**
 * @struct MFTL_RemeshSettings
 * @brief Remeshing: the schedule and the parameter block of MFTLRemeshEnsemble().
 */
typedef struct MFTL_RemeshSettings
{
  /** Remesh every this many timesteps, 0 never. Read by the caller's loop (MFTLSettingsDue()). */
  int every;

  /**
   * The block to pass to MFTLRemeshEnsemble(). `target_length`,
   * `min_edge_length` and `max_edge_length` may be 0 in a file, meaning "derive
   * it"; MFTLSettingsResolve() replaces them by the measured values, so the
   * block is valid only after that call. So may `curvature_tolerance` and
   * `min_target_length`, derived from the two factors below.
   */
  MFTL_RemeshParameters params;

  /** Curvature sizing: `params.curvature_tolerance` = this times `params.target_length`, when it is 0. */
  double curvature_tolerance_factor;

  /** Curvature sizing: `params.min_target_length` = this times `params.target_length`, when it is 0. */
  double min_target_factor;
} MFTL_RemeshSettings;

/**
 * @struct MFTL_TopologySettings
 * @brief Breakup and coalescence: the schedule, the resolution-relative
 *        defaults, and the options of MFTLEnsembleApplyTopologyChanges().
 */
typedef struct MFTL_TopologySettings
{
  /** Detect and apply topology changes every this many timesteps, 0 never. Read by the caller's loop. */
  int every;

  /** Remesh right after a reconstruction with the `remesh` block (sets `options.remesh`). */
  bool remesh_after;

  double dx_factor;  ///< `options.dx` = this times the remesh target length, when `options.dx` is 0.
  double delta_breakup_factor;  ///< `options.delta_breakup` = this times the target length, when 0.
  double delta_coalescence_factor;  ///< `options.delta_coalescence` = this times the target length, when 0.

  /**
   * The options to pass to MFTLEnsembleApplyTopologyChanges(). The three
   * lengths may be 0 in a file and are then derived by MFTLSettingsResolve(),
   * which also points `options.remesh` at THIS struct's remesh block when
   * `remesh_after` is set -- so a copy of the settings made after that call
   * still points at the original.
   */
  MFTL_TopologyOptions options;
} MFTL_TopologySettings;

/**
 * @struct MFTL_VolumeSettings
 * @brief Volume restoration: the schedule and the parameters of
 *        MFTLEnsembleCorrectVolumes().
 */
typedef struct MFTL_VolumeSettings
{
  /** Restore every body's reference volume every this many timesteps, 0 never. Read by the caller's loop. */
  int every;

  /** The parameters to pass to MFTLEnsembleCorrectVolumes(). `rim_project` cannot be set from a file. */
  MFTL_VolumeCorrectionParameters params;
} MFTL_VolumeSettings;

/**
 * @struct MFTL_UndulationSettings
 * @brief TSUR3D undulation removal: the schedule and the parameters of
 *        MFTLEnsembleRemoveUndulations().
 */
typedef struct MFTL_UndulationSettings
{
  /** Remove undulations every this many timesteps, 0 never. Read by the caller's loop. */
  int every;

  /** The parameters to pass to MFTLEnsembleRemoveUndulations(). */
  MFTL_UndulationParameters params;
} MFTL_UndulationSettings;

/**
 * @struct MFTL_ContactSettings
 * @brief Contact detection: the schedule and the parameters of
 *        MFTLEnsembleIntersectObjects().
 */
typedef struct MFTL_ContactSettings
{
  /** Run contact detection every this many timesteps, 0 never. Read by the caller's loop. */
  int every;

  /** The parameters to pass to MFTLEnsembleIntersectObjects(). `pair_filter` cannot be set from a file. */
  MFTL_BVHBuildParams params;
} MFTL_ContactSettings;

/**
 * @struct MFTL_CouplingSettings
 * @brief Coupling to a flow solver: the numerical policy of MFTLCouplerCreate().
 */
typedef struct MFTL_CouplingSettings
{
  /** The parameters to pass to MFTLCouplerCreate(). `user_kernel` cannot be set from a file. */
  MFTL_CouplerParams params;
} MFTL_CouplingSettings;

/**
 * @struct MFTL_RedistributeSettings
 * @brief The process-wide redistribution policy; applied by
 *        MFTLSettingsCreateEnsemble() through MFTLRedistributeSetSkipUnchangedBodies()
 *        and MFTLRedistributeSetShrinkSlack().
 */
typedef struct MFTL_RedistributeSettings
{
  bool skip_unchanged_bodies;  ///< Leave a body no rank saw change owner untouched (exact; default on).
  double shrink_slack;  ///< Give back pool capacity above `slack * size`; 0 off, otherwise at least 1 (4 recommended).
} MFTL_RedistributeSettings;

/**
 * @struct MFTL_GeometrySettings
 * @brief Choices a caller passes to the geometry functions.
 */
typedef struct MFTL_GeometrySettings
{
  /** For MFTLEnsembleUpdateVertexNormals(): area for a solver, angle for an exact pseudo-normal sign. */
  MFTL_NormalWeighting normal_weighting;
} MFTL_GeometrySettings;

/**
 * @struct MFTL_OutputSettings
 * @brief VTK output: the schedule, the file prefix and the options of
 *        MFTLWriteVtu().
 */
typedef struct MFTL_OutputSettings
{
  int every;  ///< Write output every this many timesteps, 0 never. Read by the caller's loop.
  char prefix[MFTL_SETTINGS_STRING_MAX];  ///< File name prefix; the caller appends the step.
  MFTL_VtkOptions vtk;  ///< The options to pass to MFTLWriteVtu().
} MFTL_OutputSettings;

/**
 * @struct MFTL_Settings
 * @brief Every runtime option of the library, one section per subsystem.
 *
 * Start from MFTLSettingsDefaults(); a zero-initialised struct is NOT a valid
 * configuration. The section and key names of the settings file are the member
 * names here, and inside a module block the field names of that block, so
 * `remesh.parallel_phase_budget` in a file is
 * `cfg.remesh.params.parallel_phase_budget` in C.
 */
typedef struct MFTL_Settings
{
  MFTL_DomainSettings domain;  ///< `[domain]`
  MFTL_SphereSettings sphere;  ///< `[sphere]`
  MFTL_RemeshSettings remesh;  ///< `[remesh]`
  MFTL_TopologySettings topology;  ///< `[topology]`
  MFTL_VolumeSettings volume;  ///< `[volume]`
  MFTL_UndulationSettings undulation;  ///< `[undulation]`
  MFTL_ContactSettings contact;  ///< `[contact]`
  MFTL_CouplingSettings coupling;  ///< `[coupling]`
  MFTL_RedistributeSettings redistribute;  ///< `[redistribute]`
  MFTL_GeometrySettings geometry;  ///< `[geometry]`
  MFTL_OutputSettings output;  ///< `[output]`
} MFTL_Settings;

// -----------------------------------------------------------------------------
// 2. THE KEY TABLE (introspection)
// -----------------------------------------------------------------------------

/** @brief The value type of a settings key. */
typedef enum MFTL_SettingType
{
  MFTL_SETTING_INT = 0,  ///< An `int`.
  MFTL_SETTING_REAL,  ///< A `double`; `inf` is accepted, `nan` is not.
  MFTL_SETTING_BOOL,  ///< `true` or `false`.
  MFTL_SETTING_KEYWORD,  ///< One of a fixed list of names, stored as the enum value it stands for.
  MFTL_SETTING_STRING,  ///< Text of at most #MFTL_SETTINGS_STRING_MAX - 1 bytes.
  MFTL_SETTING_INT3,  ///< Three `int`s.
  MFTL_SETTING_REAL3,  ///< Three `double`s.
  MFTL_SETTING_BOOL3  ///< Three `bool`s.
} MFTL_SettingType;

/**
 * @struct MFTL_SettingChoice
 * @brief One admissible name of a keyword setting.
 */
typedef struct MFTL_SettingChoice
{
  const char *name;  ///< The name written in a file; #MFTL_NULL terminates a list.
  int value;  ///< The enum value it stands for.
  const char *summary;  ///< What choosing it means, in one line.
} MFTL_SettingChoice;

/**
 * @struct MFTL_SettingInfo
 * @brief Everything the library knows about one key; see MFTLSettingsKey().
 */
typedef struct MFTL_SettingInfo
{
  const char *section;  ///< Section name, e.g. `"remesh"`.
  const char *name;  ///< Key name within the section, e.g. `"target_length"`.
  MFTL_SettingType type;  ///< Value type.
  size_t offset;  ///< Byte offset of the value in #MFTL_Settings.
  size_t size;  ///< Byte size of the value (checked against `type` by the tests).
  const char *default_text;  ///< The default, in file syntax; MFTLSettingsDefaults() parses it.
  const char *range;  ///< Admissible interval, e.g. `"[0, 1]"` or `"(0, inf)"`; empty for none.
  const MFTL_SettingChoice *choices;  ///< Admissible names of a keyword key, else #MFTL_NULL.
  const char *unit;  ///< `"length"`, `"degrees"`, or empty for a dimensionless value.
  bool advanced;  ///< Rarely changed; the manual lists it apart.
  const char *summary;  ///< What the key does, in one or two sentences.
} MFTL_SettingInfo;

/**
 * @struct MFTL_SettingSection
 * @brief One section of the settings file; see MFTLSettingsSection().
 */
typedef struct MFTL_SettingSection
{
  const char *name;  ///< Section name as written in `[name]`.
  const char *title;  ///< A short title.
  const char *summary;  ///< What the section configures and which call reads it.
} MFTL_SettingSection;

/**
 * @brief Number of keys in the table.
 * @return The key count.
 * @rank_local
 */
int MFTLSettingsKeyCount(void);

/**
 * @brief The @p i-th key, in file order (grouped by section).
 * @param i Index in `[0, MFTLSettingsKeyCount())`.
 * @return The key, or #MFTL_NULL when @p i is out of range.
 * @rank_local
 */
const MFTL_SettingInfo *MFTLSettingsKey(int i);

/**
 * @brief Looks a key up by its full name.
 * @param key `"section.name"`, e.g. `"remesh.target_length"`.
 * @return The key, or #MFTL_NULL when there is no such key.
 * @rank_local
 */
const MFTL_SettingInfo *MFTLSettingsFindKey(const char *key);

/**
 * @brief Number of sections.
 * @return The section count.
 * @rank_local
 */
int MFTLSettingsSectionCount(void);

/**
 * @brief The @p i-th section, in file order.
 * @param i Index in `[0, MFTLSettingsSectionCount())`.
 * @return The section, or #MFTL_NULL when @p i is out of range.
 * @rank_local
 */
const MFTL_SettingSection *MFTLSettingsSection(int i);

// -----------------------------------------------------------------------------
// 3. VALUES AND TEXT  (rank-local; src/settings/settings_text.c)
// -----------------------------------------------------------------------------

/**
 * @brief Fills @p s with the default of every key.
 *
 * Module blocks start from their module's own defaults (MFTLInitRemeshParams(),
 * MFTLTopologyDefaultOptions(), MFTL_BVHDefaultParams(), MFTLVtkDefaults()) and
 * every key is then set from its table default. Two defaults differ from the
 * module's on purpose, because a settings file describes a SIMULATION rather
 * than one call: `remesh.parallel_phase_budget` is 4, the measured best in a
 * timestep loop, rather than 32; and the arguments MFTLInitRemeshParams() has no
 * default for are `target_length = 0` (derive), 3 iterations, triangle-based
 * smoothing and feature detection off (180 degrees, right for a smooth
 * interface).
 *
 * @param[out] s The settings.
 * @rank_local
 */
void MFTLSettingsDefaults(MFTL_Settings *s);

/**
 * @brief Applies settings text to @p s: all of it, or nothing.
 *
 * Keys not mentioned keep their current value, so text may be layered: a file
 * over the defaults, then the command line over the file. Within one text a key
 * may appear once. After the text is applied the combination is checked with
 * MFTLSettingsCheck(); if anything fails, @p s is left exactly as it was.
 *
 * Every problem found is reported, one per line, as `origin:line: message`.
 *
 * @param[in,out] s            The settings.
 * @param         text         NUL-terminated settings text.
 * @param         origin       Name used in messages, e.g. the file name.
 * @param[out]    message      Receives the problems found, or an empty string.
 *                             May be #MFTL_NULL.
 * @param         message_size Capacity of @p message; longer reports are
 *                             truncated.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR when anything was wrong.
 * @rank_local
 */
int MFTLSettingsParse(MFTL_Settings *s, const char *text, const char *origin, char *message, int message_size);

/**
 * @brief Sets one key from its text form, as if it were a line of a file.
 *
 * The entry point for a caller with its own input format -- a solver that reads
 * `mftl.remesh.every = 5` from its own file forwards
 * `MFTLSettingsSet(&cfg, "remesh.every", "5", ...)` -- and for a binding to
 * another language, which needs no struct layout. Validated like a file line.
 *
 * @param[in,out] s            The settings.
 * @param         key          `"section.name"`.
 * @param         value        The value in file syntax, e.g. `"[2, 2, 1]"`.
 * @param[out]    message      Receives the reason on failure; may be #MFTL_NULL.
 * @param         message_size Capacity of @p message.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR (and @p s unchanged).
 * @rank_local
 */
int MFTLSettingsSet(MFTL_Settings *s, const char *key, const char *value, char *message, int message_size);

/**
 * @brief Formats the value of one key in file syntax.
 *
 * Reals are written in the shortest form that reads back to the same double,
 * so a written configuration reproduces a run bit for bit.
 *
 * @param      s          The settings.
 * @param      key        `"section.name"`.
 * @param[out] value      Receives the text.
 * @param      value_size Capacity of @p value.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for an unknown key or a buffer too small.
 * @rank_local
 */
int MFTLSettingsGet(const MFTL_Settings *s, const char *key, char *value, int value_size);

/**
 * @brief Checks what no single key can: the combinations.
 *
 * Every key within its range and among its choices; the domain box non-empty on
 * every axis; a shared-file output binary; explicit edge-length bounds ordered;
 * a shrink slack of 0 or at least 1. Called by MFTLSettingsParse() and
 * MFTLSettingsSet(); a caller that edits the struct directly may call it too.
 *
 * @param      s            The settings.
 * @param[out] message      Receives the problems found; may be #MFTL_NULL.
 * @param      message_size Capacity of @p message.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR.
 * @rank_local
 */
int MFTLSettingsCheck(const MFTL_Settings *s, char *message, int message_size);

/** @brief MFTLSettingsWrite() flag: a comment before every key -- summary, type, range, choices, default. */
#define MFTL_SETTINGS_WRITE_DOCUMENTED 1u
/** @brief MFTLSettingsWrite() flag: only the keys whose value differs from MFTLSettingsDefaults(). */
#define MFTL_SETTINGS_WRITE_CHANGED 2u

/**
 * @brief Writes @p s as a settings file.
 *
 * The output reads back to the same struct. With no flag it is the compact form
 * worth printing at the start of every run: after MFTLSettingsResolve() it holds
 * the measured lengths, so it reproduces the run on any machine. With
 * #MFTL_SETTINGS_WRITE_DOCUMENTED and the defaults it is the annotated template
 * of every key.
 *
 * @param s     The settings.
 * @param out   Destination stream.
 * @param flags A combination of the `MFTL_SETTINGS_WRITE_*` flags, or 0.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument or a write error.
 * @rank_local Call it on one rank.
 */
int MFTLSettingsWrite(const MFTL_Settings *s, FILE *out, unsigned flags);

/**
 * @brief Is an action scheduled every @p every steps due at @p step?
 * @param every The interval, e.g. `cfg.remesh.every`; 0 or less is never.
 * @param step  The caller's step counter, from 0.
 * @return `true` at steps 0, `every`, `2*every`, ...
 */
static inline bool MFTLSettingsDue(int every, int step) { return every > 0 && step % every == 0; }

// -----------------------------------------------------------------------------
// 4. THE RUN  (collective; src/settings/settings_run.c)
// -----------------------------------------------------------------------------

/**
 * @brief Reads a settings file on rank 0 and applies it on every rank.
 *
 * Rank 0 reads the file and broadcasts its TEXT; every rank parses the same
 * bytes with MFTLSettingsParse(), so every rank reaches the same result and
 * the same verdict. Problems are printed by rank 0.
 *
 * @param[in,out] s    The settings, normally fresh from MFTLSettingsDefaults().
 * @param         path The file.
 * @param         comm Communicator over which @p s must agree.
 * @return #MFTL_SUCCESS on every rank, or #MFTL_ERROR on every rank (and @p s
 *         unchanged).
 * @collective Two broadcasts.
 */
int MFTLSettingsLoad(MFTL_Settings *s, const char *path, MPI_Comm comm);

/**
 * @brief Applies `section.key=value` arguments of the command line.
 *
 * Every argument whose text before the first `=` contains a `.` is an override
 * (leading dashes are ignored, so `--remesh.every=5` works too); everything else
 * is left for the application. Overrides are parsed as one text after whatever
 * @p s already holds, so they win over the file. Rank 0's arguments are
 * broadcast.
 *
 * @param[in,out] s    The settings.
 * @param         argc Argument count, as passed to `main`.
 * @param         argv Arguments, as passed to `main`.
 * @param         comm Communicator over which @p s must agree.
 * @return #MFTL_SUCCESS on every rank, or #MFTL_ERROR on every rank.
 * @collective Two broadcasts.
 */
int MFTLSettingsLoadArgs(MFTL_Settings *s, int argc, char **argv, MPI_Comm comm);

/**
 * @brief Is @p arg a `section.key=value` override MFTLSettingsLoadArgs() takes?
 *
 * Lets an application skip the arguments that belong to the library while it
 * parses its own.
 *
 * @param arg One command-line argument.
 * @return `true` if MFTLSettingsLoadArgs() treats it as an override.
 */
bool MFTLSettingsIsOverride(const char *arg);

/**
 * @brief Chooses the process grid for `domain.cuts`.
 *
 * Non-zero entries of `s->domain.cuts` are kept. Among the factorisations of
 * @p nproc into the remaining axes, the one whose narrowest rank-box width
 * \f$\min_k L_k/c_k\f$ is LARGEST wins, then the one with the smallest widest
 * box, then the one with more cuts on the lower axis. The narrowest width is
 * what the spanning remesher's band condition is taken against and what bounds
 * the per-body reductions, so it is the quantity to protect.
 *
 * @param      s     The settings; `domain` is read.
 * @param      nproc Number of ranks.
 * @param[out] cuts  The grid.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR when the fixed entries do not divide
 *         @p nproc.
 * @rank_local Pure function of its arguments.
 */
int MFTLSettingsChooseCuts(const MFTL_Settings *s, int nproc, int cuts[3]);

/**
 * @brief Creates the ensemble the `[domain]` section describes, and applies the
 *        `[redistribute]` policy.
 *
 * Chooses the grid with MFTLSettingsChooseCuts(), sets the process-wide
 * redistribution switches, and calls MFTLEnsembleCreate(). The switches are set
 * here rather than later because the first redistribution -- the scatter of the
 * generated bodies -- is where a shrink slack matters most.
 *
 * @param s    The settings; identical on every rank.
 * @param comm Base communicator, usually `MPI_COMM_WORLD`.
 * @return The new ensemble, or #MFTL_NULL (on every rank) on invalid settings.
 * @collective
 */
MFTL_Ensemble *MFTLSettingsCreateEnsemble(const MFTL_Settings *s, MPI_Comm comm);

/**
 * @brief Turns the file's values into the ones the run uses, and checks that
 *        every rank holds the same settings.
 *
 * - `remesh.target_length` 0: MFTLEnsembleMeanEdgeLength() of @p ens (so call
 *   this after the bodies exist). On an empty ensemble it stays 0 -- and the
 *   remesher then refuses loudly -- unless remeshing or topology changes are
 *   scheduled, which is an error.
 * - `remesh.min_edge_length` / `max_edge_length` 0: \f$\tfrac45L\f$ and
 *   \f$\tfrac43L\f$ (Botsch & Kobbelt); the squared bounds are finalised.
 *   Under curvature sizing \f$L\f$ is the coarsest target and the pair is the
 *   envelope \f$\tfrac45L_\min\f$, \f$\tfrac43L\f$.
 * - `remesh.curvature_tolerance` / `min_target_length` 0: their `*_factor`
 *   times the target length.
 * - `topology.dx`, `delta_breakup`, `delta_coalescence` 0: their `*_factor`
 *   times the target length; `options.remesh` points at this struct's remesh
 *   block when `topology.remesh_after` is set.
 *
 * A resolved value is written into @p s, so the file's 0 is gone afterwards and
 * MFTLSettingsWrite() prints what the run actually uses. Idempotent.
 *
 * @param[in,out] s   The settings.
 * @param         ens The ensemble; its communicator is used.
 * @return #MFTL_SUCCESS on every rank, or #MFTL_ERROR on every rank -- for
 *         settings that differ between ranks (the first differing key is
 *         reported) or a length that cannot be derived.
 * @collective One reduction for the agreement check, plus the mean edge length
 *             when it is derived.
 */
int MFTLSettingsResolve(MFTL_Settings *s, MFTL_Ensemble *ens);

#ifdef __cplusplus
}
#endif

#endif  // MFTL_SETTINGS_H
