/**
 * @file settings_keys.c
 * @author Christian Gorges
 * @date 2026-09-29
 * @ingroup mftl_settings
 *
 * @brief THE table of settings keys: the one place a runtime option is
 *        declared.
 *
 * Everything else is derived from it -- the parser and its error messages, the
 * range checks, MFTLSettingsDefaults(), MFTLSettingsWrite() and its annotated
 * template, the `mftl_settings` tool, and the settings reference of the web
 * manual. Adding an option is one KEY(...) line here plus the struct member it
 * names; the tests check every entry's type against the member's size, every
 * default against its range, and every module-block default against the
 * module's own defaults.
 *
 * ## THE FORMAT IS READ BY A SECOND PROGRAM
 *
 * `MFTL2_Documentation/mkdocs/settings_page.py` parses this file to generate the
 * manual's settings reference, so keep to what it understands:
 *
 * - one `SECTION(name, title, summary)`, `CHOICE(name, value, summary)` or
 *   `KEY(...)` per entry, with the arguments in the order of the macros below;
 * - text arguments are string literals (adjacent literals are concatenated);
 * - a choice list is `static const MFTL_SettingChoice g_<name>[] = {CHOICE(...),
 *   ..., END_CHOICES};` and a key names it, or writes `NO_CHOICES`.
 *
 * A summary is plain text with `backticks` for code and may name library
 * functions as `Name()`; the manual links them. It is also printed in the
 * template file, so it should say what the key DOES and what its special values
 * mean in a line or two -- the rationale belongs on the struct field in the
 * header, which the manual links to.
 */

#include "mftl_settings.h"

#include <string.h>

#include "mftl_constants.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Macros
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SECTION(name, title, summary) {name, title, summary}

#define CHOICE(name, value, summary) {name, (int) (value), summary}
#define END_CHOICES {MFTL_NULL, 0, MFTL_NULL}
#define NO_CHOICES MFTL_NULL

/** The member is named by its path in MFTL_Settings; offsetof accepts nested designators. */
#define KEY(section, name, type, member, deflt, range, choices, unit, advanced, summary)                                                                     \
  {section, name, MFTL_SETTING_##type, offsetof(MFTL_Settings, member), sizeof(((MFTL_Settings *) 0)->member), deflt, range, choices, unit, advanced != 0, \
   summary}

/* Keywords are stored through an int; every enum a keyword key points at must be one. */
_Static_assert(sizeof(MFTL_RemeshParallelStrategy) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_SmoothMode) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_RemeshSizing) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_RemeshPlacement) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_RemeshProjection) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_NormalWeighting) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_IoLayout) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_VtkEncoding) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_CouplingKernel) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_Reproduction) == sizeof(int), "keyword storage");
_Static_assert(sizeof(MFTL_SamplingPolicy) == sizeof(int), "keyword storage");

/* ═══════════════════════════════════════════════════════════════════════════
 * Sections
 * ═══════════════════════════════════════════════════════════════════════════ */

// clang-format off -- the tables below are laid out by hand, one entry per macro
static const MFTL_SettingSection g_sections[] = {
    SECTION("domain", "Domain and decomposition",
            "The domain box and how it is cut into rank boxes. Read by MFTLSettingsCreateEnsemble()."),
    SECTION("sphere", "Initial sphere",
            "An icosphere for drivers that start from one: pass these to MFTLGenerateSphere(). The library never reads them on its own."),
    SECTION("remesh", "Remeshing",
            "Isotropic remeshing, serial or distributed: `MFTLRemeshEnsemble(ens, &cfg.remesh.params, &stats)`. Lengths left at 0 are derived "
            "by MFTLSettingsResolve()."),
    SECTION("topology", "Topology changes",
            "Breakup and coalescence: `MFTLEnsembleApplyTopologyChanges(ens, &cfg.topology.options, &stats)`. Lengths left at 0 are "
            "derived from the remesh target length by MFTLSettingsResolve()."),
    SECTION("volume", "Volume restoration",
            "Restores every body's reference volume, set once with MFTLEnsembleSetReferenceVolumes(): `MFTLEnsembleCorrectVolumes(ens, "
            "&cfg.volume.params, &stats)`. Outside the remesher, so its Phong projection keeps the correction."),
    SECTION("undulation", "Undulation removal (TSUR3D)",
            "Volume-conserving removal of grid-scale undulations: `MFTLEnsembleRemoveUndulations(ens, &cfg.undulation.params, &stats)`. "
            "Outside the remesher, so its Phong projection keeps the smoothing."),
    SECTION("contact", "Contact detection",
            "Triangle contact between and within bodies: `MFTLEnsembleIntersectObjects(ens, &cfg.contact.params, callback, user_data)`."),
    SECTION("coupling", "Coupling to a flow solver",
            "Interpolation to markers and spreading back: `MFTLCouplerCreate(ens, grid, &cfg.coupling.params, &coupler)`. Every tolerance is "
            "dimensionless."),
    SECTION("redistribute", "Redistribution policy",
            "Process-wide switches of MFTLRedistribute(), applied by MFTLSettingsCreateEnsemble()."),
    SECTION("geometry", "Geometry", "Choices a caller passes to the geometry functions."),
    SECTION("output", "Output", "Parallel VTK XML output: `MFTLWriteVtu(ens, name, &cfg.output.vtk)`."),
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Keyword choices
 * ═══════════════════════════════════════════════════════════════════════════ */

static const MFTL_SettingChoice g_parallel_strategy[] = {
    CHOICE("auto", MFTL_REMESH_PARALLEL_AUTO, "measure the bodies and choose per pass (MFTLRemeshChooseStrategy())"),
    CHOICE("spanning", MFTL_REMESH_PARALLEL_SPANNING, "move the cut planes; for bodies larger than a rank box"),
    CHOICE("dispersed", MFTL_REMESH_PARALLEL_DISPERSED, "move each body whole onto one rank; for bodies smaller than a rank box"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_spanning_mode[] = {
    CHOICE("once", MFTL_REMESH_SPANNING_ONCE, "placements outermost: each region gets the whole serial call in the first placement where it is unfrozen"),
    CHOICE("nested", MFTL_REMESH_SPANNING_NESTED,
           "iterations outermost: placements only for split/collapse/flip, then every vertex smoothed with a ghost exchange per sweep"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_smooth_mode[] = {
    CHOICE("triangle", MFTL_SMOOTH_TRIANGLE_BASED, "area-weighted mean of the incident triangle centroids; equalises areas"),
    CHOICE("vertex", MFTL_SMOOTH_VERTEX_BASED, "umbrella operator, the plain mean of the one-ring; equalises edge lengths"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_sizing[] = {
    CHOICE("uniform", MFTL_REMESH_SIZING_UNIFORM, "one target length everywhere (Botsch & Kobbelt)"),
    CHOICE("curvature", MFTL_REMESH_SIZING_CURVATURE, "isotropic, finer where curved: a target from the curvature (Dunyach et al. 2013)"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_placement[] = {
    CHOICE("midpoint", MFTL_REMESH_PLACE_MIDPOINT, "new vertices at the chord midpoint"),
    CHOICE("paraboloid", MFTL_REMESH_PLACE_PARABOLOID, "on a weighted least-squares paraboloid through the edge's two-ring vertices"),
    CHOICE("paraboloid_normals", MFTL_REMESH_PLACE_PARABOLOID_NORMALS, "the same, fitted to the vertex normals' slopes as well"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_projection[] = {
    CHOICE("none", MFTL_REMESH_PROJECT_NONE, "the smoother's tangential move stands"),
    CHOICE("phong", MFTL_REMESH_PROJECT_PHONG, "onto the Phong surface of the mesh the remesh call received"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_normal_weighting[] = {
    CHOICE("area", MFTL_NORMAL_WEIGHT_AREA, "faces weighted by area; what a flow solver wants"),
    CHOICE("angle", MFTL_NORMAL_WEIGHT_ANGLE, "faces weighted by their angle at the vertex; the pseudo-normal of an exact inside/outside test"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_kernel[] = {
    CHOICE("peskin4", MFTL_KERNEL_PESKIN4, "Peskin's 4-point function, half-width 2; constants and linears on a lattice"),
    CHOICE("roma3", MFTL_KERNEL_ROMA3, "Roma-Peskin-Berger 3-point function, half-width 1.5; constants and linears on a lattice"),
    CHOICE("cosine4", MFTL_KERNEL_COSINE4, "the cosine kernel, half-width 2; constants only, so LINEAR rows are corrected"),
    CHOICE("hat2", MFTL_KERNEL_HAT2, "the linear hat, half-width 1: trilinear interpolation on a lattice"),
    CHOICE("bspline3", MFTL_KERNEL_BSPLINE3, "the cubic B-spline, half-width 2; constants and linears, C2"),
    CHOICE("wendland_c2", MFTL_KERNEL_WENDLAND_C2, "radial Wendland C2, support radius 2; no lattice moments, always corrected"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_reproduction[] = {
    CHOICE("linear", MFTL_REPRODUCE_LINEAR, "constants and linear fields: force and torque conservation; weights may be negative"),
    CHOICE("constant", MFTL_REPRODUCE_CONSTANT, "normalised positive weights: constants and positivity only"),
    CHOICE("none", MFTL_REPRODUCE_NONE, "the base kernel as it is; the achieved moments are reported"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_stencil_failure[] = {
    CHOICE("error", MFTL_STENCIL_ERROR, "a row that fails its checks fails the bind"),
    CHOICE("grow", MFTL_STENCIL_GROW, "retry with a wider support, up to max_growth"),
    CHOICE("lower_order", MFTL_STENCIL_LOWER_ORDER, "accept constant reproduction where linear fails"),
    CHOICE("grow_then_lower", MFTL_STENCIL_GROW | MFTL_STENCIL_LOWER_ORDER, "grow first, then accept constant reproduction"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_sampling[] = {
    CHOICE("off", MFTL_SAMPLING_OFF, "not checked"),
    CHOICE("report", MFTL_SAMPLING_REPORT, "measured and counted, never an error"),
    CHOICE("error", MFTL_SAMPLING_ERROR, "a spreading bind of a too-coarse surface fails"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_layout[] = {
    CHOICE("file_per_rank", MFTL_IO_FILE_PER_RANK, "one .vtu per rank plus a .pvtu from rank 0"),
    CHOICE("single_file", MFTL_IO_SINGLE_FILE, "one shared .vtu written collectively with MPI-IO; needs binary encoding"),
    END_CHOICES,
};

static const MFTL_SettingChoice g_encoding[] = {
    CHOICE("binary", MFTL_VTK_BINARY, "raw binary, 23-32x faster to write than ascii"),
    CHOICE("ascii", MFTL_VTK_ASCII, "decimal text; human-readable and diffable"),
    END_CHOICES,
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Keys
 *
 *   KEY(section, name, TYPE, member in MFTL_Settings, default, range, choices,
 *       unit, advanced, summary)
 * ═══════════════════════════════════════════════════════════════════════════ */

static const MFTL_SettingInfo g_keys[] = {
    /* ── domain ─────────────────────────────────────────────────────────── */
    KEY("domain", "cuts", INT3, domain.cuts, "[0, 0, 0]", "[0, inf)", NO_CHOICES, "", 0,
        "Ranks per axis. A 0 is chosen for you: the factorisation of the rank count whose narrowest rank box is widest "
        "(MFTLSettingsChooseCuts()). Fixed entries must divide the rank count."),
    KEY("domain", "global_min", REAL3, domain.global_min, "[0.0, 0.0, 0.0]", "", NO_CHOICES, "length", 0, "Lower corner of the domain box."),
    KEY("domain", "global_max", REAL3, domain.global_max, "[1.0, 1.0, 1.0]", "", NO_CHOICES, "length", 0,
        "Upper corner of the domain box; greater than `global_min` on every axis."),
    KEY("domain", "periodic", BOOL3, domain.periodic, "[false, false, false]", "", NO_CHOICES, "", 0, "Periodicity per axis."),

    /* ── sphere ─────────────────────────────────────────────────────────── */
    KEY("sphere", "refinements", INT, sphere.refinements, "3", "[0, 10]", NO_CHOICES, "", 0,
        "Icosahedron subdivisions: 20*4^r triangles, built whole on rank 0 (about 3.4 kB * 4^r)."),
    KEY("sphere", "radius", REAL, sphere.radius, "0.25", "(0, inf)", NO_CHOICES, "length", 0, "Sphere radius."),
    KEY("sphere", "center", REAL3, sphere.center, "[0.5, 0.5, 0.5]", "", NO_CHOICES, "length", 0, "Sphere centre."),

    /* ── remesh ─────────────────────────────────────────────────────────── */
    KEY("remesh", "every", INT, remesh.every, "1", "[0, inf)", NO_CHOICES, "", 0,
        "Remesh every this many timesteps; 0 never. Read by the caller's loop through MFTLSettingsDue()."),
    KEY("remesh", "target_length", REAL, remesh.params.target_length, "0.0", "[0, inf)", NO_CHOICES, "length", 0,
        "Target edge length L. 0: the mesh's own mean edge length when MFTLSettingsResolve() runs, i.e. hold the starting resolution."),
    KEY("remesh", "min_edge_length", REAL, remesh.params.min_edge_length, "0.0", "[0, inf)", NO_CHOICES, "length", 1,
        "Edges shorter than this are collapsed. 0: 4/5 of the target length (Botsch & Kobbelt)."),
    KEY("remesh", "max_edge_length", REAL, remesh.params.max_edge_length, "0.0", "[0, inf)", NO_CHOICES, "length", 1,
        "Edges longer than this are split. 0: 4/3 of the target length."),
    KEY("remesh", "sizing", KEYWORD, remesh.params.sizing, "\"uniform\"", "", g_sizing, "", 0,
        "How the remesher judges an edge: one target length, or a per-vertex target from the curvature. "
        "The curvature mode needs `smooth_mode` = triangle."),
    KEY("remesh", "curvature_tolerance", REAL, remesh.params.curvature_tolerance, "0.0", "[0, inf)", NO_CHOICES, "length", 0,
        "Curvature sizing: the largest admitted gap between an edge and the surface it cuts. 0: `curvature_tolerance_factor` times the "
        "target length."),
    KEY("remesh", "curvature_tolerance_factor", REAL, remesh.curvature_tolerance_factor, "0.02", "(0, inf)", NO_CHOICES, "", 0,
        "Curvature sizing: the tolerance in units of the target length. 0.02 refines below the target wherever the radius of curvature "
        "is under about 8 target lengths."),
    KEY("remesh", "min_target_length", REAL, remesh.params.min_target_length, "0.0", "[0, inf)", NO_CHOICES, "length", 0,
        "Curvature sizing: the finest target; `target_length` is the coarsest. 0: `min_target_factor` times the target length."),
    KEY("remesh", "min_target_factor", REAL, remesh.min_target_factor, "0.25", "(0, 1]", NO_CHOICES, "", 1,
        "Curvature sizing: the finest target in units of the target length."),
    KEY("remesh", "split_placement", KEYWORD, remesh.params.split_placement, "\"midpoint\"", "", g_placement, "", 0,
        "Where a split of an edge between two free vertices puts the new vertex."),
    KEY("remesh", "collapse_placement", KEYWORD, remesh.params.collapse_placement, "\"midpoint\"", "", g_placement, "", 0,
        "Where a collapse of two free vertices puts the merged vertex, before `collapse_preserves_volume` moves it."),
    KEY("remesh", "collapse_preserves_volume", BOOL, remesh.params.collapse_preserves_volume, "false", "", NO_CHOICES, "", 0,
        "Move the merged vertex of a collapse of two free vertices, from the point `collapse_placement` chose, so the collapse keeps "
        "the enclosed volume exactly."),
    KEY("remesh", "fit_radius", REAL, remesh.params.fit_radius, "3.0", "(1, inf)", NO_CHOICES, "", 1,
        "Paraboloid placement: support of the Wendland weights, in local edge lengths."),
    KEY("remesh", "projection", KEYWORD, remesh.params.projection, "\"none\"", "", g_projection, "", 0,
        "What the smoother does after each tangential move of a free vertex."),
    KEY("remesh", "iterations", INT, remesh.params.iterations, "3", "[1, inf)", NO_CHOICES, "", 0,
        "Outer iterations of split, collapse, flip and smooth per pass."),
    KEY("remesh", "min_convergence_ops", INT, remesh.params.min_convergence_ops, "0", "[0, inf)", NO_CHOICES, "", 1,
        "Stop a pass early after an iteration that performed at most this many operations."),
    KEY("remesh", "feature_angle_threshold", REAL, remesh.params.feature_angle_threshold, "180.0", "[0, 180]", NO_CHOICES, "degrees", 0,
        "Dihedral angle above which an edge is a feature and kept sharp. 180 disables detection, which is right for a smooth interface: "
        "every edge flagged there is a false positive that locks a vertex. Mesh boundaries are features regardless."),
    KEY("remesh", "feature_corner_angle", REAL, remesh.params.feature_corner_angle, "45.0", "[0, 180]", NO_CHOICES, "degrees", 1,
        "Turn angle above which a vertex with two feature edges is a locked corner rather than a ridge point sliding along its curve."),
    KEY("remesh", "smooth_mode", KEYWORD, remesh.params.smooth_mode, "\"triangle\"", "", g_smooth_mode, "", 0,
        "Neighbourhood the smoother's target point is built from."),
    KEY("remesh", "smooth_iterations", INT, remesh.params.smooth_iterations, "1", "[0, inf)", NO_CHOICES, "", 0,
        "Tangential relaxation sweeps per outer iteration."),
    KEY("remesh", "smooth_relaxation", REAL, remesh.params.smooth_relaxation, "1.0", "[0, 1]", NO_CHOICES, "", 1,
        "Step scale of one sweep; 1 is the full Jacobi step."),
    KEY("remesh", "smooth_max_step_fraction", REAL, remesh.params.smooth_max_step_fraction, "0.4", "(0, 1]", NO_CHOICES, "", 1,
        "Cap on one smoothing step, as a fraction of the vertex's shortest incident edge."),
    KEY("remesh", "max_normal_deviation_cos", REAL, remesh.params.max_normal_deviation_cos, "0.2", "[-1, 1]", NO_CHOICES, "", 1,
        "A collapse or flip is refused if a surviving triangle's normal would turn by more than arccos of this (0.2 is about 78 degrees)."),
    KEY("remesh", "renumber_global_ids", BOOL, remesh.params.renumber_global_ids, "true", "", NO_CHOICES, "", 1,
        "Serial pass only: make global ids dense again afterwards. The distributed drivers always do."),
    KEY("remesh", "parallel_strategy", KEYWORD, remesh.params.parallel_strategy, "\"auto\"", "", g_parallel_strategy, "", 0,
        "Distributed strategy; ignored on one rank."),
    KEY("remesh", "parallel_phase_budget", INT, remesh.params.parallel_phase_budget, "8", "[1, inf)", NO_CHOICES, "", 0,
        "Spanning strategy: the most cut-plane placements per pass (nested: per iteration). A cap: the pass stops once every "
        "out-of-band edge was offered, and one pass does one serial call's work whatever the value."),
    KEY("remesh", "spanning_mode", KEYWORD, remesh.params.spanning_mode, "\"once\"", "", g_spanning_mode, "", 0,
        "Spanning strategy: how placements and iterations nest. Both give every region one serial call's work; nested refuses "
        "attached fields."),
    KEY("remesh", "freeze_dilation_rings", INT, remesh.params.freeze_dilation_rings, "0", "[0, inf)", NO_CHOICES, "", 1,
        "Extra triangle rings added to the frozen band. 0 is exact; a positive value exists to test that claim and costs coverage."),
    KEY("remesh", "dispersed_radius_fraction", REAL, remesh.params.dispersed_radius_fraction, "0.4", "(0, inf)", NO_CHOICES, "", 1,
        "`auto` chooses dispersed when the largest body radius is at most this fraction of the narrowest rank box."),
    KEY("remesh", "parallel_coarsen", INT, remesh.params.parallel_coarsen, "1", "[0, inf)", NO_CHOICES, "", 1,
        "Spanning strategy: merge rank boxes when a trigger fires. 0 disarmed, 1 armed (bit-identical on a healthy partition), "
        "above 1 armed with at most this many ranks per group."),
    KEY("remesh", "coarsen_frozen_fraction", REAL, remesh.params.coarsen_frozen_fraction, "1.0", "[0, inf)", NO_CHOICES, "", 1,
        "Spanning strategy: coarsen when more than this share of owned triangles is frozen. 1 or more disables; set it from a "
        "measured `frozen_fraction`, never from geometry."),
    KEY("remesh", "serial_gather_max_triangles", INT, remesh.params.serial_gather_max_triangles, "0", "[0, inf)", NO_CHOICES, "", 1,
        "Spanning strategy: remesh on one rank when the whole ensemble has fewer triangles than this. 0 disables."),
    KEY("remesh", "dispersed_balance_slack", REAL, remesh.params.dispersed_balance_slack, "1.25", "", NO_CHOICES, "", 1,
        "Dispersed strategy: a rank may carry this multiple of the mean body weight before bodies move to a neighbour. "
        "0 or less disables the balancer."),

    /* ── topology ───────────────────────────────────────────────────────── */
    KEY("topology", "every", INT, topology.every, "0", "[0, inf)", NO_CHOICES, "", 0,
        "Detect and apply breakup and coalescence every this many timesteps; 0 never. Read by the caller's loop."),
    KEY("topology", "delta_breakup", REAL, topology.options.delta_breakup, "0.0", "[0, inf)", NO_CHOICES, "length", 0,
        "A liquid thread or sheet thinner than this breaks. 0: `delta_breakup_factor` times the remesh target length."),
    KEY("topology", "delta_coalescence", REAL, topology.options.delta_coalescence, "0.0", "[0, inf)", NO_CHOICES, "length", 0,
        "A film between two surfaces thinner than this ruptures and they merge. 0: `delta_coalescence_factor` times the target length."),
    KEY("topology", "dx", REAL, topology.options.dx, "0.0", "[0, inf)", NO_CHOICES, "length", 0,
        "Lattice spacing of the reconstruction. 0: `dx_factor` times the target length."),
    KEY("topology", "delta_breakup_factor", REAL, topology.delta_breakup_factor, "1.5", "(0, inf)", NO_CHOICES, "", 0,
        "`delta_breakup` in units of the remesh target length, used when `delta_breakup` is 0."),
    KEY("topology", "delta_coalescence_factor", REAL, topology.delta_coalescence_factor, "1.0", "(0, inf)", NO_CHOICES, "", 0,
        "`delta_coalescence` in units of the remesh target length, used when `delta_coalescence` is 0."),
    KEY("topology", "dx_factor", REAL, topology.dx_factor, "1.0", "(0, inf)", NO_CHOICES, "", 0,
        "`dx` in units of the remesh target length, used when `dx` is 0."),
    KEY("topology", "remesh_after", BOOL, topology.remesh_after, "true", "", NO_CHOICES, "", 0,
        "Remesh with the [remesh] block right after a reconstruction; marching-cubes output has slivers a solver should not see."),
    KEY("topology", "conserve_volume", BOOL, topology.options.conserve_volume, "true", "", NO_CHOICES, "", 0,
        "Restore each reconstructed cluster's enclosed volume (three Newton steps along the vertex normals)."),
    KEY("topology", "min_body_triangles", INT, topology.options.min_body_triangles, "16", "[0, inf)", NO_CHOICES, "", 0,
        "Closed components with fewer triangles are deleted and their volume given to the survivors. 0 keeps everything."),
    KEY("topology", "normal_tol_deg", REAL, topology.options.normal_tol_deg, "60.0", "[0, 90]", NO_CHOICES, "degrees", 1,
        "Two triangles oppose each other when their normals are within this angle of anti-parallel, and face each other when "
        "the segment between their closest points is within it of both normals."),
    KEY("topology", "bias_factor", REAL, topology.options.bias_factor, "1.5", "(0, inf)", NO_CHOICES, "", 1,
        "Bias at each event, in units of half its threshold; above 1 so a feature at the threshold is decided with margin."),
    KEY("topology", "bias_radius_factor", REAL, topology.options.bias_radius_factor, "1.5", "(0, inf)", NO_CHOICES, "", 1,
        "Radius of the bias bump around each event, in units of max(delta, dx)."),
    KEY("topology", "mc_snap", REAL, topology.options.mc_snap, "0.02", "[0, 0.5)", NO_CHOICES, "", 1,
        "Marching-cubes snap distance relative to the lattice spacing; 0 disables snapping."),

    /* ── volume ─────────────────────────────────────────────────────────── */
    KEY("volume", "every", INT, volume.every, "0", "[0, inf)", NO_CHOICES, "", 0,
        "Restore every body's reference volume every this many timesteps; 0 never. Read by the caller's loop: once per step, after the "
        "remesh, is enough -- correcting after the advection as well changes the shape at second order only."),
    KEY("volume", "iterations", INT, volume.params.iterations, "3", "[0, inf)", NO_CHOICES, "", 1,
        "Newton steps on the exact first variation of the volume. One already reaches round-off from the per-step error of an advected "
        "front; three from a deficit of percents."),

    /* ── undulation ─────────────────────────────────────────────────────── */
    KEY("undulation", "every", INT, undulation.every, "0", "[0, inf)", NO_CHOICES, "", 0,
        "Remove undulations (TSUR3D) every this many timesteps; 0 never. Read by the caller's loop. It removes grid-scale noise and "
        "smooths under-resolved geometry alike: see `min_angle`."),
    KEY("undulation", "sweeps", INT, undulation.params.sweeps, "1", "[0, inf)", NO_CHOICES, "", 1, "Passes over the edges per call."),
    KEY("undulation", "min_angle", REAL, undulation.params.min_angle, "15.0", "[0, 90)", NO_CHOICES, "degrees", 0,
        "Leave edges whose kink -- their angle out of the plane normal to their smoothed normal -- is at most this. 0 treats every edge "
        "(the published method), which also smooths away geometry the mesh under-resolves; 15 was neutral on a resolved shape and still "
        "removed most velocity noise (docs/analysis.md §15.2)."),
    KEY("undulation", "relaxation", REAL, undulation.params.relaxation, "1.0", "(0, 1]", NO_CHOICES, "", 1,
        "Fraction of the way to the common height each end of an edge moves; any value conserves the volume exactly."),

    /* ── contact ────────────────────────────────────────────────────────── */
    KEY("contact", "every", INT, contact.every, "0", "[0, inf)", NO_CHOICES, "", 0,
        "Run contact detection every this many timesteps; 0 never. Read by the caller's loop."),
    KEY("contact", "proximity_factor", REAL, contact.params.proximity_factor, "0.001", "[0, inf)", NO_CHOICES, "", 0,
        "Contact envelope as a fraction of each body's bounding radius; the same value means the same thing at every length scale."),
    KEY("contact", "proximity_floor", REAL, contact.params.proximity_floor, "0.0", "[0, inf)", NO_CHOICES, "length", 0,
        "Absolute lower bound on the envelope. With `proximity_factor` 0 it states a physical gap."),
    KEY("contact", "self_pairs", BOOL, contact.params.self_pairs, "false", "", NO_CHOICES, "", 0,
        "Also test every body against itself: non-adjacent parts of one surface that come close."),
    KEY("contact", "leaf_size", INT, contact.params.leaf_size, "4", "[1, 64]", NO_CHOICES, "", 1, "Maximum triangles per BVH leaf."),
    KEY("contact", "include_ghosts", BOOL, contact.params.include_ghosts, "false", "", NO_CHOICES, "", 1,
        "Also use redistribution ghosts as primitives. Reports halo contacts once per holder: for inspecting the halo only."),

    /* ── coupling ───────────────────────────────────────────────────────── */
    KEY("coupling", "kernel", KEYWORD, coupling.params.kernel, "\"peskin4\"", "", g_kernel, "", 0, "The regularised delta function."),
    KEY("coupling", "reproduction", KEYWORD, coupling.params.reproduction, "\"linear\"", "", g_reproduction, "", 0,
        "What every row must reproduce exactly."),
    KEY("coupling", "failure", KEYWORD, coupling.params.failure, "\"error\"", "", g_stencil_failure, "", 0,
        "What happens to a row that fails its quality checks. Lower order is an explicit choice, counted per row."),
    KEY("coupling", "max_growth", REAL, coupling.params.max_growth, "1.5", "[1, inf)", NO_CHOICES, "", 1,
        "Largest support growth factor when failure allows growing; windows are sized for it."),
    KEY("coupling", "growth_step", REAL, coupling.params.growth_step, "1.25", "(1, inf)", NO_CHOICES, "", 1, "Growth factor per retry."),
    KEY("coupling", "support_scale", REAL, coupling.params.support_scale, "1.0", "(0, inf)", NO_CHOICES, "", 0,
        "Kernel metric as a multiple of the local fluid resolution."),
    KEY("coupling", "pilot_factor", REAL, coupling.params.pilot_factor, "1.5", "(0, inf)", NO_CHOICES, "", 1,
        "Radius of each sample's influence on a variable metric, in units of that sample's directional resolution."),
    KEY("coupling", "max_condition", REAL, coupling.params.max_condition, "100000000.0", "[1, inf)", NO_CHOICES, "", 1,
        "Largest admissible condition number of a row's equilibrated moment matrix."),
    KEY("coupling", "max_amplification", REAL, coupling.params.max_amplification, "4.0", "[1, inf)", NO_CHOICES, "", 1,
        "Largest admissible sum of the absolute weights of a row."),
    KEY("coupling", "moment_tolerance", REAL, coupling.params.moment_tolerance, "1e-10", "(0, 1)", NO_CHOICES, "", 1,
        "Moments within this of exact need no correction, and a corrected row must reach it (relaxed by the conditioning)."),
    KEY("coupling", "sampling", KEYWORD, coupling.params.sampling, "\"report\"", "", g_sampling, "", 0,
        "What a surface too coarse for the fluid resolution does."),
    KEY("coupling", "sampling_ratio", REAL, coupling.params.sampling_ratio, "1.0", "(0, inf)", NO_CHOICES, "", 0,
        "Largest admissible surface edge over the fluid resolution near it."),
    KEY("coupling", "directory_margin", REAL, coupling.params.directory_margin, "0.0", "[0, inf)", NO_CHOICES, "length", 1,
        "Extra advertisement padding beyond each provider's local influence bounds; 0 adds none. Remote certificates are looked up on demand."),

    /* ── redistribute ───────────────────────────────────────────────────── */
    KEY("redistribute", "skip_unchanged_bodies", BOOL, redistribute.skip_unchanged_bodies, "true", "", NO_CHOICES, "", 1,
        "Leave a body no rank saw change owner untouched. Exact, slot for slot, and 1.6-2.1x faster on droplet ensembles."),
    KEY("redistribute", "shrink_slack", REAL, redistribute.shrink_slack, "0.0", "[0, inf)", NO_CHOICES, "", 1,
        "Give back pool capacity above this multiple of a body's size after each redistribution. 0 off; otherwise at least 1, "
        "and 4 avoids thrashing against the growth ladder."),

    /* ── geometry ───────────────────────────────────────────────────────── */
    KEY("geometry", "normal_weighting", KEYWORD, geometry.normal_weighting, "\"area\"", "", g_normal_weighting, "", 0,
        "Face weighting of vertex normals, for `MFTLEnsembleUpdateVertexNormals(ens, cfg.geometry.normal_weighting)`."),

    /* ── output ─────────────────────────────────────────────────────────── */
    KEY("output", "every", INT, output.every, "0", "[0, inf)", NO_CHOICES, "", 0,
        "Write output every this many timesteps; 0 never. Read by the caller's loop."),
    KEY("output", "prefix", STRING, output.prefix, "\"mftl\"", "", NO_CHOICES, "", 0,
        "File name prefix, directories included; the caller appends the step number."),
    KEY("output", "layout", KEYWORD, output.vtk.layout, "\"file_per_rank\"", "", g_layout, "", 0, "How the pieces are written."),
    KEY("output", "encoding", KEYWORD, output.vtk.encoding, "\"binary\"", "", g_encoding, "", 0, "How the arrays are written."),
    KEY("output", "write_owner", BOOL, output.vtk.write_owner, "true", "", NO_CHOICES, "", 0,
        "Write `VertexOwner` and `TriangleOwner`: which rank owns each entity."),
    KEY("output", "write_body_id", BOOL, output.vtk.write_body_id, "true", "", NO_CHOICES, "", 0,
        "Write `BodyID` per cell, which lets a viewer separate the bodies again."),
    KEY("output", "write_global_id", BOOL, output.vtk.write_global_id, "false", "", NO_CHOICES, "", 0,
        "Write `VertexGID` and `TriangleGID`, to follow one entity across steps."),
    KEY("output", "include_ghosts", BOOL, output.vtk.include_ghosts, "false", "", NO_CHOICES, "", 1,
        "Write ghosts too. Every halo triangle then appears once per holder: for looking at the halo, never for integrating."),
};
// clang-format on

/* ═══════════════════════════════════════════════════════════════════════════
 * Lookup
 * ═══════════════════════════════════════════════════════════════════════════ */

enum
{
  N_KEYS = (int) (sizeof(g_keys) / sizeof(g_keys[0])),
  N_SECTIONS = (int) (sizeof(g_sections) / sizeof(g_sections[0]))
};

int MFTLSettingsKeyCount(void) { return N_KEYS; }

const MFTL_SettingInfo *MFTLSettingsKey(int i) { return (i >= 0 && i < N_KEYS) ? &g_keys[i] : MFTL_NULL; }

int MFTLSettingsSectionCount(void) { return N_SECTIONS; }

const MFTL_SettingSection *MFTLSettingsSection(int i) { return (i >= 0 && i < N_SECTIONS) ? &g_sections[i] : MFTL_NULL; }

/* A linear scan: sixty keys, looked up while parsing a file of a few dozen
 * lines, once per run. */
const MFTL_SettingInfo *MFTLSettingsFindKey(const char *key)
{
  if (!key) return MFTL_NULL;
  const char *dot = strchr(key, '.');
  if (!dot) return MFTL_NULL;
  const size_t len = (size_t) (dot - key);

  for (int i = 0; i < N_KEYS; ++i)
  {
    const MFTL_SettingInfo *k = &g_keys[i];
    if (strlen(k->section) == len && strncmp(k->section, key, len) == 0 && strcmp(k->name, dot + 1) == 0) return k;
  }
  return MFTL_NULL;
}
