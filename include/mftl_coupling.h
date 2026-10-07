/**
 * @file mftl_coupling.h
 * @author Christian Gorges
 * @date 2026-10-06
 * @ingroup mftl_coupling
 *
 * @brief Solver-independent coupling between a flow solver's degrees of freedom
 *        and the surface ensemble: interpolation to markers and its exact
 *        transpose, spreading back.
 *
 * ## The operator
 *
 * For every bound marker \f$i\f$ and every location \f$\ell\f$ (a family of
 * solver samples: cell centres, x-faces, tetrahedron centroids, ...), MFTL
 * assembles ONE row of weights \f$W_{ij}\f$ over the samples \f$j\f$ of
 * \f$\ell\f$ in the kernel support of the marker. Interpolation is
 * \f$U = W u\f$; spreading integrated marker loads is \f$b = W^T F\f$, with the
 * SAME stored rows -- never a second, approximately equal matrix -- so the
 * discrete power identity \f$u^T b = U^T F\f$ holds to round-off. A row is
 *
 * \f[
 *   \psi_{ij} = \frac{q_j}{\det \tilde H_i} \prod_k \phi\!\left(\frac{(X_j - x_i)_k}{\tilde H_{i,k}}\right),
 *   \qquad W_{ij} = \psi_{ij}\, p(d_{ij})^T c_i, \qquad M_i c_i = e_0,
 * \f]
 *
 * with \f$q_j\f$ the sample's measure, \f$\tilde H_i\f$ the marker's diagonal
 * kernel metric, \f$p(d) = [1, d_x, d_y, d_z]\f$ and
 * \f$M_i = \sum_j \psi_{ij} p p^T\f$: the LINEAR correction reproduces constants
 * and linear fields. A row whose base kernel already has those moments (Peskin
 * on a complete uniform lattice) is left exactly as the kernel made it.
 *
 * ## Who holds what
 *
 * The solver REGISTERS descriptors on the ranks that own the samples
 * (#MFTL_SolverGrid, local calls): structured blocks of a lattice, or point
 * sets. Nothing is gathered. A coupler (#MFTL_Coupler, collective) builds a
 * sparse provider directory; a SUPPORT REQUEST derived from the markers asks
 * exactly the owners of the samples that the markers' kernels (plus a motion
 * skin) can reach; a WINDOW holds that band and its forward/reverse schedules;
 * a BINDING holds the rows at one marker geometry; a FETCH holds band values of
 * one field state. Rows live with the markers' owner; every remote sample is
 * fetched from, and every spread load returned to, its unique owner.
 *
 * ## Coverage is a contract, never a guess
 *
 * A row is built only when every sample its kernel can touch is known to be
 * present. For a LATTICE location this is exact: the requester knows the
 * lattice, so it counts the samples a request box must contain and compares
 * with what the owners report (active + masked + unexported). For a POINT
 * location the owners certify regions (#MFTL_CoverageDesc). Anything missing
 * returns #MFTL_COUPLING_NEED_COVERAGE on every rank, with the regions to export
 * (MFTLCouplerGetRequests()) -- it is never renormalised away, and an
 * unprovided region is never treated as a wall.
 *
 * ## Collective calls agree their status
 *
 * Every collective call returns the SAME #MFTL_CouplingStatus on every rank,
 * agreed before any rank takes a different communication path; a failure
 * leaves previous handles and every output untouched. Theory, coverage examples
 * and the solver loops: src/coupling/README.md.
 */

#ifndef MFTL_COUPLING_H
#define MFTL_COUPLING_H

#include <mpi.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mftl_types.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Handles and status
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Opaque: the solver's registered samples on this rank (local). */
typedef struct MFTL_SolverGrid MFTL_SolverGrid;
/** @brief Opaque: one coupling between an ensemble and a solver grid (collective). */
typedef struct MFTL_Coupler MFTL_Coupler;
/** @brief Opaque: the outcome of one support request (who asked whom for what). */
typedef struct MFTL_SupportRequest MFTL_SupportRequest;
/** @brief Opaque: a certified band of samples and its transfer schedules. */
typedef struct MFTL_CouplingWindow MFTL_CouplingWindow;
/** @brief Opaque: interpolation rows for an ordered set of markers at one geometry. */
typedef struct MFTL_CouplingBinding MFTL_CouplingBinding;
/** @brief Opaque: band values of one field state. */
typedef struct MFTL_CouplingFetch MFTL_CouplingFetch;
/** @brief Opaque: a collectively agreed transfer, ready to be started. */
typedef struct MFTL_TransferReady MFTL_TransferReady;

/**
 * @brief Outcome of a coupling call, ordered by severity so that the agreed
 *        status of a collective call is the maximum over the ranks.
 */
typedef enum MFTL_CouplingStatus
{
  MFTL_COUPLING_OK = 0,  ///< Success.
  MFTL_COUPLING_NEED_COVERAGE = 1,  ///< Some required sample was not provided; export it and retry (MFTLCouplerGetRequests()).
  /** A row failed its quality checks: empty, rank-deficient, ill-conditioned, amplifying, or truncated under an ERROR boundary policy. */
  MFTL_COUPLING_BAD_STENCIL = 2,
  MFTL_COUPLING_BAD_SAMPLING = 3,  ///< The surface is too coarse for the fluid resolution under an ERROR sampling policy.
  MFTL_COUPLING_STALE = 4,  ///< A handle no longer describes the surface or the grid (moved markers outside the window, changed generation, grid updated).
  MFTL_COUPLING_INVALID = 5  ///< Invalid arguments or registration (duplicate ownership, non-finite coordinates, mismatched definitions across ranks).
} MFTL_CouplingStatus;

/**
 * @brief A short, constant name for a status ("OK", "NEED_COVERAGE", ...).
 * @param status A value of #MFTL_CouplingStatus.
 * @return A static string; "UNKNOWN" for anything else.
 */
const char *MFTLCouplingStatusName(int status);

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Kernels
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The built-in regularised delta functions. Tensor kernels are products
 *        of a 1-D \f$\phi\f$ with \f$\int\phi = 1\f$; the radial one is
 *        normalised in 3-D.
 */
typedef enum MFTL_CouplingKernel
{
  MFTL_KERNEL_PESKIN4 = 0,  ///< Peskin's 4-point function, half-width 2; constants and linears on a lattice. C1. The default.
  MFTL_KERNEL_ROMA3 = 1,  ///< Roma-Peskin-Berger 3-point function, half-width 1.5; constants and linears on a lattice. C0.
  MFTL_KERNEL_COSINE4 = 2,  ///< \f$(1+\cos(\pi r/2))/4\f$, half-width 2; constants only. C1.
  MFTL_KERNEL_HAT2 = 3,  ///< Linear hat \f$1-|r|\f$, half-width 1: trilinear interpolation on a lattice. C0.
  MFTL_KERNEL_BSPLINE3 = 4,  ///< Cubic B-spline, half-width 2; constants and linears on a lattice. C2.
  MFTL_KERNEL_WENDLAND_C2 = 5,  ///< Radial Wendland C2, support radius 2, unit 3-D integral; no lattice moments. C2.
  MFTL_KERNEL_USER = 6  ///< An application kernel, #MFTL_KernelDesc.
} MFTL_CouplingKernel;

/**
 * @brief Evaluates a kernel at @p n arguments.
 *
 * Tensor kernel: \f$\phi(r_i)\f$ in 1-D. Radial kernel: \f$\phi(\rho_i)\f$ with
 * \f$\rho = |d|\f$, normalised in 3-D. Called once per marker and axis with
 * every candidate at once -- never once per sample.
 *
 * @param r   [n] arguments, dimensionless (distance over the metric).
 * @param phi [n] values out; exactly 0 outside the declared support.
 * @param n   Number of arguments.
 * @param ctx The application context from #MFTL_KernelDesc.
 */
typedef void (*MFTL_KernelFn)(const double *r, double *phi, int n, void *ctx);

/**
 * @struct MFTL_KernelDesc
 * @brief What an application kernel declares. Nothing is inferred by sampling.
 */
typedef struct MFTL_KernelDesc
{
  MFTL_KernelFn eval;  ///< The evaluator.
  /** The derivative \f$\phi'(r)\f$ (tensor) or \f$\phi'(\rho)\f$ (radial), or #MFTL_NULL: only #MFTL_OPERATOR_VALUE_DERIVATIVE needs it. */
  MFTL_KernelFn deriv;
  void *ctx;  ///< Passed through to @c eval and @c deriv.
  double half_width;  ///< Support half-width in units of the metric; \f$\phi(r)=0\f$ for \f$|r|\ge\f$ it. Positive and finite.
  int moment_order;  ///< Moments reproduced on a complete uniform lattice: -1 none, 0 constants, 1 constants and linears.
  bool radial;  ///< `true`: \f$\phi(|d|)\f$ normalised in 3-D; `false`: a tensor product of 1-D \f$\phi\f$.
  bool nonnegative;  ///< \f$\phi \ge 0\f$ everywhere.
} MFTL_KernelDesc;

/**
 * @struct MFTL_KernelInfo
 * @brief The declared properties of a kernel.
 */
typedef struct MFTL_KernelInfo
{
  const char *name;  ///< Short name ("peskin4", ...).
  double half_width;  ///< Support half-width in metric units.
  int moment_order;  ///< -1, 0 or 1; see #MFTL_KernelDesc.
  int smoothness;  ///< Continuous derivatives: 0 for C0, 1 for C1, 2 for C2.
  bool radial;  ///< Radial (3-D) rather than tensor.
  bool nonnegative;  ///< Never negative.
} MFTL_KernelInfo;

/**
 * @brief The declared properties of a built-in kernel.
 * @param kernel A built-in kernel (not #MFTL_KERNEL_USER).
 * @param[out] info Receives the properties.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for an unknown kernel.
 * @rank_local
 */
int MFTLCouplingKernelInfo(MFTL_CouplingKernel kernel, MFTL_KernelInfo *info);

/**
 * @brief Evaluates a built-in kernel; see #MFTL_KernelFn for the arguments.
 * @param kernel A built-in kernel.
 * @param r   [n] arguments.
 * @param phi [n] values out.
 * @param n   Number of arguments.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for an unknown kernel.
 * @rank_local
 */
int MFTLCouplingKernelEval(MFTL_CouplingKernel kernel, const double *r, double *phi, int n);

/**
 * @brief Evaluates a built-in kernel's derivative: \f$\phi'(r)\f$ for a tensor
 *        kernel (odd in r), \f$\phi'(\rho)\f$ for the radial one. Where the
 *        derivative does not exist (the hat at 0, the breakpoints of a C0
 *        kernel) the value is one of the one-sided limits or 0.
 * @param kernel A built-in kernel.
 * @param r    [n] arguments.
 * @param dphi [n] derivatives out.
 * @param n    Number of arguments.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for an unknown kernel.
 * @rank_local
 */
int MFTLCouplingKernelDerivative(MFTL_CouplingKernel kernel, const double *r, double *dphi, int n);

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Parameters
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief What a row must reproduce. */
typedef enum MFTL_Reproduction
{
  MFTL_REPRODUCE_LINEAR = 0,  ///< Constants and linear fields; negative weights are possible. Force AND torque conservation. The default.
  MFTL_REPRODUCE_CONSTANT = 1,  ///< Normalised positive base weights: constants and positivity, no linear precision.
  MFTL_REPRODUCE_NONE = 2  ///< The base kernel as it is; the achieved moments are reported, not enforced.
} MFTL_Reproduction;

/**
 * @brief What a binding's rows compute from a location's samples.
 *
 * The two gradients are different operations and are named apart on purpose.
 * The FIT's slope reproduces the gradient of every affine field but is not the
 * derivative of the interpolated value; the DERIVATIVE differentiates
 * everything the value row depends on -- the kernel, the moment correction,
 * the metric -- so it is exactly \f$\partial(Wu)/\partial x\f$, and exists
 * only where that is differentiable (not at the breakpoints of a C0 kernel,
 * nor where a row's growth step or reproduction order switches).
 */
typedef enum MFTL_CouplingOperator
{
  MFTL_OPERATOR_VALUE = 0,  ///< \f$U = W u\f$: the value (a 3-vector at a directional location). The default.
  /**
   * \f$\nabla u\f$ from the weighted affine fit \f$u \approx \alpha + \beta\cdot d\f$ the
   * LINEAR value row is built on: \f$\nabla u = \tilde H^{-T}\beta\f$. Reproduces
   * the gradient of every affine field exactly; always solves the full
   * moment matrix (the reproduction policy does not apply, and there is no
   * lower-order fallback: a constant fit has no slope). Scalar locations only.
   */
  MFTL_OPERATOR_GRADIENT_FIT = 1,
  /**
   * \f$\partial U / \partial x\f$ of the value row as the reproduction policy builds
   * it, differentiated exactly: kernel (#MFTL_KernelDesc::deriv), moment
   * correction and a varying metric included. For a LINEAR row it too
   * reproduces affine gradients. Scalar locations only.
   */
  MFTL_OPERATOR_VALUE_DERIVATIVE = 2
} MFTL_CouplingOperator;

/** @brief Bit flags: what happens when a row fails its checks. 0 (the default) is an error. */
typedef enum MFTL_StencilFailure
{
  MFTL_STENCIL_ERROR = 0,  ///< Fail with #MFTL_COUPLING_BAD_STENCIL.
  MFTL_STENCIL_GROW = 1,  ///< Retry with a wider support, up to `max_growth`.
  MFTL_STENCIL_LOWER_ORDER = 2  ///< Then accept CONSTANT reproduction, counted per row.
} MFTL_StencilFailure;

/** @brief What a surface too coarse for the fluid resolution does. */
typedef enum MFTL_SamplingPolicy
{
  MFTL_SAMPLING_OFF = 0,  ///< Not checked.
  MFTL_SAMPLING_REPORT = 1,  ///< Measured and counted (MFTLCouplerSurfaceSpacing()), never an error. The default.
  MFTL_SAMPLING_ERROR = 2  ///< A spreading bind of vertex or centroid markers returns #MFTL_COUPLING_BAD_SAMPLING.
} MFTL_SamplingPolicy;

/**
 * @struct MFTL_CouplerParams
 * @brief The numerical policy of a coupler. Identical on every rank (checked).
 *
 * Every tolerance is DIMENSIONLESS: moments are taken in metric units, the
 * conditioning is that of the equilibrated moment matrix.
 */
typedef struct MFTL_CouplerParams
{
  MFTL_CouplingKernel kernel;  ///< Default #MFTL_KERNEL_PESKIN4.
  MFTL_KernelDesc user_kernel;  ///< Used only for #MFTL_KERNEL_USER.
  MFTL_Reproduction reproduction;  ///< Default #MFTL_REPRODUCE_LINEAR.
  int failure;  ///< #MFTL_StencilFailure flags, default #MFTL_STENCIL_ERROR.
  double max_growth;  ///< Largest support growth factor under #MFTL_STENCIL_GROW; >= 1. Default 1.5. Windows are sized for it.
  double growth_step;  ///< Factor per retry; > 1. Default 1.25.
  double support_scale;  ///< Kernel metric = this times the local resolution. Default 1.
  double pilot_factor;  ///< Radius of the neighbourhood whose resolution sets a variable metric, in units of the location's largest resolution. Default 1.5.
  double max_condition;  ///< Largest admissible condition number of the equilibrated moment matrix. Default 1e8.
  double max_amplification;  ///< Largest admissible \f$\sum_j |W_{ij}|\f$. Default 4.
  double moment_tolerance;  ///< Moments within this of exact need no correction, and a corrected row must reach it. Default 1e-10.
  MFTL_SamplingPolicy sampling;  ///< Default #MFTL_SAMPLING_REPORT.
  double sampling_ratio;  ///< Largest admissible surface edge over fluid resolution. Default 1.
  /** Reach of the provider directory beyond a rank box, a physical length; 0 derives it. Grown automatically when a request needs more. */
  double directory_margin;
} MFTL_CouplerParams;

/**
 * @brief Fills @p params with the defaults documented on each field.
 * @param[out] params The parameter block.
 * @rank_local
 */
void MFTLCouplerDefaultParams(MFTL_CouplerParams *params);

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Registration: the solver grid (LOCAL calls)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief How a location's samples are laid out. */
typedef enum MFTL_LatticeKind
{
  MFTL_LATTICE_NONE = 0,  ///< Point samples (MFTLSolverGridAddPoints()); coverage by certified regions.
  MFTL_LATTICE_UNIFORM = 1,  ///< A uniform Cartesian lattice; coverage by exact index counting.
  MFTL_LATTICE_RECTILINEAR = 2  ///< A tensor lattice with per-axis coordinate tables; coverage by exact index counting.
} MFTL_LatticeKind;

/**
 * @struct MFTL_LatticeDesc
 * @brief The GLOBAL lattice of a location: identical on every rank (checked).
 *
 * Sample \f$(i,j,k)\f$, \f$0 \le i_k < \f$ `dims[k]`, sits at
 * `origin[k] + i_k*spacing[k]` (uniform) or `coords[k][i_k]` (rectilinear). Its
 * measure is the product of its per-axis widths: `spacing` (uniform) or
 * `faces[k][i+1] - faces[k][i]` (rectilinear), which is also its resolution.
 * On a periodic axis the lattice must span exactly one period
 * (`dims*spacing == L`, or `faces[dims] - faces[0] == L`) with every sample in
 * `[min, max)`.
 */
typedef struct MFTL_LatticeDesc
{
  MFTL_LatticeKind kind;  ///< #MFTL_LATTICE_UNIFORM or #MFTL_LATTICE_RECTILINEAR here.
  int64_t dims[3];  ///< Samples per axis.
  double origin[3];  ///< Uniform: position of sample (0,0,0).
  double spacing[3];  ///< Uniform: spacing, positive.
  const double *coords[3];  ///< Rectilinear: `[dims[k]]` ascending sample coordinates per axis. Copied.
  const double *faces[3];  ///< Rectilinear: `[dims[k]+1]` ascending bounds of each sample's cell per axis. Copied.
} MFTL_LatticeDesc;

/**
 * @brief What a row does where its support holds a position with no sample:
 *        past a non-periodic domain wall, past the edge of a lattice smaller
 *        than the domain, or on a masked (solid) cell. Such a row is TRUNCATED;
 *        a support that crosses a wall with no position beyond it is complete.
 */
typedef enum MFTL_BoundaryPolicy
{
  MFTL_BOUNDARY_TRUNCATE = 0,  ///< Use the samples inside; the reproduction policy corrects the one-sided row. Counted. The default.
  MFTL_BOUNDARY_ERROR = 1,  ///< A truncated row is #MFTL_COUPLING_BAD_STENCIL.
  /**
   * Complete the support beyond a non-periodic domain wall with MIRROR IMAGES
   * of the location's own samples (reflected about the wall plane; across two
   * or three walls at an edge or corner, reflected in x, then y, then z). An
   * image carries its sample's value times the face's parity, plus the wall
   * value of an ODD face (#MFTL_WallParity) -- the solver's own ghost-cell
   * rule -- so the stored row is composed onto the real samples, and an ODD
   * face's share is a separate per-row coefficient of the field's
   * `wall_value`. Interpolation adds it; spreading reports it as the face's
   * reaction (`wall_reaction`): force and power balance with the wall
   * included. Rows cut by masked cells are corrected as under TRUNCATE.
   */
  MFTL_BOUNDARY_MIRROR = 2
} MFTL_BoundaryPolicy;

/** @brief The parity of a mirror face (#MFTL_BOUNDARY_MIRROR). */
typedef enum MFTL_WallParity
{
  MFTL_WALL_EVEN = 0,  ///< The image carries the sample's value: a zero normal derivative on the wall (symmetry, free slip, insulation).
  MFTL_WALL_ODD = 1  ///< The image carries \f$2 u_w - u\f$: the field equals \f$u_w\f$ (the field's `wall_value`) on the wall (no slip, a prescribed value).
} MFTL_WallParity;

/**
 * @struct MFTL_LocationDesc
 * @brief One family of samples. Same definitions, same order, on every rank --
 *        including ranks that own none of its samples.
 *
 * Every sample is a POINT SAMPLE of a scalar: a vector at one location is
 * several fields at that location, a staggered (MAC) vector is one field at
 * each of three locations.
 */
typedef struct MFTL_LocationDesc
{
  const char *name;  ///< At most 31 characters; part of the cross-rank check.
  MFTL_LatticeDesc lattice;  ///< `kind == MFTL_LATTICE_NONE` for point samples.
  MFTL_BoundaryPolicy boundary;  ///< Wall policy.
  /** #MFTL_BOUNDARY_MIRROR: the parity of each domain face, ordered x-lo, y-lo, z-lo, x-hi, y-hi, z-hi; ignored on periodic axes. */
  MFTL_WallParity wall[6];
  /**
   * Point locations: every sample declares a full CELL matrix
   * (#MFTL_PointSetDesc::cell) and rows use it -- the kernel's offsets are
   * \f$H^{-1}(X - x)\f$, so a tensor kernel follows a rotated or sheared cell's
   * own axes, and a cloud that is a complete lattice in those axes needs no
   * moment correction at all. The marker's matrix is the Shepard average of its
   * neighbours'.
   */
  bool anisotropic;
  /**
   * Point locations: every sample is DIRECTIONAL -- its value is
   * \f$q_j = s_j \cdot u(X_j)\f$ for the sample's vector
   * #MFTL_PointSetDesc::direction (a unit normal for a normal velocity, the
   * oriented area vector for a face flux) -- and a value row reconstructs the
   * vector \f$u\f$ at the marker: a field at such a location interpolates to,
   * and spreads from, THREE components per marker, with block weights
   * \f$A_j \in \mathbb{R}^3\f$ and \f$b_j = A_j^T F\f$. CONSTANT reproduction is
   * \f$A_j = \psi_j T^{-1}\hat s_j / |s_j|\f$ with \f$T = \sum\psi\hat s\hat s^T\f$;
   * LINEAR fits a 12-coefficient affine vector field, which reproduces rigid
   * motions and so conserves force and torque by work. Both need the
   * directions to span space (directional rank). Not with
   * #MFTL_BOUNDARY_MIRROR, nor with the gradient operators.
   */
  bool directional;
} MFTL_LocationDesc;

/**
 * @struct MFTL_BlockDesc
 * @brief The samples of a lattice location this rank OWNS, and where its field
 *        array keeps them. Index boxes are half-open, in lattice indices.
 *
 * Element \f$e\f$ of lattice index \f$(i,j,k)\f$ in a field array is
 * \f$\sum_k (i_k - \texttt{array\_lo}_k)\,\texttt{stride}_k\f$; a field passes
 * one base pointer per block. Owned boxes of different ranks must not overlap.
 */
typedef struct MFTL_BlockDesc
{
  int location;  ///< Index returned by MFTLSolverGridAddLocation().
  int64_t owned_lo[3];  ///< First owned index per axis.
  int64_t owned_hi[3];  ///< One past the last owned index per axis.
  int64_t array_lo[3];  ///< Lattice index of array element 0, per axis (owned_lo minus ghost layers, typically).
  int64_t array_dims[3];  ///< Array elements per axis, ghosts included; covers the owned box.
  int64_t stride[3];  ///< Elements between neighbours along each axis; all zero for x-fastest contiguous.
  int ghost_layers[3];  ///< Layers around the owned box (within the array) whose values a `halo_valid` field holds. Optional; enables direct ghost reads.
  const unsigned char *active;  ///< Optional `[owned box]` x-fastest mask, 0 = no such sample (a solid cell). Copied.
} MFTL_BlockDesc;

/**
 * @struct MFTL_PointSetDesc
 * @brief Owned point samples of a point location (an unstructured mesh's cell
 *        centroids, nodes, particles). Copied at registration.
 */
typedef struct MFTL_PointSetDesc
{
  int location;  ///< Index returned by MFTLSolverGridAddLocation().
  int64_t n;  ///< Number of samples; 0 is legal.
  const uint64_t *keys;  ///< `[n]` ids, unique across ALL ranks within the location.
  const double *pos;  ///< `[3n]` positions, inside the domain box (in `[min,max)` on periodic axes).
  const double *measure;  ///< `[n]` positive sampling measures (cell volumes).
  const double *h;  ///< `[3n]` per-axis resolution of each sample, or #MFTL_NULL to use `h_uniform`.
  double h_uniform[3];  ///< Resolution of every sample when `h` is #MFTL_NULL.
  const double *extent;  ///< `[3n]` per-axis half-extent of the cell each sample represents (bounds the ownership tiles), or #MFTL_NULL for the resolution.
  const int64_t *array_index;  ///< `[n]` element index in the field arrays, or #MFTL_NULL for 0..n-1.
  /**
   * Anisotropic location: `[9n]` each sample's cell matrix, row-major, its
   * COLUMNS the cell's edge vectors (an axis-aligned cell of widths h is
   * `diag(h)`); invertible. Its per-axis extents \f$\sum_j |H_{kj}|\f$ replace
   * `h` as the sample's resolution bound. Ignored elsewhere.
   */
  const double *cell;
  /** Directional location: `[3n]` each sample's vector \f$s_j\f$, nonzero and finite; its value is \f$s_j \cdot u\f$. Ignored elsewhere. */
  const double *direction;
} MFTL_PointSetDesc;

/** @brief Whether a rank exports everything it owns of a location. */
typedef enum MFTL_GridExposure
{
  MFTL_GRID_FULL = 0,  ///< Every owned active sample is registered. The default.
  MFTL_GRID_SELECTED = 1  ///< Only a band is; #MFTL_CoverageDesc says where it is complete.
} MFTL_GridExposure;

/**
 * @struct MFTL_CoverageDesc
 * @brief Where this rank's export of one location is COMPLETE. Replaces the
 *        previous description of that location on this rank.
 *
 * Lattice location: the exported index boxes, each inside an owned block
 * (half-open, lattice indices). Point location: closed physical boxes in which
 * every owned active sample has been registered -- the solver's own assertion,
 * which a point cloud cannot verify.
 */
typedef struct MFTL_CoverageDesc
{
  int location;  ///< The location.
  MFTL_GridExposure exposure;  ///< #MFTL_GRID_FULL ignores the boxes.
  int n_boxes;  ///< Number of boxes.
  const int64_t *index_boxes;  ///< Lattice: `[n_boxes][6]` lo xyz, hi xyz.
  const double *boxes;  ///< Points: `[n_boxes][6]` lo xyz, hi xyz.
} MFTL_CoverageDesc;

/**
 * @struct MFTL_ProviderRegions
 * @brief Where this rank OWNS samples of a point location, exported or not.
 *
 * The directory routes requests by these, so they must enclose every owned
 * sample including those not yet exported, and together the ranks' regions must
 * cover the fluid domain: a region nobody claims is reported as unprovided. A
 * region of the box with no fluid (a solid) is claimed by some rank and simply
 * holds no samples. Required for a SELECTED point location; a FULL one defaults
 * to tiles of its samples' cell extents -- 64 Morton-consecutive samples per
 * tile, so a hole in ONE registered cloud thinner than a tile is claimed by the
 * tile's box (certified empty, not reported unprovided). A gap between ranks or
 * between separately registered sets is reported; regions given here are exact.
 *
 * The resolution bounds are what a first request is sized by before anything
 * is exported: a SELECTED rank that has registered no samples yet must declare
 * them, or no rank can know how far a kernel reaches.
 */
typedef struct MFTL_ProviderRegions
{
  int location;  ///< The location.
  int n_boxes;  ///< Number of boxes.
  const double *boxes;  ///< `[n_boxes][6]` closed lo xyz, hi xyz.
  double h_min[3];  ///< Smallest resolution of the samples owned in the regions, per axis; 0 to derive it from the registered samples.
  double h_max[3];  ///< Largest, likewise; it bounds the kernel metric and therefore every request.
} MFTL_ProviderRegions;

/**
 * @struct MFTL_ConstraintDesc
 * @brief Samples whose value the solver BUILDS from other samples of the same
 *        location: hanging nodes of a composite mesh, a prescribed sample on a
 *        wall, any \f$u_c = \sum_j P_{cj} u_j + u_{bc,c}\f$ the solver keeps
 *        current in the sample's array element (as it does for its ghosts).
 *
 * Interpolation reads such a sample as stored. Spreading applies the exact
 * transpose of the construction: the whole load \f$b_c\f$ a constrained sample
 * accumulates is passed on as \f$P_{cj} b_c\f$ to each free sample j, wherever
 * it is owned, and the sample's own element receives nothing -- so a solver
 * must NOT also condense these loads itself. \f$b_c\f$ is reported through the
 * field's `constraint_load`, which is what the work of the affine part,
 * \f$\sum_c u_{bc,c} b_c\f$, needs: \f$U \cdot F = \sum_{free} u_j b_j + \sum_c
 * u_{bc,c} b_c\f$.
 *
 * Registered on the rank that OWNS the constrained samples; one level only (a
 * free sample is not itself constrained); validated collectively at
 * MFTLCouplerCreate() / MFTLCouplerUpdateGrid(), including every free-owner
 * claim. A lattice sample's key is its wrapped index `i + N0 (j + N1 k)`.
 */
typedef struct MFTL_ConstraintDesc
{
  int location;  ///< The location; its constraints on this rank are REPLACED.
  int64_t n;  ///< Constrained samples this rank owns; 0 clears them.
  const uint64_t *keys;  ///< `[n]` their keys.
  const int64_t *row_ptr;  ///< `[n+1]` rows into the entry arrays.
  const uint64_t *free_keys;  ///< `[row_ptr[n]]` the free samples each is built from.
  const double *coef;  ///< `[row_ptr[n]]` \f$P_{cj}\f$, finite.
  const int *free_owner;  ///< `[row_ptr[n]]` the rank owning each free sample, in the solver communicator passed to MFTLSolverGridCreate().
} MFTL_ConstraintDesc;

/**
 * @brief An empty solver grid.
 * @param solver_comm The solver's communicator: the ensemble's group of
 *        processes, any rank order. Checked at MFTLCouplerCreate().
 * @return The grid. Allocation failure aborts.
 * @rank_local
 */
MFTL_SolverGrid *MFTLSolverGridCreate(MPI_Comm solver_comm);

/**
 * @brief Declares a location. Same calls, same order, on every rank.
 * @param grid The grid.
 * @param desc The location; lattice tables are copied.
 * @return The location index, or #MFTL_ERROR on an invalid description.
 * @rank_local
 */
int MFTLSolverGridAddLocation(MFTL_SolverGrid *grid, const MFTL_LocationDesc *desc);

/**
 * @brief Registers an owned block of a lattice location.
 * @param grid The grid.
 * @param desc The block; the mask is copied.
 * @return The block's piece index within its location on this rank, or
 *         #MFTL_ERROR on an invalid description (reported on stderr).
 * @rank_local
 */
int MFTLSolverGridAddBlock(MFTL_SolverGrid *grid, const MFTL_BlockDesc *desc);

/**
 * @brief Registers owned point samples of a point location.
 * @param grid The grid.
 * @param desc The samples; every array is copied.
 * @return The set's piece index within its location on this rank, or
 *         #MFTL_ERROR on an invalid description (reported on stderr).
 * @rank_local
 */
int MFTLSolverGridAddPoints(MFTL_SolverGrid *grid, const MFTL_PointSetDesc *desc);

/**
 * @brief Declares where this rank's export of a location is complete.
 * @param grid The grid.
 * @param desc The coverage; copied.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an invalid description.
 * @rank_local
 */
int MFTLSolverGridSetCoverage(MFTL_SolverGrid *grid, const MFTL_CoverageDesc *desc);

/**
 * @brief Declares where this rank owns samples of a point location.
 * @param grid The grid.
 * @param desc The regions; copied.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an invalid description.
 * @rank_local
 */
int MFTLSolverGridSetProviderRegions(MFTL_SolverGrid *grid, const MFTL_ProviderRegions *desc);

/**
 * @brief Declares the constrained samples this rank owns of one location
 *        (#MFTL_ConstraintDesc), replacing earlier ones.
 * @param grid The grid.
 * @param desc The constraints; copied.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an invalid description.
 * @rank_local
 */
int MFTLSolverGridSetConstraints(MFTL_SolverGrid *grid, const MFTL_ConstraintDesc *desc);

/**
 * @brief Drops every block, point set, coverage and region; keeps the locations.
 *        The full-replacement path for a changed grid; follow with
 *        MFTLCouplerUpdateGrid().
 * @param grid The grid.
 * @rank_local
 */
void MFTLSolverGridReset(MFTL_SolverGrid *grid);

/**
 * @brief Destroys a grid. Destroy every coupler using it first.
 * @param grid Address of the grid; set to #MFTL_NULL.
 * @rank_local
 */
void MFTLSolverGridDestroy(MFTL_SolverGrid **grid);

/**
 * @brief Number of locations.
 * @param grid The grid.
 * @return The count.
 * @rank_local
 */
int MFTLSolverGridNumLocations(const MFTL_SolverGrid *grid);

/**
 * @brief Number of pieces (blocks or point sets) of a location on this rank:
 *        the length of a field's `data` array.
 * @param grid The grid.
 * @param location The location.
 * @return The count, or #MFTL_ERROR for an invalid location.
 * @rank_local
 */
int MFTLSolverGridNumPieces(const MFTL_SolverGrid *grid, int location);

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Markers, windows, fields
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Which surface points carry the rows. */
typedef enum MFTL_MarkerKind
{
  MFTL_MARKERS_VERTICES = 0,  ///< Owned vertices; weight = dual area \f$\sum_{T \ni v} A_T/3\f$.
  MFTL_MARKERS_CENTROIDS = 1,  ///< Owned triangles at their centroids; weight = area.
  MFTL_MARKERS_USER = 2,  ///< Positions and weights supplied by the caller.
  /**
   * Quadrature points of the owned triangles: each is cut into \f$m^2\f$
   * congruent sub-triangles (m per edge) with one point at each sub-centroid,
   * weight \f$A/m^2\f$ -- exact for the linear hat functions of the vertices,
   * so the points' weights sum, per vertex, to its dual area. m is chosen per
   * triangle so that a sub-triangle's longest edge stays within
   * `quadrature_ratio` of the finest fluid resolution near it: the surface
   * mesh need not be refined to the fluid. Vertex fields and point fields are
   * converted by MFTLCouplerQuadratureToVertices() and
   * MFTLCouplerQuadratureFromVertices(), an exact adjoint pair.
   */
  MFTL_MARKERS_QUADRATURE = 3
} MFTL_MarkerKind;

/**
 * @struct MFTL_MarkerSpec
 * @brief Which markers, for which locations.
 *
 * Marker order is deterministic: local bodies by ascending id, then owned
 * entities in pool order (VERTICES/CENTROIDS), or the caller's order (USER).
 */
typedef struct MFTL_MarkerSpec
{
  MFTL_MarkerKind kind;  ///< The marker kind.
  int n_locations;  ///< Number of entries in @c locations; 0 means every location in index order.
  const int *locations;  ///< Ordered location list.
  int64_t n_user;  ///< USER: number of markers on this rank.
  const double *user_pos;  ///< USER: `[3 n_user]` positions.
  const double *user_weight;  ///< USER: `[n_user]` integrated weights, or #MFTL_NULL for 1.
  const uint64_t *user_keys;  ///< USER: `[n_user]` ids, or #MFTL_NULL for the index.
  bool interpolation_only;  ///< Skip the sampling check of a spreading bind.
  int quadrature_level;  ///< QUADRATURE: points per triangle edge for EVERY triangle (m, so \f$m^2\f$ points); 0 chooses it per triangle.
  double quadrature_ratio;  ///< QUADRATURE, adaptive: largest sub-triangle edge over the local fluid resolution; 0 means the coupler's `sampling_ratio`.
  int quadrature_max_level;  ///< QUADRATURE, adaptive: largest m; 0 means 16. A triangle that needs more is #MFTL_COUPLING_BAD_SAMPLING under an ERROR policy.
  /**
   * What the rows compute (#MFTL_CouplingOperator), default the value. A
   * gradient gives THREE components per marker and field (x, y, z), and its
   * transpose spreads a vector load per marker -- the adjoint of the gradient,
   * for a variational use. Stage bindings inherit it.
   */
  MFTL_CouplingOperator operation;
} MFTL_MarkerSpec;

/**
 * @struct MFTL_WindowSpec
 * @brief The motion a window must absorb.
 *
 * A marker, or a stage position of one, displaced by at most `skin` per axis
 * from where the window was requested stays inside it: binds within it need no
 * new request and no new fetch. Exceeding it is #MFTL_COUPLING_STALE, never a
 * truncated row. Precisely: the requesting markers are binned on a grid of
 * half the largest kernel reach, anchored at the domain's lower corner, and a
 * position is inside the window when it lies within `skin` of a bin that held
 * one of them -- so motion inside such a bin is free.
 */
typedef struct MFTL_WindowSpec
{
  double skin[3];  ///< Physical displacement allowance per axis; >= 0.
  /**
   * Migration allowance per axis; >= 0, 0 for none. A rank's window then
   * also holds every bin of the OTHER ranks' markers that lies within this
   * distance of its box -- so after a redistribution has moved markers that far
   * onto it (with the skin covering how far they moved) they bind in the window
   * as it is: no request, no fetch. For vertex and centroid markers the skin
   * is the natural value; quadrature points follow their triangle's centroid,
   * so add the largest triangle. Costs the replicated bins near rank boundaries.
   */
  double migration[3];
} MFTL_WindowSpec;

/**
 * @struct MFTL_StageGeometry
 * @brief Marker positions of one stage, in the base binding's marker order.
 */
typedef struct MFTL_StageGeometry
{
  const double *pos;  ///< `[3 n_markers]`; any image -- the minimum image about the base position is used.
  const double *weight;  ///< `[n_markers]` new finite integrated weights, strictly positive for QUADRATURE, or #MFTL_NULL to keep the base ones.
} MFTL_StageGeometry;

/**
 * @struct MFTL_MarkerRef
 * @brief What a marker is: enough to write a result back.
 */
typedef struct MFTL_MarkerRef
{
  uint64_t key;  ///< `(body id << 32) | entity global id`, or the user key.
  int body_gid;  ///< Body id; -1 for USER markers.
  int slot;  ///< Local object slot at bind time; -1 for USER.
  int entity;  ///< Vertex or triangle pool index, or the user index.
  int sub;  ///< QUADRATURE: the point's index within its triangle (`key` and `sub` together are unique); 0 otherwise.
  double weight;  ///< Integrated weight (area).
} MFTL_MarkerRef;

/**
 * @struct MFTL_SolverField
 * @brief One scalar component at one location, in the solver's own arrays.
 *
 * Not retained: a transfer reads (or, spreading, adds to) these arrays during
 * the call or between Begin and End only.
 */
typedef struct MFTL_SolverField
{
  int location;  ///< The location.
  double *const *data;  ///< `[pieces of the location on this rank]` base pointers, registration order.
  ptrdiff_t stride;  ///< Doubles between consecutive elements; 0 means 1.
  uint64_t version;  ///< The solver's value token (a step and stage counter); recorded with a fetch.
  bool halo_valid;  ///< The blocks' declared ghost layers hold their owners' current values. Same on every rank.
  /**
   * SPREAD only: the SOLVER completes the transpose of the ghost route. The
   * loads of samples this rank reads from its own ghost layers (the halo route)
   * are written into those ghost elements of the target instead of being sent
   * to their owners, and every other element of the target's declared ghost
   * layers is set to ZERO -- so exactly one reverse halo accumulation by the
   * solver (each ghost element added into its owner) completes \f$W^T F\f$, and no
   * value the array held there before is added with it. Owned elements are
   * added to as always. Distinct fields of one batch must not share ghost
   * elements. Same on every rank (part of the transfer signature).
   */
  bool halo_transpose;
  /** FETCH: `[6]` values on ODD mirror faces (x-lo, y-lo, z-lo, x-hi, y-hi, z-hi), or #MFTL_NULL for zero. Kept valid through FetchBegin(), which copies them. */
  const double *wall_value;
  /**
   * SPREAD: `[6]`, or #MFTL_NULL: the job-wide integrated load each ODD mirror
   * face's prescribed value absorbs is ADDED here, identical on every rank --
   * with it, the loads the samples receive plus these sum to the markers'
   * total. May be requested independently on each rank.
   */
  double *wall_reaction;
  /**
   * SPREAD: `[constrained samples of the location on this rank]`, in
   * registration order, or #MFTL_NULL: each one's whole load \f$b_c\f$ -- before
   * it was passed on to its free samples -- is ADDED here.
   */
  double *constraint_load;
} MFTL_SolverField;

/** @brief Whether spread loads stay integrated or become densities. */
typedef enum MFTL_SpreadForm
{
  MFTL_SPREAD_INTEGRATED = 0,  ///< \f$b = W^T F\f$: integrated loads (a force), added to the target.
  MFTL_SPREAD_DENSITY = 1  ///< \f$f = b / q_j\f$: divided by each sample's measure after accumulation.
} MFTL_SpreadForm;

/** @brief Direction of a transfer. */
typedef enum MFTL_TransferDirection
{
  MFTL_TRANSFER_FETCH = 0,  ///< Solver values to the band.
  MFTL_TRANSFER_SPREAD = 1  ///< Marker loads to the solver.
} MFTL_TransferDirection;

/**
 * @struct MFTL_TransferSpec
 * @brief One batch of fields moved in one direction.
 */
typedef struct MFTL_TransferSpec
{
  MFTL_TransferDirection direction;  ///< Fetch or spread.
  const MFTL_CouplingWindow *window;  ///< FETCH: the band.
  const MFTL_CouplingBinding *binding;  ///< SPREAD: the rows.
  int n_fields;  ///< Number of fields; identical on every rank.
  const MFTL_SolverField *fields;  ///< Sources (fetch) or targets (spread). Same locations and order on every rank.
  MFTL_SpreadForm form;  ///< SPREAD only.
} MFTL_TransferSpec;

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. The coupler (COLLECTIVE unless marked local)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Creates a coupler and its provider directory.
 *
 * Checks that the solver communicator has the ensemble's group, that locations,
 * lattices and parameters agree on every rank, and validates every registered
 * piece; the grid may be empty (an initial export can come later).
 *
 * @param ens    The ensemble; borrowed, must outlive the coupler.
 * @param grid   The solver grid; borrowed, must outlive the coupler.
 * @param params The policy, or #MFTL_NULL for the defaults; copied.
 * @param[out] out The coupler, or #MFTL_NULL unless OK.
 * @return The agreed status.
 * @collective
 */
int MFTLCouplerCreate(MFTL_Ensemble *ens, MFTL_SolverGrid *grid, const MFTL_CouplerParams *params, MFTL_Coupler **out);

/**
 * @brief Destroys a coupler. No transfer may be active; windows, bindings and
 *        fetches and transfer tokens must be released first. Refused on every
 *        rank (leaving the handle intact) while dependents or a transfer live.
 * @param c Address of the coupler; set to #MFTL_NULL.
 * @collective
 */
void MFTLCouplerDestroy(MFTL_Coupler **c);

/**
 * @brief Takes up the grid's current registration after the application
 *        changed it (pieces, coverage, regions, constraints). A rank with no
 *        change passes through.
 *
 * What survives is decided by effect. A change of what some rank OWNS, or of
 * its constraint map, stales every window, binding, fetch and transfer token;
 * the directory is rebuilt
 * only if some rank's ownership ENTRIES changed (re-registering the same
 * blocks or extents keeps it). A change of EXPORTS alone keeps the directory
 * and every window whose served samples are all still exported -- checked
 * lazily by each window's next collective use, so a selection that slides
 * without touching a window's band costs that window nothing. Constraints are
 * validated here (#MFTL_ConstraintDesc).
 *
 * @param c The coupler.
 * @return The agreed status; INVALID for an invalid registration (the grid
 *         then stays un-taken-up: every consumer is STALE until it is fixed).
 * @collective
 */
int MFTLCouplerUpdateGrid(MFTL_Coupler *c);

/**
 * @brief Asks the owners of every sample the markers' kernels can reach, within
 *        the window's skin, and certifies coverage.
 *
 * @param c    The coupler.
 * @param markers The markers the window is for.
 * @param spec The motion allowance.
 * @param[out] out The request -- also on #MFTL_COUPLING_NEED_COVERAGE, when it
 *             carries the regions to export; #MFTL_NULL otherwise.
 * @return The agreed status: OK, NEED_COVERAGE or INVALID.
 * @collective
 */
int MFTLCouplerRequestSupport(MFTL_Coupler *c, const MFTL_MarkerSpec *markers, const MFTL_WindowSpec *spec, MFTL_SupportRequest **out);

/**
 * @struct MFTL_RequestRegion
 * @brief One region a request could not get.
 */
typedef struct MFTL_RequestRegion
{
  int location;  ///< The location.
  int piece;  ///< Lattice: the block owning the missing indices; -1 otherwise.
  int requester;  ///< The rank that asked.
  int64_t index_lo[3];  ///< Lattice: first missing index per axis.
  int64_t index_hi[3];  ///< Lattice: one past the last.
  double box[6];  ///< Physical lo xyz, hi xyz.
  bool by_key;  ///< MFTLCouplerRequestSamples(): the region is the one sample @c key (a point sample not registered has no box).
  uint64_t key;  ///< The missing sample's key when @c by_key.
} MFTL_RequestRegion;

/**
 * @struct MFTL_RequestView
 * @brief This rank's share of a request's outcome.
 */
typedef struct MFTL_RequestView
{
  int n_missing;  ///< Regions this rank OWNS but has not exported (provider side): what to export.
  const MFTL_RequestRegion *missing;  ///< The regions.
  int n_unprovided;  ///< Regions this rank needed that NO rank owns (requester side): a registration gap.
  const MFTL_RequestRegion *unprovided;  ///< The regions.
  long n_requests_served;  ///< Sub-requests this rank answered.
  long n_requests_sent;  ///< Sub-requests this rank made.
} MFTL_RequestView;

/**
 * @brief The regions a request could not get, on this rank.
 * @param req  The request.
 * @param[out] view Pointers into @p req, valid until it is released.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 * @rank_local
 */
int MFTLCouplerGetRequests(const MFTL_SupportRequest *req, MFTL_RequestView *view);

/**
 * @struct MFTL_SampleList
 * @brief Samples of one location named by KEY, with the rank owning each:
 *        the columns of rows the solver builds itself (MFTLCouplerImportRows()).
 */
typedef struct MFTL_SampleList
{
  int location;  ///< The location; at most one list per location.
  int64_t n;  ///< Keys this rank needs; repeats are allowed.
  const uint64_t *keys;  ///< `[n]` sample keys (a lattice sample's key is its wrapped index `i + N0 (j + N1 k)`).
  const int *owner;  ///< `[n]` the rank owning each, in the solver communicator passed to MFTLSolverGridCreate().
} MFTL_SampleList;

/**
 * @brief Asks the named owners for exactly the named samples: the coverage of
 *        an explicit column set, however far its samples lie from the markers.
 *
 * Every key must be owned by the rank named for it -- one that does not own it
 * makes the request INVALID (a solver's ghost map that disagrees with its
 * registration is a bug to report, not a sample to look for). A key owned but
 * not exported is NEED_COVERAGE, with the sample named to its owner as a
 * missing region (a one-sample index box, or the point's position). The window
 * prepared from it holds those samples and nothing else, so it serves imported
 * rows only: a kernel bind on it is INVALID.
 *
 * @param c       The coupler.
 * @param n_lists Number of lists; the same locations in the same order on every rank.
 * @param lists   `[n_lists]` the samples, by location.
 * @param[out] out The request (also on NEED_COVERAGE); #MFTL_NULL otherwise.
 * @return The agreed status: OK, NEED_COVERAGE or INVALID.
 * @collective
 */
int MFTLCouplerRequestSamples(MFTL_Coupler *c, int n_lists, const MFTL_SampleList *lists, MFTL_SupportRequest **out);

/**
 * @brief Builds the window of a fully covered request: the band, its spatial
 *        index and the forward and reverse schedules. No messages beyond the
 *        status agreement.
 * @param c   The coupler.
 * @param req A request that returned OK.
 * @param[out] out The window, or #MFTL_NULL unless OK.
 * @return The agreed status.
 * @collective
 */
int MFTLCouplerPrepareWindow(MFTL_Coupler *c, const MFTL_SupportRequest *req, MFTL_CouplingWindow **out);

/**
 * @brief Builds the rows of the current markers inside a window.
 *
 * Every marker must lie within the window's skin of a requested position;
 * otherwise the status is STALE and the caller requests a new window. Every
 * rank must bind the same kind, location list and quadrature policy: the
 * agreement covers them, so ranks that differ are INVALID, never diverging.
 * A QUADRATURE bind also builds the vertex ghost schedule (one more exchange).
 *
 * @param c       The coupler.
 * @param markers The markers; their locations must be among the window's.
 * @param win     The window; must outlive the binding.
 * @param[out] out The binding, or #MFTL_NULL unless OK.
 * @return The agreed status.
 * @collective
 */
int MFTLCouplerBind(MFTL_Coupler *c, const MFTL_MarkerSpec *markers, const MFTL_CouplingWindow *win, MFTL_CouplingBinding **out);

/**
 * @struct MFTL_NativeRows
 * @brief Rows the solver built for one location -- its own basis evaluated at
 *        the markers (FE, DG, a constrained or nonlocal operator) -- keyed by
 *        sample and given in marker order.
 *
 * The coupler stores them as it stores its own rows, so interpolation, the
 * exact transpose, constraints, routing and every transfer check apply
 * unchanged. What the rows reproduce is DECLARED and CHECKED on the rows
 * received, with the samples' positions (minimum image about the marker):
 * LINEAR -- \f$\sum_j W_{ij} = 1\f$ and \f$\sum_j W_{ij}(X_j - x_i) = 0\f$, as a
 * nodal (Lagrange) basis does; CONSTANT -- the row sum; NONE -- nothing (a
 * modal DG basis, whose coefficients are not point values). At a directional
 * location a row's weights are 3-vectors \f$A_j\f$ and the checks are
 * \f$\sum_j A_j s_j^T = I\f$ and \f$\sum_j A_j (s_j \otimes (X_j - x))^T = 0\f$.
 * Under CONSTANT and LINEAR the amplification bound applies as to assembled rows.
 */
typedef struct MFTL_NativeRows
{
  int location;  ///< The location; must be one of the window's.
  const int64_t *row_ptr;  ///< `[n_markers + 1]`, ascending from 0.
  const uint64_t *keys;  ///< `[row_ptr[n_markers]]` sample keys; a key repeated within a row is summed.
  const double *weights;  ///< `[row_ptr[n_markers] * C]`, finite; C = 3 at a directional location, 1 otherwise.
  MFTL_Reproduction reproduces;  ///< What the rows are declared to reproduce; checked.
} MFTL_NativeRows;

/**
 * @brief A binding from rows the solver built (#MFTL_NativeRows), over a
 *        window that holds their columns -- one from MFTLCouplerRequestSamples()
 *        for an arbitrary column set, or a support window when the rows are
 *        local enough to lie in it.
 *
 * The markers (#MFTL_MarkerSpec: any kind but QUADRATURE, operator VALUE) fix
 * the marker order, the positions the reproduction is checked at and the
 * surface generation the binding goes stale with; their location list is
 * ignored -- the rows name theirs. A column not in the window is INVALID;
 * a row failing its declared reproduction or the amplification bound is
 * BAD_STENCIL (`first_failed_key` in the quality). There are no stage bindings
 * of imported rows: import the rows of each stage.
 *
 * @param c       The coupler.
 * @param markers The markers the rows belong to.
 * @param win     The window; must outlive the binding.
 * @param n_rows  Number of row sets, one per location, the same locations in the same order on every rank.
 * @param rows    `[n_rows]` the rows.
 * @param[out] out The binding, or #MFTL_NULL unless OK.
 * @return The agreed status.
 * @collective
 */
int MFTLCouplerImportRows(MFTL_Coupler *c, const MFTL_MarkerSpec *markers, const MFTL_CouplingWindow *win, int n_rows, const MFTL_NativeRows *rows,
                          MFTL_CouplingBinding **out);

/**
 * @brief Rows of the same markers at stage positions, within the base binding's
 *        window: no request, no fetch, one status agreement.
 * @param c     The coupler.
 * @param base  The binding whose markers move.
 * @param stage Their stage positions (and optionally weights).
 * @param[out] out The stage binding, or #MFTL_NULL unless OK.
 * @return The agreed status; STALE when a stage position leaves the window.
 * @collective
 */
int MFTLCouplerBindStage(MFTL_Coupler *c, const MFTL_CouplingBinding *base, const MFTL_StageGeometry *stage, MFTL_CouplingBinding **out);

/**
 * @brief The rows of a stage, with NO communication: BindStage() without its
 *        status agreement, for a caller that already makes one collective
 *        decision per stage or step.
 *
 * Same rows as MFTLCouplerBindStage() to the bit when it succeeds. The status
 * is THIS rank's alone: a caller must combine it over the ranks -- in its own
 * step decision, or with MFTLCouplerAgreeStatus() -- before any communicating
 * call that depends on it, and before committing anything computed from it. A
 * failure here never starts a request, a rebuild or a fetch behind the other
 * ranks' backs. May run while a fetch is in flight (it reads only the window).
 *
 * @param c     The coupler.
 * @param base  The binding whose markers move.
 * @param stage Their stage positions (and optionally weights).
 * @param[out] out The stage binding, or #MFTL_NULL unless OK.
 * @return This rank's status: OK, STALE (a position left the window or the
 *         surface changed), BAD_STENCIL or INVALID.
 * @rank_local
 */
int MFTLCouplerBindStageLocal(MFTL_Coupler *c, const MFTL_CouplingBinding *base, const MFTL_StageGeometry *stage, MFTL_CouplingBinding **out);

/**
 * @brief Agrees a status over the coupler's ranks: the most severe one.
 *
 * The one-call completion of a local operation (MFTLCouplerBindStageLocal(),
 * MFTLCouplerInterpolateFetched()) for a caller with no step decision of its
 * own to fold it into.
 *
 * @param c     The coupler.
 * @param local This rank's #MFTL_CouplingStatus.
 * @return The maximum over the ranks.
 * @collective
 */
int MFTLCouplerAgreeStatus(MFTL_Coupler *c, int local);

/**
 * @brief Agrees a batch: same direction, same locations and flags in the same
 *        order on every rank, current handles. The token retains its window
 *        and binding until released. Begin checks its identity and currency
 *        collectively again before posting any messages.
 * @param c    The coupler.
 * @param spec The batch; the field pointers are recorded and must stay valid until End.
 * @param[out] out The token, or #MFTL_NULL unless OK.
 * @return The agreed status.
 * @collective
 */
int MFTLCouplerPrepareTransfer(MFTL_Coupler *c, const MFTL_TransferSpec *spec, MFTL_TransferReady **out);

/**
 * @brief Starts a fetch: packs owned values for the peers, posts the exchange,
 *        and snapshots the values read in place.
 * @param c     The coupler.
 * @param ready A FETCH token from MFTLCouplerPrepareTransfer().
 * @return The agreed status: OK, STALE after a grid edit/update, or INVALID on
 *         misuse or mismatched tokens. A failed Begin posts no messages.
 * @collective
 */
int MFTLCouplerFetchBegin(MFTL_Coupler *c, const MFTL_TransferReady *ready);

/**
 * @brief Completes a fetch.
 * @param c The coupler.
 * @param[out] out The fetched values, immutable; release with MFTLCouplerReleaseFetch().
 * @return #MFTL_COUPLING_OK, or #MFTL_COUPLING_INVALID when no fetch is active.
 * @collective
 */
int MFTLCouplerFetchEnd(MFTL_Coupler *c, MFTL_CouplingFetch **out);

/**
 * @brief \f$U = W u\f$: applies a binding's rows to fetched values. No MPI;
 *        allocation-free in Release (Debug checks the surface positions).
 *
 * A field contributes MFTLCouplerBindingComponents() consecutive values per
 * marker: one for a scalar value, three for a directional location's vector or
 * for a gradient. With every field scalar the layout is `[n_markers * n_fields]`.
 *
 * @param b The binding.
 * @param f A fetch from the binding's window.
 * @param[out] values `[n_markers * C]`, marker-major, C the sum of the fields' components, in the fetch's field order.
 * @return #MFTL_COUPLING_OK; #MFTL_COUPLING_STALE if the fetch is of another
 *         window, its grid changed, or the binding's surface changed; #MFTL_COUPLING_INVALID on a
 *         NULL argument. LOCAL: a caller that can fail here on one rank must
 *         agree the outcome before its next collective.
 * @rank_local
 */
int MFTLCouplerInterpolateFetched(const MFTL_CouplingBinding *b, const MFTL_CouplingFetch *f, double *values);

/**
 * @brief Starts a spread: \f$b = W^T F\f$ into private scratch, then posts the
 *        reverse exchange. The targets are untouched until End.
 * @param c     The coupler.
 * @param ready A SPREAD token.
 * @param loads `[n_markers * C]` integrated marker loads, marker-major, C the
 *              sum of the fields' components (MFTLCouplerBindingComponents());
 *              every one finite (a NaN load would poison its owner's sample).
 * @return The agreed status: OK, STALE after a grid/surface change, or INVALID
 *         on misuse, a non-finite load or mismatched tokens. A failed Begin
 *         posts no messages.
 * @collective
 */
int MFTLCouplerSpreadBegin(MFTL_Coupler *c, const MFTL_TransferReady *ready, const double *loads);

/**
 * @brief Completes a spread: every owner adds the contributions of all ranks,
 *        in ascending source-rank order, to its targets; passes constrained
 *        samples' loads on to their free samples (#MFTL_ConstraintDesc); with
 *        `halo_transpose` writes the halo loads into this rank's ghost
 *        elements; and reduces the mirror walls' reactions.
 * @param c The coupler.
 * @return #MFTL_COUPLING_OK, or #MFTL_COUPLING_INVALID when no spread is active.
 * @collective
 */
int MFTLCouplerSpreadEnd(MFTL_Coupler *c);

/**
 * @brief Checked blocking fetch: PrepareTransfer + Begin + End.
 * @param c The coupler.
 * @param win The window.
 * @param n_fields Number of fields.
 * @param fields The source fields.
 * @param[out] out The fetch, or #MFTL_NULL unless OK.
 * @return The agreed status.
 * @collective
 */
int MFTLCouplerFetch(MFTL_Coupler *c, const MFTL_CouplingWindow *win, int n_fields, const MFTL_SolverField *fields, MFTL_CouplingFetch **out);

/**
 * @brief Checked blocking interpolation: fetch the binding's window, apply, release.
 * @param c The coupler.
 * @param b The binding.
 * @param n_fields Number of fields.
 * @param fields The source fields.
 * @param[out] values `[n_markers * C]`, marker-major (see MFTLCouplerInterpolateFetched()).
 * @return The agreed status.
 * @collective
 */
int MFTLCouplerInterpolate(MFTL_Coupler *c, const MFTL_CouplingBinding *b, int n_fields, const MFTL_SolverField *fields, double *values);

/**
 * @brief Checked blocking spread: PrepareTransfer + Begin + End.
 * @param c The coupler.
 * @param b The binding.
 * @param n_fields Number of fields.
 * @param targets The target fields; contributions are ADDED.
 * @param loads `[n_markers * C]` integrated loads, marker-major (see MFTLCouplerSpreadBegin()).
 * @param form Integrated or density.
 * @return The agreed status.
 * @collective
 */
int MFTLCouplerSpread(MFTL_Coupler *c, const MFTL_CouplingBinding *b, int n_fields, const MFTL_SolverField *targets, const double *loads, MFTL_SpreadForm form);

/* ── Releases: LOCAL, no MPI ─────────────────────────────────────────────── */

/**
 * @brief Releases a request.
 * @param req Address of the request; set to #MFTL_NULL.
 * @rank_local
 */
void MFTLCouplerReleaseRequest(MFTL_SupportRequest **req);

/**
 * @brief Releases a window. Refused while a binding, fetch or transfer token of it is alive.
 * @param win Address of the window; set to #MFTL_NULL on success.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if dependents are alive (nothing freed).
 * @rank_local
 */
int MFTLCouplerReleaseWindow(MFTL_CouplingWindow **win);

/**
 * @brief Releases a binding. Refused while a stage binding or transfer token of it is alive.
 * @param b Address of the binding; set to #MFTL_NULL on success.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if dependents are alive.
 * @rank_local
 */
int MFTLCouplerReleaseBinding(MFTL_CouplingBinding **b);

/**
 * @brief Releases a fetch.
 * @param f Address of the fetch; set to #MFTL_NULL.
 * @rank_local
 */
void MFTLCouplerReleaseFetch(MFTL_CouplingFetch **f);

/**
 * @brief Releases a transfer token. Refused while its Begin/End transfer is
 *        active; the pointer is left intact so it can be released after End.
 * @param r Address of the token; set to #MFTL_NULL.
 * @rank_local
 */
void MFTLCouplerReleaseReady(MFTL_TransferReady **r);

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. Inspection
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════
 * 6b. Surface quadrature (QUADRATURE bindings)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Number of OWNED vertices of this rank's bodies: the length of a vertex
 *        field of MFTLCouplerQuadratureToVertices() and
 *        MFTLCouplerQuadratureFromVertices(), in the order of a VERTICES
 *        binding's markers (bodies by ascending id, then pool order).
 * @param b A QUADRATURE binding.
 * @return The count; 0 for anything else.
 * @rank_local
 */
int64_t MFTLCouplerQuadratureVertexCount(const MFTL_CouplingBinding *b);

/**
 * @brief Projects point values onto the vertices with the lumped surface mass:
 *        \f$U_v = \sum_q w_q \lambda_{qv} U_q \,/\, \sum_q w_q \lambda_{qv}\f$
 *        (\f$\lambda\f$ the point's barycentric coordinates, \f$w\f$ its
 *        weight), i.e. \f$M_S^{-1} B^T Q\f$ applied to interpolated velocities.
 *
 * Every rank's points contribute to the vertices of their triangles wherever
 * those are owned; one reverse exchange along the binding's ghost schedule
 * completes the sums. The denominator, a vertex's lumped mass, is the dual area
 * \f$\sum_{T \ni v} A_T/3\f$ to round-off. Constants are reproduced exactly.
 *
 * @param c     The coupler.
 * @param b     A QUADRATURE binding (a stage binding projects with its own weights).
 * @param ncomp Components per value.
 * @param Uq    `[n_markers * ncomp]` point values, marker-major.
 * @param[out] Uv `[vertex count * ncomp]` owned-vertex values, vertex-major.
 * @return The agreed status: OK, STALE if the surface changed, INVALID on misuse.
 * @collective
 */
int MFTLCouplerQuadratureToVertices(MFTL_Coupler *c, const MFTL_CouplingBinding *b, int ncomp, const double *Uq, double *Uv);

/**
 * @brief The point loads of vertex loads: \f$F_q = w_q \sum_v \lambda_{qv} g_v\f$
 *        with \f$g = M_S^{-1} F_v\f$ for INTEGRATED vertex loads (a force per
 *        vertex) or \f$g = F_v\f$ for a DENSITY (a force per area). The exact
 *        transpose of MFTLCouplerQuadratureToVertices(): for integrated loads
 *        \f$U_v \cdot F_v = U_q \cdot F_q\f$, and \f$\sum_q F_q = \sum_v F_v\f$ --
 *        so spreading \f$F_q\f$ conserves the vertex force and its power.
 *
 * @param c     The coupler.
 * @param b     A QUADRATURE binding.
 * @param ncomp Components per value.
 * @param Fv    `[vertex count * ncomp]` owned-vertex loads or densities.
 * @param form  #MFTL_SPREAD_INTEGRATED or #MFTL_SPREAD_DENSITY: what @p Fv is.
 * @param[out] Fq `[n_markers * ncomp]` integrated point loads, ready to spread.
 * @return The agreed status: OK, STALE, or INVALID.
 * @collective
 */
int MFTLCouplerQuadratureFromVertices(MFTL_Coupler *c, const MFTL_CouplingBinding *b, int ncomp, const double *Fv, MFTL_SpreadForm form, double *Fq);

/* ═══════════════════════════════════════════════════════════════════════════
 * 6c. Nearest-surface queries (for the indicator and the signed distance, plan 07)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_SurfaceCandidate
 * @brief The nearest point of the surface a solver sample sees, with what it
 *        lies on -- one record, merged as a whole.
 *
 * Merging keeps the LEAST candidate under a total order, so the result of
 * merging a set does not depend on the order of the merges, nor on how the set
 * was split among ranks (MFTLSurfaceCandidateCompare()). A scalar
 * \f$\min|\phi|\f$ merge is not enough: two triangles at the same distance
 * carry different bodies, features and signs, and the merged record must name
 * ONE of them deterministically.
 */
typedef struct MFTL_SurfaceCandidate
{
  double distance;  ///< Signed distance: negative inside (opposite the outward normals); its magnitude is the distance to the closest point.
  double offset[3];  ///< Sample minus closest point, a displacement independent of the periodic frame: the closest point is `sample - offset` in any image.
  int64_t triangle;  ///< Global id of the triangle the closest point lies on; -1 when not valid.
  int body;  ///< Its body; -1 when not valid.
  int feature;  ///< The closest point's feature on that triangle (#MFTL_TriangleFeature).
  int valid;  ///< 1: a surface lies within the query width; 0: unknown -- never "far".
} MFTL_SurfaceCandidate;

/**
 * @brief Sets a candidate to UNKNOWN, the merge identity: valid 0, body and
 *        triangle -1, distance #MFTL_DBL_MAX, everything else zero.
 * @param c The candidate.
 * @rank_local
 */
void MFTLSurfaceCandidateReset(MFTL_SurfaceCandidate *c);

/**
 * @brief The total order merges keep the least of: valid before unknown, then
 *        \f$|d|\f$, then the sign (inside first, so -0 before +0), then body,
 *        triangle and feature, then the offset (by value, -0 before +0).
 *
 * A candidate with a non-finite distance or offset is unknown. Two valid
 * candidates compare equal exactly when every field is equal, so the least of
 * a set is one record, the same however the set is merged.
 *
 * @param a A candidate.
 * @param b A candidate.
 * @return Negative, zero or positive as @p a orders before, with, or after @p b.
 * @rank_local
 */
int MFTLSurfaceCandidateCompare(const MFTL_SurfaceCandidate *a, const MFTL_SurfaceCandidate *b);

/**
 * @brief `*acc = min(*acc, *c)` under MFTLSurfaceCandidateCompare(); an unknown
 *        result is the canonical one of MFTLSurfaceCandidateReset().
 * @param acc The running candidate.
 * @param c   A new one.
 * @rank_local
 */
void MFTLSurfaceCandidateMerge(MFTL_SurfaceCandidate *acc, const MFTL_SurfaceCandidate *c);

/**
 * @struct MFTL_QueryField
 * @brief Where a query's results for one location go: one candidate per owned
 *        sample, in the solver's own element layout.
 */
typedef struct MFTL_QueryField
{
  int location;  ///< The location; one of the window's.
  MFTL_SurfaceCandidate *const *data;  ///< `[pieces of the location on this rank]` base pointers, registration order.
  ptrdiff_t stride;  ///< Candidates between consecutive elements; 0 means 1.
} MFTL_QueryField;

/**
 * @brief Requests the samples of @p locations within @p width of this rank's
 *        owned triangles: a support request for CENTROIDS markers whose skin
 *        is the width plus the largest triangle's half-extent (one MAX), so the
 *        window certifies every sample near every WHOLE triangle, not only near
 *        its centroid. Prepare the window with MFTLCouplerPrepareWindow().
 * @param c           The coupler.
 * @param n_locations Number of locations; same list on every rank.
 * @param locations   `[n_locations]` the locations.
 * @param width       Query width, a physical length, > 0; same on every rank.
 * @param[out] out    The request (also on NEED_COVERAGE), else #MFTL_NULL.
 * @return The agreed status, as MFTLCouplerRequestSupport().
 * @collective
 */
int MFTLCouplerRequestQuery(MFTL_Coupler *c, int n_locations, const int *locations, double width, MFTL_SupportRequest **out);

/**
 * @brief The nearest-surface candidate of every solver sample within @p width
 *        of this rank's owned triangles, MERGED into the solver's arrays
 *        (MFTLSurfaceCandidateMerge()) -- reset them first, or merge several
 *        queries (several ensembles) into one.
 *
 * Each (sample, triangle) candidate is computed once, by the triangle's owner,
 * from the same bits on any partition; the sign is the pseudo-normal's of the
 * closest feature (Baerentzen-Aanaes): the face normal, the sum of the two face
 * normals at an edge, the angle-weighted normal at a vertex, built here over
 * each vertex's whole fan in triangle-id order (ghost corners receive their
 * owner's) -- the caller's `verts.normal` are not read. Candidates travel to the
 * samples' owners and merge there: the merged record is the same bits at every
 * process count. A sample farther than the width from every triangle is left as
 * it was (unknown, unless something else wrote it): a query band is not a
 * far-field sign.
 *
 * Every owned triangle's box, widened by @p width, must lie in what the window
 * certifies -- true for a window from MFTLCouplerRequestQuery() at this width
 * and a surface not moved beyond its skin since; otherwise STALE.
 *
 * @param c        The coupler.
 * @param win      The window.
 * @param width    Query width, > 0; same on every rank.
 * @param n_fields Number of result arrays (distinct locations), >= 1; same on every rank.
 * @param fields   `[n_fields]` the result arrays.
 * @return The agreed status: OK, STALE or INVALID.
 * @collective
 */
int MFTLCouplerQueryNearest(MFTL_Coupler *c, const MFTL_CouplingWindow *win, double width, int n_fields, const MFTL_QueryField *fields);

/**
 * @brief Number of markers in a binding on this rank.
 * @param b The binding.
 * @return The count; 0 for #MFTL_NULL.
 * @rank_local
 */
int64_t MFTLCouplerBindingMarkerCount(const MFTL_CouplingBinding *b);

/**
 * @brief The markers of a binding, in row order.
 * @param b The binding.
 * @return `[n_markers]` references, valid while the binding is.
 * @rank_local
 */
const MFTL_MarkerRef *MFTLCouplerBindingMarkers(const MFTL_CouplingBinding *b);

/**
 * @brief The positions the rows were built at, unwrapped about the base markers.
 * @param b The binding.
 * @return `[3 n_markers]`, valid while the binding is.
 * @rank_local
 */
const double *MFTLCouplerBindingPositions(const MFTL_CouplingBinding *b);

/**
 * @struct MFTL_BindingQuality
 * @brief What a binding's rows achieved, over this rank's rows.
 */
typedef struct MFTL_BindingQuality
{
  int64_t n_rows;  ///< Rows (markers times locations).
  int64_t n_entries;  ///< Nonzero weights.
  int64_t n_corrected;  ///< Rows whose moments needed a correction.
  int64_t n_lower_order;  ///< Rows that fell back to constant reproduction.
  int64_t n_grown;  ///< Rows whose support grew.
  int64_t n_truncated;  ///< Rows missing a lattice position with weight (a wall, the lattice's edge, a masked cell); for a point location, rows crossing a wall.
  int64_t n_mirrored;  ///< Rows completed by mirror images (#MFTL_BOUNDARY_MIRROR).
  int64_t n_failed;  ///< Rows that failed (only nonzero on a failed bind's diagnostics).
  double max_amplification;  ///< Largest \f$\sum_j |W_{ij}|\f$.
  double min_weight;  ///< Most negative weight.
  double max_condition;  ///< Largest condition number among corrected rows.
  double max_moment_residual;  ///< Largest residual of the requested moments.
  uint64_t first_failed_key;  ///< Key of the first failed marker, if any.
} MFTL_BindingQuality;

/**
 * @brief The quality summary of a binding on this rank.
 * @param b The binding.
 * @param[out] q The summary.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 * @rank_local
 */
int MFTLCouplerBindingQuality(const MFTL_CouplingBinding *b, MFTL_BindingQuality *q);

/**
 * @brief Values per marker a field at @p location gives through a binding: 1
 *        for a scalar value, 3 for a directional location's vector or for a
 *        gradient (x, y, z).
 * @param b        The binding.
 * @param location The location.
 * @return The count; 0 if the binding does not cover the location.
 * @rank_local
 */
int MFTLCouplerBindingComponents(const MFTL_CouplingBinding *b, int location);

/**
 * @brief One row of a binding, with its columns as sample KEYS.
 * @param b        The binding.
 * @param loc_pos  Position of the location in the binding's location list.
 * @param marker   Marker index.
 * @param[out] n   Number of entries.
 * @param[out] keys    `[n]` sample keys, pointer into internal scratch valid until the next call on this binding.
 * @param[out] weights `[n * C]` weights, entry-major, C = MFTLCouplerBindingComponents(); same lifetime.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an invalid argument.
 * @rank_local
 */
int MFTLCouplerBindingRow(const MFTL_CouplingBinding *b, int loc_pos, int64_t marker, int *n, const uint64_t **keys, const double **weights);

/**
 * @brief The wall coefficients of one row (#MFTL_BOUNDARY_MIRROR): the weight
 *        of each face's `wall_value` in one component of the interpolated
 *        value, x-lo, y-lo, z-lo, x-hi, y-hi, z-hi. All zero away from ODD faces.
 * @param b         The binding.
 * @param loc_pos   Position of the location in the binding's location list.
 * @param marker    Marker index.
 * @param component Component, `0 <= component < MFTLCouplerBindingComponents()`.
 * @param[out] coef `[6]` the coefficients.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an invalid argument.
 * @rank_local
 */
int MFTLCouplerBindingWallCoefficients(const MFTL_CouplingBinding *b, int loc_pos, int64_t marker, int component, double coef[6]);

/** @brief Which DOFs a band mask marks. */
typedef enum MFTL_BandMaskKind
{
  MFTL_BAND_REQUESTED = 0,  ///< Owned samples some rank's request box contains.
  MFTL_BAND_EXPORTED = 1,  ///< Owned samples currently exported.
  MFTL_BAND_USED = 2  ///< Owned samples in some rank's band (served or read in place).
} MFTL_BandMaskKind;

/**
 * @brief Marks this rank's samples of one piece by band membership.
 * @param win   The window.
 * @param location The location.
 * @param piece The piece within the location on this rank.
 * @param kind  What to mark.
 * @param[out] mask `[samples of the piece]`: the owned box x-fastest (blocks) or
 *             the registration order (point sets); 1 or 0.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an invalid argument.
 * @rank_local
 */
int MFTLCouplerBandMask(const MFTL_CouplingWindow *win, int location, int piece, MFTL_BandMaskKind kind, unsigned char *mask);

/**
 * @struct MFTL_SpacingResult
 * @brief Surface resolution against fluid resolution, over the whole ensemble.
 */
typedef struct MFTL_SpacingResult
{
  double max_ratio;  ///< Largest owned-triangle edge over the finest fluid resolution near it.
  int64_t n_offending;  ///< Triangles above `sampling_ratio`.
  int worst_body;  ///< Body id of the worst triangle; -1 if none.
  int worst_triangle_gid;  ///< Its global id; -1 if none.
} MFTL_SpacingResult;

/**
 * @brief Measures the surface spacing against the fluid resolution the
 *        directory knows near each triangle. Works before any successful bind.
 * @param c The coupler.
 * @param[out] out Global result, identical on every rank.
 * @return #MFTL_COUPLING_BAD_SAMPLING if a triangle exceeds the ratio under an
 *         ERROR policy, OK otherwise.
 * @collective
 */
int MFTLCouplerSurfaceSpacing(MFTL_Coupler *c, MFTL_SpacingResult *out);

/**
 * @brief The edge length the fluid asks of the surface, per owned vertex:
 *        \f$L_v = \tfrac34\,\theta\,\texttt{sampling\_ratio}\,h_v\f$, with
 *        \f$h_v\f$ the finest fluid resolution the directory knows over the
 *        vertex's whole fan (its incident triangles' boxes).
 *
 * The 3/4 is the isotropic remesher's: it splits edges above 4/3 of its
 * target, so a CONVERGED remesh to \f$L\f$ keeps every edge within about
 * \f$\theta\f$ of the sampling bound -- \f$\theta < 1\f$ is the margin for the
 * motion and stretching until the next remesh, and absorbs the few per cent a
 * converged remesh leaves above 4/3 L. Measured from a sevenfold refinement in
 * one shot at \f$\theta = 0.9\f$: 16 serial iterations reach 0.87 h, 8 only 1.04 h
 * -- a remesh that has not converged is not sized. Until the remesher takes a
 * sizing field, the job-wide minimum is the conservative single target. Works
 * before any bind, like MFTLCouplerSurfaceSpacing().
 *
 * @param c      The coupler.
 * @param theta  Margin, in (0, 1].
 * @param[out] sizes `[owned vertices]` in VERTICES marker order, or #MFTL_NULL;
 *               #MFTL_DBL_MAX where no fluid resolution is known near the vertex.
 * @param[out] global_min The smallest over the job (#MFTL_DBL_MAX if none); identical on every rank.
 * @return The agreed status: OK, STALE (grid not taken up) or INVALID.
 * @collective
 */
int MFTLCouplerSurfaceSizing(MFTL_Coupler *c, double theta, double *sizes, double *global_min);

/**
 * @struct MFTL_CouplerStats
 * @brief Counters. Local by MFTLCouplerGetStats(), job-wide (sums, maxima) by
 *        MFTLCouplerReduceStats().
 */
typedef struct MFTL_CouplerStats
{
  long n_directory_builds;  ///< Directory constructions.
  long n_directory_kept;  ///< Grid updates that kept the directory (selection-only change).
  long n_requests;  ///< Support requests.
  long n_need_coverage;  ///< Requests that came back NEED_COVERAGE.
  long n_windows;  ///< Windows prepared.
  long n_binds;  ///< Binds.
  long n_stage_binds;  ///< Stage binds, checked and local.
  long n_local_stage_binds;  ///< Of those, local ones (MFTLCouplerBindStageLocal()), which made no agreement.
  long n_stale;  ///< Binds or transfers refused as stale.
  long n_fetches;  ///< Fetches.
  long n_spreads;  ///< Spreads.
  long band_self;  ///< Last window: samples read from this rank's own arrays.
  long band_halo;  ///< Last window: samples read from this rank's valid ghost layers.
  long band_remote;  ///< Last window: samples fetched from their owners.
  long band_images;  ///< Last window: sample images (periodic copies count separately).
  long served;  ///< Last window: samples this rank serves to others.
  long fetch_peers;  ///< Last window: ranks this rank fetches from.
  long serve_peers;  ///< Last window: ranks this rank serves.
  long bytes_discovery;  ///< Directory and request payload, sent plus received.
  long bytes_fetch;  ///< Fetch payload, sent plus received.
  long bytes_spread;  ///< Spread payload, sent plus received.
  long messages;  ///< Point-to-point messages sent.
  long bytes_memory;  ///< Live bytes held by the coupler's directory and windows.
} MFTL_CouplerStats;

/**
 * @brief This rank's counters.
 * @param c The coupler.
 * @param[out] out The counters.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 * @rank_local
 */
int MFTLCouplerGetStats(const MFTL_Coupler *c, MFTL_CouplerStats *out);

/**
 * @brief Job-wide counters: every field summed, except the band and peer
 *        fields, which are maxima over the ranks.
 * @param c The coupler.
 * @param[out] out The counters, identical on every rank.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument (on every rank).
 * @collective
 */
int MFTLCouplerReduceStats(MFTL_Coupler *c, MFTL_CouplerStats *out);

/**
 * @brief Zeroes this rank's counters (not the live memory figure).
 * @param c The coupler.
 * @rank_local
 */
void MFTLCouplerResetStats(MFTL_Coupler *c);

/**
 * @struct MFTL_RowQuality
 * @brief What one assembled row achieved.
 */
typedef struct MFTL_RowQuality
{
  int status;  ///< #MFTL_COUPLING_OK or #MFTL_COUPLING_BAD_STENCIL.
  int order;  ///< Achieved reproduction: 1 linear, 0 constant, -1 neither.
  bool corrected;  ///< The moments were corrected.
  bool lower_order;  ///< LINEAR was requested and CONSTANT delivered.
  int n_grow;  ///< Growth steps taken.
  int n_support;  ///< Samples with a nonzero base weight.
  double sum_abs;  ///< \f$\sum |W|\f$.
  double min_weight;  ///< Smallest weight.
  double max_weight;  ///< Largest weight.
  double condition;  ///< Condition number of the equilibrated moment matrix (1 when not solved).
  double moment_residual;  ///< Residual of the requested moments, relative.
  double metric[3];  ///< The kernel metric actually used (after growth).
} MFTL_RowQuality;

/**
 * @brief Assembles one row from given candidate samples: the numerical core of
 *        a bind, exposed for tests and for applications building their own
 *        candidate sets. LOCAL; no coverage check -- the caller guarantees the
 *        candidates include every sample within the grown support.
 *
 * @param params  The policy (kernel, reproduction, failure, tolerances).
 * @param x       Marker position.
 * @param H       Kernel metric per axis before `support_scale` and growth.
 * @param n       Number of candidates.
 * @param X       `[3n]` candidate positions (the image to use).
 * @param q       `[n]` candidate measures.
 * @param[out] w  `[n]` weights; 0 outside the support.
 * @param[out] quality What was achieved; may be #MFTL_NULL.
 * @return #MFTL_COUPLING_OK or #MFTL_COUPLING_BAD_STENCIL (then @p w is unspecified).
 * @rank_local
 */
int MFTLCouplingAssembleRow(const MFTL_CouplerParams *params, const double x[3], const double H[3], int n, const double *X, const double *q, double *w,
                            MFTL_RowQuality *quality);

/**
 * @brief MFTLCouplingAssembleRow() with a full cell matrix: offsets
 *        \f$\tilde H^{-1}(X - x)\f$ with \f$\tilde H\f$ = `support_scale` times the
 *        growth times @p Hc, base weights normalised by \f$|\det \tilde H|\f$.
 *        LOCAL; the caller guarantees the candidates include every sample in
 *        the grown support.
 *
 * @param params  The policy.
 * @param x       Marker position.
 * @param Hc      `[9]` cell matrix, row-major, columns the cell's edge vectors.
 * @param n       Number of candidates.
 * @param X       `[3n]` candidate positions.
 * @param q       `[n]` candidate measures.
 * @param[out] w  `[n]` weights.
 * @param[out] quality What was achieved; may be #MFTL_NULL.
 * @return #MFTL_COUPLING_OK or #MFTL_COUPLING_BAD_STENCIL.
 * @rank_local
 */
int MFTLCouplingAssembleRowCell(const MFTL_CouplerParams *params, const double x[3], const double Hc[9], int n, const double *X, const double *q, double *w,
                                MFTL_RowQuality *quality);

/**
 * @brief The block-row core of a bind for one operator, over given candidates:
 *        directional reconstruction and the two gradients as well as plain
 *        values. LOCAL; the caller guarantees the candidates include every
 *        sample in the support. No growth, no lower-order fallback.
 *
 * @param params  The policy (kernel, reproduction, tolerances).
 * @param op      The operator.
 * @param x       Marker position.
 * @param Hc      `[9]` kernel metric (cell matrix, row-major, columns the edge
 *                vectors) with `support_scale` applied by the caller; a
 *                diagonal one is an axis-aligned metric.
 * @param dHc     #MFTL_OPERATOR_VALUE_DERIVATIVE with a metric that varies with
 *                the marker position: `[27]` \f$\partial H/\partial x_m\f$, m-major
 *                (`dHc[9 m + 3 r + c]`); #MFTL_NULL for a constant metric.
 * @param n       Number of candidates.
 * @param X       `[3n]` candidate positions.
 * @param q       `[n]` candidate measures.
 * @param dir     `[3n]` directional candidates' vectors, or #MFTL_NULL for scalar samples.
 * @param[out] w  `[C n]` weights, candidate-major, C = 3 for a direction or a
 *                gradient, 1 otherwise.
 * @param[out] quality What was achieved; may be #MFTL_NULL. For a gradient,
 *                `sum_abs` and the residuals are in metric units
 *                (\f$\tilde H^T\nabla W\f$).
 * @return #MFTL_COUPLING_OK, #MFTL_COUPLING_BAD_STENCIL, or
 *         #MFTL_COUPLING_INVALID for a combination that does not exist (a
 *         gradient of directional samples, a directional NONE row, a derivative
 *         without a kernel derivative).
 * @rank_local
 */
int MFTLCouplingAssembleBlockRow(const MFTL_CouplerParams *params, MFTL_CouplingOperator op, const double x[3], const double Hc[9], const double *dHc, int n,
                                 const double *X, const double *q, const double *dir, double *w, MFTL_RowQuality *quality);

/**
 * @brief Test hook: caps every coupling message at @p bytes per MPI message, so
 *        chunking is exercised at small sizes. 0 restores the default.
 *        Must be identical on every rank.
 * @param bytes Chunk size in bytes; 0 for the default (1 GiB).
 * @rank_local
 */
void MFTLCouplerSetChunkLimit(size_t bytes);

#endif  // MFTL_COUPLING_H
