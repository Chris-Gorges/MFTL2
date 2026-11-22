/**
 * @file surface_tension.c
 * @author Christian Gorges
 * @date 2026-11-15
 * @ingroup mftl_geometry
 *
 * @brief The integrated surface-tension force on a front: Tryggvason's
 *        "Frenet" forms, per vertex and per triangle.
 *
 * Both forms evaluate the same physics, the force on a piece of surface as the
 * pull of the rest of the surface on its boundary (Tryggvason et al. 2011,
 * eqs. 7.52-7.53),
 *
 *     dF = sigma * contour integral of m dl,     m = t x n, the outward conormal,
 *
 * and differ only in the piece:
 *
 * - FRENET VERTEX (eqs. 7.54-7.59): the barycentric cell of a vertex. Inside a
 *   triangle the cell's boundary runs from an edge midpoint to the centroid to
 *   the next edge midpoint. With the triangle's normal on both segments their
 *   pulls add up to half the pull on the OPPOSITE edge: F_v += sigma_T / 2
 *   (x_k - x_j) x n_T for (v, j, k) counter-clockwise. That is
 *   -sigma_T grad_v A_T, so this vertex form is the exact gradient of
 *   E = sum_T sigma_T A_T. (Tryggvason's eq. 7.57 writes n x t where eq. 7.52
 *   has t x n; with outward normals the pull is the form given here, inward on
 *   a sphere.) With vertex normals interpolated along the segments instead,
 *   each segment's pull is evaluated once and given to the two cells it
 *   separates with opposite signs.
 * - FRENET ELEMENT (eq. 7.53 on a triangle): the triangle itself. The pull
 *   across each edge needs a normal ON the edge, from the two triangles that
 *   share it or from its two end vertices; with the triangle's own normal on
 *   all three edges the pulls cancel exactly and nothing is left.
 *
 * Both are exact on owned entities of a distributed mesh without
 * communication -- the vertex form because the halo is vertex-complete, the
 * element form because every owned triangle has its three edge neighbours --
 * given the ghost values of what they read (sigma, and vertex normals for the
 * element form's VERTICES edge normal).
 *
 * Conservation is built in rather than hoped for: per triangle the vertex
 * form's three contributions sum to zero, and an edge's two pulls in the
 * element form are computed from the SAME bits with opposite sign (the edge
 * vector as a direct minimum-image difference, the edge normal and sigma by a
 * commutative sum), so they cancel exactly.
 */

#include "mftl_geometry.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "geometry_internal.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_types.h"

MFTL_SurfaceTensionParameters MFTLSurfaceTensionDefaultParameters(void)
{
  MFTL_SurfaceTensionParameters p;
  memset(&p, 0, sizeof(p));
  p.form = MFTL_TENSION_FRENET_VERTEX;
  p.sigma = 1.0;
  p.sigma_field = -1;
  p.contour_normal = MFTL_CONTOUR_NORMAL_TRIANGLES;
  p.rim = MFTL_RIM_FORCE_FREE;
  return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Shared pieces
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Where sigma comes from: exactly one of the pointers, or neither for the uniform value. */
typedef struct
{
  const double *tri;  // [tris.n]  sigma_T
  const double *vtx;  // [verts.n] sigma_v, linear on the triangles
  double uniform;
} sigma_source;

/** @brief Resolves @p p's sigma for @p obj; MFTL_ERROR for a field that is not a one-component field of it. */
static int resolve_sigma(const MFTL_Object *obj, const MFTL_SurfaceTensionParameters *p, sigma_source *s)
{
  s->tri = MFTL_NULL;
  s->vtx = MFTL_NULL;
  s->uniform = p->sigma;
  if (p->sigma_field < 0) return MFTL_SUCCESS;
  if (obj->fields == MFTL_NULL || p->sigma_field >= obj->fields->n) return MFTL_ERROR;
  const MFTL_FieldDescriptor *const d = &obj->fields->entries[p->sigma_field];
  if (d->ncomp != 1) return MFTL_ERROR;
  if (d->entity == MFTL_FIELD_VERTEX)
    s->vtx = obj->field_data[p->sigma_field];
  else
    s->tri = obj->field_data[p->sigma_field];
  return MFTL_SUCCESS;
}

/** @brief sigma_T: the triangle's value, the mean of its corners', or the uniform one. */
static inline double sigma_triangle(const sigma_source *s, int t, int v0, int v1, int v2)
{
  if (s->tri != MFTL_NULL) return s->tri[t];
  if (s->vtx != MFTL_NULL) return (s->vtx[v0] + s->vtx[v1] + s->vtx[v2]) / 3.0;
  return s->uniform;
}

/**
 * @brief Bit k set when @p a and @p b both lie on the same wall of the
 *        non-periodic axis k (within #MFTL_BODY_WALL_TOL of it, relative to
 *        the domain length: the MFTLBodyRimWalls() rule).
 */
static int edge_walls(const MFTL_Domain *dom, const double *a, const double *b)
{
  if (dom == MFTL_NULL) return 0;
  int mask = 0;
  for (int k = 0; k < 3; ++k)
  {
    if (dom->is_periodic[k]) continue;
    const double tol = MFTL_BODY_WALL_TOL * dom->domain_size[k];
    const bool lo = fabs(a[k] - dom->global_min[k]) <= tol && fabs(b[k] - dom->global_min[k]) <= tol;
    const bool hi = fabs(a[k] - dom->global_max[k]) <= tol && fabs(b[k] - dom->global_max[k]) <= tol;
    if (lo || hi) mask |= 1 << k;
  }
  return mask;
}

/**
 * @brief \f$L\hat{\vect{m}}\f$, the length of a piece of contour times its unit
 *        outward conormal: \f$(L/|\vect{c}|)\,\vect{c}\f$ with \f$\vect{c} =
 *        \vect{e}\times\vect{s}\f$, \f$\vect{e}\f$ the piece (counter-clockwise
 *        around the cell) and \f$\vect{s}\f$ a normal, unnormalised.
 *
 * Normalising the cross product rather than @p s keeps the conormal a unit
 * vector also when @p s is not perpendicular to the piece (interpolated vertex
 * normals). A vanishing cross product -- @p s along the piece, a fold -- falls
 * back to the unit normal @p u, and to zero if that vanishes too. The same
 * bits in (e negated) give the same bits out, negated.
 *
 * The zero case leaves early instead of selecting `scale = 0`: a select puts
 * the comparison into the dependency chain of every pull, and the Frenet vertex
 * form with vertex normals measured 3 % slower with it, 5 % faster without
 * (docs/analysis.md §16.12). It writes `0 * c`, the bits the select gave.
 */
static void contour_pull(const double e[3], const double s[3], const double u[3], double out[3])
{
  double c[3];
  MFTLVectorCrossVector(c, e, s);
  double cm = MFTLVectorNorm(c);
  if (!(cm > 0.0))
  {
    MFTLVectorCrossVector(c, e, u);
    cm = MFTLVectorNorm(c);
    if (!(cm > 0.0))
    {
      for (int k = 0; k < 3; ++k) out[k] = 0.0 * c[k];
      return;
    }
  }
  const double scale = MFTLVectorNorm(e) / cm;
  out[0] = scale * c[0];
  out[1] = scale * c[1];
  out[2] = scale * c[2];
}

/**
 * @brief The rim edge of half-edge @p h on a wall: its pull \f$L\hat{\vect{m}}\f$
 *        (normal by the contour mode) and the walls it lies on; 0 walls when
 *        it lies on none.
 */
static int rim_pull(const MFTL_Object *obj, int h, MFTL_ContourNormal mode, double pull[3])
{
  const MFTL_Domain *const dom = obj->domain;
  const int a = MFTLHeOrigin(obj, h), b = MFTLHeTarget(obj, h);
  const double *const xa = MFTLVertexPos(obj, a), *const xb = MFTLVertexPos(obj, b);
  const int walls = edge_walls(dom, xa, xb);
  if (walls == 0) return 0;
  double u[3], e[3];
  MFTLGetTriangleNormal(obj, MFTLHeTriIdx(h), u);
  MFTLGeomMinimumImage(dom, e, xa, xb);
  if (mode == MFTL_CONTOUR_NORMAL_VERTICES)
  {
    const double *const na = &obj->verts.normal[3 * a], *const nb = &obj->verts.normal[3 * b];
    const double s[3] = {na[0] + nb[0], na[1] + nb[1], na[2] + nb[2]};
    contour_pull(e, s, u, pull);
  }
  else
  {
    /* The free edge's pull in the area gradient: (b - a) x n_T, of length L. */
    MFTLVectorCrossVector(pull, e, u);
  }
  return walls;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Frenet vertex
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Adds, for the vertex form, the wall-normal part of the pull on every
 *        rim edge that lies on a wall -- what #MFTL_RIM_FORCE_WALL says the
 *        wall supplies.
 *
 * The pull on a rim edge a -> b is shared by its two ends as the cells they
 * belong to share the edge: with the triangles' normals (the area gradient)
 * by the hat functions, (2 sigma_a + sigma_b)/6 and (sigma_a + 2 sigma_b)/6
 * per unit pull for a vertex field; with the contour of the barycentric cells
 * by halves, (3 sigma_a + sigma_b)/8 and (sigma_a + 3 sigma_b)/8. A triangle
 * or uniform sigma gives each end half. Either way a linear sigma on a flat
 * sheet leaves each vertex exactly grad(sigma) times its barycentric area.
 */
static void add_wall_reaction(const MFTL_Object *obj, const sigma_source *s, MFTL_ContourNormal mode, double *restrict F)
{
  const int nh = 3 * obj->tris.n;
  for (int h = 0; h < nh; ++h)
  {
    if (!(obj->tris.he[h].flags & MFTL_FLAG_BOUNDARY)) continue;
    double pull[3];
    const int walls = rim_pull(obj, h, mode, pull);
    if (walls == 0) continue;

    const int a = MFTLHeOrigin(obj, h), b = MFTLHeTarget(obj, h);
    double wa, wb;
    if (s->vtx != MFTL_NULL && mode == MFTL_CONTOUR_NORMAL_TRIANGLES)
    {
      wa = (2.0 * s->vtx[a] + s->vtx[b]) / 6.0;
      wb = (s->vtx[a] + 2.0 * s->vtx[b]) / 6.0;
    }
    else if (s->vtx != MFTL_NULL)
    {
      wa = (3.0 * s->vtx[a] + s->vtx[b]) / 8.0;
      wb = (s->vtx[a] + 3.0 * s->vtx[b]) / 8.0;
    }
    else
    {
      const int t = MFTLHeTriIdx(h);
      const int *const tv = &obj->tris.verts[3 * t];
      wa = wb = 0.5 * sigma_triangle(s, t, tv[0], tv[1], tv[2]);
    }
    for (int k = 0; k < 3; ++k)
    {
      if (!(walls & (1 << k))) continue;
      F[3 * a + k] += wa * pull[k];
      F[3 * b + k] += wb * pull[k];
    }
  }
}

/**
 * @brief The vertex form over every triangle the rank holds, then the rim.
 *
 * Corner 1 receives sigma/2 (x0 - x2) x n = -sigma/2 e2 x n, corner 2
 * sigma/2 (x1 - x0) x n = sigma/2 e1 x n, and corner 0 minus their sum, so a
 * triangle's three shares cancel exactly.
 *
 * At a rim edge (a -> b in T) the gradient contains the free edge's pull,
 * along (b - a) x n_T, on each end; #MFTL_RIM_FORCE_WALL adds its wall-normal
 * component back (add_wall_reaction()).
 */
static void frenet_vertex(const MFTL_Object *obj, const sigma_source *s, MFTL_RimForce rim, double *restrict F)
{
  memset(F, 0, 3 * (size_t) obj->verts.n * sizeof(double));

  // clang-format off
  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, obj->tris.n,
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;
    const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
    if (!(mag > 0.0)) continue;
    const double inv = 1.0 / mag;
    const double ux = nx * inv, uy = ny * inv, uz = nz * inv;
    const double hs = 0.5 * sigma_triangle(s, t, v0, v1, v2);

    /* f1 = -hs e2 x u, f2 = hs e1 x u, f0 = -(f1 + f2). */
    const double f1x = -hs * (e2y * uz - e2z * uy), f1y = -hs * (e2z * ux - e2x * uz), f1z = -hs * (e2x * uy - e2y * ux);
    const double f2x = hs * (e1y * uz - e1z * uy), f2y = hs * (e1z * ux - e1x * uz), f2z = hs * (e1x * uy - e1y * ux);
    F[3 * v0 + 0] -= f1x + f2x;
    F[3 * v0 + 1] -= f1y + f2y;
    F[3 * v0 + 2] -= f1z + f2z;
    F[3 * v1 + 0] += f1x;
    F[3 * v1 + 1] += f1y;
    F[3 * v1 + 2] += f1z;
    F[3 * v2 + 0] += f2x;
    F[3 * v2 + 1] += f2y;
    F[3 * v2 + 2] += f2z;);
  // clang-format on

  if (rim == MFTL_RIM_FORCE_FREE) return;

  if (rim == MFTL_RIM_FORCE_EXCLUDE)
  {
    for (int v = 0; v < obj->verts.n; ++v)
      if (obj->verts.rec[v].flags & MFTL_FLAG_BOUNDARY) F[3 * v + 0] = F[3 * v + 1] = F[3 * v + 2] = 0.0;
    return;
  }

  /* MFTL_RIM_FORCE_WALL: the wall supplies the wall-normal part of the pull. */
  add_wall_reaction(obj, s, MFTL_CONTOUR_NORMAL_TRIANGLES, F);
}

/**
 * @brief The vertex form with vertex normals along the contour.
 *
 * In triangle T the contour of the barycentric cells is three segments, from
 * each edge midpoint m_ab to the centroid c. Along one the normal is
 * interpolated linearly, so its mean is (n(m_ab) + n(c))/2, proportional to
 * 3(N_a + N_b) + 2(N_0 + N_1 + N_2), and the pull P = sigma L m on a's cell from
 * b's (the segment runs counter-clockwise around a) goes to a with + and to b
 * with -. sigma is sigma_T, or for a vertex field the mean of its values at the
 * segment's ends. Nothing pulls across a rim edge: that is the free edge.
 */
static void frenet_vertex_contour(const MFTL_Object *obj, const sigma_source *s, MFTL_RimForce rim, double *restrict F)
{
  memset(F, 0, 3 * (size_t) obj->verts.n * sizeof(double));
  const double *const vn = obj->verts.normal;

  // clang-format off
  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, obj->tris.n,
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;
    const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
    if (!(mag > 0.0)) continue;
    const double inv = 1.0 / mag;
    const double u[3] = {nx * inv, ny * inv, nz * inv};

    /* Corners relative to corner 0, so the arithmetic is seam- and scale-free. */
    const double x[3][3] = {{0.0, 0.0, 0.0}, {e1x, e1y, e1z}, {e2x, e2y, e2z}};
    const double c[3] = {(e1x + e2x) / 3.0, (e1y + e2y) / 3.0, (e1z + e2z) / 3.0};
    const int vv[3] = {v0, v1, v2};
    const double *const N[3] = {&vn[3 * v0], &vn[3 * v1], &vn[3 * v2]};
    const double ns[3] = {2.0 * (N[0][0] + N[1][0] + N[2][0]), 2.0 * (N[0][1] + N[1][1] + N[2][1]), 2.0 * (N[0][2] + N[1][2] + N[2][2])};
    const double sig_t = sigma_triangle(s, t, v0, v1, v2);

    for (int k = 0; k < 3; ++k)
    {
      const int a = k, b = (k + 1) % 3;
      const double seg[3] = {c[0] - 0.5 * (x[a][0] + x[b][0]), c[1] - 0.5 * (x[a][1] + x[b][1]), c[2] - 0.5 * (x[a][2] + x[b][2])};
      const double sn[3] = {3.0 * (N[a][0] + N[b][0]) + ns[0], 3.0 * (N[a][1] + N[b][1]) + ns[1], 3.0 * (N[a][2] + N[b][2]) + ns[2]};
      double P[3];
      contour_pull(seg, sn, u, P);
      const double sig = (s->vtx != MFTL_NULL) ? 0.5 * (0.5 * (s->vtx[vv[a]] + s->vtx[vv[b]]) + sig_t) : sig_t;
      F[3 * vv[a] + 0] += sig * P[0];
      F[3 * vv[a] + 1] += sig * P[1];
      F[3 * vv[a] + 2] += sig * P[2];
      F[3 * vv[b] + 0] -= sig * P[0];
      F[3 * vv[b] + 1] -= sig * P[1];
      F[3 * vv[b] + 2] -= sig * P[2];
    });
  // clang-format on

  if (rim == MFTL_RIM_FORCE_EXCLUDE)
  {
    for (int v = 0; v < obj->verts.n; ++v)
      if (obj->verts.rec[v].flags & MFTL_FLAG_BOUNDARY) F[3 * v + 0] = F[3 * v + 1] = F[3 * v + 2] = 0.0;
  }
  else if (rim == MFTL_RIM_FORCE_WALL)
    add_wall_reaction(obj, s, MFTL_CONTOUR_NORMAL_VERTICES, F);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Frenet element
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The element form: for every triangle, the pull across each of its
 *        edges, \f$\sigma_e L_e \hat{\vect{m}}_e\f$ (contour_pull()).
 *
 * The edge normal is the sum of the two triangles' unit normals, or of the two
 * end vertices' normals. The edge vector is a direct difference of the two end
 * positions and the normal a commutative sum, so the neighbour across the edge
 * computes the same pull with the opposite sign, to the bit. A ghost triangle
 * missing a neighbour uses its own normal on that edge; its row is not a force
 * anyway.
 */
static void frenet_element(const MFTL_Object *obj, const sigma_source *s, const MFTL_SurfaceTensionParameters *p, double *restrict F)
{
  const int n_t = obj->tris.n;
  const MFTL_Domain *const dom = obj->domain;
  double *const un = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(n_t, 1) * sizeof(double), "surface tension: unit normals");

  // clang-format off
  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, n_t,
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;
    const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
    const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
    un[3 * t + 0] = nx * inv;
    un[3 * t + 1] = ny * inv;
    un[3 * t + 2] = nz * inv;);
  // clang-format on

  const int *const tv = obj->tris.verts;
  const int *const nb = obj->tris.neighbors;
  const double *const vn = obj->verts.normal;
  const bool vertex_normals = p->contour_normal == MFTL_CONTOUR_NORMAL_VERTICES;

  for (int t = 0; t < n_t; ++t)
  {
    double *const f = &F[3 * t];
    f[0] = f[1] = f[2] = 0.0;
    const double *const ut = &un[3 * t];

    if (p->rim == MFTL_RIM_FORCE_EXCLUDE)
    {
      const MFTL_HalfEdge *const he = &obj->tris.he[3 * t];
      if ((he[0].flags | he[1].flags | he[2].flags) & MFTL_FLAG_BOUNDARY) continue;
    }

    for (int k = 0; k < 3; ++k)
    {
      const int a = tv[3 * t + k], b = tv[3 * t + (k + 1) % 3];
      const int other = nb[3 * t + k];

      double sig;
      if (s->vtx != MFTL_NULL)
        sig = 0.5 * (s->vtx[a] + s->vtx[b]);
      else if (s->tri != MFTL_NULL)
        sig = (other >= 0) ? 0.5 * (s->tri[t] + s->tri[other]) : s->tri[t];
      else
        sig = s->uniform;

      double pull[3];
      if (obj->tris.he[3 * t + k].flags & MFTL_FLAG_BOUNDARY)
      {
        /* A rim edge: nothing pulls across it, unless a wall does. */
        if (p->rim != MFTL_RIM_FORCE_WALL) continue;
        const int walls = rim_pull(obj, 3 * t + k, p->contour_normal, pull);
        for (int c = 0; c < 3; ++c)
          if (walls & (1 << c)) f[c] += sig * pull[c];
        continue;
      }

      double e[3], sn[3];
      MFTLGeomMinimumImage(dom, e, MFTLVertexPos(obj, a), MFTLVertexPos(obj, b));
      if (vertex_normals)
      {
        sn[0] = vn[3 * a + 0] + vn[3 * b + 0];
        sn[1] = vn[3 * a + 1] + vn[3 * b + 1];
        sn[2] = vn[3 * a + 2] + vn[3 * b + 2];
      }
      else if (other >= 0)
      {
        sn[0] = ut[0] + un[3 * other + 0];
        sn[1] = ut[1] + un[3 * other + 1];
        sn[2] = ut[2] + un[3 * other + 2];
      }
      else
      {
        sn[0] = ut[0];
        sn[1] = ut[1];
        sn[2] = ut[2];
      }
      contour_pull(e, sn, ut, pull);
      f[0] += sig * pull[0];
      f[1] += sig * pull[1];
      f[2] += sig * pull[2];
    }
  }

  free(un);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Entry point
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLComputeSurfaceTension(const MFTL_Object *obj, const MFTL_SurfaceTensionParameters *p, double *force)
{
  if (obj == MFTL_NULL || p == MFTL_NULL) return MFTL_ERROR;
  if (p->form != MFTL_TENSION_FRENET_VERTEX && p->form != MFTL_TENSION_FRENET_ELEMENT) return MFTL_ERROR;
  if (p->contour_normal != MFTL_CONTOUR_NORMAL_TRIANGLES && p->contour_normal != MFTL_CONTOUR_NORMAL_VERTICES) return MFTL_ERROR;
  if (p->rim != MFTL_RIM_FORCE_FREE && p->rim != MFTL_RIM_FORCE_WALL && p->rim != MFTL_RIM_FORCE_EXCLUDE) return MFTL_ERROR;

  sigma_source s;
  if (resolve_sigma(obj, p, &s) != MFTL_SUCCESS) return MFTL_ERROR;

  const int rows = (p->form == MFTL_TENSION_FRENET_VERTEX) ? obj->verts.n : obj->tris.n;
  if (rows == 0) return MFTL_SUCCESS;
  if (force == MFTL_NULL) return MFTL_ERROR;
  if (p->sigma_field >= 0 && s.tri == MFTL_NULL && s.vtx == MFTL_NULL) return MFTL_ERROR;
  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "MFTLComputeSurfaceTension: tris.verts is stale -- MFTLObjectBuildDerived first");

  if (p->form == MFTL_TENSION_FRENET_ELEMENT)
    frenet_element(obj, &s, p, force);
  else if (p->contour_normal == MFTL_CONTOUR_NORMAL_VERTICES)
    frenet_vertex_contour(obj, &s, p->rim, force);
  else
    frenet_vertex(obj, &s, p->rim, force);
  return MFTL_SUCCESS;
}
