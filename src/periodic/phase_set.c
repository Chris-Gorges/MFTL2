/*
 * MFTL2 phase sets — the §3.6 evaluation of plans/16 (revision 3.2, frozen by the independent
 * review of 2026-09-24, §19). Executable model: tools/phase_set_prototype/p36.py (+ p36axis.py for the
 * axis selection); the step labels R1–R10 below follow the model.
 *
 * Evaluation strategy: every query gathers the set's authoritative triangles to every rank
 * (member-gauge continuous coordinates from the image counters, relative to the set reference
 * point, §3.5) and runs the deterministic serial evaluation identically on all ranks. Crossing
 * records are therefore identical everywhere, as §3.6.4 requires. (The distributed per-line
 * gathering of §3.6.4 is a deferred performance option; the result is the same.)
 *
 * Members are the connected components of the member bodies. A member's k-coordinate is
 * single-valued in its own gauge iff it has no cut edge along k; that is checked directly on
 * the gathered data (after canonicalisation it is exactly the COMPACT_k flag of §3.3/§3.4).
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_images.h"
#include "mftl_phase_set.h"
#include "mftl_communication.h"

#define PS_PHI1 0.6180339887498949
#define PS_PHI2 0.41421356237309515
#define PS_MAXCLOS 2

struct MFTL_PhaseSet
{
  MFTL_Domain* dom;
  int n_body, cap_body;
  MFTL_Object** body;
  int* sigma;
  int n_clos;
  int clos_axis[8], clos_side[8];
  MFTL_PhaseSetTolerances tol;
};

/* ------------------------------------------------------------------------------------------
 * replicated geometry
 * ------------------------------------------------------------------------------------------ */
typedef struct
{
  int body, tgid, v[3];
  int pad;
  double Y[3][3];  // own-vertex continuous coordinates (member gauge, relative to the set reference)
} psRec;

typedef struct
{
  int body, tgid, v[3], member;
  double P[3][3];  // repaired corners: P0 = Y0, P_i = P0 + minimage(Y_i - Y0) (continuous triangle)
  double n[3];  // area vector 1/2 (P1-P0)x(P2-P0)
} psTri;

typedef struct
{
  int n;
  double* pts;  // 3 n (member gauge, continuous along the loop)
  int wind[3];
  int member;
  int wall;  // 0 min, 1 max (on the evaluation axis)
  double shoelace;  // signed lateral area (sigma-oriented)
  int sgn;  // orientation sign of the sigma-oriented loop in the wall plane
  int parent;
} psLoop;

typedef struct
{
  int wall;  // 0 / 1
  int b;  // loop index (-1: outer region)
  int* kids;
  int nkids;
  double area;
  int comp;
} psPiece;

typedef struct
{
  int nm;
  int* mem;  // members
  int npc;
  int* pc;  // pieces
  double shift;  // placement (along the evaluation axis)
  double vown;  // sigma-weighted own volume along k
  int e;
  int layer;
} psComp;

typedef struct
{
  // input
  const MFTL_Domain* dom;
  double L[3];  // periods (periodic axes) / box extents (walled axes)
  int per[3];
  double lo[3];  // domain min in the Y frame (pos_ref subtracted)
  MFTL_PhaseSetTolerances tol;
  int nb;
  int* sigma;
  // triangles
  int nt;
  psTri* t;
  // members
  int nmem;
  int* mem_body;
  int* mem_t0;  // first triangle index (triangles sorted by member, tgid)
  int* mem_nt;
  int* mem_cut;  // 3 per member: has a cut edge along axis a
  double* mem_area;  // 3 per member
  // loops / walls / pieces
  int nloop;
  psLoop* loop;
  int wall_axis;  // -1 none
  int wall_decl[2];
  double cw[2];  // wall coordinates in the Y frame
  int nW[2];
  int npiece;
  psPiece* piece;
  // components
  int ncomp;
  psComp* comp;
  int* mem_comp;
  // evaluation
  int k, i, j;  // axis and lateral pair (i, j, k right-handed)
  double tau_phi, tau_v, eta_z, Lk, Ak, Vcell;
  double pos_ref[3];
  char why[200];
  int lambda_E;
  int n_layers, n_compact;
  int* comp_parent;
  int* comp_parent_q;
  int* comp_depth;
  int tries_max;
} psEval;

static void psWhy(psEval* E, const char* fmt, const char* a)
{
  snprintf(E->why, sizeof(E->why), fmt, a ? a : "");
}

static inline void psMinImage(const psEval* E, double d[3])
{
  for (int a = 0; a < 3; ++a)
    if (E->per[a]) d[a] -= E->L[a] * round(d[a] / E->L[a]);
}

/* ------------------------------------------------------------------------------------------
 * API: creation
 * ------------------------------------------------------------------------------------------ */
MFTL_PhaseSet* MFTLPhaseSetCreate(MFTL_Domain* domain)
{
  MFTL_PhaseSet* S = (MFTL_PhaseSet*) calloc(1, sizeof(MFTL_PhaseSet));
  S->dom = domain;
  S->tol.eta_b = 1e-7;
  S->tol.eta_n = 1e-7;
  S->tol.tau_phi_rel = 1e-9;
  S->tol.tau_v_rel = 1e-10;
  S->tol.tau_d_rel = 1e-9;
  S->tol.n_try = 64;
  return S;
}

void MFTLPhaseSetDestroy(MFTL_PhaseSet** S)
{
  if (!S || !*S) return;
  free((*S)->body);
  free((*S)->sigma);
  free(*S);
  *S = MFTL_NULL;
}

int MFTLPhaseSetAddMember(MFTL_PhaseSet* S, MFTL_Object* body, int sigma)
{
  if (S->n_body == S->cap_body)
  {
    S->cap_body = S->cap_body ? 2 * S->cap_body : 4;
    S->body = realloc(S->body, (size_t) S->cap_body * sizeof(*S->body));
    S->sigma = realloc(S->sigma, (size_t) S->cap_body * sizeof(int));
  }
  S->body[S->n_body] = body;
  S->sigma[S->n_body] = sigma >= 0 ? 1 : -1;
  S->n_body++;
  return MFTL_SUCCESS;
}

int MFTLPhaseSetAddWallClosure(MFTL_PhaseSet* S, int axis, int side)
{
  if (S->n_clos >= 8) return MFTL_ERROR;
  S->clos_axis[S->n_clos] = axis;
  S->clos_side[S->n_clos] = side ? 1 : 0;
  S->n_clos++;
  return MFTL_SUCCESS;
}

void MFTLPhaseSetGetTolerances(const MFTL_PhaseSet* S, MFTL_PhaseSetTolerances* t) { *t = S->tol; }
void MFTLPhaseSetSetTolerances(MFTL_PhaseSet* S, const MFTL_PhaseSetTolerances* t) { S->tol = *t; }

/* ------------------------------------------------------------------------------------------
 * gather (collective)
 * ------------------------------------------------------------------------------------------ */
static int psCmpTri(const void* a, const void* b)
{
  const psTri* x = (const psTri*) a;
  const psTri* y = (const psTri*) b;
  if (x->body != y->body) return (x->body > y->body) - (x->body < y->body);
  return (x->tgid > y->tgid) - (x->tgid < y->tgid);
}

static int psFind(int* uf, int x)
{
  while (uf[x] != x)
  {
    uf[x] = uf[uf[x]];
    x = uf[x];
  }
  return x;
}

static int psCmpInt(const void* a, const void* b)
{
  int x = *(const int*) a, y = *(const int*) b;
  return (x > y) - (x < y);
}

static int psGather(MFTL_PhaseSet* S, psEval* E)
{
  const MFTL_Domain* dom = S->dom;
  memset(E, 0, sizeof(*E));
  E->dom = dom;
  E->tol = S->tol;
  E->nb = S->n_body;
  E->sigma = S->sigma;
  E->wall_axis = -1;
  E->eta_z = 0.1 * S->tol.tau_v_rel;  // η_z L_k A_k = 0.1 τ_V (§3.6.4)
  for (int a = 0; a < 3; ++a)
  {
    E->per[a] = dom->is_periodic[a] ? 1 : 0;
    E->L[a] = dom->domain_size[a];
  }
  // Member continuous coordinates and cut flags need canonical counters; an unknown
  // winding state is canonicalised here (collective), never taken as compact
  for (int b = 0; b < S->n_body; ++b) MFTLObjectEnsureWinding(S->body[b]);
  // set reference point: the reference vertex of the first member body (§3.5)
  double pos_ref[3] = {0, 0, 0};
  int img_ref[3] = {0, 0, 0};
  if (S->n_body > 0) MFTLObjectReference(S->body[0], pos_ref, img_ref);
  memcpy(E->pos_ref, pos_ref, sizeof(pos_ref));
  for (int a = 0; a < 3; ++a) E->lo[a] = dom->global_min[a] - pos_ref[a];
  // local records: Y = ((img - img_ref - m_b) L) + (pos - pos_ref), m_b = the body's reference
  // image offset (an integer lattice shift of the member gauge: keeps |Y| small, §19.4 E5)
  int nloc = 0, cap = 64;
  psRec* rec = malloc((size_t) cap * sizeof(psRec));
  for (int b = 0; b < S->n_body; ++b)
  {
    MFTL_Object* o = S->body[b];
    double pb[3];
    int ib[3];
    MFTLObjectReference(o, pb, ib);  // collective, identical on all ranks
    int m[3];
    for (int a = 0; a < 3; ++a) m[a] = ib[a] - img_ref[a];
    for (int ti = 0; ti < o->n_triangles; ++ti)
    {
      const MFTL_Triangle* tr = &o->triangles[ti];
      if (tr->is_ghost || (tr->flags & MFTL_FLAG_DEAD)) continue;
      if (nloc == cap)
      {
        cap *= 2;
        rec = realloc(rec, (size_t) cap * sizeof(psRec));
      }
      psRec* r = &rec[nloc++];
      memset(r, 0, sizeof(*r));
      r->body = b;
      r->tgid = tr->global_id;
      MFTL_Vertex* vv[3];
      MFTLGetTriangleVertices(o, tr, vv);
      for (int c = 0; c < 3; ++c)
      {
        r->v[c] = vv[c]->global_id;
        for (int a = 0; a < 3; ++a)
        {
          double y = vv[c]->pos[a] - pos_ref[a];
          if (dom->is_periodic[a]) y += (double) (vv[c]->img[a] - img_ref[a] - m[a]) * dom->domain_size[a];
          r->Y[c][a] = y;
        }
      }
    }
  }
  int np = dom->nproc;
  int nbytes = nloc * (int) sizeof(psRec);
  int* counts = malloc((size_t) np * sizeof(int));
  int* displs = malloc((size_t) np * sizeof(int));
  MPI_Allgather(&nbytes, 1, MPI_INT, counts, 1, MPI_INT, dom->comm_cart);
  int tot = 0;
  for (int p = 0; p < np; ++p)
  {
    displs[p] = tot;
    tot += counts[p];
  }
  psRec* all = malloc((size_t) (tot > 0 ? tot : 1));
  MPI_Allgatherv(rec, nbytes, MPI_BYTE, all, counts, displs, MPI_BYTE, dom->comm_cart);
  free(rec);
  free(counts);
  free(displs);
  int nall = tot / (int) sizeof(psRec);
  // triangles, dedup by (body, tgid)
  E->t = malloc((size_t) (nall > 0 ? nall : 1) * sizeof(psTri));
  for (int q = 0; q < nall; ++q)
  {
    psTri* t = &E->t[q];
    t->body = all[q].body;
    t->tgid = all[q].tgid;
    memcpy(t->v, all[q].v, sizeof(t->v));
    memcpy(t->P[0], all[q].Y[0], sizeof(t->P[0]));
    for (int c = 1; c < 3; ++c)
    {
      double d[3];
      for (int a = 0; a < 3; ++a) d[a] = all[q].Y[c][a] - all[q].Y[0][a];
      psMinImage(E, d);
      for (int a = 0; a < 3; ++a) t->P[c][a] = t->P[0][a] + d[a];
    }
    double u[3], w[3];
    for (int a = 0; a < 3; ++a)
    {
      u[a] = t->P[1][a] - t->P[0][a];
      w[a] = t->P[2][a] - t->P[0][a];
    }
    t->n[0] = 0.5 * (u[1] * w[2] - u[2] * w[1]);
    t->n[1] = 0.5 * (u[2] * w[0] - u[0] * w[2]);
    t->n[2] = 0.5 * (u[0] * w[1] - u[1] * w[0]);
  }
  qsort(E->t, (size_t) nall, sizeof(psTri), psCmpTri);
  int nt = 0;
  for (int q = 0; q < nall; ++q)
  {
    if (nt > 0 && E->t[nt - 1].body == E->t[q].body && E->t[nt - 1].tgid == E->t[q].tgid) continue;
    E->t[nt++] = E->t[q];
  }
  E->nt = nt;
  // per-triangle cut detection (own-vertex coordinates vs the continuous triangle)
  int* tcut = calloc((size_t) (3 * (nt > 0 ? nt : 1)), sizeof(int));
  {
    // re-scan the gathered records (same order after dedup: rebuild via bsearch)
    for (int q = 0; q < nall; ++q)
    {
      psTri key;
      key.body = all[q].body;
      key.tgid = all[q].tgid;
      psTri* f = bsearch(&key, E->t, (size_t) nt, sizeof(psTri), psCmpTri);
      if (!f) continue;
      const int ti = (int) (f - E->t);
      for (int c = 1; c < 3; ++c)
        for (int a = 0; a < 3; ++a)
          if (E->per[a] && fabs(f->P[c][a] - all[q].Y[c][a]) > 0.5 * E->L[a]) tcut[3 * ti + a] = 1;
    }
  }
  free(all);
  // members: connected components per body (union-find over vertex gids)
  int* uf = malloc((size_t) (nt > 0 ? nt : 1) * sizeof(int));
  for (int q = 0; q < nt; ++q) uf[q] = q;
  {
    // map (body, vertex gid) -> first triangle holding it
    int nkey = 3 * nt;
    int(*keys)[3] = malloc((size_t) (nkey > 0 ? nkey : 1) * sizeof(*keys));  // body, gid, triangle
    for (int q = 0; q < nt; ++q)
      for (int c = 0; c < 3; ++c)
      {
        keys[3 * q + c][0] = E->t[q].body;
        keys[3 * q + c][1] = E->t[q].v[c];
        keys[3 * q + c][2] = q;
      }
    // sort by (body, gid)
    for (int gap = nkey / 2; gap > 0; gap /= 2)  // shell sort (deterministic, no extra comparator)
      for (int x = gap; x < nkey; ++x)
      {
        int tmp[3] = {keys[x][0], keys[x][1], keys[x][2]};
        int y = x;
        while (y >= gap && (keys[y - gap][0] > tmp[0] || (keys[y - gap][0] == tmp[0] && keys[y - gap][1] > tmp[1])))
        {
          memcpy(keys[y], keys[y - gap], sizeof(keys[0]));
          y -= gap;
        }
        memcpy(keys[y], tmp, sizeof(tmp));
      }
    for (int x = 1; x < nkey; ++x)
      if (keys[x][0] == keys[x - 1][0] && keys[x][1] == keys[x - 1][1])
      {
        int ra = psFind(uf, keys[x][2]), rb = psFind(uf, keys[x - 1][2]);
        if (ra != rb) uf[ra < rb ? rb : ra] = ra < rb ? ra : rb;
      }
    free(keys);
  }
  // member ids in order of (body, smallest tgid): the root triangle index is the smallest one
  int* rootmem = malloc((size_t) (nt > 0 ? nt : 1) * sizeof(int));
  for (int q = 0; q < nt; ++q) rootmem[q] = -1;
  E->nmem = 0;
  for (int q = 0; q < nt; ++q)
  {
    int r = psFind(uf, q);
    if (rootmem[r] < 0) rootmem[r] = E->nmem++;
    E->t[q].member = rootmem[r];
  }
  free(uf);
  free(rootmem);
  // group triangles by member (stable by tgid)
  {
    psTri* sorted = malloc((size_t) (nt > 0 ? nt : 1) * sizeof(psTri));
    int* tc2 = malloc((size_t) (3 * (nt > 0 ? nt : 1)) * sizeof(int));
    E->mem_t0 = calloc((size_t) (E->nmem + 1), sizeof(int));
    E->mem_nt = calloc((size_t) (E->nmem + 1), sizeof(int));
    for (int q = 0; q < nt; ++q) E->mem_nt[E->t[q].member]++;
    for (int m = 1; m <= E->nmem; ++m) E->mem_t0[m] = E->mem_t0[m - 1] + E->mem_nt[m - 1];
    int* fill = calloc((size_t) (E->nmem + 1), sizeof(int));
    for (int q = 0; q < nt; ++q)
    {
      const int m = E->t[q].member;
      const int dst = E->mem_t0[m] + fill[m]++;
      sorted[dst] = E->t[q];
      memcpy(&tc2[3 * dst], &tcut[3 * q], 3 * sizeof(int));
    }
    free(fill);
    free(E->t);
    E->t = sorted;
    free(tcut);
    tcut = tc2;
  }
  E->mem_body = malloc((size_t) (E->nmem > 0 ? E->nmem : 1) * sizeof(int));
  E->mem_cut = calloc((size_t) (3 * (E->nmem > 0 ? E->nmem : 1)), sizeof(int));
  E->mem_area = calloc((size_t) (3 * (E->nmem > 0 ? E->nmem : 1)), sizeof(double));
  for (int m = 0; m < E->nmem; ++m)
  {
    E->mem_body[m] = E->t[E->mem_t0[m]].body;
    for (int q = E->mem_t0[m]; q < E->mem_t0[m] + E->mem_nt[m]; ++q)
      for (int a = 0; a < 3; ++a)
      {
        E->mem_area[3 * m + a] += E->t[q].n[a];
        if (tcut[3 * q + a]) E->mem_cut[3 * m + a] = 1;
      }
  }
  free(tcut);
  return MFTL_SUCCESS;
}

static void psFree(psEval* E)
{
  free(E->t);
  free(E->mem_body);
  free(E->mem_t0);
  free(E->mem_nt);
  free(E->mem_cut);
  free(E->mem_area);
  for (int l = 0; l < E->nloop; ++l) free(E->loop[l].pts);
  free(E->loop);
  for (int p = 0; p < E->npiece; ++p) free(E->piece[p].kids);
  free(E->piece);
  for (int c = 0; c < E->ncomp; ++c)
  {
    free(E->comp[c].mem);
    free(E->comp[c].pc);
  }
  free(E->comp);
  free(E->mem_comp);
  free(E->comp_parent);
  free(E->comp_parent_q);
  free(E->comp_depth);
}

/* ------------------------------------------------------------------------------------------
 * R1: boundary loops of the members (continuous along the loop, member gauge)
 * ------------------------------------------------------------------------------------------ */
typedef struct
{
  int a, b, tri, c;  // directed boundary edge a -> b = corner c -> c+1 of triangle tri
} psBEdge;

static int psCmpBE(const void* x, const void* y)
{
  const psBEdge* p = (const psBEdge*) x;
  const psBEdge* q = (const psBEdge*) y;
  return (p->a > q->a) - (p->a < q->a);
}

static int psCmpUE(const void* x, const void* y)
{
  const int* p = (const int*) x;
  const int* q = (const int*) y;
  if (p[0] != q[0]) return (p[0] > q[0]) - (p[0] < q[0]);
  if (p[1] != q[1]) return (p[1] > q[1]) - (p[1] < q[1]);
  return 0;
}

static int psLoops(psEval* E)
{
  int cap = 8;
  E->loop = malloc((size_t) cap * sizeof(psLoop));
  E->nloop = 0;
  for (int m = 0; m < E->nmem; ++m)
  {
    const int t0 = E->mem_t0[m], n = E->mem_nt[m];
    // undirected edge occurrence counts
    int(*ue)[4] = malloc((size_t) (3 * n > 0 ? 3 * n : 1) * sizeof(*ue));
    for (int q = 0; q < n; ++q)
      for (int c = 0; c < 3; ++c)
      {
        const psTri* t = &E->t[t0 + q];
        int a = t->v[c], b = t->v[(c + 1) % 3];
        ue[3 * q + c][0] = a < b ? a : b;
        ue[3 * q + c][1] = a < b ? b : a;
        ue[3 * q + c][2] = t0 + q;
        ue[3 * q + c][3] = c;
      }
    qsort(ue, (size_t) (3 * n), sizeof(*ue), psCmpUE);
    int nbe = 0, cbe = 16;
    psBEdge* be = malloc((size_t) cbe * sizeof(psBEdge));
    for (int x = 0; x < 3 * n;)
    {
      int y = x + 1;
      while (y < 3 * n && ue[y][0] == ue[x][0] && ue[y][1] == ue[x][1]) y++;
      if (y - x == 1)
      {
        if (nbe == cbe)
        {
          cbe *= 2;
          be = realloc(be, (size_t) cbe * sizeof(psBEdge));
        }
        const psTri* t = &E->t[ue[x][2]];
        const int c = ue[x][3];
        be[nbe].a = t->v[c];
        be[nbe].b = t->v[(c + 1) % 3];
        be[nbe].tri = ue[x][2];
        be[nbe].c = c;
        nbe++;
      }
      x = y;
    }
    free(ue);
    qsort(be, (size_t) nbe, sizeof(psBEdge), psCmpBE);
    char* used = calloc((size_t) (nbe > 0 ? nbe : 1), 1);
    for (int s = 0; s < nbe; ++s)
    {
      if (used[s]) continue;
      if (E->nloop == cap)
      {
        cap *= 2;
        E->loop = realloc(E->loop, (size_t) cap * sizeof(psLoop));
      }
      psLoop* L = &E->loop[E->nloop++];
      memset(L, 0, sizeof(*L));
      L->member = m;
      int lcap = 16;
      L->pts = malloc((size_t) (3 * lcap) * sizeof(double));
      double p[3], disp[3] = {0, 0, 0};
      memcpy(p, E->t[be[s].tri].P[be[s].c], sizeof(p));
      int cur = s, guard = 0;
      while (1)
      {
        used[cur] = 1;
        if (L->n == lcap)
        {
          lcap *= 2;
          L->pts = realloc(L->pts, (size_t) (3 * lcap) * sizeof(double));
        }
        memcpy(&L->pts[3 * L->n], p, sizeof(p));
        L->n++;
        const psTri* t = &E->t[be[cur].tri];
        const int c = be[cur].c;
        for (int a = 0; a < 3; ++a)
        {
          const double d = t->P[(c + 1) % 3][a] - t->P[c][a];
          p[a] += d;
          disp[a] += d;
        }
        if (be[cur].b == be[s].a) break;
        psBEdge key = {be[cur].b, 0, 0, 0};
        psBEdge* f = bsearch(&key, be, (size_t) nbe, sizeof(psBEdge), psCmpBE);
        if (!f || ++guard > nbe)
        {
          free(used);
          free(be);
          psWhy(E, "non-manifold boundary of a member%s", "");
          return MFTL_UNDEFINED;
        }
        cur = (int) (f - be);
      }
      for (int a = 0; a < 3; ++a) L->wind[a] = E->per[a] ? (int) lround(disp[a] / E->L[a]) : 0;
    }
    free(used);
    free(be);
  }
  return MFTL_DEFINED;
}

/* winding number of a closed lateral polygon around (qi, qj) in the (i, j) plane, summed over the
   lateral images (periodic lateral axes), polygon taken in minimum image w.r.t. q */
static int psWind(const psEval* E, const psLoop* L, int i, int j, double qi, double qj)
{
  double ci = 0, cj = 0;
  for (int p = 0; p < L->n; ++p)
  {
    ci += L->pts[3 * p + i];
    cj += L->pts[3 * p + j];
  }
  ci /= L->n;
  cj /= L->n;
  double q0i = qi, q0j = qj;
  if (E->per[i]) q0i += E->L[i] * round((ci - qi) / E->L[i]);
  if (E->per[j]) q0j += E->L[j] * round((cj - qj) / E->L[j]);
  double tot = 0.0;
  const int ai = E->per[i] ? 2 : 0, aj = E->per[j] ? 2 : 0;
  for (int a = -ai; a <= ai; ++a)
    for (int b = -aj; b <= aj; ++b)
    {
      const double X = q0i + a * E->L[i], Yv = q0j + b * E->L[j];
      double s = 0.0;
      for (int p = 0; p < L->n; ++p)
      {
        const int p1 = (p + 1) % L->n;
        const double a0 = atan2(L->pts[3 * p + j] - Yv, L->pts[3 * p + i] - X);
        const double a1 = atan2(L->pts[3 * p1 + j] - Yv, L->pts[3 * p1 + i] - X);
        double da = a1 - a0;
        while (da > M_PI) da -= 2 * M_PI;
        while (da < -M_PI) da += 2 * M_PI;
        s += da;
      }
      tot += s;
    }
  return (int) lround(tot / (2 * M_PI));
}

static double psLoopDist(const psEval* E, const psLoop* L, int i, int j, double qi, double qj)
{
  double best = 1e300;
  double ci = 0, cj = 0;
  for (int p = 0; p < L->n; ++p)
  {
    ci += L->pts[3 * p + i];
    cj += L->pts[3 * p + j];
  }
  ci /= L->n;
  cj /= L->n;
  double q0i = qi, q0j = qj;
  if (E->per[i]) q0i += E->L[i] * round((ci - qi) / E->L[i]);
  if (E->per[j]) q0j += E->L[j] * round((cj - qj) / E->L[j]);
  const int ai = E->per[i] ? 2 : 0, aj = E->per[j] ? 2 : 0;
  for (int a = -ai; a <= ai; ++a)
    for (int b = -aj; b <= aj; ++b)
    {
      const double X = q0i + a * E->L[i], Yv = q0j + b * E->L[j];
      for (int p = 0; p < L->n; ++p)
      {
        const int p1 = (p + 1) % L->n;
        const double x0 = L->pts[3 * p + i], y0 = L->pts[3 * p + j];
        const double dx = L->pts[3 * p1 + i] - x0, dy = L->pts[3 * p1 + j] - y0;
        const double dd = dx * dx + dy * dy;
        double t = dd > 0 ? ((X - x0) * dx + (Yv - y0) * dy) / dd : 0.0;
        t = t < 0 ? 0 : (t > 1 ? 1 : t);
        const double ex = x0 + t * dx - X, ey = y0 + t * dy - Yv;
        const double d = sqrt(ex * ex + ey * ey);
        if (d < best) best = d;
      }
    }
  return best;
}

static double psShoelace(const psLoop* L, int i, int j)
{
  double s = 0.0;
  for (int p = 0; p < L->n; ++p)
  {
    const int p1 = (p + 1) % L->n;
    s += L->pts[3 * p + i] * L->pts[3 * p1 + j] - L->pts[3 * p1 + i] * L->pts[3 * p + j];
  }
  return 0.5 * s;
}

/* ------------------------------------------------------------------------------------------
 * R1/R2: wall closures, flux selection, pieces, components
 * ------------------------------------------------------------------------------------------ */
static int psPieceContains(const psEval* E, const psPiece* P, double qi, double qj)
{
  const int i = (E->wall_axis + 1) % 3, j = (E->wall_axis + 2) % 3;
  if (P->b >= 0 && psWind(E, &E->loop[P->b], i, j, qi, qj) == 0) return 0;
  for (int c = 0; c < P->nkids; ++c)
    if (psWind(E, &E->loop[P->kids[c]], i, j, qi, qj) != 0) return 0;
  return 1;
}

static MFTL_PhaseStatus psClose(MFTL_PhaseSet* S, psEval* E)
{
  const MFTL_Domain* dom = E->dom;
  // declared closures: one axis only, non-periodic
  int w = -1;
  E->wall_decl[0] = E->wall_decl[1] = 0;
  for (int c = 0; c < S->n_clos; ++c)
  {
    if (w >= 0 && S->clos_axis[c] != w)
    {
      psWhy(E, "wall closures on two axes%s", "");
      return MFTL_UNSUPPORTED;
    }
    w = S->clos_axis[c];
    if (dom->is_periodic[w])
    {
      psWhy(E, "wall closure declared on a periodic axis (configuration error)%s", "");
      return MFTL_UNDEFINED;
    }
    E->wall_decl[S->clos_side[c]] = 1;
  }
  E->wall_axis = w;
  MFTL_PhaseStatus st = psLoops(E);
  if (st != MFTL_DEFINED) return st;
  E->nW[0] = E->nW[1] = -1;
  if (E->nloop > 0 && w < 0)
  {
    psWhy(E, "open member whose boundary is not on a declared wall closure%s", "");
    return MFTL_UNDEFINED;
  }
  if (w >= 0)
  {
    E->cw[0] = dom->global_min[w] - E->pos_ref[w];
    E->cw[1] = dom->global_max[w] - E->pos_ref[w];
    const double tau_loop = 1e-9 * (dom->global_max[w] - dom->global_min[w]);
    const int i = (w + 1) % 3, j = (w + 2) % 3;
    for (int l = 0; l < E->nloop; ++l)
    {
      psLoop* L = &E->loop[l];
      if (L->wind[0] || L->wind[1] || L->wind[2])
      {
        psWhy(E, "boundary loop winds along the wall plane%s", "");
        return MFTL_UNSUPPORTED;
      }
      int on[2] = {1, 1};
      for (int p = 0; p < L->n; ++p)
        for (int s = 0; s < 2; ++s)
          if (fabs(L->pts[3 * p + w] - E->cw[s]) > tau_loop) on[s] = 0;
      if (!on[0] && !on[1])
      {
        // a loop on no wall, or on two walls (points on both)
        int any[2] = {0, 0};
        for (int p = 0; p < L->n; ++p)
          for (int s = 0; s < 2; ++s)
            if (fabs(L->pts[3 * p + w] - E->cw[s]) <= tau_loop) any[s] = 1;
        if (any[0] && any[1])
        {
          psWhy(E, "boundary loop touches two walls%s", "");
          return MFTL_UNSUPPORTED;
        }
        psWhy(E, "open member whose boundary is not on a declared wall closure%s", "");
        return MFTL_UNDEFINED;
      }
      L->wall = on[0] ? 0 : 1;
      if (!E->wall_decl[L->wall])
      {
        psWhy(E, "open member whose boundary lies on an undeclared wall%s", "");
        return MFTL_UNDEFINED;
      }
      // Gamma = sigma * boundary: reverse for sigma = -1
      if (E->sigma[E->mem_body[L->member]] < 0)
        for (int p = 0; p < L->n / 2; ++p)
          for (int a = 0; a < 3; ++a)
          {
            double tmp = L->pts[3 * p + a];
            L->pts[3 * p + a] = L->pts[3 * (L->n - 1 - p) + a];
            L->pts[3 * (L->n - 1 - p) + a] = tmp;
          }
      L->shoelace = psShoelace(L, i, j);
      L->sgn = L->shoelace > 0 ? 1 : -1;
    }
    // nesting per wall
    for (int l = 0; l < E->nloop; ++l)
    {
      E->loop[l].parent = -1;
      int best_d = -1;
      for (int o = 0; o < E->nloop; ++o)
      {
        if (o == l || E->loop[o].wall != E->loop[l].wall) continue;
        if (psWind(E, &E->loop[o], i, j, E->loop[l].pts[i], E->loop[l].pts[j]) == 0) continue;
        // depth of o = number of loops containing o
        int d = 0;
        for (int z = 0; z < E->nloop; ++z)
          if (z != o && E->loop[z].wall == E->loop[o].wall && psWind(E, &E->loop[z], i, j, E->loop[o].pts[i], E->loop[o].pts[j]) != 0) d++;
        if (d > best_d)
        {
          best_d = d;
          E->loop[l].parent = o;
        }
      }
    }
  }
  // flux selection over the admissible multiplicities
  double F0[3] = {0, 0, 0};
  for (int m = 0; m < E->nmem; ++m)
    for (int a = 0; a < 3; ++a) F0[a] += E->sigma[E->mem_body[m]] * E->mem_area[3 * m + a];
  const double Aw = (w >= 0) ? E->L[(w + 1) % 3] * E->L[(w + 2) % 3] : 0.0;
  const double tau_phi = E->tol.tau_phi_rel * ((w >= 0) ? Aw : fmax(E->L[0] * E->L[1], fmax(E->L[1] * E->L[2], E->L[0] * E->L[2])));
  int cand[2][2] = {{0, 0}, {0, 0}};  // cand[s][n]
  for (int s = 0; s < 2; ++s)
  {
    if (w < 0 || !E->wall_decl[s]) continue;
    const int eps = s == 0 ? 1 : -1;
    for (int n = 0; n <= 1; ++n)
    {
      if (n == 1 && (!E->per[(w + 1) % 3] || !E->per[(w + 2) % 3])) continue;  // patch cannot reach other walls
      int ok = 1;
      // outer region mu = n; region of loop b: eps * sum(sgn of b and its ancestors) + n
      if (n < 0 || n > 1) ok = 0;
      for (int l = 0; l < E->nloop && ok; ++l)
      {
        if (E->loop[l].wall != s) continue;
        int sum = 0;
        for (int a = l; a >= 0; a = E->loop[a].parent) sum += E->loop[a].sgn;
        const int mu = eps * sum + n;
        if (mu != 0 && mu != 1) ok = 0;
      }
      cand[s][n] = ok;
    }
    if (!cand[s][0] && !cand[s][1])
    {
      psWhy(E, "wall loop winding numbers outside {0,1} (inconsistent loop orientation)%s", "");
      return MFTL_UNDEFINED;
    }
  }
  int nsol = 0, sol[2] = {-1, -1};
  for (int n0 = 0; n0 <= 1; ++n0)
    for (int n1 = 0; n1 <= 1; ++n1)
    {
      const int nn[2] = {n0, n1};
      int ok = 1;
      for (int s = 0; s < 2; ++s)
      {
        if (w < 0 || !E->wall_decl[s])
        {
          if (nn[s]) ok = 0;
          continue;
        }
        if (!cand[s][nn[s]]) ok = 0;
      }
      if (!ok) continue;
      double F[3] = {F0[0], F0[1], F0[2]};
      if (w >= 0)
        for (int s = 0; s < 2; ++s)
        {
          if (!E->wall_decl[s]) continue;
          const int eps = s == 0 ? 1 : -1, sW = s == 0 ? -1 : 1;
          double AG = 0.0;
          for (int l = 0; l < E->nloop; ++l)
            if (E->loop[l].wall == s) AG += E->loop[l].shoelace;
          F[w] += sW * (eps * AG + nn[s] * Aw);
        }
      if (fabs(F[0]) < tau_phi && fabs(F[1]) < tau_phi && fabs(F[2]) < tau_phi)
      {
        nsol++;
        sol[0] = n0;
        sol[1] = n1;
      }
    }
  if (nsol == 0)
  {
    psWhy(E, "Phi(S) != 0 for every admissible wall patch%s", "");
    return MFTL_UNDEFINED;
  }
  if (nsol > 1)
  {
    psWhy(E, "ambiguous wall closure (more than one patch choice closes S)%s", "");
    return MFTL_UNSUPPORTED;
  }
  E->nW[0] = (w >= 0 && E->wall_decl[0]) ? sol[0] : -1;
  E->nW[1] = (w >= 0 && E->wall_decl[1]) ? sol[1] : -1;
  // pieces: regions with mu = 1
  int pcap = 4;
  E->piece = malloc((size_t) pcap * sizeof(psPiece));
  E->npiece = 0;
  if (w >= 0)
  {
    const int i = (w + 1) % 3, j = (w + 2) % 3;
    for (int s = 0; s < 2; ++s)
    {
      if (!E->wall_decl[s]) continue;
      const int eps = s == 0 ? 1 : -1;
      for (int b = -1; b < E->nloop; ++b)
      {
        if (b >= 0 && E->loop[b].wall != s) continue;
        int sum = 0;
        for (int a = b; a >= 0; a = E->loop[a].parent) sum += E->loop[a].sgn;
        const int mu = eps * sum + E->nW[s];
        if (mu != 1) continue;
        if (E->npiece == pcap)
        {
          pcap *= 2;
          E->piece = realloc(E->piece, (size_t) pcap * sizeof(psPiece));
        }
        psPiece* P = &E->piece[E->npiece++];
        memset(P, 0, sizeof(*P));
        P->wall = s;
        P->b = b;
        P->kids = malloc((size_t) (E->nloop > 0 ? E->nloop : 1) * sizeof(int));
        double area = (b < 0) ? Aw : fabs(E->loop[b].shoelace);
        for (int l = 0; l < E->nloop; ++l)
          if (E->loop[l].wall == s && E->loop[l].parent == b)
          {
            P->kids[P->nkids++] = l;
            area -= fabs(E->loop[l].shoelace);
          }
        P->area = area;
        (void) i;
        (void) j;
      }
    }
  }
  // components: union-find over members glued through pieces
  int* uf = malloc((size_t) (E->nmem + E->npiece + 1) * sizeof(int));
  for (int x = 0; x < E->nmem + E->npiece; ++x) uf[x] = x;
  for (int p = 0; p < E->npiece; ++p)
  {
    const psPiece* P = &E->piece[p];
    const int node = E->nmem + p;
    if (P->b >= 0)
    {
      int ra = psFind(uf, node), rb = psFind(uf, E->loop[P->b].member);
      if (ra != rb) uf[ra > rb ? ra : rb] = ra < rb ? ra : rb;
    }
    for (int c = 0; c < P->nkids; ++c)
    {
      int ra = psFind(uf, node), rb = psFind(uf, E->loop[P->kids[c]].member);
      if (ra != rb) uf[ra > rb ? ra : rb] = ra < rb ? ra : rb;
    }
  }
  int* cid = malloc((size_t) (E->nmem + E->npiece + 1) * sizeof(int));
  for (int x = 0; x < E->nmem + E->npiece; ++x) cid[x] = -1;
  E->ncomp = 0;
  E->comp = calloc((size_t) (E->nmem + E->npiece + 1), sizeof(psComp));
  E->mem_comp = malloc((size_t) (E->nmem > 0 ? E->nmem : 1) * sizeof(int));
  for (int x = 0; x < E->nmem + E->npiece; ++x)
  {
    const int r = psFind(uf, x);
    if (cid[r] < 0) cid[r] = E->ncomp++;
    psComp* C = &E->comp[cid[r]];
    if (x < E->nmem)
    {
      C->mem = realloc(C->mem, (size_t) (C->nm + 1) * sizeof(int));
      C->mem[C->nm++] = x;
      E->mem_comp[x] = cid[r];
    }
    else
    {
      C->pc = realloc(C->pc, (size_t) (C->npc + 1) * sizeof(int));
      C->pc[C->npc++] = x - E->nmem;
      E->piece[x - E->nmem].comp = cid[r];
    }
  }
  free(uf);
  free(cid);
  return MFTL_DEFINED;
}

/* ------------------------------------------------------------------------------------------
 * per-axis integrals of a component (member gauge + shift along axis k)
 * ------------------------------------------------------------------------------------------ */
static void psCompIntegrals(const psEval* E, int c, int k, double shift, double* v, double* c2, double* phik)
{
  double sv = 0, s2 = 0, sf = 0;
  const psComp* C = &E->comp[c];
  for (int mi = 0; mi < C->nm; ++mi)
  {
    const int m = C->mem[mi];
    const double sg = E->sigma[E->mem_body[m]];
    for (int q = E->mem_t0[m]; q < E->mem_t0[m] + E->mem_nt[m]; ++q)
    {
      const psTri* t = &E->t[q];
      const double z0 = t->P[0][k] + shift, z1 = t->P[1][k] + shift, z2 = t->P[2][k] + shift;
      sf += sg * t->n[k];
      sv += sg * t->n[k] * (z0 + z1 + z2) / 3.0;
      s2 += sg * t->n[k] * (z0 * z0 + z1 * z1 + z2 * z2 + z0 * z1 + z1 * z2 + z0 * z2) / 6.0;
    }
  }
  if (k == E->wall_axis)
    for (int pi = 0; pi < C->npc; ++pi)
    {
      const psPiece* P = &E->piece[C->pc[pi]];
      const double sW = P->wall == 0 ? -1.0 : 1.0, z = E->cw[P->wall];
      sf += sW * P->area;
      sv += sW * z * P->area;
      s2 += sW * z * z * P->area;
    }
  if (v) *v = sv;
  if (c2) *c2 = s2;
  if (phik) *phik = sf;
}

/* ------------------------------------------------------------------------------------------
 * lines along axis k through lateral point (qi, qj)
 * ------------------------------------------------------------------------------------------ */
typedef struct
{
  double z;
  int s;  // -1 entering the phase along +k, +1 leaving
  int comp;
} psCross;

static int psCmpDbl(const void* a, const void* b)
{
  double x = *(const double*) a, y = *(const double*) b;
  return (x > y) - (x < y);
}

static int psLine(const psEval* E, int k, double qi, double qj, psCross** out, int* nout, int* generic)
{
  const int i = (k + 1) % 3, j = (k + 2) % 3;
  int cap = 32, n = 0, gen = 1;
  psCross* rec = malloc((size_t) cap * sizeof(psCross));
  const double eta_b = E->tol.eta_b, eta_n = E->tol.eta_n;
  const double det_eps = 1e-18 * E->L[i] * E->L[j];
  for (int c = 0; c < E->ncomp; ++c)
  {
    const psComp* C = &E->comp[c];
    for (int mi = 0; mi < C->nm; ++mi)
    {
      const int m = C->mem[mi];
      const int sg = E->sigma[E->mem_body[m]];
      for (int q = E->mem_t0[m]; q < E->mem_t0[m] + E->mem_nt[m]; ++q)
      {
        const psTri* t = &E->t[q];
        // per-triangle lateral minimum image relative to the line
        double sh_i = E->per[i] ? -E->L[i] * round((t->P[0][i] - qi) / E->L[i]) : 0.0;
        double sh_j = E->per[j] ? -E->L[j] * round((t->P[0][j] - qj) / E->L[j]) : 0.0;
        const double x1 = t->P[0][i] + sh_i, y1 = t->P[0][j] + sh_j;
        const double x2 = t->P[1][i] + sh_i, y2 = t->P[1][j] + sh_j;
        const double x3 = t->P[2][i] + sh_i, y3 = t->P[2][j] + sh_j;
        const double det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        const double area = sqrt(t->n[0] * t->n[0] + t->n[1] * t->n[1] + t->n[2] * t->n[2]);
        const int ia = E->per[i] ? 1 : 0, ja = E->per[j] ? 1 : 0;
        for (int a = -ia; a <= ia; ++a)
          for (int b = -ja; b <= ja; ++b)
          {
            const double X = qi + a * E->L[i], Y = qj + b * E->L[j];
            if (fabs(det) <= det_eps)
            {
              // vertical triangle containing the line: non-generic
              const double mnx = fmin(x1, fmin(x2, x3)), mxx = fmax(x1, fmax(x2, x3));
              const double mny = fmin(y1, fmin(y2, y3)), mxy = fmax(y1, fmax(y2, y3));
              if (X >= mnx - 1e-12 && X <= mxx + 1e-12 && Y >= mny - 1e-12 && Y <= mxy + 1e-12)
              {
                const double d = (x2 - x1) * (Y - y1) - (y2 - y1) * (X - x1);
                if (fabs(d) < 1e-12) gen = 0;
              }
              continue;
            }
            const double l2 = ((X - x1) * (y3 - y1) - (x3 - x1) * (Y - y1)) / det;
            const double l3 = ((x2 - x1) * (Y - y1) - (X - x1) * (y2 - y1)) / det;
            const double l1 = 1.0 - l2 - l3;
            const double lm = fmin(l1, fmin(l2, l3));
            if (lm < -1e-12) continue;
            if (!(lm > eta_b && fabs(t->n[k]) / area > eta_n)) gen = 0;
            if (n == cap)
            {
              cap *= 2;
              rec = realloc(rec, (size_t) cap * sizeof(psCross));
            }
            rec[n].z = l1 * t->P[0][k] + l2 * t->P[1][k] + l3 * t->P[2][k];
            rec[n].s = sg * (det > 0 ? 1 : -1);
            rec[n].comp = c;
            n++;
          }
      }
    }
    if (k == E->wall_axis)
      for (int pi = 0; pi < C->npc; ++pi)
      {
        const psPiece* P = &E->piece[C->pc[pi]];
        int clear = 1;
        const double tl = 1e-7 * fmax(E->L[i], E->L[j]);
        for (int l = 0; l < E->nloop; ++l)
          if (E->loop[l].wall == P->wall && E->mem_comp[E->loop[l].member] == c && psLoopDist(E, &E->loop[l], i, j, qi, qj) <= tl) clear = 0;
        if (!clear) gen = 0;
        if (psPieceContains(E, P, qi, qj))
        {
          if (n == cap)
          {
            cap *= 2;
            rec = realloc(rec, (size_t) cap * sizeof(psCross));
          }
          rec[n].z = E->cw[P->wall];
          rec[n].s = P->wall == 0 ? -1 : 1;
          rec[n].comp = c;
          n++;
        }
      }
  }
  // separation of all (reduced) heights
  if (gen && n > 1)
  {
    double* h = malloc((size_t) n * sizeof(double));
    for (int x = 0; x < n; ++x)
    {
      h[x] = rec[x].z;
      if (E->per[k]) h[x] -= E->L[k] * floor(h[x] / E->L[k]);
    }
    qsort(h, (size_t) n, sizeof(double), psCmpDbl);
    const double sep = E->eta_z * E->L[k];
    for (int x = 1; x < n; ++x)
      if (h[x] - h[x - 1] < sep) gen = 0;
    if (E->per[k] && h[0] + E->L[k] - h[n - 1] < sep) gen = 0;
    free(h);
  }
  *out = rec;
  *nout = n;
  *generic = gen;
  return MFTL_SUCCESS;
}

/* ------------------------------------------------------------------------------------------
 * helpers on crossing lists
 * ------------------------------------------------------------------------------------------ */
/* level just above the nearest crossing strictly below height z (only components with
   use[c] != 0 when use is given); returns the record index (-1: none) and its lift q */
static int psLevelBelow(const psEval* E, int k, const psCross* r, int n, double z, const int* use, int* ridx, int* q_out)
{
  double best = 0.0;
  int bi = -1, bq = 0;
  for (int x = 0; x < n; ++x)
  {
    if (use && !use[r[x].comp]) continue;
    double d;
    int q = 0;
    if (E->per[k])
    {
      d = z - r[x].z;
      d -= E->L[k] * floor(d / E->L[k]);
      q = (int) lround((z - d - r[x].z) / E->L[k]);
    }
    else
    {
      d = z - r[x].z;
      if (d <= 0) continue;
    }
    if (bi < 0 || d < best)
    {
      best = d;
      bi = x;
      bq = q;
    }
  }
  if (ridx) *ridx = bi;
  if (q_out) *q_out = bq;
  if (bi < 0) return 0;
  return r[bi].s < 0 ? 1 : 0;
}

static int psCmpCrossZ(const void* a, const void* b)
{
  const psCross* x = (const psCross*) a;
  const psCross* y = (const psCross*) b;
  return (x->z > y->z) - (x->z < y->z);
}

/* R6: cyclic strict alternation (periodic k) / linear levels in {0,1} from level 0 (walled) */
static int psAlternates(const psEval* E, int k, const psCross* r0, int n)
{
  psCross* r = malloc((size_t) (n > 0 ? n : 1) * sizeof(psCross));
  memcpy(r, r0, (size_t) n * sizeof(psCross));
  int ok = 1;
  if (E->per[k])
  {
    for (int x = 0; x < n; ++x) r[x].z -= E->L[k] * floor(r[x].z / E->L[k]);
    qsort(r, (size_t) n, sizeof(psCross), psCmpCrossZ);
    if (n % 2) ok = 0;
    for (int x = 0; x < n && ok; ++x)
      if (r[x].s == r[(x + 1) % n].s) ok = 0;
  }
  else
  {
    qsort(r, (size_t) n, sizeof(psCross), psCmpCrossZ);
    int lev = 0;
    for (int x = 0; x < n && ok; ++x)
    {
      lev += -r[x].s;
      if (lev != 0 && lev != 1) ok = 0;
    }
    if (lev != 0) ok = 0;
  }
  free(r);
  return ok;
}

/* ------------------------------------------------------------------------------------------
 * member lines (R5): interior points of the component's own triangles, deterministic
 * ------------------------------------------------------------------------------------------ */
static int psMemberLine(const psEval* E, int k, int c, double P[3], psCross** rec, int* nrec, int* tries)
{
  static const double b0[3] = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  const psComp* C = &E->comp[c];
  *tries = 0;
  for (int mi = 0; mi < C->nm; ++mi)
  {
    const int m = C->mem[mi];
    for (int q = E->mem_t0[m]; q < E->mem_t0[m] + E->mem_nt[m]; ++q)
    {
      const psTri* t = &E->t[q];
      const double area = sqrt(t->n[0] * t->n[0] + t->n[1] * t->n[1] + t->n[2] * t->n[2]);
      if (!(area > 0.0) || fabs(t->n[k]) / area <= E->tol.eta_n) continue;
      for (int w = 0; w < 8; ++w)
      {
        double bw[3];
        if (w == 0)
          memcpy(bw, b0, sizeof(bw));
        else
        {
          double f1 = w * PS_PHI1, f2 = w * PS_PHI2;
          f1 -= floor(f1);
          f2 -= floor(f2);
          bw[0] = 1.0 + f1;
          bw[1] = 1.0 + f2;
          bw[2] = 1.0;
          const double sum = bw[0] + bw[1] + bw[2];
          for (int a = 0; a < 3; ++a) bw[a] /= sum;
        }
        for (int a = 0; a < 3; ++a) P[a] = bw[0] * t->P[0][a] + bw[1] * t->P[1][a] + bw[2] * t->P[2][a];
        const int i = (k + 1) % 3, j = (k + 2) % 3;
        int gen = 0;
        psLine(E, k, P[i], P[j], rec, nrec, &gen);
        (*tries)++;
        if (gen) return MFTL_SUCCESS;
        free(*rec);
        *rec = MFTL_NULL;
        if (*tries >= E->tol.n_try) return MFTL_ERROR;
      }
    }
  }
  return MFTL_ERROR;
}

/* ------------------------------------------------------------------------------------------
 * R3–R9 along axis k
 * ------------------------------------------------------------------------------------------ */
typedef struct
{
  MFTL_PhaseStatus st;
  double V;
  double Ck;  // centre along k in the Y frame (when cst == DEFINED)
  MFTL_PhaseStatus cst;
  int lambda_E, n_layers, n_compact, n_ph, tries_max;
} psResult;

static void psAxisSetup(psEval* E, int k)
{
  E->k = k;
  E->i = (k + 1) % 3;
  E->j = (k + 2) % 3;
  E->Lk = E->L[k];
  E->Ak = E->L[E->i] * E->L[E->j];
  E->Vcell = E->Ak * E->Lk;
  E->tau_phi = E->tol.tau_phi_rel * E->Ak;
  E->tau_v = E->tol.tau_v_rel * E->Vcell;
}

static psResult psEvaluateK(psEval* E, int k, int want_centre)
{
  psResult R;
  memset(&R, 0, sizeof(R));
  R.st = MFTL_DEFINED;
  R.cst = MFTL_UNSUPPORTED;
  R.lambda_E = -1;
  psAxisSetup(E, k);
  const int nc = E->ncomp;
  // R4: classes
  int* layer = calloc((size_t) (nc > 0 ? nc : 1), sizeof(int));
  int* compact = calloc((size_t) (nc > 0 ? nc : 1), sizeof(int));
  int nlay = 0, ncmp = 0;
  double Vraw = 0.0;
  for (int c = 0; c < nc; ++c)
  {
    double v, phik;
    psCompIntegrals(E, c, k, 0.0, &v, MFTL_NULL, &phik);
    E->comp[c].vown = v;
    E->comp[c].shift = 0.0;
    Vraw += v;
    if (fabs(phik) < E->tau_phi)
    {
      compact[c] = 1;
      ncmp++;
      if (fabs(v) <= E->tau_v)
      {
        psWhy(E, "degenerate compact component (zero enclosed volume)%s", "");
        R.st = MFTL_UNSUPPORTED;
        goto out;
      }
      E->comp[c].e = v > 0 ? 1 : -1;
    }
    else if (fabs(fabs(phik) - E->Ak) < E->tau_phi)
    {
      layer[c] = 1;
      nlay++;
    }
    else
    {
      psWhy(E, "|Phi_k| of a component not in {0, A_k}%s", "");
      R.st = MFTL_UNSUPPORTED;
      goto out;
    }
  }
  R.n_layers = nlay;
  R.n_compact = ncmp;
  // R8a: degenerate volume before any line
  {
    double Vrep = Vraw;
    if (E->per[k]) Vrep -= E->Vcell * floor(Vraw / E->Vcell);
    if (Vrep <= E->tau_v || Vrep >= E->Vcell - E->tau_v)
    {
      psWhy(E, "degenerate volume (empty or full region)%s", "");
      R.st = MFTL_UNDEFINED;
      goto out;
    }
  }
  // R5/R6: canonical line
  psCross* canon = MFTL_NULL;
  int ncanon = 0;
  if (nlay)
  {
    int found = 0;
    for (int n = 0; n < E->tol.n_try && !found; ++n)
    {
      double f1 = n * PS_PHI1, f2 = n * PS_PHI2;
      f1 -= floor(f1);
      f2 -= floor(f2);
      const double qi = f1 * E->L[E->i], qj = f2 * E->L[E->j];
      int gen = 0;
      psLine(E, k, qi, qj, &canon, &ncanon, &gen);
      if (gen)
        found = 1;
      else
      {
        free(canon);
        canon = MFTL_NULL;
      }
    }
    if (!found)
    {
      psWhy(E, "no generic canonical line%s", "");
      R.st = MFTL_UNSUPPORTED;
      goto out;
    }
    if (!psAlternates(E, k, canon, ncanon))
    {
      free(canon);
      psWhy(E, "crossings of the canonical line do not alternate (inconsistent orientation)%s", "");
      R.st = MFTL_UNDEFINED;
      goto out;
    }
  }
  // R7: compact tree
  {
    free(E->comp_parent);
    free(E->comp_parent_q);
    free(E->comp_depth);
    E->comp_parent = malloc((size_t) (nc > 0 ? nc : 1) * sizeof(int));
    E->comp_parent_q = calloc((size_t) (nc > 0 ? nc : 1), sizeof(int));
    E->comp_depth = calloc((size_t) (nc > 0 ? nc : 1), sizeof(int));
    int(*enc)[2] = malloc((size_t) (nc * nc > 0 ? nc * nc : 1) * sizeof(*enc));
    int* nenc = calloc((size_t) (nc > 0 ? nc : 1), sizeof(int));
    int* lay_lev = calloc((size_t) (nc > 0 ? nc : 1), sizeof(int));
    int* lay_comp = malloc((size_t) (nc > 0 ? nc : 1) * sizeof(int));
    int* lay_q = calloc((size_t) (nc > 0 ? nc : 1), sizeof(int));
    int ok = 1;
    for (int c = 0; c < nc && ok; ++c)
    {
      E->comp_parent[c] = -1;
      lay_comp[c] = -1;
      if (!compact[c]) continue;
      double P[3];
      psCross* rec = MFTL_NULL;
      int nrec = 0, tries = 0;
      if (psMemberLine(E, k, c, P, &rec, &nrec, &tries) != MFTL_SUCCESS)
      {
        psWhy(E, "no generic line through a compact member%s", "");
        R.st = MFTL_UNSUPPORTED;
        ok = 0;
        break;
      }
      if (tries > R.tries_max) R.tries_max = tries;
      const double zP = P[k];
      for (int c2 = 0; c2 < nc; ++c2)
      {
        if (c2 == c || !compact[c2]) continue;
        double mn = 1e300, mx = -1e300;
        int cnt = 0;
        for (int x = 0; x < nrec; ++x)
          if (rec[x].comp == c2)
          {
            mn = fmin(mn, rec[x].z);
            mx = fmax(mx, rec[x].z);
            cnt++;
          }
        if (!cnt) continue;
        int plo = 0, phi = 0;
        if (E->per[k])
        {
          plo = (int) floor((zP - mx) / E->Lk) - 1;
          phi = (int) ceil((zP - mn) / E->Lk) + 1;
        }
        for (int p = plo; p <= phi; ++p)
        {
          int par = 0;
          for (int x = 0; x < nrec; ++x)
            if (rec[x].comp == c2 && rec[x].z + p * E->Lk < zP) par ^= 1;
          if (par)
          {
            enc[c * nc + nenc[c]][0] = c2;
            enc[c * nc + nenc[c]][1] = p;
            nenc[c]++;
          }
        }
      }
      if (nlay)
      {
        int ri = -1, q = 0;
        lay_lev[c] = psLevelBelow(E, k, rec, nrec, zP, layer, &ri, &q);
        lay_comp[c] = ri >= 0 ? rec[ri].comp : -1;
        lay_q[c] = q;
      }
      free(rec);
    }
    for (int c = 0; c < nc && ok; ++c)
      if (compact[c]) E->comp_depth[c] = nenc[c];
    for (int c = 0; c < nc && ok; ++c)
    {
      if (!compact[c] || E->comp_depth[c] == 0) continue;
      int npar = 0;
      for (int x = 0; x < nenc[c]; ++x)
      {
        const int c2 = enc[c * nc + x][0];
        if (E->comp_depth[c2] == E->comp_depth[c] - 1)
        {
          npar++;
          E->comp_parent[c] = c2;
          E->comp_parent_q[c] = enc[c * nc + x][1];
        }
      }
      if (npar != 1)
      {
        psWhy(E, "nesting not resolvable%s", "");
        R.st = MFTL_UNSUPPORTED;
        ok = 0;
      }
    }
    if (ok && !nlay)
    {
      int lE = -1, any = 0;
      for (int c = 0; c < nc && ok; ++c)
      {
        if (!compact[c] || E->comp_parent[c] >= 0) continue;
        const int v = E->comp[c].e > 0 ? 0 : 1;
        if (!any)
        {
          lE = v;
          any = 1;
        }
        else if (v != lE)
        {
          psWhy(E, "outermost compact components disagree on the exterior level (inconsistent orientation)%s", "");
          R.st = MFTL_UNDEFINED;
          ok = 0;
        }
      }
      if (ok && !any)
      {
        psWhy(E, "empty phase-boundary set%s", "");
        R.st = MFTL_UNDEFINED;
        ok = 0;
      }
      if (ok && lE == 1 && !(E->per[0] && E->per[1] && E->per[2]))
      {
        psWhy(E, "exterior region would be phase but touches an undeclared domain wall%s", "");
        R.st = MFTL_UNDEFINED;
        ok = 0;
      }
      R.lambda_E = lE;
    }
    // orientation rule, by increasing depth
    for (int d = 0; ok; ++d)
    {
      int anyd = 0;
      for (int c = 0; c < nc && ok; ++c)
      {
        if (!compact[c] || E->comp_depth[c] != d) continue;
        anyd = 1;
        int lout;
        if (E->comp_parent[c] >= 0)
          lout = E->comp[E->comp_parent[c]].e > 0 ? 1 : 0;
        else if (nlay)
          lout = lay_lev[c];
        else
          lout = R.lambda_E;
        if ((E->comp[c].e > 0) != (lout == 0))
        {
          psWhy(E, "orientation of a compact component inconsistent with its surroundings%s", "");
          R.st = MFTL_UNDEFINED;
          ok = 0;
        }
      }
      if (!anyd && d > nc) break;
      if (d > nc) break;
    }
    // R8: volume
    if (ok)
    {
      double V;
      if (E->per[k] && nlay)
        V = Vraw - E->Vcell * floor(Vraw / E->Vcell);
      else if (E->per[k])
        V = Vraw + R.lambda_E * E->Vcell;
      else
        V = Vraw;
      if (V <= E->tau_v || V >= E->Vcell - E->tau_v)
      {
        psWhy(E, "degenerate volume (empty or full region)%s", "");
        R.st = MFTL_UNDEFINED;
        ok = 0;
      }
      R.V = V;
      int nph = nlay / 2 + (R.lambda_E == 1 ? 1 : 0);
      for (int c = 0; c < nc; ++c)
        if (compact[c] && E->comp[c].e > 0) nph++;
      R.n_ph = nph;
    }
    // R9: placement and centre along k
    if (ok && want_centre)
    {
      if (E->per[k] && nlay)
      {
        // z0: midpoint of the first segment (reduced order, layer crossings) after a leaving crossing
        int nl = 0;
        psCross* lr = malloc((size_t) (ncanon > 0 ? ncanon : 1) * sizeof(psCross));
        for (int x = 0; x < ncanon; ++x)
          if (layer[canon[x].comp])
          {
            lr[nl] = canon[x];
            lr[nl].z -= E->Lk * floor(lr[nl].z / E->Lk);
            nl++;
          }
        qsort(lr, (size_t) nl, sizeof(psCross), psCmpCrossZ);
        double z0 = 0.0;
        for (int a = 0; a < nl; ++a)
          if (lr[a].s > 0)
          {
            double gap = lr[(a + 1) % nl].z - lr[a].z;
            gap -= E->Lk * floor(gap / E->Lk);
            if (nl == 1) gap = E->Lk;
            z0 = lr[a].z + 0.5 * gap;
            break;
          }
        free(lr);
        for (int c = 0; c < nc; ++c)
        {
          if (!layer[c]) continue;
          double mn = 1e300, mx = -1e300;
          for (int x = 0; x < ncanon; ++x)
            if (canon[x].comp == c)
            {
              mn = fmin(mn, canon[x].z);
              mx = fmax(mx, canon[x].z);
            }
          int best = 0, have = 0;
          for (int q = (int) floor((z0 - mx) / E->Lk) - 2; q <= (int) ceil((z0 - mn) / E->Lk) + 2; ++q)
          {
            int par = 0;
            for (int x = 0; x < ncanon; ++x)
              if (canon[x].comp == c && canon[x].z + q * E->Lk < z0) par ^= 1;
            if (par && (!have || q > best))
            {
              best = q;
              have = 1;
            }
          }
          E->comp[c].shift = best * E->Lk;
        }
      }
      for (int d = 0; d <= nc; ++d)
        for (int c = 0; c < nc; ++c)
        {
          if (!compact[c] || E->comp_depth[c] != d) continue;
          if (!E->per[k])
            E->comp[c].shift = 0.0;
          else if (E->comp_parent[c] >= 0)
            E->comp[c].shift = E->comp[E->comp_parent[c]].shift - E->comp_parent_q[c] * E->Lk;
          else if (nlay && lay_comp[c] >= 0)
            E->comp[c].shift = E->comp[lay_comp[c]].shift - lay_q[c] * E->Lk;
          else
            E->comp[c].shift = 0.0;
        }
      double Vp = 0.0, C2 = 0.0;
      for (int c = 0; c < nc; ++c)
      {
        double v, c2;
        psCompIntegrals(E, c, k, E->comp[c].shift, &v, &c2, MFTL_NULL);
        Vp += v;
        C2 += c2;
      }
      if (E->per[k] && R.lambda_E == 1)
        R.cst = MFTL_UNDEFINED;
      else if (E->per[k] && R.n_ph != 1)
        R.cst = MFTL_UNSUPPORTED;
      else
      {
        if (fabs(Vp - R.V) > 1e-9 * E->Vcell + 64 * 2.2e-16 * fabs(Vraw))
        {
          psWhy(E, "internal error: placed volume differs from V%s", "");
          R.st = MFTL_UNSUPPORTED;
          ok = 0;
        }
        else
        {
          R.Ck = C2 / (2.0 * R.V);
          R.cst = MFTL_DEFINED;
        }
      }
    }
    free(enc);
    free(nenc);
    free(lay_lev);
    free(lay_comp);
    free(lay_q);
  }
  free(canon);
out:
  E->n_layers = nlay;
  E->n_compact = ncmp;
  free(layer);
  free(compact);
  return R;
}

/* ------------------------------------------------------------------------------------------
 * the full evaluation
 * ------------------------------------------------------------------------------------------ */
typedef struct
{
  MFTL_PhaseStatus st;
  psResult Rk;
  int k;
  double Phi[3];
} psTop;

static psTop psRun(MFTL_PhaseSet* S, psEval* E, int want_centre)
{
  psTop T;
  memset(&T, 0, sizeof(T));
  T.k = -1;
  psGather(S, E);
  for (int m = 0; m < E->nmem; ++m)
    for (int a = 0; a < 3; ++a) T.Phi[a] += E->sigma[E->mem_body[m]] * E->mem_area[3 * m + a];
  if (E->nmem == 0)
  {
    psWhy(E, "empty phase-boundary set%s", "");
    T.st = MFTL_UNDEFINED;
    return T;
  }
  // R1/R2: closure and flux (precedence E2: closure errors, then Phi, then the axis)
  T.st = psClose(S, E);
  if (T.st != MFTL_DEFINED) return T;
  // R3: admissible axis
  int k = -1;
  if (E->wall_axis >= 0)
    k = E->wall_axis;
  else
  {
    for (int a = 0; a < 3 && k < 0; ++a)
      if (!E->per[a]) k = a;
    for (int a = 0; a < 3 && k < 0; ++a)
    {
      int wr = 0;
      for (int m = 0; m < E->nmem; ++m)
        if (E->mem_cut[3 * m + a]) wr = 1;
      if (!wr) k = a;
    }
  }
  if (k < 0)
  {
    psWhy(E, "no admissible axis (every periodic axis wrapped by some member)%s", "");
    T.st = MFTL_UNSUPPORTED;
    return T;
  }
  for (int m = 0; m < E->nmem; ++m)
    if (E->mem_cut[3 * m + k])
    {
      psWhy(E, "a member wraps the evaluation axis%s", "");
      T.st = MFTL_UNSUPPORTED;
      return T;
    }
  T.k = k;
  T.Rk = psEvaluateK(E, k, want_centre);
  T.st = T.Rk.st;
  return T;
}

/* ------------------------------------------------------------------------------------------
 * API
 * ------------------------------------------------------------------------------------------ */
MFTL_PhaseStatus MFTLPhaseSetClassify(MFTL_PhaseSet* S, MFTL_PhaseSetInfo* info)
{
  psEval E;
  psTop T = psRun(S, &E, 0);
  if (info)
  {
    memset(info, 0, sizeof(*info));
    info->status = T.st;
    snprintf(info->reason, sizeof(info->reason), "%s", T.st == MFTL_DEFINED ? "" : E.why);
    memcpy(info->Phi, T.Phi, sizeof(T.Phi));
    info->axis = T.k;
    info->n_components = E.ncomp;
    info->n_layers = T.Rk.n_layers;
    info->n_compact = T.Rk.n_compact;
    info->lambda_E = T.Rk.lambda_E;
    info->n_phase_components = T.Rk.n_ph;
    info->wall_n[0] = E.nW[0];
    info->wall_n[1] = E.nW[1];
    info->volume = T.st == MFTL_DEFINED ? T.Rk.V : 0.0;
    info->member_line_tries_max = T.Rk.tries_max;
  }
  psFree(&E);
  return T.st;
}

MFTL_PhaseStatus MFTLPhaseSetVolume(MFTL_PhaseSet* S, double* V)
{
  psEval E;
  psTop T = psRun(S, &E, 0);
  if (T.st == MFTL_DEFINED && V) *V = T.Rk.V;
  psFree(&E);
  return T.st;
}

static double psWrapAxis(const MFTL_Domain* dom, int a, double x)
{
  if (!dom->is_periodic[a]) return x;
  x -= dom->domain_size[a] * floor((x - dom->global_min[a]) / dom->domain_size[a]);
  if (x >= dom->global_max[a]) x -= dom->domain_size[a];
  return x;
}

MFTL_PhaseStatus MFTLPhaseSetCentre(MFTL_PhaseSet* S, double C[3], MFTL_PhaseStatus status[3])
{
  psEval E;
  psTop T = psRun(S, &E, 1);
  for (int a = 0; a < 3; ++a)
  {
    status[a] = T.st;
    C[a] = 0.0;
  }
  if (T.st != MFTL_DEFINED)
  {
    psFree(&E);
    return T.st;
  }
  const int k = T.k;
  status[k] = T.Rk.cst;
  if (T.Rk.cst == MFTL_DEFINED) C[k] = psWrapAxis(S->dom, k, E.pos_ref[k] + T.Rk.Ck);
  const double V = T.Rk.V;
  const int nph = T.Rk.n_ph, lE = T.Rk.lambda_E;
  for (int a = 0; a < 3; ++a)
  {
    if (a == k) continue;
    if (!E.per[a])
    {
      // non-periodic axis: no placement ambiguity along a (own gauges are absolute)
      double c2 = 0.0;
      for (int c = 0; c < E.ncomp; ++c)
      {
        double x;
        psCompIntegrals(&E, c, a, 0.0, MFTL_NULL, &x, MFTL_NULL);
        c2 += x;
      }
      C[a] = E.pos_ref[a] + c2 / (2.0 * V);
      status[a] = MFTL_DEFINED;
      continue;
    }
    int wr = (lE == 1);
    for (int m = 0; m < E.nmem; ++m)
      if (E.mem_cut[3 * m + a]) wr = 1;
    if (wr)
    {
      status[a] = MFTL_UNDEFINED;
      continue;
    }
    if (nph != 1)
    {
      status[a] = MFTL_UNSUPPORTED;
      continue;
    }
    // the same construction with a as the line axis (every component is compact along a)
    const int save_k = E.k;
    double* save_shift = malloc((size_t) (E.ncomp > 0 ? E.ncomp : 1) * sizeof(double));
    int* save_e = malloc((size_t) (E.ncomp > 0 ? E.ncomp : 1) * sizeof(int));
    for (int c = 0; c < E.ncomp; ++c)
    {
      save_shift[c] = E.comp[c].shift;
      save_e[c] = E.comp[c].e;
    }
    psResult Ra = psEvaluateK(&E, a, 1);
    if (Ra.st == MFTL_DEFINED && Ra.cst == MFTL_DEFINED && fabs(Ra.V - V) <= 1e-9 * E.Vcell)
    {
      status[a] = MFTL_DEFINED;
      C[a] = psWrapAxis(S->dom, a, E.pos_ref[a] + Ra.Ck);
    }
    else
      status[a] = MFTL_UNSUPPORTED;
    for (int c = 0; c < E.ncomp; ++c)
    {
      E.comp[c].shift = save_shift[c];
      E.comp[c].e = save_e[c];
    }
    free(save_shift);
    free(save_e);
    psAxisSetup(&E, save_k);
  }
  psFree(&E);
  return T.st;
}

/* closest distance from point p to triangle (a, b, c) (Ericson, Real-Time Collision Detection) */
static double psPointTri(const double p[3], const double a[3], const double b[3], const double c[3])
{
  double ab[3], ac[3], ap[3], q[3];
  for (int x = 0; x < 3; ++x)
  {
    ab[x] = b[x] - a[x];
    ac[x] = c[x] - a[x];
    ap[x] = p[x] - a[x];
  }
#define PS_DOT(u, v) ((u)[0] * (v)[0] + (u)[1] * (v)[1] + (u)[2] * (v)[2])
  const double d1 = PS_DOT(ab, ap), d2 = PS_DOT(ac, ap);
  if (d1 <= 0 && d2 <= 0)
    memcpy(q, a, sizeof(q));
  else
  {
    double bp[3], cp[3];
    for (int x = 0; x < 3; ++x)
    {
      bp[x] = p[x] - b[x];
      cp[x] = p[x] - c[x];
    }
    const double d3 = PS_DOT(ab, bp), d4 = PS_DOT(ac, bp), d5 = PS_DOT(ab, cp), d6 = PS_DOT(ac, cp);
    const double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    if (d3 >= 0 && d4 <= d3)
      memcpy(q, b, sizeof(q));
    else if (vc <= 0 && d1 >= 0 && d3 <= 0)
    {
      const double v = d1 / (d1 - d3);
      for (int x = 0; x < 3; ++x) q[x] = a[x] + v * ab[x];
    }
    else if (d6 >= 0 && d5 <= d6)
      memcpy(q, c, sizeof(q));
    else if (vb <= 0 && d2 >= 0 && d6 <= 0)
    {
      const double w = d2 / (d2 - d6);
      for (int x = 0; x < 3; ++x) q[x] = a[x] + w * ac[x];
    }
    else if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    {
      const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      for (int x = 0; x < 3; ++x) q[x] = b[x] + w * (c[x] - b[x]);
    }
    else
    {
      const double den = 1.0 / (va + vb + vc), v = vb * den, w = vc * den;
      for (int x = 0; x < 3; ++x) q[x] = a[x] + ab[x] * v + ac[x] * w;
    }
  }
#undef PS_DOT
  double d[3] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
  return sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

MFTL_PhaseStatus MFTLPhaseSetInside(MFTL_PhaseSet* S, const double Q[3], int* where)
{
  psEval E;
  psTop T = psRun(S, &E, 0);
  if (T.st != MFTL_DEFINED)
  {
    psFree(&E);
    return T.st;
  }
  const int k = T.k, i = (k + 1) % 3, j = (k + 2) % 3;
  psAxisSetup(&E, k);
  double Y[3];
  for (int a = 0; a < 3; ++a) Y[a] = Q[a] - E.pos_ref[a];
  // clearance: distance to Ŝ (every periodic image; patches: distance to their wall plane)
  double d = 1e300;
  for (int q = 0; q < E.nt; ++q)
  {
    const psTri* t = &E.t[q];
    double sh[3];
    for (int a = 0; a < 3; ++a) sh[a] = E.per[a] ? -E.L[a] * round((t->P[0][a] - Y[a]) / E.L[a]) : 0.0;
    const int ra = E.per[0], rb = E.per[1], rc = E.per[2];
    for (int a = -ra; a <= ra; ++a)
      for (int b = -rb; b <= rb; ++b)
        for (int c = -rc; c <= rc; ++c)
        {
          double A0[3], B0[3], C0[3];
          const int nn[3] = {a, b, c};
          for (int x = 0; x < 3; ++x)
          {
            const double o = sh[x] + nn[x] * E.L[x];
            A0[x] = t->P[0][x] + o;
            B0[x] = t->P[1][x] + o;
            C0[x] = t->P[2][x] + o;
          }
          const double dd = psPointTri(Y, A0, B0, C0);
          if (dd < d) d = dd;
        }
  }
  for (int p = 0; p < E.npiece; ++p)
    d = fmin(d, fabs(Y[E.wall_axis] - E.cw[E.piece[p].wall]));
  const double tau_d = E.tol.tau_d_rel * fmax(E.L[0], fmax(E.L[1], E.L[2]));
  if (d <= tau_d)
  {
    *where = MFTL_PHASE_ON_BOUNDARY;
    psFree(&E);
    return MFTL_DEFINED;
  }
  for (int n = 0; n < E.tol.n_try; ++n)
  {
    double f1 = n * PS_PHI1, f2 = n * PS_PHI2;
    f1 -= floor(f1);
    f2 -= floor(f2);
    const double r = (n == 0) ? 0.0 : 0.5 * d * f1, th = 2.0 * M_PI * f2;
    psCross* rec = MFTL_NULL;
    int nrec = 0, gen = 0;
    psLine(&E, k, Y[i] + r * cos(th), Y[j] + r * sin(th), &rec, &nrec, &gen);
    if (gen)
    {
      int ri = -1;
      int lev = psLevelBelow(&E, k, rec, nrec, Y[k], MFTL_NULL, &ri, MFTL_NULL);
      if (ri < 0 && E.per[k]) lev = T.Rk.lambda_E > 0 ? 1 : 0;
      free(rec);
      *where = lev ? MFTL_PHASE_INSIDE : MFTL_PHASE_OUTSIDE;
      psFree(&E);
      return MFTL_DEFINED;
    }
    free(rec);
  }
  psFree(&E);
  return MFTL_UNSUPPORTED;
}

/* ------------------------------------------------------------------------------------------
 * Set-level volume recovery (plans/16 §4.4.4)
 * ------------------------------------------------------------------------------------------ */
MFTL_PhaseStatus MFTLPhaseSetVolumeRecovery(MFTL_PhaseSet* S, struct MFTL_Ensemble* ens, double V_target, double* V_out)
{
  const MFTL_Domain* dom = S->dom;
  double Vc = 1.0;
  for (int a = 0; a < 3; ++a) Vc *= dom->domain_size[a];
  double V = 0.0;
  MFTL_PhaseStatus st = MFTL_DEFINED;
  for (int it = 0; it < 8; ++it)
  {
    st = MFTLPhaseSetVolume(S, &V);
    if (st != MFTL_DEFINED) return st;
    const double dV = V_target - V;
    if (fabs(dV) <= 1e-12 * Vc) break;
    // area of the set (owned triangles)
    double A = 0.0, gA = 0.0;
    for (int b = 0; b < S->n_body; ++b)
    {
      MFTL_Object* o = S->body[b];
      MFTLComputeTriangleProperties(o);
      for (int t = 0; t < o->n_triangles; ++t)
        if (!o->triangles[t].is_ghost && !(o->triangles[t].flags & MFTL_FLAG_DEAD)) A += o->triangles[t].area;
    }
    MPI_Allreduce(&A, &gA, 1, MPI_DOUBLE, MPI_SUM, dom->comm_cart);
    if (!(gA > 0.0)) return MFTL_UNSUPPORTED;
    const double delta = dV / gA;
    for (int b = 0; b < S->n_body; ++b)
    {
      MFTL_Object* o = S->body[b];
      MFTLComputeVertexNormals(o);
      for (int i = 0; i < o->n_vertices; ++i)
      {
        MFTL_Vertex* v = &o->vertices[i];
        if (v->is_ghost || (v->flags & MFTL_FLAG_DEAD) || v->out_he_idx < 0) continue;
        int prev, next;
        if (MFTLBoundaryNeighbours(o, i, &prev, &next) == MFTL_SUCCESS) continue;  // boundary vertex: fixed
        for (int k = 0; k < 3; ++k) v->pos[k] += delta * S->sigma[b] * v->normal[k];
        MFTLWrapPosImg(dom, v->pos, v->img);
      }
      MFTLComputeTriangleProperties(o);
    }
    MFTLUpdateGhostPositions(ens, NULL);
  }
  st = MFTLPhaseSetVolume(S, &V);
  if (st == MFTL_DEFINED && V_out) *V_out = V;
  return st;
}
