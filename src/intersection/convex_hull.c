/*
 * MFTL2 intersection — 2D convex hull (Graham scan) and polygon helpers for the
 * overlap tessellation and as public API. Replaces MFTL1's graham.c with a
 * direct XY implementation (MFTL1 rotated points onto the XY plane first;
 * the overlap pipeline projects into a 2D frame already, so the rotation is
 * unnecessary here).
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_overlap.h"
#include "mftl_constants.h"

// Squared distance, for the coincident-point culling.
static double mftlHullDistSq(const MFTL_Point2D* a, const MFTL_Point2D* b)
{
  double dx = a->x - b->x, dy = a->y - b->y;
  return dx * dx + dy * dy;
}

// qsort comparator: polar angle about `origin`, CCW; ties broken by
// distance (nearer first). Context passed via a file-static (qsort_r is
// non-portable).
static MFTL_Point2D g_hull_origin;
static int mftlHullAngleCompare(const void* pa, const void* pb)
{
  const MFTL_Point2D* a = (const MFTL_Point2D*) pa;
  const MFTL_Point2D* b = (const MFTL_Point2D*) pb;
  double cax = a->x - g_hull_origin.x, cay = a->y - g_hull_origin.y;
  double cbx = b->x - g_hull_origin.x, cby = b->y - g_hull_origin.y;
  double cross = cax * cby - cay * cbx;
  if (cross > 0.0) return -1;
  if (cross < 0.0) return 1;
  double da = cax * cax + cay * cay, db = cbx * cbx + cby * cby;
  return (da < db) ? -1 : (da > db) ? 1 : 0;
}

int MFTLConvexHull2D(MFTL_Point2D* pts, int* n)
{
  int np = *n;
  if (np <= 2)
  {
    *n = np;
    return np >= 2 ? 1 : 0;
  }

  // 1. Lowest (then rightmost) point to index 0.
  int low = 0;
  for (int i = 1; i < np; ++i)
    if (pts[i].y < pts[low].y || (pts[i].y == pts[low].y && pts[i].x > pts[low].x)) low = i;
  MFTL_Point2D tmp = pts[0];
  pts[0] = pts[low];
  pts[low] = tmp;
  g_hull_origin = pts[0];

  // 2. Sort by polar angle about the origin.
  qsort(pts + 1, (size_t) (np - 1), sizeof(MFTL_Point2D), mftlHullAngleCompare);

  // 3. Coincident-with-origin removal happens naturally in the scan.
  // Graham scan with a dynamic stack in the output prefix.
  MFTL_Point2D* stack = malloc((size_t) np * sizeof(MFTL_Point2D));
  if (!stack) return 0;
  int top = 0;
  stack[top++] = pts[0];
  stack[top++] = pts[1];
  for (int i = 2; i < np; ++i)
  {
    while (top >= 2)
    {
      MFTL_Point2D* o = &stack[top - 2];
      MFTL_Point2D* a = &stack[top - 1];
      double cross = (a->x - o->x) * (pts[i].y - o->y) - (a->y - o->y) * (pts[i].x - o->x);
      if (cross > 0.0) break;  // left turn: keep
      --top;                   // right turn or collinear: pop
    }
    stack[top++] = pts[i];
  }

  memcpy(pts, stack, (size_t) top * sizeof(MFTL_Point2D));
  free(stack);
  *n = top;
  return top >= 3 ? 1 : 0;
}

double MFTLPolygonArea2D(const MFTL_Point2D* pts, int n)
{
  if (n < 3) return 0.0;
  double a = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const MFTL_Point2D* p = &pts[i];
    const MFTL_Point2D* q = &pts[(i + 1) % n];
    a += p->x * q->y - p->y * q->x;
  }
  return 0.5 * a;
}

void MFTLPolygonCentroid2D(const MFTL_Point2D* pts, int n, double centre[2])
{
  centre[0] = centre[1] = 0.0;
  if (n < 3) return;
  double a = 0.0, cx = 0.0, cy = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const MFTL_Point2D* p = &pts[i];
    const MFTL_Point2D* q = &pts[(i + 1) % n];
    double w = p->x * q->y - p->y * q->x;
    a += w;
    cx += (p->x + q->x) * w;
    cy += (p->y + q->y) * w;
  }
  if (fabs(a) < MFTL_VERYSMALL)
  {
    for (int i = 0; i < n; ++i)
    {
      cx += pts[i].x;
      cy += pts[i].y;
    }
    centre[0] = cx / (double) n;
    centre[1] = cy / (double) n;
    return;
  }
  centre[0] = cx / (3.0 * a);
  centre[1] = cy / (3.0 * a);
}
