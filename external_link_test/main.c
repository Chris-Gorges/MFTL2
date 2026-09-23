/*
 * External consumer smoke test for the MFTL2 CMake package.
 *
 * Verifies that a downstream project can, using only find_package(MFTL2) +
 * link MFTL2::MFTL2:
 *   - include a public MFTL2 header (which transitively pulls in khash.h),
 *   - compile against the public API, and
 *   - link an out-of-line symbol from libMFTL2.a.
 *
 * MFTLPolygonArea2D is a pure-computational public function (no MPI/mesh state),
 * so the check is self-contained: the signed area of the unit square is 1.
 */
#include <stdio.h>
#include <math.h>

#include "mftl_overlap.h" /* public header; pulls mftl_core.h -> khash.h via the target's include interface */

int main(void)
{
  MFTL_Point2D square[4] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
  double area = MFTLPolygonArea2D(square, 4);

  if (fabs(area - 1.0) > 1e-12)
  {
    fprintf(stderr, "MFTL2 link test FAILED: unit-square area = %.17g (expected 1)\n", area);
    return 1;
  }
  printf("MFTL2 link test OK: unit-square area = %.17g\n", area);
  return 0;
}
