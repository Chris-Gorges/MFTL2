/*
 * MFTL2 parallel — the MPI datatypes of the packed transport records (PackedVertex,
 * PackedTriangle, PackedHalfEdge, PackedTriangleComplex, GhostUpdatePayload), committed while
 * at least one ensemble exists.
 *
 * Wire format: the Packed* records (mftl_communication.h) are committed as MPI struct types
 * resized to sizeof(record), so arrays of records have the C stride.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <mpi.h>
#include <khash.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_communication.h"
#include "mftl_twin_internal.h"
#include "mftl_arrayfunctions.h"
#include "mftl_topology.h"
#include "mftl_memory.h"

// committed while at least one ensemble exists (MFTLRegisterMpiTypes / MFTLFreeMpiTypes)
MPI_Datatype MPI_PackedVertex;
MPI_Datatype MPI_PackedTriangle;
MPI_Datatype MPI_PackedHalfEdge;
MPI_Datatype MPI_PackedTriangleComplex;
MPI_Datatype MPI_GhostUpdatePayload;

// Fix every packed datatype's extent to the C struct size. The MPI extent of a struct type is
// implementation-defined in the presence of trailing padding; arrays of packed records are
// exchanged with counts > 1, so the stride MUST equal sizeof(record).
static void mftlResizeToSizeof(MPI_Datatype* type, size_t size)
{
  MPI_Datatype resized;
  MPI_Type_create_resized(*type, 0, (MPI_Aint) size, &resized);
  MPI_Type_free(type);
  MPI_Type_commit(&resized);
  *type = resized;
}

void MFTLRegisterMpiTypes()
{
  // PackedVertex
  {
#define NITEMS_VERTEX 14
    int blocklengths[] = {1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3};

    MPI_Datatype types[] = {
        MPI_INT,  // global_id
        MPI_INT,  // owner_proc
        MPI_DOUBLE,  // pos[3]
        MPI_DOUBLE,  // normal[3]
        MPI_DOUBLE,  // meancurvature
        MPI_DOUBLE,  // gaussiancurvature
        MPI_DOUBLE,  // area
        MPI_DOUBLE,  // roughness_mean
        MPI_DOUBLE,  // roughness_variance
        MPI_C_BOOL,  // is_ghost
        MPI_INT,  // flags
        MPI_INT,  // id_he_out
        MPI_INT,  // body_id
        MPI_INT  // img[3]
    };

    MPI_Aint offsets[NITEMS_VERTEX];
    offsets[0] = offsetof(PackedVertex, global_id);
    offsets[1] = offsetof(PackedVertex, owner_proc);
    offsets[2] = offsetof(PackedVertex, pos);
    offsets[3] = offsetof(PackedVertex, normal);
    offsets[4] = offsetof(PackedVertex, meancurvature);
    offsets[5] = offsetof(PackedVertex, gaussiancurvature);
    offsets[6] = offsetof(PackedVertex, area);
    offsets[7] = offsetof(PackedVertex, roughness_mean);
    offsets[8] = offsetof(PackedVertex, roughness_variance);
    offsets[9] = offsetof(PackedVertex, is_ghost);
    offsets[10] = offsetof(PackedVertex, flags);
    offsets[11] = offsetof(PackedVertex, id_he_out);
    offsets[12] = offsetof(PackedVertex, body_id);
    offsets[13] = offsetof(PackedVertex, img);

    MPI_Type_create_struct(NITEMS_VERTEX, blocklengths, offsets, types, &MPI_PackedVertex);
    mftlResizeToSizeof(&MPI_PackedVertex, sizeof(PackedVertex));
  }

  // PackedTriangle
  {
#define NITEMS_TRIANGLE 10
    int blocklengths[] = {1, 1, 1, 3, 1, 3, 1, 1, 1, 1};

    MPI_Datatype types[] = {
        MPI_INT,  // global_id
        MPI_INT,  // owner_proc
        MPI_INT,  // he_id
        MPI_DOUBLE,  // normal[3]
        MPI_DOUBLE,  // area
        MPI_DOUBLE,  // center[3]
        MPI_C_BOOL,  // is_ghost
        MPI_INT,  // flags
        MPI_INT,  // body_id
        MPI_DOUBLE  // surface_scalar
    };

    MPI_Aint offsets[NITEMS_TRIANGLE];
    offsets[0] = offsetof(PackedTriangle, global_id);
    offsets[1] = offsetof(PackedTriangle, owner_proc);
    offsets[2] = offsetof(PackedTriangle, he_id);
    offsets[3] = offsetof(PackedTriangle, normal);
    offsets[4] = offsetof(PackedTriangle, area);
    offsets[5] = offsetof(PackedTriangle, center);
    offsets[6] = offsetof(PackedTriangle, is_ghost);
    offsets[7] = offsetof(PackedTriangle, flags);
    offsets[8] = offsetof(PackedTriangle, body_id);
    offsets[9] = offsetof(PackedTriangle, surface_scalar);

    MPI_Type_create_struct(NITEMS_TRIANGLE, blocklengths, offsets, types, &MPI_PackedTriangle);
    mftlResizeToSizeof(&MPI_PackedTriangle, sizeof(PackedTriangle));
  }

  // PackedHalfEdge (owner_proc is not transported)
  {
#define NITEMS_HALFEDGE 7
    int blocklengths[] = {1, 1, 1, 1, 1, 1, 1};

    MPI_Datatype types[] = {
        MPI_INT,  // global_id
        MPI_INT,  // vertex_gid
        MPI_INT,  // twin_gid
        MPI_INT,  // next_gid
        MPI_INT,  // triangle_gid
        MPI_C_BOOL,  // boundary
        MPI_INT  // flags
    };

    MPI_Aint offsets[NITEMS_HALFEDGE];
    offsets[0] = offsetof(PackedHalfEdge, global_id);
    offsets[1] = offsetof(PackedHalfEdge, vertex_gid);
    offsets[2] = offsetof(PackedHalfEdge, twin_gid);
    offsets[3] = offsetof(PackedHalfEdge, next_gid);
    offsets[4] = offsetof(PackedHalfEdge, triangle_gid);
    offsets[5] = offsetof(PackedHalfEdge, boundary);
    offsets[6] = offsetof(PackedHalfEdge, flags);

    MPI_Type_create_struct(NITEMS_HALFEDGE, blocklengths, offsets, types, &MPI_PackedHalfEdge);
    mftlResizeToSizeof(&MPI_PackedHalfEdge, sizeof(PackedHalfEdge));
  }

  // PackedTriangleComplex: nested record; member displacements from MPI_Get_address so any
  // padding between the members is honoured
  {
#define NITEMS_COMPLEX 2
    int blocklengths[] = {1, 3};
    MPI_Datatype types[] = {MPI_PackedTriangle, MPI_PackedHalfEdge};

    MPI_Aint offsets[NITEMS_COMPLEX];
    PackedTriangleComplex dummy;
    MPI_Aint base_addr, tri_addr, he_addr;

    MPI_Get_address(&dummy, &base_addr);
    MPI_Get_address(&dummy.triangle, &tri_addr);
    MPI_Get_address(&dummy.half_edges, &he_addr);

    offsets[0] = MPI_Aint_diff(tri_addr, base_addr);
    offsets[1] = MPI_Aint_diff(he_addr, base_addr);

    MPI_Type_create_struct(NITEMS_COMPLEX, blocklengths, offsets, types, &MPI_PackedTriangleComplex);
    mftlResizeToSizeof(&MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex));
  }

  // GhostUpdatePayload
  {
#define NITEMS_GHOST 10
    int blocklengths[] = {1, 1, 3, 3, 1, 1, 1, 1, 1, 3};

    MPI_Datatype types[] = {
        MPI_INT,  // global_id
        MPI_INT,  // found
        MPI_DOUBLE,  // pos[3]
        MPI_DOUBLE,  // normal[3]
        MPI_DOUBLE,  // meancurvature
        MPI_DOUBLE,  // gaussiancurvature
        MPI_DOUBLE,  // area
        MPI_DOUBLE,  // roughness_mean
        MPI_DOUBLE,  // roughness_variance
        MPI_INT,  // img[3]
    };

    MPI_Aint offsets[NITEMS_GHOST];
    offsets[0] = offsetof(GhostUpdatePayload, global_id);
    offsets[1] = offsetof(GhostUpdatePayload, found);
    offsets[2] = offsetof(GhostUpdatePayload, pos);
    offsets[3] = offsetof(GhostUpdatePayload, normal);
    offsets[4] = offsetof(GhostUpdatePayload, meancurvature);
    offsets[5] = offsetof(GhostUpdatePayload, gaussiancurvature);
    offsets[6] = offsetof(GhostUpdatePayload, area);
    offsets[7] = offsetof(GhostUpdatePayload, roughness_mean);
    offsets[8] = offsetof(GhostUpdatePayload, roughness_variance);
    offsets[9] = offsetof(GhostUpdatePayload, img);

    MPI_Type_create_struct(NITEMS_GHOST, blocklengths, offsets, types, &MPI_GhostUpdatePayload);
    mftlResizeToSizeof(&MPI_GhostUpdatePayload, sizeof(GhostUpdatePayload));
  }
}

void MFTLFreeMpiTypes()
{
  MPI_Type_free(&MPI_PackedVertex);
  MPI_Type_free(&MPI_PackedTriangle);
  MPI_Type_free(&MPI_PackedHalfEdge);
  MPI_Type_free(&MPI_PackedTriangleComplex);
  MPI_Type_free(&MPI_GhostUpdatePayload);
}
