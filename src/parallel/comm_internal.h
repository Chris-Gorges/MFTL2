/*
 * MFTL2 parallel (internal) — declarations shared by the transport sources of src/parallel/
 * (redistribute.c, ghost_refresh.c, ghost_layers.c): the per-rank index list, the
 * variable-length buffer exchange and the payload-size lookup. Not part of the public API and
 * not installed.
 */
#ifndef MFTL_COMM_INTERNAL_H
#define MFTL_COMM_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_memory.h"

// growable int list (index lists per destination rank)
typedef struct
{
  int* data;
  int size;
  int capacity;
} IntBuffer;

void mftlBufferInit(IntBuffer* buf, int initial_cap);

static inline void mftlBufferAdd(IntBuffer* buf, int val)
{
  MFTL_ENSURE_CAPACITY(buf->data, &buf->size, &buf->capacity, 1, int);
  buf->data[buf->size++] = val;
}

// empties the list, keeps the allocation
static inline void mftlBufferReset(IntBuffer* buf) { buf->size = 0; }

/**
 * @brief Payload size (data_size) of the object with the given body id (linear search over
 * the objects, as the object lookups of the redistribution rebuild passes).
 */
static inline size_t mftlDataSizeOfBody(MFTL_Object** objects, int num_objects, int body_id)
{
  for (int o = 0; o < num_objects; ++o)
  {
    if (objects[o]->global_id == body_id) return objects[o]->data_size;
  }
  return 0;
}

// Variable-length per-rank buffer exchange (redistribute.c).
void mftlMpiExchangeBuffers(MFTL_Domain* domain, MFTL_SendBuffer* send_bufs, MPI_Datatype type, size_t type_size, bool use_global_path,
                            void** out_recv_buf, int* out_recv_count);

#endif  // MFTL_COMM_INTERNAL_H
