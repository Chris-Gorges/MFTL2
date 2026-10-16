/* Bounded-memory initialization for the scaling harness. Included by
 * scaling_common.h after its local mesh checks and configuration types. */
#ifndef MFTL_SCALING_INIT_H
#define MFTL_SCALING_INIT_H

typedef struct
{
  double generation_s;
  double scatter_s;
  double wall_s;
  double reference[3];
  long rss_generation;
  long rss_scatter;
  long batch_pool_peak;
  int batches;
  unsigned long long signature;
} sc_setup;

/* Optional order-independent checksum of owned entities. Global IDs, exact
 * vertex coordinates and triangle connectivity must survive batching. */
static unsigned long long sc_init_signature(const MFTL_Ensemble *ens, int body_offset)
{
  uint64_t signature = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      uint64_t h = sc_seed(obj->global_id + body_offset, (uint64_t) obj->verts.rec[v].global_id);
      for (int d = 0; d < 3; ++d)
      {
        uint64_t bits;
        memcpy(&bits, &obj->verts.pos[3 * v + d], sizeof bits);
        h ^= bits;
        h = sc_splitmix64(&h);
      }
      signature ^= h;
    }
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      uint64_t h = sc_seed(obj->global_id + body_offset, (uint64_t) obj->tris.rec[t].global_id ^ UINT64_C(0xD1B54A32D192ED03));
      for (int d = 0; d < 3; ++d)
      {
        h ^= (uint64_t) obj->verts.rec[obj->tris.verts[3 * t + d]].global_id;
        h = sc_splitmix64(&h);
      }
      signature ^= h;
    }
  }
  return (unsigned long long) signature;
}

/* Only NEW spheres enter the staging ensemble. Redistributing the growing
 * final ensemble after every batch would repeatedly revisit all earlier mesh.
 * The temporary header borrows the final domain's communicators; we explicitly
 * free only its directory and workspace, never its borrowed domain resources.
 * Each batch has a fresh dense ID space; after scattering, offset its IDs and
 * append the local pieces to the final directory in ascending order.
 *
 * Release workspace after each batch: persistent per-destination send buffers
 * would otherwise accumulate their historical maxima over ALL destinations. */
static sc_setup sc_initialize(MFTL_Ensemble *ens, const sc_problem *prob, const sc_options *opt)
{
  sc_setup setup;
  memset(&setup, 0, sizeof setup);
  double reference_local[3] = {0.0, 0.0, 0.0}, area_correction = 0.0;
  unsigned long long expected_signature = 0;
  const int rank = ens->domain.rank;
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  for (int first = 0; first < prob->n_bodies;)
  {
    const int count = MFTL_MIN(opt->init_batch, prob->n_bodies - first);
    MFTL_Ensemble stage;
    memset(&stage, 0, sizeof stage);
    stage.domain = ens->domain;  // borrowed, identical geometry and rank numbering
    stage.capacity_objects = count;
    stage.local_objects = (MFTL_Object **) MFTLMallocOrDie((size_t) count * sizeof(MFTL_Object *), "scaling initialization batch");
    double begin = MPI_Wtime();
    for (int b = 0; b < count; ++b)
    {
      MFTL_Object *obj = MFTLObjectCreate(b);
      sc_require(MFTLEnsembleAddObject(&stage, obj), "add staged body");
      double centre[3];
      prob->centre(prob->centre_ctx, first + b, centre);
      sc_require(MFTLGenerateSphereLocal(obj, centre, prob->radius, opt->refine), "generate staged sphere");
    }
    sc_require(MFTLPublishGeneratedBodies(&stage), "publish staged bodies");
    double local[3];
    sc_local_totals(&stage, local);
    const double y = local[0] - area_correction;
    const double sum = reference_local[0] + y;
    area_correction = (sum - reference_local[0]) - y;
    reference_local[0] = sum;
    reference_local[1] += local[1];
    reference_local[2] += local[2];
    if (opt->verify_init) expected_signature ^= sc_init_signature(&stage, first);
    long pool = 0;
    for (int o = 0; o < stage.n_local_objects; ++o) pool += (long) MFTLObjectPoolBytes(stage.local_objects[o]);
    setup.batch_pool_peak = MFTL_MAX(setup.batch_pool_peak, pool);
    setup.rss_generation = MFTL_MAX(setup.rss_generation, sc_current_rss());
    setup.generation_s += MPI_Wtime() - begin;

    begin = MPI_Wtime();
    sc_require(MFTLRedistribute(&stage), "scatter initialization batch");
    setup.rss_scatter = MFTL_MAX(setup.rss_scatter, sc_current_rss());
    for (int o = 0; o < stage.n_local_objects; ++o)
    {
      MFTL_Object *obj = stage.local_objects[o];
      obj->global_id += first;
      obj->domain = &ens->domain;
      sc_require(MFTLEnsembleInsertLocalObject(ens, obj), "append distributed body");
    }
    MFTLCommWorkspaceDestroy(&stage.comm_workspace);
    free(stage.local_objects);  // bodies now belong to ens, do not destroy them
    setup.scatter_s += MPI_Wtime() - begin;
    ++setup.batches;
    first += count;
    if (rank == 0 && (setup.batches % 100 == 0 || first == prob->n_bodies))
    {
      printf("   initialized %d/%d spheres in %d batches (%.1f s)\n", first, prob->n_bodies, setup.batches, MPI_Wtime() - start);
      fflush(stdout);
    }
  }
  ens->n_bodies_global = prob->n_bodies;
  MPI_Allreduce(reference_local, setup.reference, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (opt->verify_init)
  {
    const unsigned long long actual = sc_init_signature(ens, 0);
    unsigned long long expected;
    MPI_Allreduce(&expected_signature, &expected, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);
    MPI_Allreduce(&actual, &setup.signature, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);
    if (setup.signature != expected)
    {
      if (rank == 0) fprintf(stderr, "initialization changed body/entity IDs, positions or connectivity\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  const double timing[2] = {setup.generation_s, setup.scatter_s};
  double maximum[2];
  MPI_Allreduce(timing, maximum, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  setup.generation_s = maximum[0];
  setup.scatter_s = maximum[1];
  setup.wall_s = MPI_Wtime() - start;
  return setup;
}

#endif
