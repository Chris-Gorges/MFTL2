# Site configuration for the MFTL scaling sweeps -- THE ONLY FILE TO EDIT FOR A CLUSTER.
#
# Sourced by launch_strong.sh / launch_weak.sh on the login node and by
# job_scaling.sh on the compute nodes, so both sides agree on it.
# shellcheck shell=bash

# ---- Machine -----------------------------------------------------------------
CORES_PER_NODE="${CORES_PER_NODE:-36}"
MAX_NODES="${MAX_NODES:-30}"   # points needing more nodes are skipped, not queued forever

# ---- Slurm -------------------------------------------------------------------
# Extra sbatch flags: partition, account, QoS, constraint ... e.g.
#   SBATCH_EXTRA="--partition=compute --account=myproj"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

# Whole nodes for every point (1 = --exclusive). A partly filled node shared
# with another job is the commonest way to get a scaling curve that measures
# somebody else's memory bandwidth. Keep it on.
EXCLUSIVE="${EXCLUSIVE:-1}"
# --exclusive allocates CPUs, not all RAM. Allow room for mesh and MPI workspace.
# 0 requests all Slurm-configured node memory; use e.g. 110G if required locally.
MEMORY_PER_NODE="${MEMORY_PER_NODE:-0}"

# ---- Software ----------------------------------------------------------------
# The same modules must be loaded to BUILD the library (python3 build_all.py
# --comm-profile) and to RUN it, or CMake finds a different MPI than mpirun uses.
site_modules()
{
  if command -v module >/dev/null 2>&1; then
    module purge
    module load os mvapich2
  fi
}

# MPI launcher; the rank count is appended. MVAPICH2's mpirun (Hydra) reads the
# Slurm allocation. If MVAPICH2 was built with Slurm PMI support, srun also works:
#   LAUNCHER=(srun --mpi=pmi2 -n)
LAUNCHER=(mpirun -np)

# Runtime environment of every point.
site_runtime_env()
{
  export OMP_NUM_THREADS=1
  # MVAPICH2: one rank per core, spread over both sockets when a node is only
  # partly filled (np < 36), and print the binding into the job log so the
  # placement can be checked afterwards. Harmless under another MPI.
  export MV2_ENABLE_AFFINITY=1
  export MV2_CPU_BINDING_POLICY=scatter
  export MV2_CPU_BINDING_LEVEL=core
  export MV2_SHOW_CPU_BINDING=1
  # Optional MPI-level profiler for every point, e.g. mpiP:
  #   export LD_PRELOAD=/path/to/libmpiP.so MPIP="-f ."
}
