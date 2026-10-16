#!/bin/bash
#SBATCH --job-name=a_ro_0_75
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=2400:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

exec 2>&1     # send errors into stdout stream
echo "DEBUG: SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES=$SLURM_NNODES"
echo "DEBUG: SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE"
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID # show slurm-command and more for DBGmodule load gnu8

module purge
module load os mvapich2


export CASE_DIR=$PWD
HOSTFILE=${CASE_DIR}/slurm-$SLURM_JOBID.hosts

echo "check for free space on /dev/shm (needed for MPI jobs)" # 2018-01-30
# start one check process per node, independend from ntasks-per-node
# srun within sbatch can not use --ntasks-per-node=1, see slurm doc
srun -l bash -c "[ \$SLURM_LOCALID != 0 ] || \
 (df -k /dev/shm | tail -1;uname -n)|xargs" |sort -k 4n|tail -2
# SLURM_LOCALID is the task-id per node 0..15
echo "DEBUG: LOADEDMODULES=$LOADEDMODULES" # module listHOSTFILE=${RUN_DIR}/slurm-$SLURM_JOBID.hosts
scontrol show hostnames $SLURM_JOB_NODELIST > $HOSTFILE # one entry per host


mpirun build/MF3 -o examples/DEM/DEM-Drum/DEM-Drum-a_ro_0_75_101.mf -d results-a_ro_0_75_101 > outfile-a_ro_0_75_101


