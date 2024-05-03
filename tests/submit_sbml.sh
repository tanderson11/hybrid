#!/bin/bash
#SBATCH --job-name=SBMLTest
#SBATCH --time=4:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=end
#SBATCH --mail-user=tander64@jhu.edu

#### load and unload modules
module purge
module load gcc/11.4.0
module load python/3.9.15
module load foss/2022b
module load poetry

#### execute code
poetry env use 3.9
echo $SLURM_JOBID
echo $SLURM_ARRAY_JOB_ID
echo $SLURM_ARRAY_TASK_ID
pwd
#poetry run which python
poetry run python -u clustertest.py $SLURM_ARRAY_TASK_ID
echo "Finished with job $SLURM_JOBID"
