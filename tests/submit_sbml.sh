#!/bin/bash
#SBATCH --job-name=SubmitTest$SLURM_JOBID
#SBATCH --time=4:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=tander64@jhu.edu

#### load and unload modules you may need
module load python/3.9.15
module load foss/2023a
module load poetry

#### execute code and write output file to OUT-24log.
poetry env use 3.9
poetry run python clustertest.py $SLURM_JOBID
echo "Finished with job $SLURM_JOBID"
