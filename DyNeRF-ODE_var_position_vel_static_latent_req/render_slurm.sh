#!/bin/sh
 
#SBATCH --job-name=JOBNAME
#SBATCH --output=JOBNAME-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=JOBNAME-%A.err  # Standard error of the script
#SBATCH --time=1-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=48G
#SBATCH --nodelist=mandalore
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/guests/hiran_sarkar/envs/nerf_ode
python render.py --config configs/config.txt