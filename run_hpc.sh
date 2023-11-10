#!/usr/bin/bash
# Script for executing the experiment on the HPC

#SBATCH --job-name=adv_nlp_ebv # Job name
#SBATCH --output=simple.out # Name of output file
#SBATCH --cpus-per-task=8 # Schedule one core
#SBATCH --time=16:00:00 # Run time (hh:mm:ss)
#SBATCH --gres=gpu:rtx8000:1 # Request a GPU
#SBATCH --partition=brown
#SBATCH --output=jobLogs/job.%j.out # (%j expands to jobId)
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load Anaconda3
source activate /home/$user/condaenv/advNlp