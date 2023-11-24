#!/bin/bash
#SBATCH --job-name=HelloHPC # Job name
#SBATCH --output=simple.out # Name of output file
#SBATCH --cpus-per-task=8 # Schedule one core
#SBATCH --time=24:00:00 # Run time (hh:mm:ss)
#SBATCH --gres=gpu:rtx8000:1 # Request a GPU
#SBATCH --partition=brown
#SBATCH --output=job.%j.out # (%j expands to jobId)
#SBATCH --mail-type=FAIL

# Queue to HPC `sbatch ./run_hpc.sh`; view queue `squeue`

module purge
conda init
conda activate /home/$USER/.conda/envs/hackathon

echo "Starting python"
python3 main.py --args_path conf/args.yaml
echo "Done ./run_hpc.sh"