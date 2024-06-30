#!/bin/bash

#SBATCH --job-name='preprocess'
#SBATCH --nodes=1                       
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 0-04:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base


source='/data' # path to the raw data
python ./scripts/preprocess.py --source $source



