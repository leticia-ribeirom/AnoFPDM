#!/bin/bash

#SBATCH --job-name='train'
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 01-12:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base


noise_type=simplex
# noise_type=gaussian

in_channels=1
batch_size=64
save_interval=10000

num_classes=0 # unguided
image_size=128

# log directory
export OPENAI_LOGDIR="./logs/logs_atlas_normal_99_11_128/logs_anoddpm_${noise_type}"
# data directory
data_dir="/data/amciilab/yiming/DATA/ATLAS/preprocessed_data_t1_00_128"
image_dir="$OPENAI_LOGDIR/images"

DATA_FLAGS="--image_size $image_size --num_classes $num_classes \
                --class_cond False --ret_lab False --mixed False
                --n_unhealthy_patients 0" # only train on non-tumour slices

MODEL_FLAGS="--unet_ver v1\
            --in_channels $in_channels \
             --num_channels 128 \
             --attention_resolutions 32,16,8 \
             --learn_sigma False\
             --dropout 0"

DIFFUSION_FLAGS="--diffusion_steps 1000\
                --noise_type $noise_type \
                    --noise_schedule linear \
                    --rescale_learned_sigmas False \
                    --rescale_timesteps False"

TRAIN_FLAGS="--data_dir $data_dir --image_dir $image_dir --batch_size $batch_size"


EVA_FLAGS="--save_interval $save_interval --sample_shape 12 $in_channels $image_size $image_size \
            --timestep_respacing ddim1000" # ignore this for non-gaussian noise and we use ddpm sampling for visual checking


# slurm setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR


export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

NUM_GPUS=2
torchrun --nproc-per-node $NUM_GPUS \
         --nnodes=1\
         --rdzv-backend=c10d\
         --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        ./scripts/train.py --name ATLAS \
                            $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $GUI_FLAGS $EVA_FLAGS 



