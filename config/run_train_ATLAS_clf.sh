#!/bin/bash

#SBATCH --job-name='clf'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 00-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base

num_classes=2
image_size=128
version=v1
in_channels=1


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

export OPENAI_LOGDIR="./logs/logs_atlas_clf"
echo $OPENAI_LOGDIR

image_dir="$OPENAI_LOGDIR/images"

data_dir="/data/amciilab/yiming/DATA/ATLAS/preprocessed_data_t1_00_128"


CLASSIFIER_FLAGS="--unet_ver $version --image_size $image_size --classifier_attention_resolutions 32,16,8 \
                --in_channels $in_channels --out_channels $num_classes \
                --classifier_depth 2 --classifier_width 128 \
                --classifier_pool attention \
                --classifier_resblock_updown True --classifier_use_scale_shift_norm True\
                --data_dir $data_dir --image_dir $image_dir --batch_size 64 --iterations 250000"

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS \
         --nnodes=1\
         --rdzv-backend=c10d\
         --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        ./scripts/train_classifier.py --name atlas --save_interval 5000\
                                    $CLASSIFIER_FLAGS