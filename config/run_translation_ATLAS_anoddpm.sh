#!/bin/bash

#SBATCH --job-name='trans_ano'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH -p htc               
#SBATCH -q public
            
#SBATCH -t 0-04:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base



image_size=128
num_classes=0
in_channels=1
num_channels=128
seed=0 # for data loader only

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

w=-1 # unguided
diffusion_steps=1000

# setup1
# noise_type=gaussian
# # sample_steps=100 # for gaussian noise
# model_num=226100 # model steps
# use_ddpm=False # ddpm or ddim sampling 
# 300 0.32 for ddim; 300 0.43 for ddpm

# setup2
noise_type=simplex
model_num=156100
use_ddpm=True

if [ $use_ddpm = "True" ]
then
    model_name="anoddpm"
else
    model_name="anoddim"
fi


for round in 1
do
    for sample_steps in 200 
    do
        export OPENAI_LOGDIR="./logs_atlas/translation_${model_name}_${noise_type}_${sample_steps}_${round}_plot"
        echo $OPENAI_LOGDIR

        data_dir="/data/amciilab/yiming/DATA/ATLAS/preprocessed_data_t1_00_128" 
        model_dir="./logs/logs_atlas_normal_99_11_128/logs_anoddpm_${noise_type}"
        
        image_dir="$OPENAI_LOGDIR"

        MODEL_FLAGS="--unet_ver v1 --image_size $image_size --num_classes $num_classes \
                        --in_channels $in_channels --clf_free False\
                        --dropout 0 --attention_resolutions 32,16,8\
                        --num_channels $num_channels --model_num $model_num --ema True\
                        --use_ddpm $use_ddpm --noise_type $noise_type"

        DATA_FLAGS="--batch_size 100 --num_batches 5\
                    --batch_size_val 100 --num_batches_val 0\
                    --modality 0 --seed $seed --use_weighted_sampler False"


        DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps\
                            --sample_steps $sample_steps\
                            --noise_schedule linear\
                            --rescale_learned_sigmas False\
                            --rescale_timesteps False\
                            --dynamic_clip False"

        DIR_FLAGS="--save_data True --data_dir $data_dir\
                    --image_dir $image_dir --model_dir $model_dir"


        NUM_GPUS=1
        torchrun --nproc-per-node $NUM_GPUS\
                    --nnodes=1\
                    --rdzv-backend=c10d\
                    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
                ./scripts/translation_HEALTHY.py --name ATLAS $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS
    done
done