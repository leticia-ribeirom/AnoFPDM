#!/bin/bash

#SBATCH --job-name='translation_test'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -p general                
#SBATCH -q grp_twu02 
            
#SBATCH -t 01-00:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT


threshold=0.1
num_classes=2
seed=0 # for data loader only
in_channels=4
num_channels=128
image_size=128
rev_steps=600
diffusion_steps=1000
model_num=210000
version=v2


d_reverse=True # set d_reverse to True for ddim reverse (deterministic encoding) 
                # or will be ddpm reverse (stochastic encoding)
for w in 2 
do
    
    export OPENAI_LOGDIR="./logs_brats/translation_fpdm_${w}_${model_num}_${rev_steps}_last"
    echo $OPENAI_LOGDIR

    data_dir="/data/amciilab/yiming/DATA/BraTS21_training/preprocessed_data_all_00_128"
    model_dir="./logs/logs_brats_normal_99_11_128/logs_guided_${threshold}_all_00_${version}_128_norm"

    image_dir="$OPENAI_LOGDIR"

    MODEL_FLAGS="--image_size $image_size --num_classes $num_classes --in_channels $in_channels  \
                    --w $w --attention_resolutions 32,16,8 \
                    --num_channels $num_channels --model_num $model_num --ema True\
                    --rev_steps $rev_steps --d_reverse $d_reverse" 

    DATA_FLAGS="--batch_size 100 --num_batches 2 \
                --batch_size_val 100 --num_batches_val 0\
                --modality 0 3 --use_weighted_sampler False --seed $seed"

    DIFFUSION_FLAGS="--null True \
                        --dynamic_clip False \
                        --diffusion_steps $diffusion_steps \
                        --noise_schedule linear \
                        --rescale_learned_sigmas False --rescale_timesteps False"

    DIR_FLAGS="--save_data False --data_dir $data_dir  --image_dir $image_dir --model_dir $model_dir"

    ABLATION_FLAGS="--last_only True --subset_interval -1 --t_e_ratio 1"


    NUM_GPUS=1
    # torchrun --nproc-per-node $NUM_GPUS \
    #             --nnodes=1\
    #             --rdzv-backend=c10d\
    #             --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
    #         ./scripts/translation_FPDM.py --name brats $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $ABLATION_FLAGS

    torchrun --nproc-per-node $NUM_GPUS \
            ./scripts/translation_FPDM.py --name brats $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $ABLATION_FLAGS

done