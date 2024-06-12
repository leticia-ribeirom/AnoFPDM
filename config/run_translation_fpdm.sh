#!/bin/bash

#SBATCH --job-name='translation_test'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 00-05:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base


modality=all
suffix=00
threshold=0.1

w_reverse=-1

num_classes=2

seed=0


in_channels=4
num_channels=128


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

# w=2
image_size=128
rev_steps=600
diffusion_steps=1000
# model_num=210000
version=v2


tuned=False # set tuned to true to use hyperparameter tuned by pixel-level label 
            # (paras are already obtained not tuned in this code)

d_reverse=True # set d_reverse to True for ddim reverse (deterministic encoding) 
                # or will be ddpm reverse (stochastic encoding)

w=2
for model_num in 210000 230000 250000 270000 290000 310000
do
    
    export OPENAI_LOGDIR="./logs_test/eva_${diffusion_steps}_${rev_steps}_${w}_${modality}_${model_num}_FPDM_clamp"
    echo $OPENAI_LOGDIR

    data_dir="/data/amciilab/yiming/DATA/BraTS21_training/preprocessed_data_${modality}_${suffix}_${image_size}"
    # data_dir="/data/amciilab/yiming/DATA/mmbrain/preprocessed_data_all_00_128"
    
    # model_dir="./trained_weights/clf-free"
    model_dir="./logs/logs_normal_99_11/logs_guided_0.1_all_00_v2_128_clamp"

    image_dir="$OPENAI_LOGDIR"

    MODEL_FLAGS="--image_size $image_size --num_classes $num_classes --in_channels $in_channels  \
                    --w $w --attention_resolutions 32,16,8 \
                    --num_channels $num_channels --model_num $model_num --ema True\
                    --rev_steps $rev_steps --d_reverse $d_reverse --tuned $tuned" 

    DATA_FLAGS="--batch_size 100 --num_batches 10 \
                --batch_size_val 100 --num_batches_val 10\
                --modality 0 3"

    DIFFUSION_FLAGS="--null True \
                        --dynamic_clip False \
                        --diffusion_steps $diffusion_steps \
                        --noise_schedule linear \
                        --rescale_learned_sigmas False --rescale_timesteps False\
                        --seed $seed"

    DIR_FLAGS="--save_data False --data_dir $data_dir  --image_dir $image_dir --model_dir $model_dir"

    ABLATION_FLAGS="--last_only False --subset_interval -1"


    NUM_GPUS=1
    # torchrun --nproc-per-node $NUM_GPUS \
    #             --nnodes=1\
    #             --rdzv-backend=c10d\
    #             --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
    #         ./scripts/translation_FPDM.py $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $ABLATION_FLAGS

    torchrun ./scripts/translation_FPDM.py $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $ABLATION_FLAGS
done