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
threshold=0.1 # for model trained with p=0.1
image_size=128

w_reverse=-1
num_classes=2

if [ "$modality" == "all" ]; then
    in_channels=4
    num_channels=128
elif [ "$modality" == "flair" ]; then
    in_channels=1
    num_channels=128
fi



master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

# w=1.1
diffusion_steps=1000
version=v2


for model_num in 210000
do
    for w in 1.1 1.5
    do
        for sample_steps in 450
        do
            # export OPENAI_LOGDIR="./logs_samples/eva_${diffusion_steps}_${sample_steps}_${w}_${version}_${model_num}_ddib"
            
            export OPENAI_LOGDIR="./logs_test/eva_${diffusion_steps}_${sample_steps}_${w}_${version}_${model_num}_ddib_mmbrain"
            echo $OPENAI_LOGDIR
            # data_dir="/data/amciilab/yiming/DATA/BraTS21_training/preprocessed_data_${modality}_${suffix}_${image_size}"
            # model_dir="./logs_normal_99_11/logs_guided_${threshold}_${modality}_${suffix}_${version}_${image_size}"
            data_dir="/data/amciilab/yiming/DATA/mmbrain/preprocessed_data_all_00_128"
            model_dir="./trained_weights"
           
            image_dir="$OPENAI_LOGDIR"

            MODEL_FLAGS="--unet_ver $version --image_size $image_size --num_classes $num_classes \
                        --in_channels $in_channels --clf_free True\
                            --w $w  --attention_resolutions 32,16,8\
                            --num_channels $num_channels --model_num $model_num --ema True\
                            --learn_sigma False"

            DATA_FLAGS="--batch_size 100 --num_batches 5\
                        --batch_size_val 100 --num_batches_val 0\
                        --modality 0 3"


            DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps\
                                --sample_steps $sample_steps\
                                --noise_schedule linear\
                                --rescale_learned_sigmas False\
                                --rescale_timesteps False\
                                --dynamic_clip False"

            DIR_FLAGS="--save_data False --data_dir $data_dir\
                        --image_dir $image_dir --model_dir $model_dir"


            NUM_GPUS=1
            # torchrun --nproc-per-node $NUM_GPUS\
            #             --nnodes=1\
            #             --rdzv-backend=c10d\
            #             --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
            #         ./scripts/translation_DDIB.py $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS
            torchrun ./scripts/translation_DDIB.py $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS
        done
    done
done