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
image_size=128
version=v1
num_classes=2

if [ "$modality" == "all" ]; then
    in_channels=4
elif [ "$modality" == "flair" ]; then
    in_channels=1
fi


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT


diffusion_steps=1000
model_num=500000
clf_num=149999

for round in 1
do
    for sample_steps in 200
    do  
        for classifier_scale in 1000
        do
            # export OPENAI_LOGDIR="./logs_samples/eva_${diffusion_steps}_${sample_steps}_${classifier_scale}_${model_num}_clf_${round}"
            export OPENAI_LOGDIR="./logs_test/eva_${diffusion_steps}_${sample_steps}_${classifier_scale}_${model_num}_clf_${round}"
            echo $OPENAI_LOGDIR
            
            # data_dir="/data/amciilab/yiming/DATA/BraTS21_training/preprocessed_data_${modality}_${suffix}_${image_size}"
            # model_dir="./logs_normal_99_11/logs_clf_guided_${modality}_${suffix}_${version}_${image_size}"
            # classifier_dir="./logs_normal_99_11/logs_clf_${version}_${image_size}"

            data_dir="/data/amciilab/yiming/DATA/mmbrain/preprocessed_data_all_00_128"
            model_dir="./trained_weights/clf-guided"
            classifier_dir="./trained_weights/clf"

            image_dir="$OPENAI_LOGDIR"

            MODEL_FLAGS="--unet_ver $version --image_size $image_size --clf_free False\
                        --num_classes $num_classes --in_channels $in_channels\
                        --classifier_scale $classifier_scale  --dropout 0
                        --attention_resolutions 32,16,8 --learn_sigma True\
                        --num_channels 128 --model_num $model_num --ema True"

            
            CLASSIFIER_FLAGS="--clf_dir $classifier_dir --clf_num $clf_num\
                                --classifier_attention_resolutions 32,16,8\
                                --out_channels $num_classes\
                                --classifier_depth 2 --classifier_width 128\
                                --classifier_pool attention\
                                --classifier_resblock_updown True\
                                --classifier_use_scale_shift_norm True"

            DATA_FLAGS="--batch_size 100 --num_batches 5\
                        --batch_size_val 100 --num_batches_val 0\
                        --modality 0 3"


            DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps\
                                --sample_steps $sample_steps\
                                --noise_schedule linear\
                                --rescale_learned_sigmas False\
                                --rescale_timesteps False"

            DIR_FLAGS="--save_data False --data_dir $data_dir\
                        --image_dir $image_dir --model_dir $model_dir"


            NUM_GPUS=1
            torchrun --nproc-per-node $NUM_GPUS \
                        --nnodes=1\
                        --rdzv-backend=c10d\
                        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
                    ./scripts/translation_CLF.py $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $DATA_FLAGS $CLASSIFIER_FLAGS
        done
    done
done