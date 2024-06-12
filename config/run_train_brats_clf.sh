#!/bin/bash

#SBATCH --job-name='clf'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 01-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base


modality=all
suffix=00
num_classes=2
image_size=128
version=v1

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


export OPENAI_LOGDIR="./logs_normal_99_11/logs_clf_${version}_${image_size}"
echo $OPENAI_LOGDIR

image_dir="$OPENAI_LOGDIR/images"

data_dir="/data/amciilab/yiming/DATA/BraTS21_training/preprocessed_data_${modality}_${suffix}_${image_size}"


CLASSIFIER_FLAGS="--unet_ver $version --image_size $image_size --classifier_attention_resolutions 32,16,8 \
                --in_channels $in_channels --out_channels $num_classes \
                --classifier_depth 2 --classifier_width 128 \
                --classifier_pool attention \
                --classifier_resblock_updown True --classifier_use_scale_shift_norm True\
                --data_dir $data_dir --image_dir $image_dir --batch_size 64 --iterations 250000"

resume_checkpoint="$OPENAI_LOGDIR/model149999.pt"
NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS \
         --nnodes=1\
         --rdzv-backend=c10d\
         --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        ./scripts/train_classifier.py --name brats --save_interval 10000\
                                    $CLASSIFIER_FLAGS