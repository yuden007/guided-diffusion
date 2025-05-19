#!/bin/bash
#SBATCH --job-name=sunflower-horse-classifier
#SBATCH --mail-user=1155172258@link.cuhk.edu.hk
#SBATCH --mail-type=ALL
##SBATCH --output=output/log/%x_%j.out            #    Standard output log as $job_name_$job_id.out
##SBATCH --error=output/log/%x_%j.err             #    Standard error log as $job_name_$job_id.err
#SBATCH --gres=gpu:1

## Below is the commands to run , for this example,
## Create a sample helloworld.py and Run the sample python file
## Result are stored at your defined --output location

source /research/d2/fyp24/yhlin2/miniconda3/etc/profile.d/conda.sh
conda activate env3.10
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
python scripts/classifier_train.py $TRAIN_FLAGS $CLASSIFIER_FLAGS --data_dir ../denoising-diffusion/input/50pinkredflower-150horse