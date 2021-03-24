#!/usr/bin/env bash
#SBATCH --partition=batch_default
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
###SBATCH --gres=gpu:titanrtx:1
###SBATCH --gres=gpu:rtx6000:1
###SBATCH --gres=gpu:a100:1
###SBATCH --gres=gpu:v100:1
###SBATCH --gres=gpu:rtx2080ti:1
###SBATCH --nodelist=hopper
###SBATCH --nodelist=euler56
#SBATCH --exclude=euler50,euler54
#SBATCH --time=96:0:0
#SBATCH -o slurm.%j.%N.out # STDOUT
#SBATCH -e slurm.%j.%N.err # STDERR
#SBATCH --job-name=monodepth


module load anaconda/3
bootstrap_conda
conda activate minienv

which python
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
top -b -d1 -n1 | grep -i "%Cpu" #This will show cpu utilization at the start of the script

#LOG_FILE=train.txt
#python -u ../../../train.py --data-folder /nobackup/nyuv2/data_average1/ --bs 20 --epochs 20 --num-instance 5 --label-file nyu2_train_4frame+cl256_updated.csv --lamb 0.001 --pretrained-weights /u/b/h/bhavya/Documents/quantavisioninference/experiments/imagenet/resnet34/a256_Xframe/a256_1frame/models/model_best.pth.tar  2>&1 | tee $LOG_FILE 
#python -u ../../../train.py --data-folder /nobackup/nyuv2/data_average256/ --bs 4 --epochs 20 --label-file nyu2_train_updated.csv --pretrained-weights /u/b/h/bhavya/Documents/quantavisioninference/experiments/imagenet/resnet34/a256_Xframe/a256_1frame/models/model_best.pth.tar 2>&1 | tee $LOG_FILE 
LOG_FILE=val.txt
python -u ../../../train.py --data-folder /nobackup/nyuv2/data_average10/  --evaluate models_19.pth.tar 2>&1 | tee $LOG_FILE 

