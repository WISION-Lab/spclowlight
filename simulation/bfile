#!/usr/bin/env bash
#SBATCH --partition=research
#SBATCH --cpus-per-task=24
#SBATCH --time=100:0:0
#SBATCH -o log/slurm.%j.%N.out # STDOUT
#SBATCH -e log/slurm.%j.%N.err # STDERR
#SBATCH --job-name=simdata

module load anaconda/mini/4.9.2
module load nvidia/cuda/11.3.1
bootstrap_conda
conda activate minienv

which python
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
top -b -d1 -n1 | grep -i "%Cpu" #This will show cpu utilization at the start of the script
date

#START=6000
#TOTAL=3000
#EACH=300
#SPLITS=$TOTAL/$EACH
#for (( i = 0; i < $SPLITS; i++ ))
#do
#	BEGIN=$(($i*$EACH+$START))
#	echo $BEGIN
#	python -u create_data.py $BEGIN $(($BEGIN+$EACH)) &
#done

python -u create_data.py $1 $2

wait
