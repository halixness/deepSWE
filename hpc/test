#!/bin/bash
#SBATCH --job-name=SWE_benchmark
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --time=23:00:00

# al piu' 128gb di ram per gpu in media
# con il comando 'seff' controllo efficienza memoria

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh" 
conda activate swe-cv-pytorch

#python -u keras_3dconv.py | tee output # -a in caso di append al file
# man tee
# python -u 3dconv_deconv.py > train.out

cd ..
python -u test.py -r ../datasets/baganza/ -npy ../datasets/arda.npy -weights runs/train_45_10_08_2021_19_28_41/model.weights -ls 2048 
