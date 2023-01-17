#!/bin/bash

#SBATCH --partition training
#SBATCH --gres gpu:1
#SBATCH -o training.out
#SBATCH --mail-type ALL
#SBATCH --mail-user Lukas.Drews@student.hpi.de

export PYTHONUNBUFFERED='x'

# create venv
python -m venv env
source env/bin/activate
pip install --upgrade setuptools
pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu111.html
pip install torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install transformers
pip install torch-geometric-temporal
pip install torchmetrics
pip install notebook
pip install matplotlib
pip install pandas

# run training
srun python train_with_batches.py

echo 'Finished'