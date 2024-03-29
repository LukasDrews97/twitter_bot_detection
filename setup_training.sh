#!/bin/bash

#SBATCH --partition training
#SBATCH --gres gpu:3090:1
#SBATCH  -o Experiment_2_5_all_relations.out
#SBATCH --job-name=Experiment_2_5_all_relations
#SBATCH --mail-type ALL
#SBATCH --mail-user xxx@student.xxx.de

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
srun python train.py

echo 'Finished'
exit 0