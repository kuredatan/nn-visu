#!/bin/bash

## 1: model
## 2: number of tries
## 3: method
#git clone https://github.com/kuredatan/nn-visu.git
cd nn-visu/
ls
git pull
#conda create -n py36 python=3.6 anaconda
conda activate py36
python3.6 final_pipeline.py --tmodel $1 --tmethod $3 --ntry $2 --py 3.6
conda deactivate
exit
