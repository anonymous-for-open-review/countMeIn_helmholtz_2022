#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=hai_countmein

ml --force  purge

module load  Stages/2022  GCCcore/.11.2.0 Singularity-Tools/2022

# srun python train.py 
#srun singularity run /p/project/hai_countmein/countmein_sklearn_1.0.sif  python3 starter-pack.py --data_path_So2Sat_pop_part1 '/p/project/hai_countmein/data/So2Sat_POP_Part1' --data_path_So2Sat_pop_part2 '/p/project/hai_countmein/data/So2Sat_POP_Part2'


data_1='/p/project/hai_countmein/data/So2Sat_POP_Part1'
data_2='/p/project/hai_countmein/data/So2Sat_POP_Part2'

learning_algo=$1
tuning_method=$2

echo "Learning method: " $learning_algo
echo "Tuning method: " $tuning_method
echo "data path1 : " $data_1
echo "data path2 : " $data_2

srun singularity run /p/project/hai_countmein/countmein_sklearn_1.0.sif  python3 starter-pack-v2.py --data_path_So2Sat_pop_part1  $data_1 --data_path_So2Sat_pop_part2 $data_2 --learning_method $learning_algo --tuning_method $tuning_method --seed 10