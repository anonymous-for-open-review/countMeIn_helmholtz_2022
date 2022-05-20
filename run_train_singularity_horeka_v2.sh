#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --gres gpu:4
#SBATCH --partition accelerated
#SBATCH --exclusive

ml --force  purge

data_1='/p/project/hai_countmein/data/So2Sat_POP_Part1'
data_2='/p/project/hai_countmein/data/So2Sat_POP_Part2'
feature_folder='/local_home/trao_ka/projects/helmholtz_challenges/countMeIn/starter-pack/So2Sat_POP_features/'

learning_algo=$1
tuning_method=$2
training_no_feature=$3 


echo "Learning method: " $learning_algo
echo "Tuning method: " $tuning_method
echo "data path1: " $data_1
echo "data path2: " $data_2
echo "Training (and not feature engineering)': " $training_no_feature
echo "feature folder: " $feature_folder

#srun \
singularity run --bind /hkfs /hkfs/home/dataset/datasets/So2Sat_POP/countmein_sklearn_1.0.sif \
python3 starter-pack-v3.py --data_path_So2Sat_pop_part1 $data_1 \
--data_path_So2Sat_pop_part2 $data_2 \
--learning_method $learning_algo  \
--tuning_method $tuning_method \
--seed 10 \
--training_no_engineering $training_no_feature \
--data_path_feature_folder $feature_folder 