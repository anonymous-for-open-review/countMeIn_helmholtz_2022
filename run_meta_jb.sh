#!/usr/bin/env bash
# `-x` for detailed Shell debugging


## Build features only


JOBID=$(sbatch --nodes=1 --account=hai_countmein --time=24:00:00--partition=booster --gres=gpu:1 run_train_singularity_jb_v2.sh adaboost grid 0 2>&1 | awk '{print $(NF)}')
echo "Done with: ${JOBID} - feature eng"

sbatch --dependency=singleton --nodes=1 --account=hai_countmein --time=24:00:00--partition=booster --gres=gpu:1 run_train_singularity_jb_v2.sh adaboost grid 1

sbatch --dependency=singleton --nodes=1 --account=hai_countmein --time=24:00:00--partition=booster --gres=gpu:1 run_train_singularity_jb_v2.sh adaboost halving 1

sbatch --dependency=singleton --nodes=1 --account=hai_countmein --time=24:00:00--partition=booster --gres=gpu:1 run_train_singularity_jb_v2.sh gradient_boosting grid 1

sbatch --dependency=singleton --nodes=1 --account=hai_countmein --time=24:00:00--partition=booster --gres=gpu:1 run_train_singularity_jb_v2.sh gradient_boosting halving 1

sbatch --dependency=singleton --nodes=1 --account=hai_countmein --time=24:00:00--partition=booster --gres=gpu:1 run_train_singularity_jb_v2.sh voting grid 1

sbatch --dependency=singleton --nodes=1 --account=hai_countmein --time=24:00:00--partition=booster --gres=gpu:1 run_train_singularity_jb_v2.sh voting halving 1

sbatch --dependency=singleton --nodes=1 --account=hai_countmein --time=24:00:00--partition=booster --gres=gpu:1 run_train_singularity_jb_v2.sh random_forest grid 1

sbatch --dependency=singleton --nodes=1 --account=hai_countmein --time=24:00:00--partition=booster --gres=gpu:1 run_train_singularity_jb_v2.sh random_forest halving 1