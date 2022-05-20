 #!/usr/bin/env bash
# `bash -x` for detailed Shell debugging


## Build features only
run_train_singularity_jb_v2.sh adaboost grid 0

## Build train following ...
run_train_singularity_jb_v2.sh adaboost grid 1
run_train_singularity_jb_v2.sh adaboost halving 1

run_train_singularity_jb_v2.sh gradient_boosting grid 1
run_train_singularity_jb_v2.sh gradient_boosting halving 1

run_train_singularity_jb_v2.sh random_forest grid 1
run_train_singularity_jb_v2.sh random_forest halving 1

run_train_singularity_jb_v2.sh voting grid 1
run_train_singularity_jb_v2.sh voting halving 1


   