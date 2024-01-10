#!/bin/bash
set -e

# Set the dataset and models
dataset="set-mnist-100"
models=("EinsumNet" "RatSPN")

rm -r ../experiments/$dataset

# Set the number of trials
trials=3

# Loop over the models
for model in "${models[@]}"; do
    # Loop over the trials
    for ((trial=1; trial<=$trials; trial++)); do
        echo "Running trial $trial for model $model"

        # Run the train_ucpc.py script
        CUDA_VISIBLE_DEVICES=0 python train_ucpc.py --config_file=configs/$dataset/$model"_unconstrained.py" &
        CUDA_VISIBLE_DEVICES=1 python train_ucpc.py --config_file=configs/$dataset/$model"_constrained.py"

        echo "Trial $trial for model $model completed"
        echo
    done
done

