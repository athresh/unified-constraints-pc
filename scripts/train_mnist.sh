#!/bin/bash


# Set the dataset and models
dataset="set-mnist-100"
models=("EinsumNet" "RatSPN")

rm -r ../experiments/$dataset
mkdir ../console/$dataset
# Set the number of trials
trials=3
set -e
# # Loop over the models
# for model in "${models[@]}"; do
#     # Loop over the trials
#     for ((trial=1; trial<=$trials; trial++)); do
#         echo "Running trial $trial for model $model"

#         # Run the train_ucpc.py script
#         CUDA_VISIBLE_DEVICES=0 python train_ucpc.py --config_file=configs/$dataset/$model"_unconstrained.py" > console/$dataset/$model"_unconstrained.txt" &
#         CUDA_VISIBLE_DEVICES=1 python train_ucpc.py --config_file=configs/$dataset/$model"_constrained.py" > console/$dataset/$model"_constrained.txt"

#         echo "Trial $trial for model $model completed"
#         echo
#     done
# done


# Loop over the trials
for ((trial=1; trial<=$trials; trial++)); do
    echo "Running trial $trial for model $model"

    # CUDA_VISIBLE_DEVICES=0 python train_ucpc.py --config_file=configs/$dataset/"RatSPN_unconstrained.py" > ../console/$dataset/"RatSPN_unconstrained.txt" &
    # CUDA_VISIBLE_DEVICES=1 python train_ucpc.py --config_file=configs/$dataset/"EinsumNet_unconstrained.py" > ../console/$dataset/"EinsumNet_unconstrained.txt" 

    CUDA_VISIBLE_DEVICES=0 python train_ucpc.py --config_file=configs/$dataset/"RatSPN_constrained.py" > ../console/$dataset/"RatSPN_constrained.txt" &
    CUDA_VISIBLE_DEVICES=1 python train_ucpc.py --config_file=configs/$dataset/"EinsumNet_constrained.py" > ../console/$dataset/"EinsumNet_constrained.txt" 
    
    echo "Trial $trial for model $model completed"
    echo
done

