# Script performing the grid search. 

#!/bin/bash
pre_process_data_gaydhani=("True" "False")

for pre in "${pre_process_data_gaydhani[@]}"
do
  python logistic_regression.py -preprocess $pre
done

reg_vals=(0.1 0.001 0.01 1)
act_funcs=("relu")
emb_dims=(25 50 100 200)
use_pretrained_vecs=("True" "False")

for use_pretrained_vec in "${use_pretrained_vecs[@]}"
do
  for emb_dim in "${emb_dims[@]}"
  do
    for reg_val in "${reg_vals[@]}"
    do
      for activation_function in "${act_funcs[@]}"
      do
        echo "Trying with regval = $reg_val and activationfunction = $activation_function"
        python cnn.py -regval $reg_val -activationfunction $activation_function -emb_dim $emb_dim -use_pretrained_vecs $use_pretrained_vec
      done
    done
  done
done
