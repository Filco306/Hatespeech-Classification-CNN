#!/bin/bash

reg_vals=(0.1 0.001 0.05 0.01 0.5)
act_funcs=("relu" "tanh")
for reg_val in "${reg_vals[@]}"
do
  for activation_function in "${act_funcs[@]}"
  do
    echo "Trying with regval = $reg_val and activationfunction = $activation_function"
    python cnn.py -regval $reg_val -activationfunction $activation_function
  done
done
