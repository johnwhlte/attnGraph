#!/bin/bash
DATA="${1-baffle}"
CONFIG=configs/${DATA}
source configs/${DATA}

run() {

    python src/main_nets.py \
    -batch_size $batch_size \
    -data_path $data_path \
    -num_epochs $num_epochs \
    -device $device \
    -lr $lr \
    -num_snaps $num_snaps \
    -funcs_list $funcs_list \
    -alpha_vals $alpha_vals \
    -early_stop $early_stop \
    -seed $seed \
    -feat_dim $feat_dim \
    -in_dim $in_dim \
    -out_dim $out_dim \
    -pred_dim $pred_dim \
    -n_attn $n_attn \
    -n_convs $n_convs

}


    run