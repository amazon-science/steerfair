#!/bin/bash
cd ../../social_bias_dataset

split="test"
result_dir="../../vgg2/generalization/math_arts/qr_test"
output_dir="../../vgg2/generalization/math_arts/qr_test_outputs"
echo $result_dir
find_str="${split}"
for entry in ls "$result_dir"/*
do
if grep -q "$find_str" <<< "$entry"; then
    IFS='/' read -ra ADDR <<< "$entry"
    result_file_name=${ADDR[-1]}
    IFS='j' read -ra ADDR2 <<< "$result_file_name"
    result_file_name=${ADDR2[0]}
    result_file_name=${result_file_name:0:-1} 
    echo $result_file_name
    python eval_tuning.py \
    --result-file $entry \
    --data-dir ~/VGG-Face2 \
    --split $split 
fi
done