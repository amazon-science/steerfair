#!/bin/bash
data_dir="/home/ubuntu/VGG-Face2/"
split="test"
bias_type="math_arts"
result_dir="/home/ubuntu/llava_probing/vgg2/pca_tuning_NEW/${bias_type}/qr_${split}"
output_dir="/home/ubuntu/llava_probing/vgg2/pca_tuning_NEW/${bias_type}/qr_${split}_outputs"
# result_dir="/home/ubuntu/llava_probing/vgg2/instructblip/baseline_${split}"
# output_dir="/home/ubuntu/llava_probing/vgg2/instructblip/baseline_${split}_outputs"
mkdir -p $output_dir
for entry in ls "$result_dir"/*
do
    if grep -q ".jsonl" <<< "$entry"; then
        IFS='/' read -ra ADDR <<< "$entry"
        filename=${ADDR[-1]}
        # IFS='.' read -ra ADDR2 <<< "$filename"
        # filename=${ADDR2[0]}
        echo $filename >> ${output_dir}/output.txt 
        python eval_tuning.py --result-file $entry \
        --split $split \
        --data-dir $data_dir >> ${output_dir}/output.txt 
    fi
done
# python eval_tuning.py --result-file $result_file \
# --split $split \
# --data-dir $data_dir >> $output_file
