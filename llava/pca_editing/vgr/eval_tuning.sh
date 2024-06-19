#!/bin/bash
cd ../../vg_relation
search_dir="/home/ubuntu/VG_Relation"

combine_mode="qr"
split="minival"

result_dir="../../vgr/results/generalization/llava"
output_dir="../../vgr/results/generalization/llava_outputs"
mkdir -p $output_dir

find_str="vgr_${split}_QCM"

# ls $search_dir
for entry in ls "$search_dir"/*
do
if grep -q "$find_str" <<< "$entry"; then
    for entry2 in ls "$result_dir"/*
    do
        IFS='/' read -ra ADDR <<< "$entry"
        attack_file_name=${ADDR[-1]}
        IFS='.' read -ra ADDR2 <<< "$attack_file_name"
        attack_file_name=${ADDR2[0]}
        if grep -q "$attack_file_name" <<< "$entry2"; then
            IFS='/' read -ra ADDR <<< "$entry2"
            result_file_name=${ADDR[-1]}
            IFS='j' read -ra ADDR2 <<< "$result_file_name"
            result_file_name=${ADDR2[0]}
            result_file_name=${result_file_name:0:-1}
            echo $result_file_name >> ${output_dir}/${attack_file_name}.txt
            python eval_vgr_yesno.py \
            --data-file $entry \
            --result-file ${entry2} \
            --output-file ${output_dir}/${result_file_name}_output.json \
            --output-result  ${output_dir}/${result_file_name}_result.json >> ${output_dir}/${attack_file_name}.txt
        fi
    done
fi
done

