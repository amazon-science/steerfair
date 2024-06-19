#!/bin/bash
cd ../eval
split="test"
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
result_dir=~/llava_probing/vqa/idefics/ScienceQA/iti_100
output_dir=~/llava_probing/vqa/idefics/ScienceQA/iti_100_outputs
mkdir -p $output_dir


split="test"
for noption in {2,3,4,5,}
do
find_str="noption_${noption}_${split}"
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
            IFS='.' read -ra ADDR2 <<< "$result_file_name"
            result_file_name=${ADDR2[0]}
            echo $result_file_name >> ${output_dir}/${attack_file_name}.txt
            python eval_sqa_base.py \
            --base-dir ~/ScienceQA/data/scienceqa/ \
            --attack-file ${entry} \
            --result-file ${entry2} \
            --output-file ${output_dir}/${result_file_name}_output.json \
            --output-result  ${output_dir}/${result_file_name}_result.json >> ${output_dir}/${attack_file_name}.txt
        fi
    done
fi
done
done

