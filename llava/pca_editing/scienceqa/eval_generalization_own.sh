#!/bin/bash
cd ../../eval
search_dir=~/ScienceQA/data/scienceqa/attack_by_category

combine_mode="qr"
split="test"

result_dir="../../vqa/results/instructblip/ScienceQA/generalization/own/test"
output_dir="../../vqa/results/instructblip/ScienceQA/generalization/own/test_outputs"
mkdir -p $output_dir


# for category in {"language_science","social_science","natural_science"}
# do
# dir="$result_dir/dir_${category}"
# noption="category_$category"
# find_str="${noption}_${split}"

# for entry in ls "$search_dir"/*
# do
    # echo $entry
# for category in {"language_science","social_science","natural_science"}
# do
# dir="$result_dir/dir_${category}"
# mkdir -p ${output_dir}/dir_${category}/
for entry2 in ls "$result_dir"/*
do
echo $entry2
IFS='/' read -ra ADDR <<< "$entry2"
attack_file_name=${ADDR[-1]}
attack_file_name=${attack_file_name%??????????????????}
attack_file_name="${search_dir}/${attack_file_name}.json"
IFS='/' read -ra ADDR <<< "$entry2"
result_file_name=${ADDR[-1]}
IFS='j' read -ra ADDR2 <<< "$result_file_name"
result_file_name=${ADDR2[0]}
result_file_name=${result_file_name:0:-1}
# echo ${output_dir}/dir_${category}/${result_file_name}_output.json 
echo $result_file_name >> ${output_dir}/${result_file_name}.txt
python eval_sqa_base.py \
--base-dir ~/ScienceQA/data/scienceqa/ \
--attack-file $attack_file_name \
--result-file ${entry2} \
--output-file  ${output_dir}/${result_file_name}_output.json  \
--output-result   ${output_dir}/${result_file_name}_result.json >>  ${output_dir}/${result_file_name}.txt
done
# done
# done


