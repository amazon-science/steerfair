#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
split=$1
combine_mode="qr"
result_dir="../../../vqa/results/ScienceQA/circular_without_answer/${combine_mode}_${split}"
echo $result_dir


for n in {2,3,4,5,}
do
noption="noption_$n"
find_str="${noption}_${split}"
for alpha in {1,}
do
    for entry in ls "$search_dir"/*
    do
        if grep -q "$find_str" <<< "$entry"; then
            orig_filepath="$entry"
            IFS='/' read -ra ADDR <<< "$entry"
            attack_file_name=${ADDR[-1]}
            IFS='.' read -ra ADDR2 <<< "$attack_file_name"
            filepath=${ADDR2[0]}
            echo $filepath
            python apply_steering.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_alpha${alpha}.jsonl \
            --alpha $alpha \
            --vector-direction-dir pca_with_answer \
            --reverse \
            --combine-mode $combine_mode \
            --split $split
        fi
    done
done
done
