#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
split=$1
result_dir="../../../vqa/instructblip/ScienceQA/iti_100"

for n in {2,3,4,5,}
do
noption="noption_$n"
find_str="${noption}_${split}"
echo $find_str
for alpha in {1,}
do
    for k in {50,}
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
            python iti.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_alpha${alpha}_k${k}.jsonl \
            --alpha $alpha --k $k \
            --probe-split "train" \
            --split $split
        fi
        done
    done
done
done