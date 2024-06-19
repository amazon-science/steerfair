#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
split=$1
result_dir="../vqa/instructblip/ScienceQA/baseline_${split}"

for n in {2,3,4,5}
do
    noption="noption_$n"
    find_str="${noption}_${split}"
    for entry in ls "$search_dir"/*
    do
        if grep -q "$find_str" <<< "$entry"; then
            orig_filepath="$entry"
            filepath=${orig_filepath:56:-5}
            echo $filepath
            python baseline.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}.jsonl
        fi
    done
done