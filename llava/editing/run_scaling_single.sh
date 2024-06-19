#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
split="test"
result_dir=~/llava_probing/vqa/results/ScienceQA/scaling_${split}_BUGFIX
scale=$1
k=$2
find_str="_$split"
for entry in ls "$search_dir"/*
    do
    if grep -q "$find_str" <<< "$entry"; then
        orig_filepath="$entry"
        filepath=${orig_filepath:56:-5}
        echo $filepath 
        python scaling.py --question-file $orig_filepath \
        --answers-file ${result_dir}/${filepath}_scale${scale}_k${k}.jsonl \
        --scaling-strength $scale --k $k \
        --probing-split train --n-options 3 \
        --split $split
    fi
done