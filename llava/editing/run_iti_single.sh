#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
split="test"
alpha=$1
k=$2
find_str="_$split"
for entry in ls "$search_dir"/*
    do
    if grep -q "$find_str" <<< "$entry"; then
        orig_filepath="$entry"
        filepath=${orig_filepath:56:-5}
        echo $filepath 
        python iti.py --question-file $orig_filepath \
        --answers-file ../../vqa/results/ScienceQA/iti_100/${filepath}_alpha${alpha}_k${k}.jsonl \
        --alpha $alpha --k $k \
        --probe-split train \
        --split $split
    fi
done