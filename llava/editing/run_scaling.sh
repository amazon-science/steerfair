#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
split="minival"
result_dir="../../vqa/results/ScienceQA/scaling_${split}_ALL"

for n in {2,3,4,5}
do
noption="noption_$n"
find_str="${noption}_${split}"
for scale in {0.001,0.01,0.05,0.1,0.5,0.8}
do
    for k in {5,10,20,30,40,50}
    do
        for entry in ls "$search_dir"/*
        do
        if grep -q "$find_str" <<< "$entry"; then
            orig_filepath="$entry"
            filepath=${orig_filepath:56:-5}

            echo $orig_filepath
            python scaling.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_scale${scale}_k${k}.jsonl \
            --k $k --scaling-strength $scale \
            --split $split \
            --n-options $n
        fi
        done
    done
done
done
