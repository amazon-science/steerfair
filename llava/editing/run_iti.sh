#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
split="test"
result_dir="../../vqa/results/ScienceQA/iti_BEST_PARAM_${split}"
for n in {2,}
do
noption="noption_$n"
find_str="attack_choice_1_${noption}_${split}"
echo $find_str
for alpha in {5,}
do
    for k in {10,}
    do
        for entry in ls "$search_dir"/*
        do
        if grep -q "$find_str" <<< "$entry"; then
            orig_filepath="$entry"
            filepath=${orig_filepath:56:-5}

            echo $orig_filepath
            python iti.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_alpha${alpha}_k${k}.jsonl \
            --alpha $alpha --k $k \
            --probe-split train \
            --split $split
        fi
        done
    done
done
done
