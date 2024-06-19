#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
split="minival"
result_dir="../../vqa/results/ScienceQA/iti_bias_2_${split}_ALL"

for n in {2,3,4,5}
do
noption="noption_$n"
find_str="${noption}_${split}"

for alpha in {1,5,10,15}
do
    for k in {5,10,20,30,40,50}
    do
        for entry in ls "$search_dir"/*
        do
        if grep -q "$find_str" <<< "$entry"; then
            orig_filepath="$entry"
            filepath=${orig_filepath:56:-5}

            echo $orig_filepath
            python iti_bias.py --question-file \
            $orig_filepath --answers-file ${result_dir}/${filepath}_alpha${alpha}_k${k}.jsonl \
            --alpha $alpha --k $k \
            --split $split
        fi
        done
    done
done
done