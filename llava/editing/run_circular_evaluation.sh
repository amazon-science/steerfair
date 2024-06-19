#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/debias_baseline
split="test"
# noption=$1

for noption in {2,3,4,5}
do
find_str="noption_${noption}_${split}"
# ls $search_dir
for alpha in {0,}
do
    for k in {0,}
    do
        for entry in ls "$search_dir"/*
        do
        if grep -q "$find_str" <<< "$entry"; then
            orig_filepath="$entry"
            filepath=${orig_filepath:54:-5}

            echo $filepath

            python iti.py --question-file $orig_filepath \
            --answers-file ../../vqa/results/ScienceQA/circular_eval/${split}/iti_0/${filepath}_alpha${alpha}_k${k}.jsonl \
            --alpha $alpha --k $k \
            --probe-split train \
            --split $split
        fi
        done
    done
done
done
