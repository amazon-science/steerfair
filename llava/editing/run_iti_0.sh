#!/bin/bash
search_dir=~/VGG-Face2
# /ScienceQA/data/scienceqa/stratified_attack
split="test"
# noption=$1

# for noption in {5,}
# do
find_str="baseline_2_vgg_${split}_"
# echo $find_str

for alpha in {0,}
do
    for k in {0,}
    do
        for entry in ls "$search_dir"/*
        do
        if grep -q "$find_str" <<< "$entry"; then
            orig_filepath="$entry"
            filepath=${orig_filepath:23:-5}
            echo $orig_filepath
            echo $filepath
            python iti.py --question-file $orig_filepath \
            --answers-file ../../vgg2/baseline_2/${filepath}_alpha${alpha}_k${k}.jsonl \
            --alpha $alpha --k $k \
            --probe-split train \
            --split $split
        fi
        done
    done
done
# done
