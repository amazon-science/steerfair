#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
find_str="noption_$1_test"

for scale in {1.,2.,5.,10.}
do
    for thr in {0.9,0.95,0.98}
    do
        for entry in ls "$search_dir"/*
        do
        if grep -q "$find_str" <<< "$entry"; then
            orig_filepath="$entry"
            filepath=${orig_filepath:56:-5}

            echo $orig_filepath
            python scaling.py --question-file $orig_filepath --answers-file ../../vqa/results/ScienceQA/scaling_up/${filepath}_scale${scale}_thr${thr}.jsonl --cosine-threshold $thr --scaling-strength $scale --n-options $1
        fi
        done
    done
done
