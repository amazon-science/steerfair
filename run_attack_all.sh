#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/stratified_attack_sampled_0.2
find_str="noption_$1_test"

for entry in ls "$search_dir"/*
do
if grep -q "$find_str" <<< "$entry"; then
orig_filepath="$entry"
filepath=${orig_filepath:56:-5}

echo $orig_filepath
python -m llava.eval.model_vqa_science --model-path liuhaotian/llava-v1.5-13b --question-file $orig_filepath --image-folder ~/ScienceQA/test --answers-file vqa/results/base_model_sampled_0.2/${filepath}.jsonl  --conv-mode llava_v1 
fi
done