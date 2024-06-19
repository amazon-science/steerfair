#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/debias_baseline
find_str="noption_$1_minival"

# ls $search_dir

for entry in ls "$search_dir"/*
do
if grep -q "$find_str" <<< "$entry"; then
orig_filepath="$entry"
filepath=${orig_filepath:56:-5}

echo $orig_filepath
python collect_probe_invariant.py --question-file $orig_filepath --split minival

fi
done