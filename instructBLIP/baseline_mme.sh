#!/bin/bash
search_dir=~/MME_benchmark/llava
split=$1
find_str="${split}"
result_dir="../vqa/instructblip/mme/baseline_${split}"
echo $result_dir

for alpha in {0,}
do
    for k in {0,}
    do
        for entry in ls "$search_dir"/*
        do
        if grep -q "$find_str" <<< "$entry"; then
            orig_filepath="$entry"
            IFS='/' read -ra ADDR <<< "$entry"
            attack_file_name=${ADDR[-1]}
            IFS='.' read -ra ADDR2 <<< "$attack_file_name"
            filepath=${ADDR2[0]}
            echo $filepath
            python baseline.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}.jsonl
        fi
        done
    done
done