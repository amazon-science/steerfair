#!/bin/bash
search_dir=~/ScienceQA/data/scienceqa/attack_by_category
split=$1
combine_mode="qr"
result_dir="../../../vqa/results/ScienceQA/idefics/generalization/own/${split}"
echo $result_dir

for category in {"language_science","social_science","natural_science"}
do
find_str="${category}_${split}"
# echo $find_str
for k in {50,}
do
for alpha in {1,}
do
    for entry in ls "$search_dir"/*
    do
        if grep -q "$find_str" <<< "$entry"; then
            # echo $entry
            orig_filepath="$entry"
            IFS='/' read -ra ADDR <<< "$entry"
            attack_file_name=${ADDR[-1]}
            IFS='.' read -ra ADDR2 <<< "$attack_file_name"
            filepath=${ADDR2[0]}
            echo $filepath
            python apply_steering_select_head.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_alpha${alpha}_k${k}.jsonl \
            --alpha $alpha \
            --k $k \
            --vector-direction-dir pca_by_category/${category} \
            --reverse \
            --combine-mode $combine_mode \
            --normalize \
            --split $split
        fi
    done
done
done
done
