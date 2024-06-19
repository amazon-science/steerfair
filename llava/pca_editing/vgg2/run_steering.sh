#!/bin/bash
search_dir=~/VGG-Face2
split=$1
combine_mode="qr"
bias_type=$2
find_str="_${split}_${bias_type}"
result_dir="../../../vgg2/pca/pca_${split}_${bias_type}_direction2/${combine_mode}_${split}"
echo $result_dir

for alpha in {1,}
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
            python apply_steering_debug.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_alpha${alpha}.jsonl \
            --alpha $alpha \
            --vector-direction-dir pca_2 --reverse \
            --combine-mode $combine_mode
        fi
    done
done
