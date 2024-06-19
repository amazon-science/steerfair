#!/bin/bash
search_dir=/home/ubuntu/VGG-Face2
split=$1
combine_mode="qr"
find_str="_${split}"
result_dir="../../../vgg2_open/instructblip/pca_tuning/${combine_mode}_${split}"
echo $result_dir

for k in {1000,}
do
for alpha in {1,}
do
    for entry in ls "$search_dir"/*
    do
        # echo $entry
        if grep -q "$find_str" <<< "$entry"; then
        # echo "here"
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
            --vector-direction-dir pca --reverse \
            --combine-mode $combine_mode \
            --normalize
        fi
    done
done
done