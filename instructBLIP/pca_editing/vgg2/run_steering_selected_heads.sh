#!/bin/bash
search_dir=~/VGG-Face2
split=$1
bias_type=$2
combine_mode="qr"
find_str="llava_vgg_${split}_${bias_type}"
result_dir="../../../vgg2/instructblip/${bias_type}/pca_tuning/${combine_mode}_${split}"
echo $result_dir

for k in {30,}
do
for alpha in {0.5,}
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
            python apply_steering_select_head.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_alpha${alpha}_k${k}.jsonl \
            --alpha $alpha \
            --k $k \
            --vector-direction-dir pca_${bias_type} \
            --reverse \
            --combine-mode $combine_mode \
            --normalize
        fi
    done
done
done