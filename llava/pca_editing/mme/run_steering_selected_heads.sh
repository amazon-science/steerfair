#!/bin/bash
search_dir=/home/ubuntu/MME_benchmark/llava
split=$1
combine_mode="qr"
find_str="${split}"
result_dir="../../../vqa/results/MME_Benchmark/generalization/llava/tuning"
echo $result_dir

for k in {10,30,50,100,200,500,1000,1600}
do
for alpha in {0.1,0.5,1,2,5,10,15,20,25}
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
            --vector-direction-dir ../vgr/pca_TEST_val \
            --reverse \
            --combine-mode $combine_mode --normalize
        fi
    done
done
done
