#!/bin/bash
# search_dir=~/VG_Relation
search_dir=~/MME_benchmark/llava
split=$1
find_str="_${split}"
# result_dir="../../vgr/results/iti_100"
result_dir="../../vqa/results/MME_Benchmark/iti_q00"
echo $result_dir
cd ..
for alpha in {10,}
do
    for k in {50,}
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
            python iti_vgr.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_alpha${alpha}_k${k}.jsonl \
            --alpha $alpha --k $k \
            --probe-split train \
            --split $split
        fi
        done
    done
done
