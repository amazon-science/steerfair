#!/bin/bash
search_dir=~/VG_Relation
split=$1
find_str="vgr_${split}_QCM"
result_dir="../../../vgr/results/instructblip/iti_100"

for alpha in {25,}
do
    for k in {10,}
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
            python iti.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_alpha${alpha}_k${k}.jsonl \
            --alpha $alpha --k $k \
            --probe-split "train" \
            --split $split
        fi
        done
    done
done