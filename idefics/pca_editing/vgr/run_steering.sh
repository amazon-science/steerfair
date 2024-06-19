#!/bin/bash
search_dir=~/VG_Relation
split=$1
combine_mode="qr"
find_str="vgr_${split}_QCM_yesno"
result_dir="../../../vgr/results/idefics/no_tuning"
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
            python apply_steering.py --question-file $orig_filepath \
            --answers-file ${result_dir}/${filepath}_alpha${alpha}.jsonl \
            --alpha $alpha \
            --vector-direction-dir pca --reverse \
            --combine-mode $combine_mode
        fi
    done
done
