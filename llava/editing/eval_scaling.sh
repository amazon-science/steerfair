#!/bin/bash
cd ../eval
search_dir=~/ScienceQA/data/scienceqa/stratified_attack
result_dir=~/llava_probing/vqa/results/ScienceQA/scaling_minival
find_str="noption_$1_minival"
for entry in ls "$search_dir"/*
do
if grep -q "$find_str" <<< "$entry"; then
    for entry2 in ls "$result_dir"/*
    do
        filepath=${entry:68:-5}
        # echo $filepath
        # echo $entry2
        if grep -q "$filepath" <<< "$entry2"; then
            orig_filepath="$entry2"
            filepath=${orig_filepath:65:-6}
            if [[ $filepath ]]; then 
                attack_file="$entry"
                result_file=${attack_file:68:-5}
                echo $filepath >> ~/llava_probing/vqa/results/ScienceQA/scaling_minival_outputs/${result_file}.txt 
                python eval_sqa_base.py \
                --base-dir ~/ScienceQA/data/scienceqa/ \
                --attack-file ${attack_file}\
                --result-file ~/llava_probing/vqa/results/ScienceQA/scaling_minival/${filepath}.jsonl \
                --output-file ~/llava_probing/vqa/results/ScienceQA/scaling_minival_outputs/${filepath}_output.json \
                --output-result  ~/llava_probing/vqa/results/ScienceQA/scaling_minival_outputs/${filepath}_result.json >> ~/llava_probing/vqa/results/ScienceQA/scaling_minival_outputs/${result_file}.txt
            fi
        fi
    done
fi
done

