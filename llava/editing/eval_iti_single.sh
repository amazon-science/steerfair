cd ../eval
result_dir="../../vqa/results/ScienceQA/iti_0_test_CLEAN"
output_dir="../../vqa/results/ScienceQA/iti_0_test_CLEAN_outputs"
mkdir -p $output_dir
attack_files_dir="/home/ubuntu/ScienceQA/data/scienceqa/stratified_attack"
split="test"

strength=0
k=0
strength_param="alpha"

for n in {2,3,4,5}
do
    noption="noption_$n"
    find_str="${noption}_${split}"
    for entry in ls "$attack_files_dir"/*
    do
        if grep -q "$find_str" <<< "$entry"; then
            IFS='/' read -ra ADDR <<< "$entry"
            attack_file_name=${ADDR[-1]}
            IFS='.' read -ra ADDR2 <<< "$attack_file_name"
            attack_file_name=${ADDR2[0]}
            echo $attack_file_name >> ${output_dir}/${attack_file_name}_${strength_param}${strength}_k${k}.txt 
            python eval_sqa_base.py \
            --base-dir ~/ScienceQA/data/scienceqa/ \
            --attack-file $entry \
            --result-file ${result_dir}/${attack_file_name}_${strength_param}${strength}_k${k}.jsonl \
            --output-file ${output_dir}/${attack_file_name}_${strength_param}${strength}_k${k}_output.json \
            --output-result ${output_dir}/${attack_file_name}_${strength_param}${strength}_k${k}_result.json >> ${output_dir}/${attack_file_name}_${strength_param}${strength}_k${k}.txt 
        fi
    done
done