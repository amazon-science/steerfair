result_dir="../../vqa/results/SEED_Bench/iti_0"
output_dir="../../vqa/results/SEED_Bench/iti_0_outputs"
mkdir -p $output_dir
attack_files_dir="/home/ubuntu/SEED-Bench/stratified_attack"
ls $attack_files_dir
split="test"

strength=0
k=0
strength_param="alpha"

for n in {4,}
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
            python eval.py \
            --attack-file $entry \
            --result-file ${result_dir}/${attack_file_name}_${strength_param}${strength}_k${k}.jsonl \
            --output-file ${output_dir}/${attack_file_name}_${strength_param}${strength}_k${k}_output.json \
            --output-result ${output_dir}/${attack_file_name}_${strength_param}${strength}_k${k}_result.json >> ${output_dir}/${attack_file_name}_${strength_param}${strength}_k${k}.txt 
        fi
    done
done