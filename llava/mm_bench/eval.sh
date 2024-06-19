result_dir="../../vqa/results/MM_Bench/iti_0_val"
output_dir="../../vqa/results/MM_Bench/iti_0_val_outputs"
mkdir -p $output_dir
attack_files_dir="/home/ubuntu/MM_Bench/stratified_attack"
split="val"

strength=0
k=0
strength_param="alpha"

for n in {2,3,4}
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