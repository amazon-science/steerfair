cd ../../mme_benchmark
pwd
combine_mode="qr"
split="test"
result_dir="../../vqa/results/MME_Benchmark/generalization/llava"
output_dir="../../vqa/results/MME_Benchmark/generalization/llava_outputs"

mkdir -p $output_dir
attack_files_dir="/home/ubuntu/MME_benchmark/llava"

alpha=10
k=50

find_str="${split}"
for entry in ls "$attack_files_dir"/*
do
    if grep -q "$find_str" <<< "$entry"; then
        IFS='/' read -ra ADDR <<< "$entry"
        attack_file_name=${ADDR[-1]}
        IFS='.' read -ra ADDR2 <<< "$attack_file_name"
        attack_file_name=${ADDR2[0]}
        echo $attack_file_name
        echo $attack_file_name >> ${output_dir}/${attack_file_name}_alpha${alpha}_k${k}.txt 
        python eval_mme.py \
            --data-file $entry \
            --result-file ${result_dir}/${attack_file_name}_alpha${alpha}_k${k}.jsonl \
            --output-file ${output_dir}/${attack_file_name}_alpha${alpha}_k${k}_output.json \
            --output-result ${output_dir}/${attack_file_name}_alpha${alpha}_k${k}_result.json >> ${output_dir}/${attack_file_name}_alpha${alpha}_k${k}.txt 
    fi
done