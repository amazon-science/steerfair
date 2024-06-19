split="test"

result_dir="/home/ubuntu/llava_probing/vqa/idefics/mme/generalization"
output_dir="/home/ubuntu/llava_probing/vqa/idefics/mme/generalization_outputs"
mkdir -p $output_dir

attack_files_dir="/home/ubuntu/MME_benchmark/llava"

alpha=15
k=30

find_str="${split}"

cd ../llava/mme_benchmark
for entry in ls "$attack_files_dir"/*
do
    if grep -q "$find_str" <<< "$entry"; then
        IFS='/' read -ra ADDR <<< "$entry"
        attack_file_name=${ADDR[-1]}
        IFS='.' read -ra ADDR2 <<< "$attack_file_name"
        attack_file_name=${ADDR2[0]}
        echo $attack_file_name
        echo $attack_file_name >> ${output_dir}/${attack_file_name}.txt 
        python convert_answer_to_mme.py \
            --data-file $entry \
            --result-file ${result_dir}/${attack_file_name}_alpha${alpha}_k${k}.jsonl \
            --output-dir ${output_dir}/${attack_file_name}
        
        python calculation.py --results_dir ${output_dir}/${attack_file_name}  >> ${output_dir}/${attack_file_name}.txt 
    fi
done