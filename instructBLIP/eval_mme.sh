split="test"

result_dir="/home/ubuntu/llava_probing/vqa/instructblip/mme/generalization"
output_dir="/home/ubuntu/llava_probing/vqa/instructblip/mme/generalization_outputs"
mkdir -p $output_dir

attack_files_dir="/home/ubuntu/MME_benchmark/llava"

find_str="${split}"

cd ../llava/mme_benchmark
for entry in ls "$attack_files_dir"/*
do
    if grep -q "$find_str" <<< "$entry"; then
        for entry2 in ls "$result_dir"/*
        do
            IFS='/' read -ra ADDR <<< "$entry"
            attack_file_name=${ADDR[-1]}
            IFS='.' read -ra ADDR2 <<< "$attack_file_name"
            attack_file_name=${ADDR2[0]}
            if grep -q "$attack_file_name" <<< "$entry2"; then
                IFS='/' read -ra ADDR <<< "$entry2"
                result_file_name=${ADDR[-1]}
                IFS='j' read -ra ADDR2 <<< "$result_file_name"
                result_file_name=${ADDR2[0]}
                result_file_name=${result_file_name:0:-1}
                echo $result_file_name >> ${output_dir}/${attack_file_name}.txt
                python convert_answer_to_mme.py \
                    --data-file $entry \
                    --result-file ${entry2} \
                    --output-dir  ${output_dir}/${attack_file_name}
                
                python calculation.py --results_dir ${output_dir}/${attack_file_name} >> ${output_dir}/${attack_file_name}.txt 
            fi
        done
    fi
done