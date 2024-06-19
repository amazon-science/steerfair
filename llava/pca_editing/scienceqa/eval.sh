cd ../../eval

split="test"
combine_mode="qr"

result_dir="../../vqa/results/ScienceQA/pca_best_param_3/${combine_mode}_${split}"
output_dir="../../vqa/results/ScienceQA/pca_best_param_3/${combine_mode}_${split}_outputs"

attack_files_dir="/home/ubuntu/ScienceQA/data/scienceqa/stratified_attack"

mkdir -p $output_dir

alpha=1

for n in {2,3,4,5,}
do
noption="noption_$n"
find_str="${noption}_${split}"
for entry in ls "$attack_files_dir"/*
do
# echo $entry
    if grep -q "$find_str" <<< "$entry"; then
        IFS='/' read -ra ADDR <<< "$entry"
        attack_file_name=${ADDR[-1]}
        IFS='.' read -ra ADDR2 <<< "$attack_file_name"
        attack_file_name=${ADDR2[0]}
        echo $attack_file_name
        echo $attack_file_name >> ${output_dir}/${attack_file_name}_alpha${alpha}.txt 
        python eval_sqa_base.py \
        --base-dir ~/ScienceQA/data/scienceqa/ \
        --attack-file $entry \
        --result-file ${result_dir}/${attack_file_name}_alpha${alpha}_k${k}.jsonl \
        --output-file ${output_dir}/${attack_file_name}_alpha${alpha}_output.json \
        --output-result ${output_dir}/${attack_file_name}_alpha${alpha}_result.json >> ${output_dir}/${attack_file_name}_alpha${alpha}.txt 
    fi
done
done