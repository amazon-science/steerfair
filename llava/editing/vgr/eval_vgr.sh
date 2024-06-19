cd ../../vg_relation
split="test"

result_dir="../../vgr/results/iti_0_${split}_CLEAN"
output_dir="../../vgr/results/ti_0_${split}_CLEAN_outputs"

mkdir -p $output_dir
attack_files_dir="/home/ubuntu/VG_Relation"

alpha=0
k=0

find_str="vgr_${split}_QCM"
for entry in ls "$attack_files_dir"/*
do
    if grep -q "$find_str" <<< "$entry"; then
        IFS='/' read -ra ADDR <<< "$entry"
        attack_file_name=${ADDR[-1]}
        IFS='.' read -ra ADDR2 <<< "$attack_file_name"
        attack_file_name=${ADDR2[0]}
        echo $attack_file_name
        echo $attack_file_name >> ${output_dir}/${attack_file_name}_alpha${alpha}_k${k}.txt 
        python eval_vgr_yesno.py \
            --data-file $entry \
            --result-file ${result_dir}/${attack_file_name}_alpha${alpha}_k${k}.jsonl \
            --output-file ${output_dir}/${attack_file_name}_alpha${alpha}_k${k}_output.json \
            --output-result ${output_dir}/${attack_file_name}_alpha${alpha}_k${k}_result.json >> ${output_dir}/${attack_file_name}_alpha${alpha}_k${k}.txt 
    fi
done