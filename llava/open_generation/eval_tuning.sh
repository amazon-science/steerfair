combine_mode="qr"
split="test"
result_dir="../../vgg2_open/instructblip/"
# pca_tuning/${combine_mode}_${split}"

for entry2 in ls "$result_dir"/*
do
    if grep -q "$split" <<< "$entry2"; then
        IFS='/' read -ra ADDR <<< "$entry2"
        result_file_name=${ADDR[-1]}
        IFS='j' read -ra ADDR2 <<< "$result_file_name"
        result_file_name=${ADDR2[0]}
        result_file_name=${result_file_name:0:-1}
        echo $result_file_name
        python eval.py --result-file $entry2
    fi
done