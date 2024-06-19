#!/bin/bash
search_dir=~/VG_Relation
split=$1
combine_mode="qr"
find_str="vgr_${split}_QCM"
result_dir="../../../vgr/results/ablation_REPEAT/"
echo $result_dir

k=500
alpha=2

for i in {1,2,3,4,5,6,7,8,9,10}
do
for n in {50,100,300,500,1000,2000,5000,10000}
do
# python get_head_values.py --n-samples $n --save-dir head_ablation_${i}/train_${n}
cd ../
python get_pca_direction.py --n-samples $n --head-values-dir vgr/head_ablation_${i}/train_${n} --save-dir vgr/pca_ablation/pca_i${i}_n${n}
cd vgr
done
done

# for i in {2,3,4,5,6,7,8,9,10}
# do
# for n in {10,50,100,300,500,1000,2000,5000,10000}
# do
# for entry in ls "$search_dir"/*
#     do
#         if grep -q "$find_str" <<< "$entry"; then
#             orig_filepath="$entry"
#             IFS='/' read -ra ADDR <<< "$entry"
#             attack_file_name=${ADDR[-1]}
#             IFS='.' read -ra ADDR2 <<< "$attack_file_name"
#             filepath=${ADDR2[0]}
#             echo $filepath
#             python apply_steering_select_head.py --question-file $orig_filepath \
#             --answers-file ${result_dir}/${i}/${filepath}_n_${n}.jsonl \
#             --alpha $alpha \
#             --k $k \
#             --vector-direction-dir pca_i${i}_n${n} \
#             --reverse \
#             --combine-mode $combine_mode \
#             --normalize
#         fi
#     done
# done
# done
