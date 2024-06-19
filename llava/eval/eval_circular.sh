result_dir="../../vqa/results/ScienceQA/circular_eval/test/iti_0"
output_dir="../../vqa/results/ScienceQA/circular_eval_outputs/test_iti_0/"
mkdir -p $output_dir
attack_files_dir="/home/ubuntu/ScienceQA/data/scienceqa/debias_baseline"
split="test"

strength=0
k=0
strength_param="alpha"

for n in {2,3,4,5}
do
    echo $n
    python eval_circular.py \
    --base-dir ~/ScienceQA/data/scienceqa/ \
    --n-options $n \
    --eval-result-dir $result_dir \
    --attack-files-dir $attack_files_dir \
    --split test 
done