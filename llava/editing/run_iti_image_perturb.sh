#!/bin/bash
filepath="/home/ubuntu/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json"
python iti.py --question-file $filepath \
--answers-file ../../vqa/results/ScienceQA/iti_0_test_perturb_2/llava_test_QCM-LEA.jsonl \
--alpha 0 --k 0 \
--probe-split train \
--split test \
--perturb-image
