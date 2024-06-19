import argparse
import json
import os
import re
import random
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--output-dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    output_dir = args.output_dir

    data_file = args.data_file
    with open(data_file, "r") as data_file:
        json_data = data_file.read()
    data_obj = json.loads(json_data)

    options = ["No", "Yes"]
    predictions = [json.loads(line) for line in open(args.result_file)]

    output_obj = {}

    for i, pred_obj in enumerate(predictions):
        prob_id = pred_obj["question_id"]
        pred_text = pred_obj["text"]
        if "idefics" in args.output_dir:
            pred_text = pred_text.split("Assistant: ")[-1].replace("\n", "")
        category = prob_id.split("_")[0]
        image_name = f"{prob_id.split(f'{category}_')[-1][:-2]}.jpg"
        gt = data_obj[i]["answer_idx"]
        gt_answer = options[gt]
        if "idefics" in args.output_dir or "instructblip" in args.output_dir:
            question = question = pred_obj["prompt"]
        else:
            question = pred_obj["prompt"].split("<image>\n")[-1]

        if category not in output_obj:
            output_obj[category] = []

        # Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
        output_str = image_name + "\t" + question + "\t" + gt_answer + "\t" + pred_text + "\n"

        output_obj[category].append(output_str)

    if not os.path.isdir(os.path.join(output_dir)):
        os.makedirs(os.path.join(output_dir))

    # print(output_obj.keys())
    for category in output_obj:
        # print(category)
        filepath = os.path.join(output_dir, f"{category}.txt")
        file_arr = output_obj[category]
        myfile = open(filepath, "w")
        for line in file_arr:
            myfile.write(line)
        myfile.close()
