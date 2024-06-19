import json
import os
import numpy as np
from tqdm import tqdm

import argparse

"""
format \in {QCM, QM}
adopted from SQA -> LLaVA
if QCM, give context,
if QM, only questionand options
"""


def convert_to_mcq(base_dir, format, split=None):
    if split == None:
        problems = json.load(open(os.path.join(base_dir, "visual_genome_relation.json")))
    else:
        problems = json.load(open(os.path.join(base_dir, f"vgr_{split}.json")))

    image_dir = os.path.join(base_dir, "images")
    target_format = []
    for problem in tqdm(problems):
        prob_id = problem["image_id"]
        image_path = os.path.join(image_dir, problem["image_path"])
        true_caption = problem["true_caption"]
        false_caption = problem["false_caption"]

        question = "Does this caption correctly describes the image? Answer with a no or yes."
        main_object = problem["primary_object_name"]
        # context = f"Context: Look at the {main_object}."

        if format == "QCM":
            input_true = f"{question}\n{true_caption}"
            input_false = f"{question}\n{false_caption}"
        elif format == "QM":
            input_true = f"{question}\n{true_caption}"
            input_false = f"{question}\n{false_caption}"
        else:
            raise NotImplementedError
        input_true = input_true.replace("  ", " ").strip()
        input_false = input_false.replace("  ", " ").strip()
        target_format.append(
            {
                "id": prob_id,
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": f"{input_true}\n<image>"},
                ],
                "answer_idx": 1,
            }
        )

        target_format.append(
            {
                "id": prob_id,
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": f"{input_false}\n<image>"},
                ],
                "answer_idx": 0,
            }
        )

    print(f"Number of samples: {len(target_format)}")
    if split == None:
        with open(os.path.join(base_dir, f"llava_vgr_{format}_noyes.json"), "w") as f:
            json.dump(target_format, f, indent=2)
    else:
        with open(os.path.join(base_dir, f"vgr_{split}_{format}_noyes.json"), "w") as f:
            json.dump(target_format, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="QCM")
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir
    format = args.format
    convert_to_mcq(base_dir, format, args.split)
