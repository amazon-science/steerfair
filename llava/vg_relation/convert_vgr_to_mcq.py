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


def convert_to_mcq(base_dir, format):
    problems = json.load(open(os.path.join(base_dir, "visual_genome_relation.json")))

    image_dir = os.path.join(base_dir, "images")
    target_format = []
    for problem in tqdm(problems):
        prob_id = problem["image_id"]
        image_path = os.path.join(image_dir, problem["image_path"])
        true_caption = problem["true_caption"]
        false_caption = problem["false_caption"]

        question = "Pick the most appropriate caption for the image."
        p = np.random.rand()
        if p > 0.5:
            captions = [true_caption, false_caption]
            answer_idx = 0
        else:
            captions = [false_caption, true_caption]
            answer_idx = 1

        options = f"Options: (A) {captions[0]} (B) {captions[1]}"
        main_object = problem["primary_object_name"]
        context = f"Context: Look at the {main_object}."

        if format == "QCM":
            input = f"{question}\n{context}\n{options}"
        elif format == "QM":
            input = f"{question}\n{options}"
        else:
            raise NotImplementedError
        input = input.replace("  ", " ").strip()
        target_format.append(
            {
                "id": prob_id,
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": f"{input}\n<image>"},
                ],
                "answer_idx": answer_idx,
            }
        )

    print(f"Number of samples: {len(target_format)}")
    with open(os.path.join(base_dir, f"llava_vgr_{format}.json"), "w") as f:
        json.dump(target_format, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="QM")
    args = parser.parse_args()

    base_dir = args.base_dir
    format = args.format
    convert_to_mcq(base_dir, format)
