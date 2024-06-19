import json
import os
import numpy as np
from tqdm import tqdm

import argparse
from loader import MME_Dataset
from sklearn.model_selection import train_test_split


def convert_to_llava(base_dir):
    answer_dict = {
        "no": 0,
        "yes": 1,
    }
    problems_orig = MME_Dataset()
    problems_test, problems_val = train_test_split(problems_orig, test_size=0.1)
    problems_all = [problems_val, problems_test]
    splits = ["val", "test"]
    prompts = ["Please answer no or yes.", "Please answer yes or no."]

    for split_idx, split in enumerate(splits):
        filenames = [f"llava_mme_noyes_{split}.json", f"llava_mme_yesno_{split}.json"]
        problems = problems_all[split_idx]
        for f_idx, fname in enumerate(filenames):
            target_format = []
            prompt = prompts[f_idx]
            fname = filenames[f_idx]
            for problem in tqdm(problems):
                prob_id = problem["index"]
                image_path = problem["image"]
                question1 = problem["question1"]
                input1 = f"{question1} {prompt}"
                answer1 = problem["answer1"].lower()
                answer_idx1 = answer_dict[answer1]

                question2 = problem["question2"]
                input2 = f"{question2} {prompt}"
                answer2 = problem["answer2"].lower()
                answer_idx2 = answer_dict[answer2]
                target_format.append(
                    {
                        "id": f"{prob_id}_1",
                        "image": image_path,
                        "conversations": [
                            {"from": "human", "value": f"{input1}\n<image>"},
                        ],
                        "answer_idx": answer_idx1,
                    }
                )

                target_format.append(
                    {
                        "id": f"{prob_id}_2",
                        "image": image_path,
                        "conversations": [
                            {"from": "human", "value": f"{input2}\n<image>"},
                        ],
                        "answer_idx": answer_idx2,
                    }
                )

            print(f"Split = {split} | Number of samples: {len(target_format)}")
            # with open(os.path.join(base_dir, fname), "w") as f:
            #     json.dump(target_format, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True)
    # parser.add_argument("--format", type=str, default="QCM")
    # parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir
    # format = args.format
    convert_to_llava(base_dir)
