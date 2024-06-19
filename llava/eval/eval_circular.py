import argparse
import json
import os
import re
import random
from tqdm import tqdm
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str)
    parser.add_argument("--n-options", type=int)
    parser.add_argument("--eval-result-dir", type=str)
    parser.add_argument("--attack-files-dir", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--options", type=list, default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


def get_pred_idx(prediction, choices, options, gt):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[: len(choices)]:
        return options.index(prediction)
    else:
        other_choices = [i for i in range(len(choices))]
        other_choices.pop(gt)
        return random.choice(other_choices)


if __name__ == "__main__":
    args = get_args()
    options = args.options
    base_dir = args.base_dir
    attack_files_dir = args.attack_files_dir
    eval_result_dir = args.eval_result_dir
    n_options = args.n_options

    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    attack_files_all = os.listdir(attack_files_dir)
    attack_files_all = [
        os.path.join(attack_files_dir, f) for f in attack_files_all if (f"noption_{n_options}_{args.split}" in f)
    ]
    result_files_all = [
        os.path.join(eval_result_dir, f)
        for f in os.listdir(eval_result_dir)
        if (f"noption_{n_options}_{args.split}" in f)
    ]
    result_files_all.sort()
    attack_files_all.sort()

    new_gt_all = []
    split_problems_all = []
    for attack_file in attack_files_all:
        attack_obj = json.load(open(attack_file))
        split_indices = [obj_["id"] for obj_ in attack_obj]
        split_problems = {idx: problems[idx] for idx in split_indices}
        split_problems_all.append(split_problems)
        new_gt = {obj_["id"]: obj_["new_gt"] for obj_ in attack_obj}
        new_gt_all.append(new_gt)

    predictions_all = []
    for result_file in result_files_all:
        predictions = [json.loads(line) for line in open(result_file)]
        predictions = {pred["question_id"]: pred for pred in predictions}
        predictions_all.append(predictions)

    results = {"correct": [], "incorrect": []}
    sqa_results = {}
    sqa_results["acc"] = None
    sqa_results["correct"] = None
    sqa_results["count"] = None
    sqa_results["results"] = {}
    sqa_results["outputs"] = {}

    pred_obj = {}

    for i, split_problems in tqdm(enumerate(split_problems_all)):
        failed = 0
        predictions = predictions_all[i]
        for prob_id, prob in split_problems.items():
            if prob_id not in predictions:
                continue
            pred = predictions[prob_id]
            pred_text = pred["text"]

            if len(pred_text) == 1:
                answer = pred_text[0].upper()
                if answer not in options[: len(prob["choices"])]:
                    answer = "FAILED"
                    failed += 1
                if new_gt != None:
                    pred_idx = get_pred_idx(answer, prob["choices"], args.options, new_gt[prob_id])
                else:
                    pred_idx = get_pred_idx(answer, prob["choices"], args.options, prob["answer"])
            else:
                re_A = [
                    r"\(A\)",
                    r"\(A",
                    r"A\)",
                    r"is A.",
                ]
                re_B = [
                    r"\(B\)",
                    r"\(B",
                    r"B\)",
                    r"is B.",
                ]
                re_C = [
                    r"\(C\)",
                    r"\(C",
                    r"C\)",
                    r"is C.",
                ]
                re_D = [
                    r"\(D\)",
                    r"\(D",
                    r"D\)",
                    r"is D.",
                ]
                re_E = [
                    r"\(E\)",
                    r"\(E",
                    r"E\)",
                    r"is E.",
                ]
                re_all = [re_A, re_B, re_C, re_D, re_E]
                re_all = re_all[: len(prob["choices"])]

                re_all = [item for sublist in re_all for item in sublist]
                found = False
                for regex in re_all:
                    pattern = re.compile(regex)
                    res = pattern.findall(pred_text)
                    if len(res) > 0:
                        if regex in re_A:
                            answer = "A"
                            found = True
                            break
                        elif regex in re_B:
                            answer = "B"
                            found = True
                            break
                        elif regex in re_C:
                            answer = "C"
                            found = True
                            break
                        elif regex in re_D:
                            answer = "D"
                            found = True
                            break
                        elif regex in re_E:
                            answer = "E"
                            found = True
                            break
                if found == False:
                    try:
                        for i, option in enumerate(prob["choices"]):
                            pattern = re.compile(option)
                            res = pattern.findall(pred_text)
                            if len(res) == 1:
                                found = True
                                answer = args.options[i]
                                break
                    except:
                        answer = "FAILED"
                        failed += 1
                    if found == False:
                        answer = "FAILED"
                        failed += 1
                if new_gt != None:
                    pred_idx = get_pred_idx(answer, prob["choices"], args.options, new_gt[prob_id])
                else:
                    pred_idx = get_pred_idx(answer, prob["choices"], args.options, prob["answer"])

            if new_gt != None:
                gt = new_gt[prob_id]
            else:
                gt = prob["answer"]
            analysis = {
                "question_id": prob_id,
                "parsed_ans": answer,
                "pred_idx": pred_idx,
                "ground_truth": gt,
                "question": pred["prompt"],
                "pred": pred_text,
                "is_multimodal": "<image>" in pred["prompt"],
            }

            if pred_idx == gt:
                if prob_id not in pred_obj:
                    pred_obj[prob_id] = [1]
                else:
                    pred_obj[prob_id].append(1)
            else:
                if prob_id not in pred_obj:
                    pred_obj[prob_id] = [0]
                else:
                    pred_obj[prob_id].append(0)

    # print(pred_obj)
    correct = 0
    for key in pred_obj:
        arr_ = np.array(pred_obj[key])
        if np.sum(arr_) == len(arr_):
            correct += 1
    acc = correct / len(pred_obj)
    print(f"Accuracy = {acc:.3f}")
