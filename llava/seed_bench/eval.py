import argparse
import json
import os
import re
import random
import numpy as np

from tqdm import tqdm
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--attack-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--output-result", type=str)
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
    attack_file = args.attack_file

    attack_obj = json.load(open(attack_file))
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {int(pred["question_id"]): pred for pred in predictions}
    dataset = load_dataset("AILab-CVC/SEED-Bench")["test"]
    try:
        new_gt = {obj_["id"]: obj_["new_gt"] for obj_ in attack_obj}
    except:
        new_gt = None

    results = {"correct": [], "incorrect": []}
    sqa_results = {}
    sqa_results["acc"] = None
    sqa_results["correct"] = None
    sqa_results["count"] = None
    sqa_results["results"] = {}
    sqa_results["outputs"] = {}

    pred_obj = []

    failed = 0
    choice_keys = ["choice_a", "choice_b", "choice_c", "choice_d"]
    choice_str = ["A", "B", "C", "D"]

    for prob in tqdm(dataset):
        # print(prob)
        if prob["data_type"] != "image":
            continue
        prob_id = int(prob["question_id"])
        if prob_id not in predictions:
            continue
        pred = predictions[prob_id]
        pred_text = pred["text"]

        choices = np.array(choice_str)
        if len(pred_text) == 1:
            answer = pred_text[0].upper()
            if answer not in options[: len(choices)]:
                answer = "FAILED"
                failed += 1
            if new_gt != None:
                pred_idx = get_pred_idx(
                    answer, choices, args.options, np.argwhere(choices == new_gt[str(prob_id)]).flatten()[0]
                )
            else:
                pred_idx = get_pred_idx(answer, choices, args.options, prob["answer"])
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
            re_all = re_all[: len(choices)]

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
                    for i, option in enumerate(choice_str):
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
                pred_idx = get_pred_idx(
                    answer, choices, args.options, np.argwhere(choices == new_gt[str(prob_id)]).flatten()[0]
                )
            else:
                pred_idx = get_pred_idx(answer, choices, args.options, prob["answer"])

        if new_gt != None:
            gt = new_gt[str(prob_id)]
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

        if new_gt != None:
            sqa_results["results"][prob_id] = get_pred_idx(
                answer, choices, args.options, np.argwhere(choices == new_gt[str(prob_id)]).flatten()[0]
            )
        else:
            sqa_results["results"][prob_id] = get_pred_idx(answer, choices, args.options, prob["answer"])

        sqa_results["outputs"][prob_id] = pred_text

        if choices[pred_idx] == gt:
            results["correct"].append(analysis)
        else:
            results["incorrect"].append(analysis)

        if new_gt != None:
            pred_obj.append(
                {
                    "question_id": prob_id,
                    "pred_idx": pred_idx,
                    "ground_truth": new_gt[str(prob_id)],
                }
            )
        else:
            pred_obj.append(
                {
                    "question_id": prob_id,
                    "pred_idx": pred_idx,
                    "ground_truth": prob["answer"],
                }
            )
    # print('len correct', len(results['correct']))
    # print('len incorect', len(results['incorrect']))

    try:
        correct = len(results["correct"])
        total = len(results["correct"]) + len(results["incorrect"])
        print(f"Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%")
    except:
        print(results)

    sqa_results["acc"] = correct / total * 100
    sqa_results["correct"] = correct
    sqa_results["count"] = total
    print(f"% failed = {failed/total:.3f} | n fail = {failed} | total = {total}")
    print("")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, "w") as f:
        json.dump(sqa_results, f, indent=2)
