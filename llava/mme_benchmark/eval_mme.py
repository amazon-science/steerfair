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
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--output-result", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--options", type=list, default=[0, 1])
    return parser.parse_args()


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[: len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


if __name__ == "__main__":
    args = get_args()
    options = args.options
    data_file = args.data_file

    with open(data_file, "r") as data_file:
        json_data = data_file.read()
    data_obj = json.loads(json_data)
    predictions = [json.loads(line) for line in open(args.result_file)]
    results = {"correct_yes": [], "incorrect_yes": [], "correct_no": [], "incorrect_no": []}

    results_yes = {}
    results_no = {}

    sqa_results = {}
    sqa_results["acc"] = None
    sqa_results["correct"] = None
    sqa_results["count"] = None
    sqa_results["results"] = {}
    sqa_results["outputs"] = {}

    failed = 0
    preds = []
    gt_all = []

    clean_preds = []
    clean_gt = []

    results_by_category_all = {}
    results_by_category_yes = {}
    results_by_category_no = {}

    for i, pred_obj in enumerate(predictions):
        if data_obj[i]["id"] != pred_obj["question_id"]:
            print("NO MATCH")
        prob_id = pred_obj["question_id"]
        pred_text = pred_obj["text"]
        if len(pred_text) == 1:
            answer = pred_text[0].upper()
            if answer not in options:
                answer = "FAILED"
                failed += 1
            pred_idx = get_pred_idx(answer, [0, 1], args.options)
        else:
            re_yes = [r"Yes", r"yes"]
            re_no = [r"No", r"no"]
            re_all = [re_yes, re_no]
            re_all = [item for sublist in re_all for item in sublist]
            found = False
            for regex in re_all:
                pattern = re.compile(regex)
                res = pattern.findall(pred_text)
                if len(res) == 1:
                    if regex in re_yes:
                        answer = 1
                    elif regex in re_no:
                        answer = 0
                    found = True
                    break
            if found == False:
                answer = "FAILED"
                failed += 1
            pred_idx = get_pred_idx(answer, [0, 1], args.options)
        gt = data_obj[i]["answer_idx"]

        analysis = {
            "question_id": prob_id,
            "parsed_ans": answer,
            "pred_idx": pred_idx,
            "ground_truth": gt,
            "question": pred_obj["prompt"],
            "pred": pred_text,
            "is_multimodal": "<image>" in pred_obj["prompt"],
        }

        prob_id_both = prob_id[:-2]
        category = prob_id.split("_")[0]
        if category not in results_by_category_all:
            results_by_category_all[category] = {}
        results_by_category_all[category][prob_id] = {
            "parsed_ans": answer,
            "pred_idx": pred_idx,
            "correct": gt == pred_idx,
        }

        if category not in results_by_category_yes:
            results_by_category_yes[category] = {}
        if category not in results_by_category_no:
            results_by_category_no[category] = {}
        if gt == 1:
            results_yes[prob_id_both] = {"parsed_ans": answer, "pred_idx": pred_idx, "correct": gt == pred_idx}
            results_by_category_yes[category][prob_id_both] = {
                "parsed_ans": answer,
                "pred_idx": pred_idx,
                "correct": gt == pred_idx,
            }
        else:
            results_no[prob_id_both] = {"pred_idx": pred_idx, "ground_truth": gt, "correct": gt == pred_idx}
            results_by_category_no[category][prob_id_both] = {
                "parsed_ans": answer,
                "pred_idx": pred_idx,
                "correct": gt == pred_idx,
            }

        sqa_results["results"][data_obj[i]["id"]] = get_pred_idx(answer, [0, 1], args.options)
        sqa_results["outputs"][data_obj[i]["id"]] = pred_text

        if pred_idx == gt:
            if gt == 0:
                results["correct_no"].append(analysis)
            else:
                results["correct_yes"].append(analysis)
        else:
            if gt == 0:
                results["incorrect_no"].append(analysis)
            else:
                results["incorrect_yes"].append(analysis)

        preds.append(pred_idx)
        gt_all.append(gt)

    correct_yes = len(results["correct_yes"])
    correct_no = len(results["correct_no"])
    total_yes = len(results["correct_yes"]) + len(results["incorrect_yes"])
    total_no = len(results["correct_no"]) + len(results["incorrect_no"])

    acc_yes = correct_yes / total_yes * 100
    acc_no = correct_no / total_no * 100
    print(f"YES: Total {total_yes}, Correct: {correct_yes}, Accuracy: {acc_yes:.2f}%")
    print(f"NO: Total {total_no}, Correct: {correct_no}, Accuracy: {acc_no:.2f}%")

    acc_total = (correct_yes + correct_no) / (total_yes + total_no) * 100
    print(f"Accuracy: {acc_total:.2f}%")
    acc_plus = 0
    for id in results_yes:
        r_yes = results_yes[id]
        r_no = results_no[id]
        if r_yes["correct"] == 1 and r_no["correct"] == 1:
            acc_plus += 2
    acc_plus = acc_plus / (len(results_yes) + len(results_no)) * 100
    print(f"Accuracy+: {acc_plus:.2f}%")

    print(f"% failed = {failed/(total_yes+total_no):.3f} | n fail = {failed} | total = {total_yes+total_no}")
    print("")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, "w") as f:
        json.dump(sqa_results, f, indent=2)

    print("By Category")
    acc_by_category = {}
    acc_plus_by_category = {}
    # acc_all = 0
    for category in results_by_category_all:
        acc_category = 0
        for id in results_by_category_all[category]:
            pred = results_by_category_all[category][id]
            if pred["correct"] == 1:
                acc_category += 1
        acc_category = 100 * acc_category / len(results_by_category_all[category])
        acc_by_category[category] = acc_category
        # acc_all += acc_category
        print(f"{category} Accuracy = {acc_category:.3}%")
    print("")
    for category in results_by_category_yes:
        acc_plus_category = 0
        for id in results_by_category_yes[category]:
            pred_yes = results_by_category_yes[category][id]
            pred_no = results_by_category_no[category][id]
            if pred_yes["correct"] == 1 and pred_no["correct"] == 1:
                acc_plus_category += 2
        acc_plus_category = (
            100 * acc_plus_category / (len(results_by_category_no[category]) + len(results_by_category_yes[category]))
        )
        acc_plus_by_category[category] = acc_plus_category
        # acc_all += acc_category
        print(f"{category} Accuracy+ = {acc_plus_category:.3}%")

    print("")
    print("Scores")
    score_total = 0
    for category in acc_by_category:
        acc = acc_by_category[category]
        acc_plus = acc_plus_by_category[category]
        score = acc + acc_plus
        print(f"{category} score = {score:3f}")
        score_total += score
    print("")
    print(f"Total Score = {score_total}")
    print("")
