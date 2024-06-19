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
    results_clean = {"correct_yes": 0, "incorrect_yes": 0, "correct_no": 0, "incorrect_no": 0}

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

    both_yes = 0
    both_no = 0
    correct_yes_clean = 0
    correct_no_clen = 0
    incorrect_yes_clean = 0
    incorrect_no = 0

    for i, pred_obj in enumerate(predictions):
        if data_obj[i]["id"] != pred_obj["question_id"]:
            print("NO MATCH")
        prob_id = pred_obj["question_id"]
        pred_text = pred_obj["text"]
        if "idefics" in args.result_file:
            pred_text = pred_text.split("\nAssistant: ")[-1].strip().rstrip()
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

        if (i % 2) != 0:
            if preds[-1] == preds[-2]:
                # print('BOTH', i, i-1, preds[-1], preds[-2])
                if pred_idx == 0:
                    both_no += 2
                else:
                    both_yes += 2
            else:
                # print('NOT', i, i-1, preds[-1], preds[-2])
                prev_pred = preds[-2]
                prev_gt = gt_all[-2]

                # if prev_pred == prev_gt:
                clean_preds.extend([prev_pred, pred_idx])
                clean_gt.extend([prev_gt, gt])
                # if prev_gt == 0:
                #     results_clean['correct_no'] +=1
                # else:
                #     results_clean['correct_yes'] +=1
                # else:
                #     if prev_gt == 0:
                #         results_clean['incorrect_no'] +=1
                #     else:
                #         results_clean['incorrect_yes'] +=1

                # if pred_idx == gt:
                #     if gt == 0:
                #         results_clean['correct_no'] +=1
                #     else:
                #         results_clean['correct_yes'] +=1
                # else:
                #     if gt == 0:
                #         results_clean['incorrect_no'] +=1
                #     else:
                #         results_clean['incorrect_yes'] +=1

    correct_yes = len(results["correct_yes"])
    correct_no = len(results["correct_no"])
    total_yes = len(results["correct_yes"]) + len(results["incorrect_yes"])
    total_no = len(results["correct_no"]) + len(results["incorrect_no"])
    # print(correct_yes, correct_no, total_yes, total_no)
    acc_yes = correct_yes / total_yes * 100
    acc_no = correct_no / total_no * 100
    print(f"YES: Total {total_yes}, Correct: {correct_yes}, Accuracy: {acc_yes:.2f}%")
    print(f"NO: Total {total_no}, Correct: {correct_no}, Accuracy: {acc_no:.2f}%")
    if "yesno" in args.data_file:
        print(f"DIFF: {acc_yes-acc_no:.3f}%")
    else:
        print(f"DIFF: {acc_no-acc_yes:.3f}%")
    acc_total = (correct_yes + correct_no) / (total_yes + total_no) * 100
    print(f"ALL Accuracy: {acc_total:.2f}%")
    sqa_results["acc_yes"] = correct_yes / total_yes * 100
    sqa_results["acc_no"] = correct_no / total_no * 100
    sqa_results["correct_yes"] = correct_yes
    sqa_results["correct_no"] = correct_no
    sqa_results["count_yes"] = total_yes
    sqa_results["count_no"] = total_no
    sqa_results["acc dif"] = (correct_yes / total_yes * 100) - (correct_no / total_no * 100)
    print(f"% failed = {failed/(total_yes+total_no):.3f} | n fail = {failed} | total = {total_yes+total_no}")
    print("")
    # print(f"BOTH YES | count {both_yes} | acc % {100*both_yes/(total_yes+total_no):.3f}%")
    # print(f"BOTH NO | count {both_no} | acc % {100*both_no/(total_yes+total_no):.3f}%")
    # print("")
    # print("DISCARDING SAMPLES WITH BOTH YES/NO")

    # print(len(clean_gt), len(clean_preds), len(preds), len(gt_all))
    clean_preds = np.array(clean_preds)
    clean_gt = np.array(clean_gt)
    total_yes = len(np.argwhere(clean_gt == 1))
    correct_yes = len(np.argwhere((clean_gt == 1) & (clean_preds == clean_gt)))
    total_no = len(np.argwhere(clean_gt == 0))
    correct_no = len(np.argwhere((clean_gt == 0) & (clean_preds == clean_gt)))
    # print(total_yes, total_no)
    # correct_yes = results_clean['correct_yes']
    # correct_no = results_clean['correct_no']
    # total_no = results_clean['correct_no'] + results_clean['incorrect_no']
    # print(correct_yes, correct_no, total_yes, total_no)
    # print(f'YES: Total {total_yes}, Correct: {correct_yes}, Accuracy: {correct_yes / total_yes * 100:.2f}%')
    # print(f'NO: Total {total_no}, Correct: {correct_no}, Accuracy: {correct_no / total_no * 100:.2f}%')

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, "w") as f:
        json.dump(sqa_results, f, indent=2)
