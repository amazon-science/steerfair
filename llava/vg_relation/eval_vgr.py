import argparse
import json
import os
import re
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--output-result", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--options", type=list, default=["A", "B"])
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
    new_gt = {obj_["id"]: obj_["answer_idx"] for obj_ in data_obj}

    results = {"correct": [], "incorrect": []}
    sqa_results = {}
    sqa_results["acc"] = None
    sqa_results["correct"] = None
    sqa_results["count"] = None
    sqa_results["results"] = {}
    sqa_results["outputs"] = {}

    failed = 0
    for pred_obj in predictions:
        # print(pred_obj.keys())
        prob_id = pred_obj["question_id"]
        # pred = predictions[prob_id]
        # print(prob.keys())
        pred_text = pred_obj["text"]

        if len(pred_text) == 1:
            answer = pred_text[0].upper()
            if answer not in options:
                answer = "FAILED"
                failed += 1
            pred_idx = get_pred_idx(answer, ["A", "B"], args.options)
        else:
            re_A = [r"is \(A\)", r"\(A", r"A\)", r"A.", r"option A", r"choice A"]
            re_B = [r"is \(B\)", r"\(B", r"B\)", r"B.", r"option B", r"choice B"]
            re_all = [re_A, re_B]
            re_all = [item for sublist in re_all for item in sublist]
            found = False
            for regex in re_all:
                pattern = re.compile(regex)
                res = pattern.findall(pred_text)
                if len(res) == 1:
                    if regex in re_A:
                        answer = "A"
                    elif regex in re_B:
                        answer = "B"
                    found = True
                    break
            if found == False:
                answer = "FAILED"
                failed += 1
            pred_idx = get_pred_idx(answer, ["A", "B"], args.options)

        analysis = {
            "question_id": prob_id,
            "parsed_ans": answer,
            "pred_idx": pred_idx,
            "ground_truth": new_gt[prob_id],
            "question": pred_obj["prompt"],
            "pred": pred_text,
            "is_multimodal": "<image>" in pred_obj["prompt"],
        }

        sqa_results["results"][prob_id] = get_pred_idx(answer, ["A", "B"], args.options)
        sqa_results["outputs"][prob_id] = pred_text

        if pred_idx == new_gt[prob_id]:
            results["correct"].append(analysis)
        else:
            results["incorrect"].append(analysis)

    correct = len(results["correct"])
    total = len(results["correct"]) + len(results["incorrect"])
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%")

    sqa_results["acc"] = correct / total * 100
    sqa_results["correct"] = correct
    sqa_results["count"] = total
    print(f"% failed = {failed/total:.3f} | n fail = {failed} | total = {total}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, "w") as f:
        json.dump(sqa_results, f, indent=2)
