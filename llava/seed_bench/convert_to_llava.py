import sys

sys.path.insert(0, "/home/ubuntu")

import json
import os
import fire
import re

from datasets import load_dataset

from tqdm import tqdm
from PIL import Image
import numpy as np


def create_one_example_chatbot(format, question, context, choice, answer, lecture, solution, test_example=True):
    input_format, output_format = format.split("-")
    ## Inputs
    input = f"Question: {question}\nOptions: {choice}\n"
    # if input_format == "CQM":
    #     input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    # elif input_format == "QCM":
    #     input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # # upper bound experiment
    # elif input_format == "QCML":
    #     input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    # elif input_format == "QCME":
    #     input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    # elif input_format == "QCMLE":
    #     input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    # elif input_format == "QCLM":
    #     input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    # elif input_format == "QCEM":
    #     input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    # elif input_format == "QCLEM":
    #     input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == "A":
        output = f"Answer: The answer is {answer}."

    elif output_format == "AL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == "AE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == "ALE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == "AEL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == "LA":
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == "EA":
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == "LEA":
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == "ELA":
        output = f"Answer: {solution} {lecture} The answer is {answer}."
    elif output_format == "LEPA":
        output = ""
        if len(lecture.strip()) > 0:
            output += f"LECTURE: {lecture}\n"
        if len(solution.strip()) > 0:
            output += f"SOLUTION: {solution}\n"
        output += "###\n"
        output += f"ANSWER: {answer}."

    input = input.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    if input.endswith("BECAUSE:"):
        input = input.replace("BECAUSE:", "").strip()
    if output.endswith("BECAUSE:"):
        output = output.replace("BECAUSE:", "").strip()
    return input, output


def convert_to_llava(base_dir, split, prompt_format="QCM-LEA"):
    target_format = []
    choice_keys = ["choice_a", "choice_b", "choice_c", "choice_d"]
    dataset = load_dataset("AILab-CVC/SEED-Bench")["test"]
    image_store = os.path.join(base_dir, split)
    os.makedirs(image_store, exist_ok=True)
    for sample in tqdm(dataset):
        image_path = os.path.join("/home/ubuntu/SEED-Bench/SEED-Bench-image", sample["data_id"])
        image = Image.open(image_path)
        prob_id = sample["question_id"]
        answer = sample["answer"]
        if answer == None:
            continue
        image_save_path = os.path.join(image_store, f"{prob_id}.png")
        image.save(image_save_path)

        question = sample["question"]
        context = ""
        choice = [sample[c] for c in choice_keys]

        lecture = ""
        solution = ""
        input, output = create_one_example_chatbot(prompt_format, question, context, choice, answer, lecture, solution)
        if input.startswith("Question: "):
            input = input.replace("Question: ", "")
        if output.startswith("Answer: "):
            output = output.replace("Answer: ", "")
        raw_prob_data = sample
        if raw_prob_data["data_id"] is None:
            target_format.append(
                {
                    "id": str(prob_id),
                    "conversations": [
                        {"from": "human", "value": f"{input}"},
                        {"from": "gpt", "value": f"{output}"},
                    ],
                }
            )

        else:
            target_format.append(
                {
                    "id": str(prob_id),
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": f"{input}\n<image>"},
                        {"from": "gpt", "value": f"{output}"},
                    ],
                }
            )

    print(f"Number of samples: {len(target_format)}")

    with open(os.path.join(base_dir, f"llava_{split}_{prompt_format}.json"), "w") as f:
        json.dump(target_format, f, indent=2)


def attack_choice_text(options_dict, new_idx, answer_idx):
    def swap_choices(choices, pos1, pos2):
        choices_swapped = np.copy(choices).tolist()
        choices_swapped[pos1], choices_swapped[pos2] = choices_swapped[pos2], choices_swapped[pos1]
        return choices_swapped

    choices = [options_dict[key] for key in options_dict]  # ex choices: ['chiasmus', 'apostrophe']
    options = [key for key in options_dict]
    # answer_idx = problem['answer'] # ex answer = 1 (starts from 0)
    default_idx = answer_idx
    choices_swapped = swap_choices(choices, default_idx, new_idx)

    choice_list = []
    for i, c in enumerate(choices_swapped):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt


def convert_attack(base_dir, dataset, split, prompt_format="QCM-LEA", attack_idx=None, n_options=2):
    choice_keys = ["choice_a", "choice_b", "choice_c", "choice_d"]
    option_candidate = np.array(["A", "B", "C", "D", "E"])
    image_store = os.path.join(base_dir, split)
    target_format = []
    for sample in tqdm(dataset):
        if sample["data_type"] != "image":
            continue
        image_path = os.path.join("/home/ubuntu/SEED-Bench/SEED-Bench-image", sample["data_id"])
        # image = Image.open(image_path)
        prob_id = sample["question_id"]
        answer = sample["answer"]
        if answer == None:
            continue
        # image_save_path = os.path.join(image_store, f'{prob_id}.png')
        # image.save(image_save_path)

        question = sample["question"]
        context = ""
        choice = [sample[c] for c in choice_keys]
        choice_text = ""
        for i, c in enumerate(choice_keys):
            choice_text += f"({option_candidate[i]}) {sample[c]} "
        choice_text = choice_text.strip().rstrip()
        lecture = ""
        solution = ""

        answer_idx = np.argwhere(option_candidate == answer).flatten()[0]
        options_dict = {option_candidate[i]: c for i, c in enumerate(choice)}
        if attack_idx == None or (attack_idx == answer_idx):
            choice = choice_text
            new_gt = answer
        else:
            new_gt = option_candidate[attack_idx]
            choice = attack_choice_text(options_dict, attack_idx, answer_idx)
        input, output = create_one_example_chatbot(prompt_format, question, context, choice, answer, lecture, solution)

        if input.startswith("Question: "):
            input = input.replace("Question: ", "")
        if output.startswith("Answer: "):
            output = output.replace("Answer: ", "")
        raw_prob_data = sample
        if raw_prob_data["question_id"] is None:
            target_format.append(
                {
                    "id": str(prob_id),
                    "conversations": [
                        {"from": "human", "value": f"{input}"},
                        {"from": "gpt", "value": f"{output}"},
                    ],
                    "new_gt": new_gt,
                }
            )

        else:
            target_format.append(
                {
                    "id": str(prob_id),
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": f"{input}\n<image>"},
                        {"from": "gpt", "value": f"{output}"},
                    ],
                    "new_gt": new_gt,
                }
            )

    print(f"Number of samples: {len(target_format)}")

    save_dir = "stratified_attack"
    if not os.path.isdir(os.path.join(base_dir, save_dir)):
        os.makedirs(os.path.join(base_dir, save_dir))
    if attack_idx == None:
        file_path = os.path.join(base_dir, save_dir, f"original_noption_{n_options}_{split}.json")
    else:
        file_path = os.path.join(base_dir, save_dir, f"attack_choice_{attack_idx}_noption_{n_options}_{split}.json")
    with open(os.path.join(base_dir, file_path), "w") as f:
        json.dump(target_format, f, indent=2)


def attack_llava(base_dir, split, prompt_format="QCM-LEA"):
    dataset = load_dataset("AILab-CVC/SEED-Bench")["test"]
    image_store = os.path.join(base_dir, split)
    os.makedirs(image_store, exist_ok=True)
    n_options = 4
    convert_attack(base_dir, dataset, split, prompt_format, n_options=n_options)  # save original (unattacked) one
    for j in range(n_options):
        convert_attack(
            base_dir, dataset, split, prompt_format, attack_idx=j, n_options=n_options
        )  # save original (unattacked) one


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
