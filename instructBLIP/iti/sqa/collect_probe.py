import argparse
import os
import json
from tqdm import tqdm
import numpy as np

import math

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from baukit import TraceDict
from PIL import Image


def get_single_activation(model, HEADS, MLPS, inputs):
    with torch.inference_mode():
        with TraceDict(model, HEADS + MLPS, retain_input=True) as ret:
            generated_ids = model(**inputs, output_hidden_states=True)
    layer_wise_activations = generated_ids["language_model_outputs"].hidden_states
    layer_wise_activations = torch.stack(layer_wise_activations, dim=0).squeeze()
    layer_wise_activations = layer_wise_activations.detach().cpu().numpy()

    head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
    head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

    return layer_wise_activations, head_wise_activations


def collect_probe(args):
    options = ["A", "B", "C", "D", "E"]

    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    questions = json.load(open(os.path.expanduser(question_file), "r"))
    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    questions = [q for q in questions if q["id"] in split_indices if "image" in q]
    questions = np.random.choice(questions, n_samples)
    all_head_wise_activations_truth = []
    all_head_wise_activations_non_truth = []
    y_all_truth = []
    y_all_non_truth = []

    for line in tqdm(questions):
        idx = line["id"]
        n_choices = len(problems[idx]["choices"])

        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip().rstrip("\n").rstrip()

        answer_idx = problems[idx]["answer"]
        y_non_truth = 0
        wrong_answer = np.random.choice(np.delete(np.arange(n_choices), answer_idx), 1)[0]
        answer_non_truth = f"{options[wrong_answer]}"

        y_truth = 1
        answer_truth = f"{options[answer_idx]}"

        if "image" in line:
            image = Image.open(os.path.join(image_folder, line["image"]))
        else:
            image = Image.new("RGB", (20, 20))

        prompt_truth = qs + " " + answer_truth
        prompt_non_truth = qs + " " + answer_non_truth

        inputs = processor(images=image, text=prompt_truth, return_tensors="pt").to(device="cuda")
        _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
        all_head_wise_activations_truth.append(head_wise_activations[:, -1, :])
        y_all_truth.append(y_truth)

        inputs = processor(images=image, text=prompt_non_truth, return_tensors="pt").to(device="cuda")
        _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
        all_head_wise_activations_non_truth.append(head_wise_activations[:, -1, :])
        y_all_non_truth.append(y_non_truth)

    print("Saving labels")
    np.save(f"{save_dir_truth}/labels_{n_samples}.npy", y_all_truth)

    print("Saving labels")
    np.save(f"{save_dir_non_truth}/labels_{n_samples}.npy", y_all_non_truth)

    print("Saving head wise activations")
    np.save(f"{save_dir_truth}/head_{n_samples}.npy", all_head_wise_activations_truth)

    print("Saving head wise activations")
    np.save(f"{save_dir_non_truth}/head_{n_samples}.npy", all_head_wise_activations_non_truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    device = "cuda"
    n_samples = args.n_samples
    question_file = f"~/ScienceQA/data/scienceqa/llava_{args.split}_QCM-LEA.json"
    if "mini" in args.split:
        split = args.split.split("mini")[-1]
        image_folder = f"/home/ubuntu/ScienceQA/{split}"
    else:
        image_folder = f"/home/ubuntu/ScienceQA/{args.split}"
    base_dir = "/home/ubuntu/ScienceQA/data/scienceqa"

    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-13b", device_map="auto"
    )
    model.tie_weights()

    HEADS = [
        f"language_model.model.layers.{i}.self_attn.o_proj"
        for i in range(model.language_model.config.num_hidden_layers)
    ]
    MLPS = [f"language_model.model.layers.{i}.mlp" for i in range(model.language_model.config.num_hidden_layers)]

    save_dir_non_truth = f"features/{args.split}/non_truth"
    save_dir_truth = f"features/{args.split}/truth"
    os.makedirs(save_dir_truth, exist_ok=True)
    os.makedirs(save_dir_non_truth, exist_ok=True)

    collect_probe(args)
