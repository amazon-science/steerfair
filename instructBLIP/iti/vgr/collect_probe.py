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
    options = ["no", "yes"]

    questions = json.load(open(os.path.join(base_dir, f"vgr_{args.split}_QCM_yesno.json"), "r"))
    questions = np.random.choice(questions, n_samples)

    all_head_wise_activations_truth = []
    all_head_wise_activations_non_truth = []
    y_all_truth = []
    y_all_non_truth = []

    for line in tqdm(questions):
        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "")

        answer_idx = line["answer_idx"]
        y_non_truth = 0
        wrong_answer = np.abs(answer_idx - 1)
        answer_non_truth = f"{options[wrong_answer]}"

        y_truth = 1
        answer_truth = f"{options[answer_idx]}"

        if "image" in line:
            image = Image.open(line["image"])
        else:
            image = None

        prompt_truth = qs + " " + answer_truth
        prompt_non_truth = qs + " " + answer_non_truth
        image = Image.open(line["image"])

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

    base_dir = "/home/ubuntu/VG_Relation"

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

    save_dir_non_truth = f"features_{n_samples}/{args.split}/non_truth"
    save_dir_truth = f"features_{n_samples}/{args.split}/truth"
    os.makedirs(save_dir_truth, exist_ok=True)
    os.makedirs(save_dir_non_truth, exist_ok=True)

    collect_probe(args)
