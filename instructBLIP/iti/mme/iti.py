import numpy as np
from einops import rearrange

from tqdm import tqdm
import os

from baukit import Trace, TraceDict

from utils import *

import torch

from PIL import Image
import json
import math
from functools import partial
import shortuuid

import argparse

from tqdm import tqdm

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

import sys

sys.path.append("../")
from utils import train_single_prob


def get_probe_accuracies():
    val_ratio = 0.2
    n_samples = 50

    head_truth = np.load(f"{probe_dir}/truth/head_{n_samples}.npy")
    labels_truth = np.load(f"{probe_dir}/truth/labels_{n_samples}.npy")
    head_truth = rearrange(head_truth, "b l (h d) -> b l h d", h=num_heads)

    head_non_truth = np.load(f"{probe_dir}/non_truth/head_{n_samples}.npy")
    labels_non_truth = np.load(f"{probe_dir}/non_truth/labels_{n_samples}.npy")
    head_non_truth = rearrange(head_non_truth, "b l (h d) -> b l h d", h=num_heads)

    X_all = np.vstack((head_truth, head_non_truth))
    y_all = np.hstack((labels_truth, labels_non_truth))

    head_perf_dict = {f"l{l}_h{h}": [] for l in range(num_layers) for h in range(num_heads)}
    probes_dict = {f"l{l}_h{h}": None for l in range(num_layers) for h in range(num_heads)}
    for l in tqdm(range(num_layers)):
        for h in range(num_heads):
            X_probe = X_all[:, l, h, :]
            y_probe = y_all[:]
            probe, val_acc = train_single_prob(X_probe, y_probe, val_size=val_ratio)
            head_perf_dict[f"l{l}_h{h}"] = val_acc
            probes_dict[f"l{l}_h{h}"] = probe

    head_perf_dict_mean = {k: np.mean(v) for k, v in head_perf_dict.items()}
    l_h_means = np.array(list(head_perf_dict_mean.values())).reshape(
        num_heads, num_layers
    )  # row = heads | colums = layers
    return l_h_means, probes_dict, X_all


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def get_top_heads(l_h_means, num_to_intervene):
    top_accs = np.argsort(l_h_means.reshape(num_heads * num_layers))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    return top_heads


def get_interventions_dict(top_heads, probes_dict, tuning_activations):
    interventions = {}
    probes = np.array(list(probes_dict.values()))
    for layer, head in top_heads:
        interventions[f"language_model.model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in top_heads:
        direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:, layer, head, :]  # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"language_model.model.layers.{layer}.self_attn.o_proj"].append(
            (head, direction.squeeze(), proj_val_std)
        )
    for layer, head in top_heads:
        interventions[f"language_model.model.layers.{layer}.self_attn.o_proj"] = sorted(
            interventions[f"language_model.model.layers.{layer}.self_attn.o_proj"], key=lambda x: x[0]
        )
    return interventions


def lt_modulated_vector_add(head_output, layer_name):
    head_output = rearrange(head_output, "b s (h d) -> b s h d", h=num_heads)
    head_output = head_output.cuda()
    for head, direction, proj_val_std in interventions[layer_name]:
        direction_to_add = torch.tensor(direction).cuda()
        head_output[:, -1, head, :] += alpha * proj_val_std * direction_to_add
    head_output = rearrange(head_output, "b s h d -> b s (h d)")
    head_output = head_output.cuda()
    return head_output


def get_answer_with_intervention(model, processor, q, image, interventions={}, intervention_fn=None):
    # --- intervention code --- #
    inputs = processor(images=image, text=q, return_tensors="pt").to(device="cuda")

    def id(head_output, layer_name):
        return head_output

    if interventions == {}:
        intervene = id
        layers_to_intervene = []
    else:
        intervene = partial(intervention_fn)
        layers_to_intervene = list(interventions.keys())
    # --- intervention code --- #
    with torch.inference_mode():
        with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
            generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    torch.cuda.empty_cache()
    return generated_text


def edit_model(args):
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    question_file = args.question_file
    questions = json.load(open(question_file, "r"))

    for obj_ in tqdm(questions):
        idx = obj_["id"]
        q = obj_["conversations"][0]["value"].split("<image>")[0].rstrip().replace("\n", " ")
        image = Image.open(obj_["image"])

        intervened_answer = get_answer_with_intervention(
            model, processor, q, image, interventions=interventions, intervention_fn=lt_modulated_vector_add
        )

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": q,
                    "text": intervened_answer,
                    "answer_id": ans_id,
                }
            )
            + "\n"
        )
        ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--probe-split", type=str, default="val")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--alpha", type=float, default=15)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    alpha = args.alpha
    k = args.k
    probe_dir = f"features_50/{args.probe_split}"

    num_heads = 32
    num_layers = 32

    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-13b", device_map="auto"
    )
    model.tie_weights()

    device = "cuda"
    print("getting probe activations accuracies...")
    l_h_means, probes_dict, head_vals = get_probe_accuracies()
    l_h_means_flattened = l_h_means.flatten()

    num_to_intervene = args.k
    top_heads = get_top_heads(l_h_means, num_to_intervene)
    interventions = get_interventions_dict(top_heads, probes_dict, head_vals)

    edit_model(args)
