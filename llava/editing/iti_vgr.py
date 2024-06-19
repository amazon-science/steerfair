import numpy as np
from einops import rearrange

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import linalg

import os

from baukit import Trace, TraceDict

from llava.model import *
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from utils import *

from transformers import AutoTokenizer
import torch

from PIL import Image
import json
import math
from functools import partial
import shortuuid

import argparse

from tqdm import tqdm


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_models():
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    return tokenizer, model, image_processor


def lt_modulated_vector_add(head_output, layer_name, start_edit_location="lt"):
    head_output = rearrange(head_output, "b s (h d) -> b s h d", h=num_heads)
    head_output = head_output.cuda()
    for head, direction, proj_val_std in interventions[layer_name]:
        direction_to_add = torch.tensor(direction).cuda()
        if start_edit_location == "lt":
            head_output[:, -1, head, :] += alpha * proj_val_std * direction_to_add
        else:
            if not reverse:
                head_output[:, start_edit_location:, head, :] += alpha * proj_val_std * direction_to_add
            else:
                head_output[:, start_edit_location:, head, :] -= alpha * proj_val_std * direction_to_add
    head_output = rearrange(head_output, "b s h d -> b s (h d)")
    head_output = head_output.cuda()
    return head_output


def get_com_directions(num_layers, num_heads, head_activations, labels):
    com_directions = []
    for layer in range(num_layers):
        for head in range(num_heads):
            true_mass_mean = np.mean(head_activations[labels == 1], axis=0)
            false_mass_mean = np.mean(head_activations[labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)
    return com_directions


def get_interventions_dict(top_heads, probes_dict, tuning_activations, use_center_of_mass=False, com_directions=None):
    interventions = {}
    probes = np.array(list(probes_dict.values()))
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in top_heads:
        if use_center_of_mass:
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        else:
            direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:, layer, head, :]  # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, direction.squeeze(), proj_val_std))
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(
            interventions[f"model.layers.{layer}.self_attn.o_proj"], key=lambda x: x[0]
        )
    return interventions


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def get_top_heads(l_h_means, num_to_intervene):
    top_accs = np.argsort(l_h_means.reshape(num_heads * num_layers))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    return top_heads


def get_answer_with_intervention(
    model, tokenizer, prompt, images, stopping_criteria, stop_str, interventions={}, intervention_fn=None
):
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # --- intervention code --- #
    def id(head_output, layer_name):
        return head_output

    if interventions == {}:
        intervene = id
        layers_to_intervene = []
    else:
        intervene = partial(intervention_fn, start_edit_location="lt")
        layers_to_intervene = list(interventions.keys())
    # --- intervention code --- #
    input_token_len = input_ids.shape[1]
    if interventions == {}:
        with torch.inference_mode():
            model_output = model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=stopping_criteria,
                use_cache=True,
            )
    else:
        with torch.inference_mode():
            with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
                model_output = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=stopping_criteria,
                    use_cache=True,
                )
    model_gen_str = tokenizer.batch_decode(model_output[:, input_token_len:], skip_special_tokens=True)[0]
    model_gen_str = model_gen_str.strip()
    if model_gen_str.endswith(stop_str):
        model_gen_str = model_gen_str[: -len(stop_str)]
    model_gen_str = model_gen_str.strip()
    torch.cuda.empty_cache()
    return model_gen_str


def build_prompt(line, model, image_processor, tokenizer, perturb_image=False):
    question = line["conversations"][0]
    qs = question["value"].replace("<image>", "").strip()
    cur_prompt = qs

    image_file = line["image"]
    # image = Image.open(os.path.join(image_folder, image_file))
    image = Image.open(image_file)
    if perturb_image:
        image = random_perturb_image(image)

    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    images = image_tensor.unsqueeze(0).half().cuda()
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    cur_prompt = "<image>" + "\n" + cur_prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

    prompt = conv.get_prompt()
    return cur_prompt, prompt, images, stopping_criteria, stop_str


def get_probe_accuracies():
    val_ratio = 0.2
    n_samples = 50

    head_truth = np.load(f"{probe_dir}/truth/head_wise_{n_samples}.npy")
    labels_truth = np.load(f"{probe_dir}/truth/labels_{n_samples}.npy")
    head_truth = rearrange(head_truth, "b l (h d) -> b l h d", h=num_heads)

    head_non_truth = np.load(f"{probe_dir}/non_truth/head_wise_{n_samples}.npy")
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


def edit_model(args):
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    question_file = args.question_file

    questions = json.load(open(question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    tokenizer, model, image_processor = get_models()

    for obj_ in tqdm(questions):
        idx = obj_["id"]
        cur_prompt, prompt, images, stopping_criteria, stop_str = build_prompt(
            obj_, model, image_processor, tokenizer, args.perturb_image
        )

        intervened_answer = get_answer_with_intervention(
            model,
            tokenizer,
            prompt,
            images,
            stopping_criteria,
            stop_str,
            interventions=interventions,
            intervention_fn=lt_modulated_vector_add,
        )

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": intervened_answer,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--probe-split", type=str, default="train")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--alpha", type=float, default=15)
    parser.add_argument("--use-center-of-mass", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--perturb-image", action="store_true")
    args = parser.parse_args()

    alpha = args.alpha
    # image_folder = f'/home/ubuntu/VG_Relation/images'
    model_path = "liuhaotian/llava-v1.5-13b"
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    conv_mode = "llava_v1"

    probe_dir = f"../probing/vgr_probing/mme/features_truthful/{args.probe_split}"

    device = "cuda"
    # HARDCODED FOR LLAVA
    num_heads = 40
    num_layers = 40
    reverse = args.reverse

    if args.k > 0:
        print("getting probe activations accuracies...")
        l_h_means, probes_dict, head_vals = get_probe_accuracies()
        l_h_means_flattened = l_h_means.flatten()

        num_to_intervene = args.k
        top_heads = get_top_heads(l_h_means, num_to_intervene)
        interventions = get_interventions_dict(top_heads, probes_dict, head_vals)
    else:
        interventions = {}
    edit_model(args)
