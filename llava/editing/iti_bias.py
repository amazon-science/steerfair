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
from iti import get_answer_with_intervention

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


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def get_top_heads(l_h_means, num_to_intervene):
    top_accs = np.argsort(l_h_means.reshape(num_heads * num_layers))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    return top_heads


# def get_answer_with_intervention(model, tokenizer, prompt, images, stopping_criteria, interventions={}, intervention_fn=None):
#     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
#     # --- intervention code --- #
#     def id(head_output, layer_name):
#         return head_output
#     if interventions == {}:
#         intervene = id
#         layers_to_intervene = []
#     else:
#         # print("INTERVENING!")
#         intervene = partial(intervention_fn, start_edit_location='lt')
#         layers_to_intervene = list(interventions.keys())
#     # --- intervention code --- #
#     sequences = []
#     with torch.no_grad():
#         max_len = input_ids.shape[-1] + 50
#         # --- intervention code --- #
#         with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
#             model_output = model.generate(input_ids,
#                                           images=images,
#                                           do_sample=True,
#                                           max_new_tokens=1024,
#                                           stopping_criteria=stopping_criteria,
#                                          )
#             model_gen_tokens = model_output['sequences'][:, input_ids.shape[-1]:]
#         model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
#         model_gen_str = model_gen_str.strip()
#         # --- intervention code --- #
#     if device:
#         torch.cuda.empty_cache()
#     return model_gen_str


def build_prompt(line, model, image_processor, tokenizer):
    question = line["conversations"][0]
    qs = question["value"].replace("<image>", "").strip()
    cur_prompt = qs
    if "image" in line:
        image_file = line["image"]
        image = Image.open(os.path.join(image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        images = image_tensor.unsqueeze(0).half().cuda()
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        cur_prompt = "<image>" + "\n" + cur_prompt
    else:
        images = None

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

    prompt = conv.get_prompt()
    return cur_prompt, prompt, images, stopping_criteria, stop_str


def lt_modulated_vector_add(head_output, layer_name, start_edit_location="lt"):
    head_output = rearrange(head_output, "b s (h d) -> b s h d", h=num_heads).cuda()
    for head, direction, proj_val_std in interventions[layer_name]:
        direction_to_add = torch.tensor(direction).cuda()
        if start_edit_location == "lt":
            head_output[:, -1, head, :] += alpha * proj_val_std * direction_to_add
        else:
            if not reverse:
                head_output[:, start_edit_location:, head, :] += alpha * proj_val_std * direction_to_add
            else:
                head_output[:, start_edit_location:, head, :] -= (1 / alpha) * proj_val_std * direction_to_add
    head_output = rearrange(head_output, "b s h d -> b s (h d)")
    return head_output


def get_interventions_dict(top_heads, probes_dict, tuning_activations, num_to_intervene):
    interventions = {}
    probes = np.array(list(probes_dict.values()))
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in top_heads:
        direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_[0, :]
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


def get_probe_accuracies():
    val_ratio = 0.2
    n_samples = 200

    # head_truth = np.load(f"{probe_dir}/truth/head_wise_{n_samples}_llm.npy")
    # labels_truth = np.load(f"{probe_dir}/truth/labels_{n_samples}.npy")
    # head_truth = rearrange(head_truth, 'b l (h d) -> b l h d', h = num_heads)

    head_bias = np.load(f"{probe_dir}/bias/head_wise_{n_samples}_llm.npy")
    labels_bias = np.load(f"{probe_dir}/bias/labels_{n_samples}.npy")
    head_bias = rearrange(head_bias, "b l (h d) -> b l h d", h=num_heads)

    # X_all = np.vstack((head_truth, head_bias))
    # y_all = np.hstack((labels_truth, labels_bias))

    X_all = head_bias
    y_all = np.array(labels_bias) - 1
    y_all = y_all.tolist()

    head_perf_dict = {f"l{l}_h{h}": [] for l in range(num_layers) for h in range(num_heads)}
    probes_dict = {f"l{l}_h{h}": None for l in range(num_layers) for h in range(num_heads)}
    for l in tqdm(range(num_layers)):
        for h in range(num_heads):
            X_probe = X_all[:, l, h, :]
            y_probe = y_all[:]
            probe, val_acc = train_single_prob(X_probe, y_probe, val_size=val_ratio)
            head_perf_dict[f"l{l}_h{h}"] = val_acc
            probes_dict[f"l{l}_h{h}"] = probe

    head_perf_dict = {k: v for k, v in head_perf_dict.items()}
    perf_matrix = np.array(list(head_perf_dict.values())).reshape(
        num_heads, num_layers
    )  # row = heads | colums = layers
    return perf_matrix, probes_dict, X_all


def edit_model(args):
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    options = ["A", "B", "C", "D", "E"]
    question_file = args.question_file

    questions = json.load(open(question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    tokenizer, model, image_processor = get_models()
    for obj_ in tqdm(questions):
        idx = obj_["id"]
        question = obj_["conversations"][0]
        answer_idx = obj_["new_gt"]
        # prompt, images, stopping_criteria = build_prompt(question)
        cur_prompt, prompt, images, stopping_criteria, stop_str = build_prompt(obj_, model, image_processor, tokenizer)
        # try:
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
                    "prompt": prompt,
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
    # parser.add_argument("--acc-threshold", type=float, required=True)
    parser.add_argument("--probe-split", type=str, default="train")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--alpha", type=int, default=15)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    alpha = args.alpha

    if "mini" in args.split:
        split = args.split.split("mini")[-1]
        image_folder = f"/home/ubuntu/ScienceQA/{split}"
    else:
        image_folder = f"/home/ubuntu/ScienceQA/{args.split}"
    model_path = "liuhaotian/llava-v1.5-13b"
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    conv_mode = "llava_v1"
    probe_dir = f"../probing/features_bias_2/{args.probe_split}"

    device = "cuda"
    # HARDCODED FOR LLAVA
    num_heads = 40
    num_layers = 40

    # acc_threshold = args.acc_threshold
    reverse = args.reverse

    print("getting probe activations accuracies...")
    l_h_means, probes_dict, head_vals = get_probe_accuracies()
    l_h_means_flattened = l_h_means.flatten()
    num_to_intervene = args.k

    # print(f"Number of heads with probe acc > {acc_threshold} = {len(np.argwhere(l_h_means_flattened > acc_threshold))}")

    top_heads = get_top_heads(l_h_means, num_to_intervene)
    interventions = get_interventions_dict(top_heads, probes_dict, head_vals, num_to_intervene)

    edit_model(args)
