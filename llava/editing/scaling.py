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

from sklearn.metrics.pairwise import cosine_similarity
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


# same default alpha as ITI
def lt_scale(head_output, layer_name, start_edit_location="lt"):
    head_output = rearrange(head_output, "b s (h d) -> b s h d", h=num_heads)
    for head, cos in interventions[layer_name]:
        if start_edit_location == "lt":
            head_output[:, -1, head, :] = head_output[:, -1, head, :] * cos * scaling_power
        else:
            if not upscale:
                head_output[:, start_edit_location:, head, :] = head_output[:, -1, head, :] * cos * scaling_power
            else:
                head_output[:, start_edit_location:, head, :] = head_output[:, -1, head, :] * (1 / cos) * scaling_power
    head_output = rearrange(head_output, "b s h d -> b s (h d)")
    return head_output


def get_interventions_dict(top_heads, cosine_matrix):
    interventions = {}
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, cosine_matrix[layer, head]))
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(
            interventions[f"model.layers.{layer}.self_attn.o_proj"], key=lambda x: x[0]
        )
    return interventions


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def get_top_heads(cosine_mtrx, num_to_intervene):
    top_cosine = np.argsort(cosine_mtrx.reshape(num_heads * num_layers))[:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_cosine]
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
#         print("INTERVENING!")
#         intervene = partial(intervention_fn, start_edit_location='lt')
#         layers_to_intervene = list(interventions.keys())
#     # --- intervention code --- #
#     sequences = []
#     with torch.no_grad():
#         # for idx, input_ids in enumerate(tqdm(tokens)):
#         max_len = input_ids.shape[-1] + 50

#         # --- intervention code --- #

#         with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
#             # input_ids = input_ids.cuda()

#             model_output = model.generate(input_ids,
#                                           images=images,
#                                           do_sample=True,
#                                           max_new_tokens=1024,
#                                           stopping_criteria=stopping_criteria,
#                                          )
#             model_gen_tokens = model_output['sequences'][:, input_ids.shape[-1]:]
#         # print(model_gen_tokens[0])
#         model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
#         model_gen_str = model_gen_str.strip()
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


def get_cosine_matrix(head_value_matrix):
    head_sim_dict = {f"l{l}_h{h}": [] for l in range(num_layers) for h in range(num_heads)}
    # ^ each dict value with be an array of length #n_question
    for q_idx in tqdm(range(head_value_matrix.shape[1])):
        q_head_vals = head_value_matrix[:, q_idx, :, :, :]
        for l in range(num_layers):
            for h in range(num_heads):
                vectors = q_head_vals[:, l, h, :]
                cosine_sim = cosine_similarity(vectors)
                pairwise_sim = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
                avg_sim = np.mean(pairwise_sim)  # taking mean across permutations
                head_sim_dict[f"l{l}_h{h}"].append(avg_sim)
    mean_sim_noption = [
        np.mean(head_sim_dict[key]) for key in list(head_sim_dict.keys())
    ]  # taking mean across questions
    l_h_sim_avg = np.array(mean_sim_noption).reshape(num_heads, num_layers)  # row = heads | colums = layers
    return l_h_sim_avg


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
            intervention_fn=lt_scale,
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
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--scaling-strength", type=float, default=1.0)
    parser.add_argument("--n-options", type=int, default=3)
    parser.add_argument("--upscale", action="store_true")
    parser.add_argument("--split", type=str, default="minival")
    parser.add_argument("--probing-split", type=str, default="test")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    if "mini" in args.split:
        split = args.split.split("mini")[-1]
        image_folder = f"/home/ubuntu/ScienceQA/{split}"
    else:
        image_folder = f"/home/ubuntu/ScienceQA/{args.split}"
    model_path = "liuhaotian/llava-v1.5-13b"
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    conv_mode = "llava_v1"

    device = 1
    # HARDCODED FOR LLAVA
    num_heads = 40
    num_layers = 40

    # cosine_threshold = args.cosine_threshold
    scaling_power = args.scaling_strength
    upscale = args.upscale
    if upscale:
        assert scaling_power >= 1.0

    if not os.path.isdir("cache"):
        os.makedirs("cache")
    if not os.path.exists("cache/cosine_matrix.npy"):
        print("getting cosine similarity matrix...")
        base_dir = "../probing/features_attack/"
        n_options = args.n_options
        features_dir = os.listdir(base_dir)
        features_dir = [
            os.path.join(base_dir, dir)
            for dir in features_dir
            if f"noption_{args.n_options}_{args.probing_split}" in dir
        ]

        head_wise_activations_all = []
        for dir in features_dir:
            head_wise_activations = np.load(f"{dir}/head_wise.npy")
            head_wise_activations = rearrange(head_wise_activations, "b l (h d) -> b l h d", h=num_heads)
            head_wise_activations_all.append(head_wise_activations)
        head_wise_activations_all = np.stack(head_wise_activations_all)
        cosine_matrix = get_cosine_matrix(head_wise_activations_all)
        np.save("cache/cosine_matrix.npy", cosine_matrix)
    else:
        cosine_matrix = np.load("cache/cosine_matrix.npy")

    num_to_intervene = args.k
    top_heads = get_top_heads(cosine_matrix, num_to_intervene)
    interventions = get_interventions_dict(top_heads, cosine_matrix)

    edit_model(args)
