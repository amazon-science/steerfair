import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
import math

from baukit import TraceDict


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def collect_probe(args):
    options = ["A", "B", "C", "D", "E"]
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, 1, 0)
    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    all_layer_wise_activations = []
    all_head_wise_activations = []

    y_all = []
    for line in tqdm(questions):
        idx = line["id"]
        n_choices = len(problems[idx]["choices"])
        y_one_hot = [0 for i in range(n_choices)]

        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip()
        answer_idx = line["new_gt"]

        y_one_hot[answer_idx] = 1
        answer = f"The answer is ({options[answer_idx]})"
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
        prompt = conv.get_prompt()
        prompt = prompt + " " + answer

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with torch.inference_mode():
            with TraceDict(model, HEADS + MLPS, retain_input=True) as ret:
                model_output = model(input_ids, images=images, output_hidden_states=True)

        # layer_wise_activations = model_output.hidden_states
        # layer_wise_activations = torch.stack(layer_wise_activations, dim = 0).squeeze()
        # layer_wise_activations = layer_wise_activations.detach().cpu().numpy()

        head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
        head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

        # all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
        all_head_wise_activations.append(head_wise_activations[:, -1, :])

        y_all.append(y_one_hot)

    print("Saving labels")
    np.save(f"{save_dir}/labels.npy", y_all)

    # print("Saving layer wise activations")
    # np.save(f'{save_dir}/layer_wise.npy', all_layer_wise_activations)

    print("Saving head wise activations")
    np.save(f"{save_dir}/head_wise.npy", all_head_wise_activations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    # model_path = 'liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3'
    model_path = "liuhaotian/llava-v1.5-13b"
    if "mini" in args.split:
        split = args.split.split("mini")[-1]
        image_folder = f"/home/ubuntu/ScienceQA/{split}"
    else:
        image_folder = f"/home/ubuntu/ScienceQA/{args.split}"
    base_dir = "/home/ubuntu/ScienceQA/data/scienceqa"
    conv_mode = "llava_v1"

    save_dir = f"features_attack/features_{args.question_file.split('/')[-1].split('.')[0]}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    collect_probe(args)
