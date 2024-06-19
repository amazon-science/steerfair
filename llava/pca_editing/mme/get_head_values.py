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


def get_single_activation(model, HEADS, MLPS, input_ids, images):
    with torch.inference_mode():
        with TraceDict(model, HEADS + MLPS, retain_input=True) as ret:
            model_output = model(input_ids, images=images, output_hidden_states=True)
    layer_wise_activations = model_output.hidden_states
    layer_wise_activations = torch.stack(layer_wise_activations, dim=0).squeeze()
    layer_wise_activations = layer_wise_activations.detach().cpu().numpy()

    head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
    head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

    return layer_wise_activations, head_wise_activations


def build_prompt(line, appended_answer):
    question = line["conversations"][0]
    qs = question["value"].replace("<image>", "").strip()
    cur_prompt = qs

    image_file = line["image"]
    image = Image.open(image_file)
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
    prompt = conv.get_prompt()

    prompt_with_answer = prompt + " " + appended_answer
    return prompt_with_answer, images


def get_head_values(args, save=True):
    yesno_file = os.path.join(args.file_dir, f"llava_mme_yesno_{split}.json")
    noyes_file = os.path.join(args.file_dir, f"llava_mme_noyes_{split}.json")

    yesno_questions = json.load(open(yesno_file))
    noyes_questions = json.load(open(noyes_file))

    yesno_sample_idxs = np.random.choice(len(yesno_questions), n_samples // 2)
    yesno_questions = np.array(yesno_questions)[yesno_sample_idxs]
    noyes_sample_idxs = np.random.choice(len(noyes_questions), n_samples // 2)
    noyes_questions = np.array(noyes_questions)[noyes_sample_idxs]

    question_both = [yesno_questions, noyes_questions]
    head_values_dict = {}
    for option in options:
        head_values = []
        answer = option
        for i, question_all in enumerate(question_both):
            for question in tqdm(question_all):
                prompt, images = build_prompt(question, answer)
                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
                )
                _, head_wise_activations = get_single_activation(model, HEADS, MLPS, input_ids, images)
                head_values.append(head_wise_activations[:, -1, :])
            if save:
                save_dir = f"{head_values_save_dir}/{option}_{i}"
                os.makedirs(save_dir, exist_ok=True)
                np.save(f"{save_dir}/head_{n_samples}.npy", head_values)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-dir", type=str, default="/home/ubuntu/MME_benchmark/llava")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    n_samples = args.n_samples
    split = args.split
    model_path = "liuhaotian/llava-v1.5-13b"
    conv_mode = "llava_v1"

    options = ["no", "yes"]
    head_values_save_dir = f"head_values_{n_samples}/{split}"
    os.makedirs(head_values_save_dir, exist_ok=True)

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    n_heads = 40
    n_layers = 40

    head_values_dict = get_head_values(args, save=True)
