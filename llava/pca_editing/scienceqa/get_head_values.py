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

    answer_str = f" ({appended_answer})"
    q = question["value"].replace("<image>", "").strip()
    answer_str += q.split(answer_str)[-1].split("(")[0]
    prompt_with_answer = prompt + answer_str
    return prompt_with_answer, images


def get_head_values(args, save=True):
    # dict_ = {
    #     'A': [2,],
    #     'B': [2,],
    #     # 'C': [3,4,5],
    #     # 'D': [4,5],
    #     # 'E': [5],
    # }
    categories = ["language_science", "natural_science", "social_science"]
    original_files = [os.path.join(args.file_dir, f) for f in os.listdir(args.file_dir) if args.split in f]
    original_files.sort()

    for options_idx, option in enumerate(options):
        # noption_applicable = dict_[option]
        answer = option
        for category in categories:
            # for noption in noption_applicable:

            files_option = [f for f in original_files if f"category_{category}" in f]
            head_values = []
            # print(option, noption)
            # exit()
            for i, question_file in enumerate(files_option):
                questions = json.load(open(question_file))
                try:
                    sample_idxs = np.random.choice(len(questions), n_samples // len(files_option), replace=False)
                except:
                    sample_idxs = np.arange(len(questions))
                questions = np.array(questions)[sample_idxs]
                for question in tqdm(questions):
                    prompt, images = build_prompt(question, answer)
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )
                    _, head_wise_activations = get_single_activation(model, HEADS, MLPS, input_ids, images)
                    head_values.append(head_wise_activations[:, -1, :])
            if save:
                save_dir = f"{head_values_save_dir}/{category}_{option}"
                os.makedirs(save_dir, exist_ok=True)
                np.save(f"{save_dir}/head_{n_samples}.npy", head_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-dir", type=str, default="/home/ubuntu/ScienceQA/data/scienceqa/permute_by_category")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()

    n_samples = args.n_samples
    model_path = "liuhaotian/llava-v1.5-13b"
    image_folder = f"/home/ubuntu/ScienceQA/{args.split}"
    conv_mode = "llava_v1"

    options = [
        "A",
        "B",
    ]

    head_values_save_dir = f"head_by_category/{args.split}"
    os.makedirs(head_values_save_dir, exist_ok=True)

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    n_heads = 40
    n_layers = 40

    get_head_values(args, save=True)
