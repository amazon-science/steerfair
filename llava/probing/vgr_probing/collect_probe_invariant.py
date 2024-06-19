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
    options = ["no", "yes"]
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    questions_yesno = json.load(open(os.path.join(args.base_dir, "llava_vgr_QCM_yesno.json"), "r"))
    questions_noyes = json.load(open(os.path.join(args.base_dir, "llava_vgr_QCM_noyes.json"), "r"))

    questions_yesno = np.array(questions_yesno)
    questions_noyes = np.array(questions_noyes)

    random_idx = np.random.choice(len(questions_yesno), n_samples)
    questions_yesno = questions_yesno[random_idx]
    questions_noyes = questions_noyes[random_idx]

    all_head_wise_activations_yesno = []
    y_all_yesno = []

    all_head_wise_activations_noyes = []
    y_all_noyes = []

    for i, line in tqdm(enumerate(questions_yesno)):
        idx = line["id"]
        n_choices = len(options)

        question_yesno = line["conversations"][0]
        question_noyes = questions_noyes[i]["conversations"][0]
        qs_yesno = question_yesno["value"].replace("<image>", "").strip()
        qs_noyes = question_noyes["value"].replace("<image>", "").strip()
        answer_idx_yesno = line["answer_idx"]
        answer_idx_noyes = questions_noyes[i]["answer_idx"]

        answer_yesno = f"{options[answer_idx_yesno]}"
        answer_noyes = f"{options[answer_idx_noyes]}"
        cur_prompt_yesno = qs_yesno
        cur_prompt_noyes = qs_noyes

        if "image" in line:
            image_file = line["image"]
            image = Image.open(os.path.join(image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, "mm_use_im_start_end", False):
                qs_yesno = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs_yesno
                qs_noyes = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs_noyes
            else:
                qs_yesno = DEFAULT_IMAGE_TOKEN + "\n" + qs_yesno
                qs_noyes = DEFAULT_IMAGE_TOKEN + "\n" + qs_noyes
            cur_prompt_yesno = "<image>" + "\n" + cur_prompt_yesno
            cur_prompt_noyes = "<image>" + "\n" + cur_prompt_noyes
        else:
            images = None

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs_yesno)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt_yesno = prompt + " " + answer_yesno
        input_ids = (
            tokenizer_image_token(prompt_yesno, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        )

        with torch.inference_mode():
            with TraceDict(model, HEADS + MLPS, retain_input=True) as ret:
                model_output = model(input_ids, images=images, output_hidden_states=True)

        head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
        head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

        all_head_wise_activations_yesno.append(head_wise_activations[:, -1, :])
        y_all_yesno.append(answer_idx_yesno)

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs_noyes)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt_noyes = prompt + " " + answer_noyes
        input_ids = (
            tokenizer_image_token(prompt_noyes, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        )

        with torch.inference_mode():
            with TraceDict(model, HEADS + MLPS, retain_input=True) as ret:
                model_output = model(input_ids, images=images, output_hidden_states=True)

        head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
        head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

        all_head_wise_activations_noyes.append(head_wise_activations[:, -1, :])
        y_all_noyes.append(answer_idx_noyes)

    print("Saving labels")
    np.save(f"{save_dir}/labels_yesno.npy", y_all_yesno)

    print("Saving head wise activations")
    np.save(f"{save_dir}/head_wise_yesno.npy", all_head_wise_activations_yesno)

    print("Saving labels")
    np.save(f"{save_dir}/labels_noyes.npy", y_all_noyes)

    print("Saving head wise activations")
    np.save(f"{save_dir}/head_wise_noyes.npy", all_head_wise_activations_noyes)


if __name__ == "__main__":
    n_samples = 500
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    model_path = "liuhaotian/llava-v1.5-13b"
    image_folder = f"/home/ubuntu/VG_Relation/images"
    conv_mode = "llava_v1"
    base_dir = "/home/ubuntu/VG_Relation"

    save_dir = f"features_attack"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    collect_probe(args)
