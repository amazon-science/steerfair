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
    # print(image_file)
    # print(image)
    # exit()
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


def get_head_values(args):
    """
    induced assosciations:
    1. female <=> family
    2. male <=> career
    3. female <=> career
    4. male <=> family
    """

    def prompt_model(question_set1, question_set2, gender, attribute_group):
        answer_set1 = "Yes"
        answer_set2 = "No"
        head_values = []
        for question in tqdm(question_set1):
            prompt, images = build_prompt(question, answer_set1)
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            )
            _, head_wise_activations = get_single_activation(model, HEADS, MLPS, input_ids, images)
            head_values.append(head_wise_activations[:, -1, :])
        for question in tqdm(question_set2):
            prompt, images = build_prompt(question, answer_set2)
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            )
            _, head_wise_activations = get_single_activation(model, HEADS, MLPS, input_ids, images)
            head_values.append(head_wise_activations[:, -1, :])
        save_dir = f"{head_values_save_dir}/{gender}_{attribute_group}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/head_{n_samples}.npy", head_values)
        return

    questions = json.load(open(args.question_file))
    if "family_career" in args.question_file:
        attributes = ["family", "career"]
    elif "math_arts" in args.question_file:
        attributes = ["math", "arts"]
    else:
        attributes = ["pleasant", "unpleasant"]

    if "family_career" in args.question_file or "math_arts" in args.question_file:
        female_0_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[0]) and (obj_["gender_label"] == "f")
        ]
        female_1_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[1]) and (obj_["gender_label"] == "f")
        ]

        male_0_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[0]) and (obj_["gender_label"] == "m")
        ]
        male_1_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[1]) and (obj_["gender_label"] == "m")
        ]

        ff_idxs = np.random.choice(len(female_0_questions), n_samples // 4)
        female_0_questions = np.array(female_0_questions)[ff_idxs]

        fc_idxs = np.random.choice(len(female_1_questions), n_samples // 4)
        female_1_questions = np.array(female_1_questions)[fc_idxs]

        mf_idxs = np.random.choice(len(male_0_questions), n_samples // 4)
        male_0_questions = np.array(male_0_questions)[mf_idxs]

        mc_idxs = np.random.choice(len(male_1_questions), n_samples // 4)
        male_1_questions = np.array(male_1_questions)[mc_idxs]

        prompt_model(female_0_questions, female_1_questions, "f", attributes[0])
        prompt_model(female_1_questions, female_0_questions, "f", attributes[1])
        prompt_model(male_0_questions, male_1_questions, "m", attributes[0])
        prompt_model(male_1_questions, male_0_questions, "m", attributes[1])
    else:
        young_0_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[0]) and (obj_["age_label"] == 0)
        ]
        young_1_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[1]) and (obj_["age_label"] == 0)
        ]

        old_0_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[0]) and (obj_["age_label"] == 1)
        ]
        old_1_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[1]) and (obj_["age_label"] == 1)
        ]

        ff_idxs = np.random.choice(len(young_0_questions), n_samples // 4)
        young_0_questions = np.array(young_0_questions)[ff_idxs]

        fc_idxs = np.random.choice(len(young_1_questions), n_samples // 4)
        young_1_questions = np.array(young_1_questions)[fc_idxs]

        mf_idxs = np.random.choice(len(old_0_questions), n_samples // 4)
        old_0_questions = np.array(old_0_questions)[mf_idxs]

        mc_idxs = np.random.choice(len(old_1_questions), n_samples // 4)
        old_1_questions = np.array(old_1_questions)[mc_idxs]

        prompt_model(young_0_questions, 0, attributes[0])
        prompt_model(young_1_questions, 0, attributes[1])
        prompt_model(old_0_questions, 1, attributes[0])
        prompt_model(old_1_questions, 1, attributes[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()

    n_samples = args.n_samples
    model_path = "liuhaotian/llava-v1.5-13b"
    conv_mode = "llava_v1"

    head_values_save_dir = f"head_values_2/{args.split}/math_arts"
    os.makedirs(head_values_save_dir, exist_ok=True)

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    n_heads = 40
    n_layers = 40

    head_values_dict = get_head_values(args)
