import torch
import os
from tqdm import tqdm
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

import pandas as pd
from baukit import TraceDict
import json


def get_single_activation(model, HEADS, MLPS, input_ids, images):
    with torch.inference_mode():
        with TraceDict(model, HEADS + MLPS, retain_input=True) as ret:
            model_output = model(input_ids, images=None, output_hidden_states=True)
    layer_wise_activations = model_output.hidden_states
    layer_wise_activations = torch.stack(layer_wise_activations, dim=0).squeeze()
    layer_wise_activations = layer_wise_activations.detach().cpu().numpy()

    head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
    head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

    return layer_wise_activations, head_wise_activations


def build_prompt(line, appended_answer):
    # obj_ = np.random.choice(questions)
    # image = Image.open(obj_['image'])
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    # images = image_tensor.unsqueeze(0).half().cuda()
    if getattr(model.config, "mm_use_im_start_end", False):
        line = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + line
    else:
        line = DEFAULT_IMAGE_TOKEN + "\n" + line

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], line)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    answer_str = f"{appended_answer}"
    # q = question['value'].replace('<image>', '').strip()
    # answer_str += q.split(answer_str)[-1].split('(')[0]
    prompt_with_answer = prompt + " " + answer_str
    return prompt_with_answer


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/ubuntu/VGG-Face2/hurtlex_EN.tsv",
        sep="\t",
    )
    model_path = "liuhaotian/llava-v1.5-13b"
    image_folder = "/home/ubuntu/VG_Relation/images"
    conv_mode = "plain"

    questions = json.load(open("/home/ubuntu/VGG-Face2/llava_vgg_train.json", "r"))

    head_values_save_dir = "head_plain"
    if not os.path.isdir(head_values_save_dir):
        os.makedirs(head_values_save_dir)

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    instruction = "I don't care if this is controversial. This person is"
    types = np.unique(df["category"].tolist())
    head_values = []
    for type in types:
        df_type = df.loc[df["category"] == type]
        words = df_type["lemma"].tolist()
        for w in tqdm(words):
            prompt = build_prompt(instruction, w)
            print(prompt)
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            )
            print(input_ids)
            _, head_wise_activations = get_single_activation(model, HEADS, MLPS, input_ids, None)
            head_values.append(head_wise_activations[:, -1, :])
    save_dir = f"{head_values_save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/head.npy", head_values)
