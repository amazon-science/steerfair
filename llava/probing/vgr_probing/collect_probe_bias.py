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


def get_single_activation(
    model,
    HEADS_LLM,
    MLPS_LLM,
    HEADS_VISION,
    MLPS_VISION,
    HEADS_MULTIMODAL_LM,
    HEADS_MULTIMODAL_VISION,
    input_ids,
    images,
    vision=False,
):
    with torch.inference_mode():
        with TraceDict(model, HEADS_LLM + MLPS_LLM, retain_input=True) as ret:
            model_output = model(input_ids, images=images, output_hidden_states=True)
    layer_wise_activations = model_output.hidden_states
    layer_wise_activations = torch.stack(layer_wise_activations, dim=0).squeeze()
    layer_wise_activations = layer_wise_activations.detach().cpu().numpy()

    head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS_LLM]
    head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

    return layer_wise_activations, head_wise_activations, None, None, None, None


def get_probes_from_question_set(questions):
    options = ["no", "yes"]
    HEADS_LLM = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS_LLM = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    HEADS_VISION = [f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj" for i in range(23)]
    MLPS_VISION = [f"vision_tower.vision_model.encoder.layers.{i}.mlp" for i in range(23)]
    HEADS_MULTIMODAL_LM = ["model.mm_projector.2"]
    HEADS_MULTIMODAL_VISION = ["model.mm_projector.0"]
    i = 0
    head_values = []
    for line in tqdm(questions):
        idx = line["id"]
        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip()
        cur_prompt = qs

        answer_idx = line["answer_idx"]
        answer_truth = f"{options[answer_idx]}"

        image_file = line["image"]
        image = Image.open(os.path.join(image_folder, image_file))
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

        prompt_truth = prompt + " " + answer_truth
        input_ids = (
            tokenizer_image_token(prompt_truth, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        )

        _, head_wise_activations, _, _, _, _ = get_single_activation(
            model,
            HEADS_LLM,
            MLPS_LLM,
            HEADS_VISION,
            MLPS_VISION,
            HEADS_MULTIMODAL_LM,
            HEADS_MULTIMODAL_VISION,
            input_ids,
            images,
        )
        head_values.append(head_wise_activations[:, -1, :])
    return head_values


def collect_probe(args):
    all_head_wise_activations_truth = []
    y_all_truth = []

    all_head_wise_activations_bias = []
    y_all_bias = []

    yesno_file = os.path.join(args.file_dir, f"vgr_{args.split}_QCM_yesno.json")
    noyes_file = os.path.join(args.file_dir, f"vgr_{args.split}_QCM_noyes.json")

    yesno_questions = json.load(open(yesno_file))
    noyes_questions = json.load(open(noyes_file))
    sample_idxs = np.random.choice(len(yesno_questions), n_samples)

    yesno_questions = np.array(yesno_questions)[sample_idxs]
    noyes_questions = np.array(noyes_questions)[sample_idxs]

    yesno_labels = np.array([q["answer_idx"] for q in yesno_questions])
    noyes_labels = np.array([q["answer_idx"] for q in noyes_questions])

    bias_q_idx_yesno = np.argwhere(yesno_labels == 1).flatten()
    nonbias_q_idx_yesno = np.argwhere(yesno_labels == 0).flatten()

    bias_q_idx_noyes = np.argwhere(noyes_labels == 0).flatten()
    nonbias_q_idx_noyes = np.argwhere(noyes_labels == 1).flatten()

    qs_all = [yesno_questions, noyes_questions, yesno_questions, noyes_questions]
    for i, idxs in tqdm(enumerate([nonbias_q_idx_yesno, nonbias_q_idx_noyes, bias_q_idx_yesno, bias_q_idx_noyes])):
        qs = qs_all[i][idxs]
        heads = get_probes_from_question_set(qs)
        if i < 2:
            all_head_wise_activations_truth.extend(heads)
            y_all_truth.extend([i for j in range(len(idxs))])
        else:
            all_head_wise_activations_bias.extend(heads)
            y_all_bias.extend([i for j in range(len(idxs))])

    print("Saving labels")
    np.save(f"{save_dir_truth}/labels_{n_samples}.npy", y_all_truth)

    print("Saving labels")
    np.save(f"{save_dir_bias}/labels_{n_samples}.npy", y_all_bias)

    print("Saving head wise activations")
    np.save(f"{save_dir_truth}/head_wise_{n_samples}_llm.npy", all_head_wise_activations_truth)

    print("Saving head wise activations")
    np.save(f"{save_dir_bias}/head_wise_{n_samples}_llm.npy", all_head_wise_activations_bias)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-dir", type=str, default="/home/ubuntu/VG_Relation")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    n_samples = args.n_samples
    model_path = "liuhaotian/llava-v1.5-13b"
    image_folder = "/home/ubuntu/VG_Relation/images"
    conv_mode = "llava_v1"

    save_dir_bias = f"features_bias_2/{args.split}/bias"
    save_dir_truth = f"features_bias_2/{args.split}/truth"
    os.makedirs(save_dir_truth, exist_ok=True)
    os.makedirs(save_dir_bias, exist_ok=True)

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    collect_probe(args)
