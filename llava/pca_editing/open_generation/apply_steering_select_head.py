import argparse
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import torch

from llava.model import *
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from einops import rearrange
from functools import partial
import json
from tqdm import tqdm
import shortuuid

from PIL import Image

from baukit import Trace, TraceDict


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def lt_modulated_vector_proj(head_output, layer_name, start_edit_location="lt"):
    head_output = rearrange(head_output, "b s (h d) -> b s h d", h=num_heads).cuda()
    for head, proj_mtrx in interventions[layer_name]:
        # proj_mtrx = torch.tensor(proj_mtrx).cuda()
        head_orig = head_output[:, -1, head, :].detach().cpu().numpy()
        head_new = np.matmul(proj_mtrx, head_orig.T).T
        # head_new = head_orig - (alpha * head_proj)
        head_new = torch.tensor(head_new).cuda()
        head_output[:, -1, head, :] = head_new
        if normalize:
            head_output[:, -1, head, :] = head_new * (
                torch.linalg.norm(torch.Tensor(head_orig).cuda()) / torch.linalg.norm(head_new)
            )
    head_output = rearrange(head_output, "b s h d -> b s (h d)")
    return head_output


def lt_modulated_vector_add(head_output, layer_name, start_edit_location="lt"):
    head_output = rearrange(head_output, "b s (h d) -> b s h d", h=num_heads).cuda()
    for head, direction in interventions[layer_name]:
        direction_to_add = torch.tensor(direction).cuda()
        head_orig = torch.clone(head_output[:, -1, head, :])
        if not reverse:
            head_new = head_orig + (alpha * direction_to_add)
        else:
            head_new = head_orig - (alpha * direction_to_add)
        head_output[:, -1, head, :] = head_new
        if normalize:
            head_output[:, -1, head, :] = head_new * (torch.linalg.norm(head_orig) / torch.linalg.norm(head_new))
    head_output = rearrange(head_output, "b s h d -> b s (h d)")
    return head_output


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


def get_top_heads(per_head_component, num_to_intervene):
    top_component_val = np.argsort(per_head_component.reshape(num_heads * num_layers))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_component_val]
    return top_heads


def get_interventions_dict(top_heads, pca_directions):
    interventions = {}
    # all_heads = [(j, i) for j in range(num_heads) for i in range(num_layers)]
    pca_directions = rearrange(pca_directions, "(l h) s d -> l h s d", h=num_heads)
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in top_heads:
        direction = pca_directions[layer, head]
        interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, direction.squeeze()))
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(
            interventions[f"model.layers.{layer}.self_attn.o_proj"], key=lambda x: x[0]
        )
    return interventions


def build_prompt(line, model, image_processor, tokenizer):
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

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

    prompt = conv.get_prompt()
    return cur_prompt, prompt, images, stopping_criteria, stop_str


def get_models():
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    return tokenizer, model, image_processor


def edit_model(args):
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    question_file = args.question_file

    questions = json.load(open(question_file, "r"))
    # questions = get_chunk(questions, 1, 0

    tokenizer, model, image_processor = get_models()

    if args.combine_mode == "orth":
        intervention_fn = lt_modulated_vector_proj
    else:
        intervention_fn = lt_modulated_vector_add

    for obj_ in tqdm(questions):
        idx = obj_["id"]
        try:
            cur_prompt, prompt, images, stopping_criteria, stop_str = build_prompt(
                obj_, model, image_processor, tokenizer
            )
        except:
            continue
        intervened_answer = get_answer_with_intervention(
            model,
            tokenizer,
            prompt,
            images,
            stopping_criteria,
            stop_str,
            interventions=interventions,
            intervention_fn=intervention_fn,
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


def combine_qr(head_directions):
    if head_directions.shape[0] > 1:
        Q, _ = np.linalg.qr(head_directions.T)
        basis = Q.T
        return np.mean(basis, axis=0)
    else:
        return head_directions


def combine_svd(head_directions):
    tSVD = TruncatedSVD(n_components=len(head_directions))
    embeddings_ = tSVD.fit_transform(head_directions)
    basis = tSVD.components_
    return np.mean(basis, axis=0)


def combine_avg(head_directions):
    return np.mean(head_directions, axis=0)


# def combine_orth(head_directions):
#     A = head_directions.T
#     a_term = np.linalg.pinv(np.matmul(A.T, A))
#     a_term = np.matmul(A, a_term)
#     a_term = np.matmul(a_term, A.T)
#     # I = np.identity(a_term.shape[0])
#     # P0 = I - a_term
#     return a_term


def combine_orth(head_directions):
    # print(head_directions.shape)
    tSVD = TruncatedSVD(n_components=len(head_directions))
    embeddings_ = tSVD.fit_transform(head_directions)
    basis = tSVD.components_.T
    # basis, _ = np.linalg.qr(head_directions.T)

    # orthogonal projection
    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj = np.eye(proj.shape[0]) - proj
    return proj


def combine_rejection(head_directions):
    Q, _ = np.linalg.qr(head_directions.T)
    return Q.T


def combine_all_directions(pca_direction_all):
    print("combining bias direction from all options..")
    combine_fn_dict = {
        "avg": combine_avg,
        "qr": combine_qr,
        "orth": combine_orth,
        "rejection": combine_rejection,
        "svd": combine_svd,
    }
    combined_direction = []
    combine_fn = combine_fn_dict[combine_mode]
    for l in range(num_layers):
        for h in range(num_heads):
            combined_head_value = combine_fn(pca_direction_all[l, h, :, :])
            if len(combined_head_value.shape) == 1:
                combined_head_value = combined_head_value.reshape(1, -1)
            combined_direction.append(combined_head_value)
    return np.array(combined_direction)


if __name__ == "__main__":
    combine_mode_all = ["avg", "orth", "qr", "rejection", "svd", "none"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--vector-direction-dir", type=str, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--combine-mode", default="avg")

    args = parser.parse_args()
    alpha = args.alpha
    k = args.k
    reverse = args.reverse
    normalize = args.normalize
    combine_mode = args.combine_mode
    assert combine_mode in combine_mode_all
    vector_direction_dir = args.vector_direction_dir

    model_path = "liuhaotian/llava-v1.5-13b"
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)

    conv_mode = "llava_v1"

    device = "cuda"
    num_heads = 40
    num_layers = 40
    pca_directions = np.load(os.path.join(vector_direction_dir, f"pca_direction.npy"))
    pca_directions = rearrange(pca_directions, "(l h) s d -> l h s d", h=num_heads)

    pca_values = np.load(os.path.join(vector_direction_dir, f"pca_values.npy")).squeeze()
    pca_values = rearrange(pca_values, "(l h) d -> l h d", h=num_heads)
    pca_values = np.mean(pca_values, axis=-1)

    if combine_mode != "none":
        pca_direction_combined = combine_all_directions(pca_directions)
    else:
        pca_direction_combined = pca_directions

    top_heads = get_top_heads(pca_values, k)
    interventions = get_interventions_dict(top_heads, pca_direction_combined)

    edit_model(args)
