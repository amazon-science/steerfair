import argparse
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import torch

from einops import rearrange
from functools import partial
import json
from tqdm import tqdm
import shortuuid

from PIL import Image

from baukit import Trace, TraceDict
from transformers import IdeficsForVisionText2Text, AutoProcessor


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
        if not no_normalize:
            head_output[:, -1, head, :] = head_new * (
                torch.linalg.norm(torch.Tensor(head_orig).cuda()) / torch.linalg.norm(head_new)
            )
    head_output = rearrange(head_output, "b s h d -> b s (h d)")
    return head_output


# def lt_modulated_vector_rejection(head_output, layer_name, start_edit_location='lt'):
#     head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads).cuda()
#     for head, rejection_vectors in interventions[layer_name]:
#         head_orig = head_output[:, -1, head, :]
#         head_new = np.copy(head_orig.detach().cpu().numpy())
#         head_new = head_new/np.linalg.norm(head_new).reshape(-1, 1)
#         for orthonormal_vector in rejection_vectors:
#             cos = np.squeeze(cosine_similarity(head_new, orthonormal_vector.reshape(1, -1)))
#             rejection_features = cos.reshape(-1, 1) * np.repeat(orthonormal_vector.reshape(1, -1), 1, axis=0) / np.linalg.norm(orthonormal_vector)
#             head_new = head_new - rejection_features
#             head_new = head_new / np.linalg.norm(head_new, axis=1).reshape(-1, 1)
#         head_new = torch.tensor(head_new).cuda()
#         head_output[:, -1, head, :] = head_new
#         if not no_normalize:
#             head_output[:, -1, head, :] = head_new * (torch.linalg.norm(torch.Tensor(head_orig).cuda())/torch.linalg.norm(head_new))
#     head_output = rearrange(head_output, 'b s h d -> b s (h d)')
#     return head_output


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
        if not no_normalize:
            head_output[:, -1, head, :] = head_new * (torch.linalg.norm(head_orig) / torch.linalg.norm(head_new))
    head_output = rearrange(head_output, "b s h d -> b s (h d)")
    return head_output


def get_answer_with_intervention(
    model, processor, q, image, exit_condition, bad_words_ids, interventions={}, intervention_fn=None
):
    # --- intervention code --- #
    prompts = [
        [
            image,
            f"User: {q}",
            "<end_of_utterance>",
            "\nAssistant:",
        ]
    ]
    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)

    def id(head_output, layer_name):
        return head_output

    if interventions == {}:
        intervene = id
        layers_to_intervene = []
    else:
        intervene = partial(intervention_fn, start_edit_location="lt")
        layers_to_intervene = list(interventions.keys())
    # --- intervention code --- #
    with torch.inference_mode():
        with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
            generated_ids = model.generate(
                **inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100
            )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = generated_text[0].split(q)[-1].split("Assistant:")[-1].strip().rstrip()
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    torch.cuda.empty_cache()
    return generated_text


def get_interventions_dict(pca_directions):
    interventions = {}
    all_heads = [(j, i) for j in range(num_heads) for i in range(num_layers)]
    pca_directions = rearrange(pca_directions, "(l h) s d -> l h s d", h=num_heads)
    for layer, head in all_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in all_heads:
        direction = pca_directions[layer, head]
        interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, direction.squeeze()))
    for layer, head in all_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(
            interventions[f"model.layers.{layer}.self_attn.o_proj"], key=lambda x: x[0]
        )
    return interventions


def edit_model(args):
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    question_file = args.question_file

    questions = json.load(open(question_file, "r"))

    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    if args.combine_mode == "orth":
        intervention_fn = lt_modulated_vector_proj
    elif args.combine_mode == "rejection":
        intervention_fn = lt_modulated_vector_rejection
    else:
        intervention_fn = lt_modulated_vector_add

    for obj_ in tqdm(questions):
        try:
            idx = obj_["id"]

            q = obj_["conversations"][0]["value"].split("<image>")[0].rstrip().replace("\n", " ")
            try:
                image = Image.open(obj_["image"])
            except:
                continue
            intervened_answer = get_answer_with_intervention(
                model,
                processor,
                q,
                image,
                exit_condition,
                bad_words_ids,
                interventions=interventions,
                intervention_fn=intervention_fn,
            )

            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": q,
                        "text": intervened_answer,
                        "answer_id": ans_id,
                    }
                )
                + "\n"
            )
            ans_file.flush()
        except Exception as e:
            raise e


def combine_qr(head_directions):
    Q, _ = np.linalg.qr(head_directions.T)
    basis = Q.T
    return np.mean(basis, axis=0)


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
            head_all = []
            for direction_set in pca_direction_all:
                head_all.append(direction_set[l, h, :, :])
            head_all = np.vstack(head_all)
            combined_head_value = combine_fn(head_all)
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
    parser.add_argument("--vector-direction-dir", type=str, required=True)
    parser.add_argument("--n-direction-samples", type=int, default=1000)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--combine-mode", default="avg")

    args = parser.parse_args()
    alpha = args.alpha
    reverse = args.reverse
    no_normalize = args.no_normalize
    combine_mode = args.combine_mode
    assert combine_mode in combine_mode_all
    vector_direction_dir = args.vector_direction_dir

    device = "cuda"
    num_heads = 32
    num_layers = 32

    bias_type = args.question_file.split("/")[-1].split(".json")[0]
    if "test" in args.question_file:
        bias_type = bias_type.split("test_")[-1]
    elif "val" in args.question_file:
        bias_type = bias_type.split("val_")[-1]
    else:
        bias_type = bias_type.split("train_")[-1]
    direction_dict = {
        "family_career": ["f_family", "f_career", "m_family", "m_career"],
        "math_arts": ["f_arts", "m_math", "f_math", "m_arts"],
        "pleasant_unpleasant": ["1_pleasant", "0_unpleasant"],
    }
    bias_dir = direction_dict[bias_type]

    # interventions_per_option = {}
    pca_direction_all = []
    for subdir in os.listdir(vector_direction_dir):
        if subdir in bias_dir:
            load_dir = os.path.join(vector_direction_dir, subdir)
            pca_directions = np.load(os.path.join(load_dir, f"pca_direction_{args.n_direction_samples}.npy"))
            pca_directions = rearrange(pca_directions, "(l h) s d -> l h s d", h=num_heads)
            pca_direction_all.append(pca_directions)

    if combine_mode != "none":
        pca_direction_combined = combine_all_directions(pca_direction_all)
    else:
        pca_direction_combined = pca_direction_all
    interventions = get_interventions_dict(pca_direction_combined)

    edit_model(args)
    # interventions_per_option[subdir] = interventions
