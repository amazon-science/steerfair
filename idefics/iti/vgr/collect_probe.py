import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np

from PIL import Image
import math

from baukit import TraceDict
from transformers import IdeficsForVisionText2Text, AutoProcessor


def get_single_activation(model, HEADS, MLPS, inputs):
    with torch.inference_mode():
        with TraceDict(model, HEADS + MLPS, retain_input=True) as ret:
            generated_ids = model(**inputs, output_hidden_states=True)
    layer_wise_activations = generated_ids.hidden_states
    layer_wise_activations = torch.stack(layer_wise_activations, dim=0).squeeze()
    layer_wise_activations = layer_wise_activations.detach().cpu().numpy()

    head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
    head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

    return layer_wise_activations, head_wise_activations


def collect_probe(args):
    options = ["no", "yes"]

    questions = json.load(open(os.path.join(base_dir, "vgr_train_QCM_yesno.json"), "r"))
    questions = np.random.choice(questions, n_samples)

    all_head_wise_activations_truth = []
    all_head_wise_activations_non_truth = []
    y_all_truth = []
    y_all_non_truth = []

    for line in tqdm(questions):
        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip().rstrip("\n").rstrip()

        answer_idx = line["answer_idx"]
        y_non_truth = 0
        wrong_answer = np.abs(answer_idx - 1)
        answer_non_truth = f"{options[wrong_answer]}"

        y_truth = 1
        answer_truth = f"{options[answer_idx]}"

        if "image" in line:
            image = Image.open(os.path.join(image_folder, line["image"]))
        else:
            image = None

        prompt_truth = qs + " " + answer_truth
        if image != None:
            prompt_truth = [
                [
                    image,
                    f"User: {qs}",
                    "<end_of_utterance>",
                    f"\nAssistant: {answer_truth}",
                ],
            ]
            prompt_non_truth = [
                [
                    image,
                    f"User: {qs}",
                    "<end_of_utterance>",
                    f"\nAssistant: {answer_non_truth}",
                ],
            ]
        else:
            prompt_truth = [
                [
                    f"User: {qs}",
                    "<end_of_utterance>",
                    f"\nAssistant: {answer_truth}",
                ],
            ]
            prompt_non_truth = [
                [
                    f"User: {qs}",
                    "<end_of_utterance>",
                    f"\nAssistant: {answer_non_truth}",
                ],
            ]
        inputs = processor(prompt_truth, add_end_of_utterance_token=False, return_tensors="pt").to(device)
        _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
        all_head_wise_activations_truth.append(head_wise_activations[:, -1, :])
        y_all_truth.append(y_truth)

        inputs = processor(prompt_non_truth, add_end_of_utterance_token=False, return_tensors="pt").to(device)
        _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
        all_head_wise_activations_non_truth.append(head_wise_activations[:, -1, :])
        y_all_non_truth.append(y_non_truth)

    print("Saving labels")
    np.save(f"{save_dir_truth}/labels_{n_samples}.npy", y_all_truth)

    print("Saving labels")
    np.save(f"{save_dir_non_truth}/labels_{n_samples}.npy", y_all_non_truth)

    print("Saving head wise activations")
    np.save(f"{save_dir_truth}/head_{n_samples}.npy", all_head_wise_activations_truth)

    print("Saving head wise activations")
    np.save(f"{save_dir_non_truth}/head_{n_samples}.npy", all_head_wise_activations_non_truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    device = "cuda"
    n_samples = args.n_samples
    base_dir = "/home/ubuntu/VG_Relation"

    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    image_folder = "/home/ubuntu/VG_Relation/images"

    save_dir_non_truth = f"features_{n_samples}/{args.split}/non_truth"
    save_dir_truth = f"features_{n_samples}/{args.split}/truth"
    os.makedirs(save_dir_truth, exist_ok=True)
    os.makedirs(save_dir_non_truth, exist_ok=True)

    collect_probe(args)
