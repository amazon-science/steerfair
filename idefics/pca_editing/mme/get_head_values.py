import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np

from PIL import Image
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
                q = question["conversations"][0]["value"].split("<image>")[0].rstrip().replace("\n", " ")
                image = Image.open(question["image"])
                prompts = [
                    [
                        image,
                        f"User: {q}",
                        "<end_of_utterance>",
                        f"\nAssistant: {answer}",
                    ],
                ]
                inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
                _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
                head_values.append(head_wise_activations[:, -1, :])
            if save:
                save_dir = f"{head_values_save_dir}/{option}_{i}"
                os.makedirs(save_dir, exist_ok=True)
                np.save(f"{save_dir}/head_{n_samples}.npy", head_values)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-dir", type=str, default="/home/ubuntu/MME_benchmark/llava")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    device = "cuda"
    n_samples = args.n_samples
    split = args.split

    options = ["no", "yes"]
    # pca_save_dir = f"pca_bias_vector/{args.split}/"
    head_values_save_dir = f"head_NEW/{args.split}"
    os.makedirs(head_values_save_dir, exist_ok=True)

    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    head_values_dict = get_head_values(args, save=True)
