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
    # dict_ = {
    #     'A': [2,3,4,5],
    #     'B': [2,3,4,5],
    #     'C': [3,4,5],
    #     'D': [4,5],
    #     'E': [5],
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
            for i, question_file in enumerate(files_option):
                questions = json.load(open(question_file))
                try:
                    sample_idxs = np.random.choice(len(questions), n_samples // len(files_option), replace=False)
                except:
                    sample_idxs = np.arange(len(questions))
                questions = np.array(questions)[sample_idxs]
                for question in tqdm(questions):
                    q = question["conversations"][0]["value"].split("<image>")[0].strip().rstrip("\n")
                    q_ans = question["conversations"][0]["value"].replace("<image>", "").strip()
                    answer_str = q_ans.split(f"({answer})")[-1].split("(")[0]
                    if "image" in question:
                        image = Image.open(os.path.join(image_folder, question["image"]))
                        prompts = [
                            [
                                image,
                                f"User: {q}",
                                "<end_of_utterance>",
                                f"\nAssistant: ({answer}) {answer_str}",
                            ],
                        ]
                    else:
                        prompts = [
                            [
                                f"User: {q}",
                                "<end_of_utterance>",
                                f"\nAssistant: ({answer}) {answer_str}",
                            ],
                        ]
                    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
                    _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
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

    device = "cuda"
    n_samples = args.n_samples
    split = args.split

    image_folder = f"/home/ubuntu/ScienceQA/{args.split}"

    options = ["A", "B"]
    # pca_save_dir = f"pca_bias_vector/{args.split}/"
    head_values_save_dir = f"head_by_category/{args.split}"
    os.makedirs(head_values_save_dir, exist_ok=True)

    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    head_values_dict = get_head_values(args, save=True)
