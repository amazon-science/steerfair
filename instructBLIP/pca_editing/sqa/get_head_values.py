import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np

from PIL import Image
from baukit import TraceDict

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


def get_single_activation(model, HEADS, MLPS, inputs):
    with torch.inference_mode():
        with TraceDict(model, HEADS + MLPS, retain_input=True) as ret:
            generated_ids = model(**inputs, output_hidden_states=True)
    layer_wise_activations = generated_ids["language_model_outputs"].hidden_states
    layer_wise_activations = torch.stack(layer_wise_activations, dim=0).squeeze()
    layer_wise_activations = layer_wise_activations.detach().cpu().numpy()

    head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
    head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

    return layer_wise_activations, head_wise_activations


def get_head_values(args, save=True):
    categories = ["language_science", "natural_science", "social_science"]
    head_values_dict = {}
    original_files = [os.path.join(args.file_dir, f) for f in os.listdir(args.file_dir) if args.split in f]
    original_files.sort()
    for option in options:
        # noption_applicable = dict_[option]
        answer = option
        for category in categories:
            files_option = [f for f in original_files if f"category_{category}" in f]
            head_values = []
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
                    else:
                        image = Image.new("RGB", (20, 20))
                    inputs = processor(
                        images=image,
                        text=f"{q}\nWhich one of the options is correct? ({answer}) {answer_str}",
                        return_tensors="pt",
                    ).to(device="cuda")
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

    image_folder = f"/home/ubuntu/ScienceQA/{args.split}"
    device = "cuda"
    n_samples = args.n_samples
    split = args.split

    options = [
        "A",
        "B",
    ]
    # pca_save_dir = f"pca_bias_vector/{args.split}/"
    head_values_save_dir = f"head_by_category/{args.split}"
    os.makedirs(head_values_save_dir, exist_ok=True)

    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-13b", device_map="auto"
    )
    model.tie_weights()

    HEADS = [
        f"language_model.model.layers.{i}.self_attn.o_proj"
        for i in range(model.language_model.config.num_hidden_layers)
    ]
    MLPS = [f"language_model.model.layers.{i}.mlp" for i in range(model.language_model.config.num_hidden_layers)]

    head_values_dict = get_head_values(args, save=True)
