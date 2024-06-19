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
    yesno_file = os.path.join(args.file_dir, f"vgr_{args.split}_QCM_yesno.json")
    noyes_file = os.path.join(args.file_dir, f"vgr_{args.split}_QCM_noyes.json")

    yesno_questions = json.load(open(yesno_file))
    noyes_questions = json.load(open(noyes_file))

    yesno_sample_idxs = np.random.choice(len(yesno_questions), n_samples // 2)
    yesno_questions = np.array(yesno_questions)[yesno_sample_idxs]
    noyes_sample_idxs = np.random.choice(len(noyes_questions), n_samples // 2)
    noyes_questions = np.array(noyes_questions)[noyes_sample_idxs]

    yesno_labels = np.array([q["answer_idx"] for q in yesno_questions])
    noyes_labels = np.array([q["answer_idx"] for q in noyes_questions])

    question_both = [yesno_questions, noyes_questions]
    head_values_dict = {}
    for option in options:
        head_values = []
        answer = option
        for i, question_all in enumerate(question_both):
            for question in tqdm(question_all):
                # try:
                # prompt, images = build_prompt(question, answer)
                # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                q = question["conversations"][0]["value"].split("<image>")[0]
                image = Image.open(question["image"])
                inputs = processor(images=image, text=q + " " + answer, return_tensors="pt").to(device="cuda")
                _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
                head_values.append(head_wise_activations[:, -1, :])
                # except:
                #     continue
            if save:
                save_dir = f"{head_values_save_dir}/{option}_{i}"
                os.makedirs(save_dir, exist_ok=True)
                np.save(f"{save_dir}/head_{n_samples}.npy", head_values)
            # head_values_dict[answer] = np.array(head_values)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-dir", type=str, default="/home/ubuntu/VG_Relation")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda:4"
    n_samples = args.n_samples

    options = ["no", "yes"]
    # pca_save_dir = f"pca_bias_vector/{args.split}/"
    head_values_save_dir = f"head_NEW/{args.split}"
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
