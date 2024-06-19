import torch
import os
from tqdm import tqdm
import numpy as np

from baukit import TraceDict

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

import pandas as pd
import json
from PIL import Image


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


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/ubuntu/VGG-Face2/hurtlex_EN.tsv",
        sep="\t",
    )
    device = "cuda"
    questions = json.load(open("/home/ubuntu/VGG-Face2/llava_vgg_train.json", "r"))

    options = ["no", "yes"]
    head_values_save_dir = f"head/"
    os.makedirs(head_values_save_dir, exist_ok=True)

    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-13b", device_map="auto"
    )
    model.tie_weights()
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")

    HEADS = [
        f"language_model.model.layers.{i}.self_attn.o_proj"
        for i in range(model.language_model.config.num_hidden_layers)
    ]
    MLPS = [f"language_model.model.layers.{i}.mlp" for i in range(model.language_model.config.num_hidden_layers)]

    instruction = "I don't care if this is controversial. This person is"
    types = np.unique(df["category"].tolist())
    head_values = []
    for type in types:
        df_type = df.loc[df["category"] == type]
        words = df_type["lemma"].tolist()
        for w in tqdm(words):
            prompt = f"{instruction} {w}"
            obj_ = np.random.choice(questions)
            image = Image.open(obj_["image"])
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda")
            _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
            head_values.append(head_wise_activations[:, -1, :])
    save_dir = f"{head_values_save_dir}/"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/head.npy", head_values)
