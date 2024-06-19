import torch
import os
from tqdm import tqdm
import numpy as np

from baukit import TraceDict

from transformers import IdeficsForVisionText2Text, AutoProcessor

import pandas as pd


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


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/ubuntu/VGG-Face2/hurtlex_EN.tsv",
        sep="\t",
    )
    device = "cuda"

    options = ["no", "yes"]
    head_values_save_dir = f"head_1/"
    os.makedirs(head_values_save_dir, exist_ok=True)

    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    instruction = "I don't care if this is controversial. This person is"
    types = np.unique(df["category"].tolist())
    head_values = []
    for type in types:
        df_type = df.loc[df["category"] == type]
        words = df_type["lemma"].tolist()
        for w in tqdm(words):
            prompt = f"{instruction}"
            prompts = [
                [
                    f"User: {prompt}",
                    "<end_of_utterance>",
                    f"\nAssistant: {w}",
                ],
            ]
            inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
            _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
            head_values.append(head_wise_activations[:, -1, :])
    save_dir = f"{head_values_save_dir}/"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/head.npy", head_values)
