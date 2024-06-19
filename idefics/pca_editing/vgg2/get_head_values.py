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
    # print(generated_ids.keys())
    # print(generated_ids['language_model_outputs'].keys())
    layer_wise_activations = generated_ids.hidden_states
    layer_wise_activations = torch.stack(layer_wise_activations, dim=0).squeeze()
    layer_wise_activations = layer_wise_activations.detach().cpu().numpy()

    head_wise_activations = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
    head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()

    return layer_wise_activations, head_wise_activations


def get_head_values(args):
    """
    induced assosciations:
    1. female <=> family
    2. male <=> career
    3. female <=> career
    4. male <=> family
    """

    def prompt_model(question_set1, question_set2, gender, attribute_group):
        answer_set1 = "Yes"
        answer_set2 = "No"
        head_values = []
        for question in tqdm(question_set1):
            q = question["conversations"][0]["value"].split("<image>")[0].rstrip().replace("\n", " ")
            image = Image.open(question["image"])
            prompts = [
                [
                    image,
                    f"User: {q}",
                    "<end_of_utterance>",
                    f"\nAssistant: {answer_set1}",
                ],
            ]
            inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
            _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
            head_values.append(head_wise_activations[:, -1, :])
        for question in tqdm(question_set2):
            q = question["conversations"][0]["value"].split("<image>")[0].rstrip().replace("\n", " ")
            image = Image.open(question["image"])
            prompts = [
                [
                    image,
                    f"User: {q}",
                    "<end_of_utterance>",
                    f"\nAssistant: {answer_set2}",
                ],
            ]
            inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
            _, head_wise_activations = get_single_activation(model, HEADS, MLPS, inputs)
            head_values.append(head_wise_activations[:, -1, :])
        save_dir = f"{head_values_save_dir}/{gender}_{attribute_group}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/head_{n_samples}.npy", head_values)
        return

    questions = json.load(open(args.question_file))
    if "family_career" in args.question_file:
        attributes = ["family", "career"]
    elif "math_arts" in args.question_file:
        attributes = ["math", "arts"]
    else:
        attributes = ["pleasant", "unpleasant"]

    if "family_career" in args.question_file or "math_arts" in args.question_file:
        female_0_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[0]) and (obj_["gender_label"] == "f")
        ]
        female_1_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[1]) and (obj_["gender_label"] == "f")
        ]

        male_0_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[0]) and (obj_["gender_label"] == "m")
        ]
        male_1_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[1]) and (obj_["gender_label"] == "m")
        ]

        ff_idxs = np.random.choice(len(female_0_questions), n_samples // 4)
        female_0_questions = np.array(female_0_questions)[ff_idxs]

        fc_idxs = np.random.choice(len(female_1_questions), n_samples // 4)
        female_1_questions = np.array(female_1_questions)[fc_idxs]

        mf_idxs = np.random.choice(len(male_0_questions), n_samples // 4)
        male_0_questions = np.array(male_0_questions)[mf_idxs]

        mc_idxs = np.random.choice(len(male_1_questions), n_samples // 4)
        male_1_questions = np.array(male_1_questions)[mc_idxs]

        prompt_model(female_0_questions, female_1_questions, "f", attributes[0])
        prompt_model(female_1_questions, female_0_questions, "f", attributes[1])
        prompt_model(male_0_questions, male_1_questions, "m", attributes[0])
        prompt_model(male_1_questions, male_0_questions, "m", attributes[1])
    else:
        young_0_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[0]) and (obj_["age_label"] == 0)
        ]
        young_1_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[1]) and (obj_["age_label"] == 0)
        ]

        old_0_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[0]) and (obj_["age_label"] == 1)
        ]
        old_1_questions = [
            obj_ for obj_ in questions if (obj_["attribute_type"] == attributes[1]) and (obj_["age_label"] == 1)
        ]

        ff_idxs = np.random.choice(len(young_0_questions), n_samples // 4)
        young_0_questions = np.array(young_0_questions)[ff_idxs]

        fc_idxs = np.random.choice(len(young_1_questions), n_samples // 4)
        young_1_questions = np.array(young_1_questions)[fc_idxs]

        mf_idxs = np.random.choice(len(old_0_questions), n_samples // 4)
        old_0_questions = np.array(old_0_questions)[mf_idxs]

        mc_idxs = np.random.choice(len(old_1_questions), n_samples // 4)
        old_1_questions = np.array(old_1_questions)[mc_idxs]

        prompt_model(young_0_questions, 0, attributes[0])
        prompt_model(young_1_questions, 0, attributes[1])
        prompt_model(old_0_questions, 1, attributes[0])
        prompt_model(old_1_questions, 1, attributes[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda"
    n_samples = args.n_samples
    split = args.split

    options = ["no", "yes"]
    head_values_save_dir = f"head_NEW_math_arts"
    os.makedirs(head_values_save_dir, exist_ok=True)

    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    head_values_dict = get_head_values(args)
