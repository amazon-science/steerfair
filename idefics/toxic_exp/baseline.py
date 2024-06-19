import numpy as np
from PIL import Image
import json
import shortuuid
import argparse
from tqdm import tqdm
import os

import torch

from transformers import IdeficsForVisionText2Text, AutoProcessor


def get_models():
    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    args = parser.parse_args()

    device = "cuda"

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    question_file = args.question_file
    questions = json.load(open(question_file, "r"))

    model, processor = get_models()

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    for obj_ in tqdm(questions):
        idx = obj_["id"]
        q = "Complete this sentence succinctly. " + obj_["prompt"].rstrip() + " "

        image = Image.open(obj_["image"])
        prompts = [
            [
                f"User: {q}",
                "<end_of_utterance>",
                "\nAssistant:",
            ],
        ]

        inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
        generated_ids = model.generate(
            **inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=500
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text[0].split(q)[-1].split("Assistant:")[-1].strip().rstrip()
        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": q,
                    "text": generated_text,
                    "answer_id": ans_id,
                }
            )
            + "\n"
        )
        ans_file.flush()
