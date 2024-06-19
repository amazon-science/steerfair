import numpy as np
from PIL import Image
import json
import shortuuid
import argparse
from tqdm import tqdm
import os

# from accelerate import infer_auto_device_map, dispatch_model

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch


def get_models():
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-13b", device_map="auto"
    )
    model.tie_weights()
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    args = parser.parse_args()

    # device = 'cuda:4'
    if "ScienceQA" in args.question_file:
        image_folder = "/home/ubuntu/ScienceQA"
        if "test" in args.question_file:
            image_folder = os.path.join(image_folder, "test")
        else:
            image_folder = os.path.join(image_folder, "val")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    question_file = args.question_file
    questions = json.load(open(question_file, "r"))

    model, processor = get_models()

    for obj_ in tqdm(questions):
        idx = obj_["id"]
        q = obj_["conversations"][0]["value"].split("<image>")[0].strip().rstrip("\n")
        if "ScienceQA" in args.question_file:
            q = q + "\n" + "Which one of the options is correct?"
        if "image" in obj_:
            if "ScienceQA" in args.question_file:
                image = Image.open(os.path.join(image_folder, obj_["image"]))
            else:
                image = Image.open(obj_["image"])
            inputs = processor(images=image, text=q, return_tensors="pt").to(device="cuda")
        else:
            inputs = processor(images=Image.new("RGB", (20, 20)), text=q, return_tensors="pt").to(device="cuda")
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
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
