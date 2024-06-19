from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import json


def file_to_list(filepath):
    my_file = open(filepath, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    return data_into_list


img_file = "/home/ubuntu/VGG-Face2/data/train_list.txt"
img_file_list = file_to_list(img_file)
img_file_list = [os.path.join("/home/ubuntu/VGG-Face2/data/train", f_) for f_ in img_file_list]
dataset = load_dataset("allenai/real-toxicity-prompts")

categories = [
    "sexually_explicit",
    "profanity",
    "sexually_explicit",
    "identity_attack",
    "flirtation",
    "threat",
    "insult",
    "toxicity",
    "severe_toxicity",
]

idxs_all = []

for category in tqdm(categories):
    toxic_idxs = [
        idx
        for idx, obj_ in enumerate(dataset["train"]["prompt"])
        if (
            (" her " in obj_["text"])
            or (" his " in obj_["text"])
            or (" she is " in obj_["text"])
            or (" he is " in obj_["text"])
            or ("this person" in obj_["text"])
        )
        and (category in obj_)
        and (obj_[category])
        and (obj_[category] > 0.8)
    ]
    idxs_all.extend(toxic_idxs)
idxs_all = np.unique(idxs_all)
toxic_prompts = np.array([obj_ for obj_ in dataset["train"]["prompt"]])
toxic_cont = np.array([obj_ for obj_ in dataset["train"]["continuation"]])

train_idxs, test_idxs = train_test_split(idxs_all, test_size=0.4)
train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.2)
print(f"train = {len(train_idxs)}, test = {len(test_idxs)}, val = {len(val_idxs)}")
train_objects = []
for idx in tqdm(train_idxs):
    prompt = toxic_prompts[idx]["text"]
    cont = toxic_cont[idx]["text"]
    img = np.random.choice(img_file_list)
    obj_ = {
        "id": f"train_{idx}",
        "image": img,
        "prompt": prompt,
        "continuation": cont,
    }
    train_objects.append(obj_)

train_fname = "/home/ubuntu/VGG-Face2/toxic_prompt_train.json"
with open(train_fname, "w") as f:
    json.dump(train_objects, f, indent=2)


test_objects = []
for idx in tqdm(test_idxs):
    prompt = toxic_prompts[idx]["text"]
    cont = toxic_cont[idx]["text"]
    img = np.random.choice(img_file_list)
    obj_ = {
        "id": f"test_{idx}",
        "image": img,
        "prompt": prompt,
        "continuation": cont,
    }
    test_objects.append(obj_)

test_fname = "/home/ubuntu/VGG-Face2/toxic_prompt_test.json"
with open(test_fname, "w") as f:
    json.dump(test_objects, f, indent=2)

val_objects = []
for idx in tqdm(val_idxs):
    prompt = toxic_prompts[idx]["text"]
    cont = toxic_cont[idx]["text"]
    img = np.random.choice(img_file_list)
    obj_ = {
        "id": f"val_{idx}",
        "image": img,
        "prompt": prompt,
        "continuation": cont,
    }
    val_objects.append(obj_)

val_fname = "/home/ubuntu/VGG-Face2/toxic_prompt_val.json"
with open(val_fname, "w") as f:
    json.dump(val_objects, f, indent=2)
# for i, p,c in tqdm(enumerate(zip(toxic_prompts, toxic_cont))):
