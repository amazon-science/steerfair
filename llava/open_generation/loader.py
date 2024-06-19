from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np


class VGG_Dataset(Dataset):
    def __init__(self, root_dir="/home/ubuntu/VGG-Face2", split="test"):
        self.df_all = pd.read_csv(os.path.join(root_dir, "meta", "identity_meta.csv"), on_bad_lines="skip")
        class_id_all = set(self.df_all["Class_ID"].tolist())

        filepath = os.path.join(root_dir, "splits", f"{split}.txt")
        self.image_list = []
        my_file = open(filepath, "r")
        data = my_file.read()
        self.image_list = np.array(data.split("\n"))
        my_file.close()
        dir_list = np.array([i.split("/")[0] for i in self.image_list])
        available_idxs = []
        for i, dir in enumerate(dir_list):
            if dir in class_id_all:
                available_idxs.append(i)

        self.image_list = self.image_list[available_idxs]
        self.image_dir = os.path.join(root_dir, "data", "test")

        age_annotation_file = os.path.join(root_dir, "meta", f"test_agetemp_imglist.txt")
        my_file = open(age_annotation_file, "r")
        data = my_file.read()
        self.age_annotation = np.array(data.split("\n"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        data_id = self.image_list[idx]
        image_path = os.path.join(self.image_dir, data_id)

        class_id = data_id.split("/")[0]
        gender_label = self.df_all.loc[self.df_all["Class_ID"] == class_id][" Gender"].values[0].strip().rstrip()

        age = 0
        subject_in_file_list = np.array([f for f in self.age_annotation if f.split("/")[0] == class_id])
        sample_idx = np.argwhere((subject_in_file_list == data_id)).flatten()[0]
        if sample_idx >= 10:
            age = 1
        data = {"data_id": data_id, "image": image_path, "gender": gender_label, "age": age}
        return data


if __name__ == "__main__":
    dataset = VGG_Dataset(split="val")
    print(dataset[30])
