from torch.utils.data import Dataset
import os
from PIL import Image


class MME_Dataset(Dataset):
    def __init__(self, root_dir="/home/ubuntu/MME_benchmark/MME_Benchmark_release_version"):
        self.categories = os.listdir(root_dir)
        self.images_path_all, self.qa_path_all, self.indexes_all = self.load_data_all(root_dir)

    def load_data_all(self, root_dir):
        images_path_all = []
        qa_path_all = []
        indexes_all = []
        for cat in self.categories:
            category_path = os.path.join(root_dir, cat)
            if not os.path.isdir(category_path):
                continue
            # directories = [dir for dir in os.listdir(category_path) if os.path.isdir(category_path)]
            if "images" in os.listdir(category_path):
                images_path = os.path.join(category_path, "images")
                questions_path = os.path.join(category_path, "questions_answers_YN")
                file_name_all = os.listdir(images_path)
                file_name_all = [f.split(".jpg")[0] for f in file_name_all if ".jpg" in f]
                images_path_all.extend([os.path.join(images_path, f"{p}.jpg") for p in file_name_all])
                qa_path_all.extend([os.path.join(questions_path, f"{p}.txt") for p in file_name_all])
                indexes_all.extend([f"{cat}_{p}" for p in file_name_all])
            else:
                file_name_all = [f.split(".jpg")[0] for f in os.listdir(category_path) if ".jpg" in f]
                images_path_all.extend([os.path.join(category_path, f"{p}.jpg") for p in file_name_all])
                qa_path_all.extend([os.path.join(category_path, f"{p}.txt") for p in file_name_all])
                indexes_all.extend([f"{cat}_{p}" for p in file_name_all])
        return images_path_all, qa_path_all, indexes_all

    def __len__(self):
        return len(self.images_path_all)

    def __getitem__(self, idx):
        text_split = "Please answer yes or no."
        index = self.indexes_all[idx]
        image = Image.open(self.images_path_all[idx])
        data = []
        with open(self.qa_path_all[idx], "r") as file:
            data = []
            for line in file:
                line = line.strip().rstrip()
                data.append(line)
        qa1 = data[0]
        qa2 = data[1]
        q1, a1 = qa1.split(text_split)
        q1 = q1.strip().rstrip()
        a1 = a1.strip().rstrip()

        q2, a2 = qa2.split(text_split)
        q2 = q2.strip().rstrip()
        a2 = a2.strip().rstrip()
        data = {
            "index": index,
            "image": self.images_path_all[idx],
            "question1": q1,
            "answer1": a1,
            "question2": q2,
            "answer2": a2,
        }
        return data


if __name__ == "__main__":
    dataset = MME_Dataset()
    print(dataset[0])
