import os
import json
import argparse
from tqdm import tqdm


def split_by_category(
    category,
    problems,
    all_indices,
):
    # return indices and problems by number of options
    idxs_n = []
    for i, idx in enumerate(all_indices):
        p = problems[idx]
        subject = p["subject"]
        if subject != category:
            continue
        n_choices = len(p["choices"])
        if n_choices != 2:
            continue
        idxs_n.append(idx)
    return set(idxs_n)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    filename = args.result_file.split("/")[-1]
    save_dir = args.result_file[: args.result_file.rindex("/")] + "_by_category"
    # print(save_dir)
    # exit()
    base_dir = "/home/ubuntu/ScienceQA/data/scienceqa"
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))["test"]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred["question_id"]: pred for pred in predictions}

    categories = ["language_science", "natural_science", "social_science"]
    idx_dict = {c: [] for c in categories}
    for category in categories:
        idxs = split_by_category(category.replace("_", " "), problems, split_indices)
        idx_dict[category] = idxs
        # print(category, len(idxs))

    pred_by_categories = {c: {} for c in categories}
    for pred_id in predictions:
        for cat in idx_dict:
            if pred_id in idx_dict[cat]:
                pred_by_categories[cat][pred_id] = predictions[pred_id]
                break
    for cat in pred_by_categories:
        save_dir_cat = os.path.join(save_dir, cat)
        if not os.path.isdir(save_dir_cat):
            os.makedirs(save_dir_cat)
        save_path = os.path.join(save_dir_cat, filename)
        ans_file = open(save_path, "w")
        # with open(save_path, "w") as f:
        #     json.dump(pred_by_categories[cat], f, indent=2)
        for key in tqdm(pred_by_categories[cat]):
            obj_ = pred_by_categories[cat][key]
            ans_file.write(json.dumps(obj_) + "\n")
            ans_file.flush()
