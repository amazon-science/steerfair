"""
get n_option
get directory
get all .txt file
for each file:
    1. parse to list
    2. get performance, group by hyperparams
per hyperparam, report performance as usual
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_filename(str, split):
    return str.split(f"_{split}")[0], str.split(f"noption_{n_options}_{split}_")[-1].split("_result")[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    n_options = args.n_options
    result_dir = args.result_dir

    hparams_name = ["alpha", "k"]
    txt_files_to_parse = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if ".txt" in f]

    perf_dict = {}
    for f in txt_files_to_parse:
        print(f)
        step = 4
        file = open(f, "r")
        lines = [line.split("\n")[:-1] for line in file.readlines()]
        for i in range(0, len(lines), step):
            try:
                if len(lines[i][0]) > 0:
                    filename = lines[i]
                    performance = lines[i + 1]
                else:
                    filename = lines[i + 1]
                    performance = lines[i + 2]
                #     i = i+1
                # print(filename)
                # print(performance)
                # print("")

                fname, params = parse_filename(filename[0], args.split)
                acc = performance[0].split("Accuracy: ")[-1]
                if params not in perf_dict:
                    perf_dict[params] = {fname: acc}
                else:
                    perf_dict[params][fname] = acc
            except:
                continue
    # exit()
    # print(perf_dict)
    # exit()
    perf_by_hparams = {}

    for key in perf_dict:
        print(key)
        perf_dict[key] = dict(sorted(perf_dict[key].items()))
        nums = []
        for key2 in perf_dict[key]:
            hparams_str = key.split("_")[-2:]
            hparams_val = []
            for i, name in enumerate(hparams_name):
                hparams_val.append(float(hparams_str[i].split(name)[-1]))
            print(key2, perf_dict[key][key2])
            nums.append(float(perf_dict[key][key2][:-1]) / 100)
        std = np.std(nums)
        mean = np.mean(nums)
        if hparams_val[0] in perf_by_hparams:
            perf_by_hparams[hparams_val[0]].append({hparams_val[1]: {"mean": mean, "std": std}})
        else:
            perf_by_hparams[hparams_val[0]] = [{hparams_val[1]: {"mean": mean, "std": std}}]
        print((f"mean = {mean:.3f}"))
        print(f"std = {std:.3f}")
        print("")
    alpha_all = list(perf_by_hparams.keys())
    alpha_all.sort()
    perf_by_hparams = {i: perf_by_hparams[i] for i in alpha_all}
    # print(perf_by_hparams)

    mean_matrix = []
    std_matrix = []

    for alpha in perf_by_hparams:
        # print(alpha)
        k_all = [list(obj_.keys())[0] for obj_ in perf_by_hparams[alpha]]
        val_all = [list(obj_.values())[0] for obj_ in perf_by_hparams[alpha]]
        argsort = np.argsort(k_all).tolist()
        k_all = np.array(k_all)[argsort]
        val_all = np.array(val_all)[argsort]
        # print(k_all)
        # print(val_all)
        mean_ = [obj_["mean"] for obj_ in val_all]
        std_ = [obj_["std"] for obj_ in val_all]
        mean_matrix.append(mean_)
        std_matrix.append(std_)

    mean_matrix = np.array(mean_matrix)
    std_matrix = np.array(std_matrix)

    title = args.result_dir.split("/")[-2]

    plt.imshow(std_matrix, cmap="Blues")
    for (j, i), label in np.ndenumerate(std_matrix):
        plt.text(i, j, f"{label:.4f}", ha="center", va="center")
    plt.yticks(range(len(alpha_all)), alpha_all)
    plt.xticks(range(len(k_all)), k_all)
    plt.xlabel("k")
    plt.ylabel("scale")
    plt.title("std")
    plt.colorbar()
    plt.tight_layout()
    plt.title(title)
    plt.savefig(f"plot_results/inverse.png")
