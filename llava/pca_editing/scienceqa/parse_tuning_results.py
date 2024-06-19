import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_filename(filename, split, noption):
    return filename.split(f"_{split}")[0], filename.split(f"noption_{str(noption)}_{split}_")[-1].split("_result")[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    result_dir = args.result_dir

    hparams_name = ["alpha", "k"]
    txt_files_to_parse = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if ".txt" in f]
    perf_dict = {}
    for f in txt_files_to_parse:
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
                noption = int(filename[0].split("noption_")[-1].split("_")[0])
                fname, params = parse_filename(filename[0], args.split, noption)
                acc = performance[0].split("Accuracy: ")[-1]
                if params not in perf_dict:
                    perf_dict[params] = {fname: acc}
                else:
                    perf_dict[params][fname] = acc
            except Exception as e:
                print("failed", filename)
                print(f)
                raise e
                continue
    # print(perf_dict)
    perf_by_hparams = {}
    perf_by_hparams_stratified = {i: {} for i in range(2, 6)}
    n_dict = {
        2: 221,
        3: 102,
        4: 97,
        5: 4,
    }
    for key in perf_dict:
        # print(key)
        try:
            hparams_str = key.split("_")[-2:]
            hparams_val = []
            for i, name in enumerate(hparams_name):
                hparams_val.append(int(hparams_str[i].split(name)[-1]))
            perf_dict[key] = dict(sorted(perf_dict[key].items()))
            nums_by_noption = {i: [] for i in range(2, 6)}
            # print(key)
            for key2 in perf_dict[key]:
                noption = int(key2[-1])
                nums_by_noption[noption].append(float(perf_dict[key][key2][:-1]) / 100)

            std = 0
            mean = 0
            std_all = []
            mean_all = []
            for n in nums_by_noption:
                nums = nums_by_noption[n]
                weight = n_dict[n]
                std_n = np.std(nums)
                mean_n = np.mean(nums)
                std_all.append(std_n)
                mean_all.append(mean_n)
                if hparams_val[0] in perf_by_hparams_stratified[n]:
                    perf_by_hparams_stratified[n][hparams_val[0]].append(
                        {hparams_val[1]: {"mean": mean_n, "std": std_n}}
                    )
                else:
                    perf_by_hparams_stratified[n][hparams_val[0]] = [{hparams_val[1]: {"mean": mean_n, "std": std_n}}]
                std += weight * std_n
                mean += weight * mean_n
            alpha_all = list(perf_by_hparams_stratified[2].keys())
            alpha_all.sort()
            for n in nums_by_noption:
                perf_by_hparams_stratified[n] = {i: perf_by_hparams_stratified[n][i] for i in alpha_all}

            std /= np.sum(list(n_dict.values()))
            mean /= np.sum(list(n_dict.values()))

            # print(nums_by_noption)
            print(key)
            print("weighted")
            # print((f"mean = {mean:.3f}"))
            print(f"std = {std:.3f}")
            print("unweighted")
            # print((f"mean = {np.mean(mean_all):.3f}"))
            print((f"mean = {mean:.3f}"))
            print(f"std = {std:.3f}")
            print("unweighted")
            print((f"mean = {np.mean(mean_all):.3f}"))
            print(f"std = {np.mean(std_all):.3f}")
            print(nums_by_noption)
            print("")

            if hparams_val[0] in perf_by_hparams:
                perf_by_hparams[hparams_val[0]].append({hparams_val[1]: {"mean": mean, "std": std}})
            else:
                perf_by_hparams[hparams_val[0]] = [{hparams_val[1]: {"mean": mean, "std": std}}]
        except Exception as e:
            # print(key)
            # raise e
            continue
    alpha_all = list(perf_by_hparams.keys())
    alpha_all.sort()
    perf_by_hparams = {i: perf_by_hparams[i] for i in alpha_all}

    # print(perf_by_hparams_stratified)

    # mean_matrix_all = []
    # std_matrix_all = []

    # for alpha in perf_by_hparams:
    #     k_all = [list(obj_.keys())[0] for obj_ in perf_by_hparams[alpha]]
    #     val_all = [list(obj_.values())[0] for obj_ in perf_by_hparams[alpha]]
    #     argsort = np.argsort(k_all).tolist()
    #     k_all = np.array(k_all)[argsort]
    #     val_all = np.array(val_all)[argsort]
    #     # print(alpha, len(k_all), len(val_all))
    #     mean_ = [obj_['mean'] for obj_ in val_all]
    #     std_ = [obj_['std'] for obj_ in val_all]
    #     mean_matrix_all.append(mean_)
    #     std_matrix_all.append(std_)

    # # print(mean_matrix_all)
    # mean_matrix_all = np.array(mean_matrix_all)
    # std_matrix_all = np.array(std_matrix_all)
