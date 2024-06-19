import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_filename(str):
    split_str = f"_{split}_"
    return str.split(split_str)
    # if 'noyes' in str:
    #     return 'no/yes', str.split(f"noyes_")[-1]
    # else:
    #     return 'yes/no', str.split(f"yesno_")[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="minival")
    args = parser.parse_args()

    split = args.split
    result_dir = args.result_dir

    hparams_name = ["alpha", "k"]
    txt_files_to_parse = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if ".txt" in f]
    # print(txt_files_to_parse)
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
                    break
                fname, params = parse_filename(lines[i][0].split("/")[-1])

                # print('fname', fname, 'params', params)
                acc = float(performance[0].split("Accuracy: ")[-1][:-1])
                if params not in perf_dict:
                    perf_dict[params] = {fname: acc}
                else:
                    perf_dict[params][fname] = acc
            except Exception as e:
                raise e
                # continue
    print(perf_dict)
    # exit()
    perf_by_hparams = {}
    for key in perf_dict:
        hparams_val = []
        hparams_str = key.split("_")
        for i, name in enumerate(hparams_name):
            hparams_val.append(float(hparams_str[i].split(name)[-1]))
        print(hparams_val)
        acc_all = []
        # acc_yes = []
        # acc_no = []
        for key2 in perf_dict[key]:
            acc_all.append(perf_dict[key][key2])
        std = np.std(acc_all)
        mean = np.mean(acc_all)
        if hparams_val[0] in perf_by_hparams:
            perf_by_hparams[hparams_val[0]].append({hparams_val[1]: {"mean": mean, "std": std}})
        else:
            perf_by_hparams[hparams_val[0]] = [{hparams_val[1]: {"mean": mean, "std": std}}]
        print(key)
        print(acc_all)
        print((f"mean = {mean:.3f}"))
        print(f"std = {std:.3f}")
        print("")
