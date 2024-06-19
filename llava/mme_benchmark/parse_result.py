import os
import argparse
import numpy as np

# import matplotlib.pyplot as plt


def parse_filename(str):
    if "noyes" in str:
        return "noyes", str.split(f"test_")[-1]
    else:
        return "yesno", str.split(f"test_")[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    # n_options = args.n_options
    result_dir = args.result_dir

    hparams_name = ["alpha", "k"]
    txt_files_to_parse = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if ".txt" in f]

    perf_dict = {}
    for f in txt_files_to_parse:
        print(f)
        step = 3
        file = open(f, "r")
        lines = [line.split("\n")[:-1] for line in file.readlines()]
        for i in range(0, len(lines), step):
            try:
                if len(lines[i][0]) > 0:
                    filename = lines[i]
                    performance = lines[i + 2]
                    # avg_performance = lines[i+3]
                else:
                    break
                fname, params = parse_filename(lines[i][0].split("/")[-1])

                # print('fname', fname, 'params', params)
                acc = float(performance[0].split("score: ")[-1])
                # avg_acc = float(avg_performance[0].split("score: ")[-1])
                if params not in perf_dict:
                    perf_dict[params] = {fname: {"total": acc}}
                else:
                    perf_dict[params][fname] = {"total": acc}
            except Exception as e:
                # raise e
                continue
    print(perf_dict)
    perf_by_hparams = {}
    for key in perf_dict:
        hparams_val = []
        hparams_str = key.split("_")
        for i, name in enumerate(hparams_name):
            hparams_val.append(float(hparams_str[i].split(name)[-1]))
        # print(hparams_val)
        acc_total = []
        # acc_avg = []
        # acc_yes = []
        # acc_no = []
        for key2 in perf_dict[key]:
            # print(key2)
            acc_total.append(perf_dict[key][key2]["total"])
            # acc_avg.append(perf_dict[key][key2]['avg'])
        # exit()
        std_total = np.std(acc_total)
        mean_total = np.mean(acc_total)
        # std_avg = np.std(acc_avg)
        # mean_avg = np.mean(acc_avg)
        # if hparams_val[0] in perf_by_hparams:
        #     perf_by_hparams[hparams_val[0]].append({hparams_val[1]: {'mean': mean, 'std': std}})
        # else:
        #     perf_by_hparams[hparams_val[0]] = [{hparams_val[1]: {'mean': mean, 'std': std}}]
        print(key)
        # print(acc_all)
        # print("total")
        # print(f"mean = {mean_total:.3f}")
        # print(f"std = {std_total:.3f}")
        print("avg")
        print(f"mean = {mean_total:.3f}")
        print(f"std = {std_total:.3f}")
        print(acc_total)
        print("")
