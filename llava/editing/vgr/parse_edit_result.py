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


def parse_filename(str):
    if "noyes" in str:
        return "no/yes", str.split(f"noyes_")[-1]
    else:
        return "yes/no", str.split(f"yesno_")[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    result_dir = args.result_dir

    hparams_name = ["alpha", "k"]
    txt_files_to_parse = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if ".txt" in f]

    perf_dict = {}
    for f in txt_files_to_parse:
        print(f)
        step = 7
        file = open(f, "r")
        lines = [line.split("\n")[:-1] for line in file.readlines()]
        for i in range(0, len(lines), step):
            filename = lines[i]
            performance_yes = lines[i + 1]
            performance_no = lines[i + 2]
            performance_diff = lines[i + 3]
            performance_all = lines[i + 4]
            # print(filename, performance_diff, )
            fname, params = parse_filename(filename[0])
            acc_yes = float(performance_yes[0].split("Accuracy: ")[-1][:-1])
            acc_no = float(performance_no[0].split("Accuracy: ")[-1][:-1])
            acc_all = float(performance_all[0].split("Accuracy: ")[-1][:-1])
            acc_diff = np.abs(float(performance_diff[0].split(" ")[-1][:-1]))
            if params not in perf_dict:
                perf_dict[params] = {
                    fname: {"acc_yes": acc_yes, "acc_no": acc_no, "acc_all": acc_all, "diff": acc_diff}
                }
            else:
                perf_dict[params][fname] = {"acc_yes": acc_yes, "acc_no": acc_no, "acc_all": acc_all, "diff": acc_diff}
    # exit()
    perf_by_hparams = {}

    for key in perf_dict:
        print(key)
        perf_dict[key] = dict(sorted(perf_dict[key].items()))
        acc_yes_all = []
        acc_no_all = []
        acc_diff_all = []
        acc_all_all = []
        for key2 in perf_dict[key]:
            hparams_str = key.split("_")[-2:]
            hparams_val = []
            for i, name in enumerate(hparams_name):
                hparams_val.append(int(hparams_str[i].split(name)[-1]))
            acc_yes_all.append(perf_dict[key][key2]["acc_yes"] / 100)
            acc_no_all.append(perf_dict[key][key2]["acc_no"] / 100)
            acc_diff_all.append(perf_dict[key][key2]["diff"] / 100)
            acc_all_all.append(perf_dict[key][key2]["acc_all"] / 100)

        std_yes = np.std(acc_yes_all)
        std_no = np.std(acc_no_all)
        acc_all = acc_yes_all
        acc_all.extend(acc_no_all)
        std_all = np.std(acc_all)
        mean_all = np.mean(acc_all)
        delta_yes = np.abs(acc_yes_all[0] - acc_yes_all[1])
        delta_no = np.abs(acc_no_all[0] - acc_no_all[1])
        diff_diff = np.sum(acc_diff_all)
        std_combined = np.abs(acc_all_all[0] - acc_all_all[1])
        if hparams_val[0] in perf_by_hparams:
            perf_by_hparams[hparams_val[0]].append(
                {
                    hparams_val[1]: {
                        "std_yes": std_yes,
                        "std_no": std_no,
                        "std_all": std_all,
                        "mean_all": mean_all,
                        "discrepancy": diff_diff,
                    }
                }
            )
        else:
            perf_by_hparams[hparams_val[0]] = [
                {
                    hparams_val[1]: {
                        "std_yes": std_yes,
                        "std_no": std_no,
                        "std_all": std_all,
                        "mean_all": mean_all,
                        "discrepancy": diff_diff,
                    }
                }
            ]
        #     print((f"mean = {mean:.3f}"))
        # print(f"std_yes = {std_yes:.3f}")
        # print(f"std_no = {std_no:.3f}")
        # print(f"std_all = {std_all:.3f}")
        # print(f"mean_all = {mean_all:.3f}")
        # print(f"delta yes = {delta_yes}")
        # print(f"delta no = {delta_no}")
        print(f"delta total = {delta_yes+delta_no}")
        # print(f"diff total = {diff_diff}")
        # print(f"mean total = {((delta_yes+delta_no)+diff_diff)/2}")
        # print(f"std combined = {std_combined}")
        print("")
    # alpha_all = list(perf_by_hparams.keys())
    # alpha_all.sort()
    # perf_by_hparams = {i: perf_by_hparams[i] for i in alpha_all}
    # print(perf_by_hparams)

    # # mean_matrix = []
    # # std_matrix = []

    # # for alpha in perf_by_hparams:
    # #     k_all = [list(obj_.keys())[0] for obj_ in perf_by_hparams[alpha]]
    # #     k_all.sort()
    # #     mean_ = [list(obj_.values())[0]['mean'] for i, obj_ in enumerate(perf_by_hparams[alpha])]
    # #     std_ = [list(obj_.values())[0]['std'] for i, obj_ in enumerate(perf_by_hparams[alpha])]
    # #     mean_matrix.append(mean_)
    # #     std_matrix.append(std_)

    # # mean_matrix = np.array(mean_matrix)
    # # std_matrix = np.array(std_matrix)

    # # title = args.result_dir.split('/')[-2]

    # # plt.imshow(std_matrix, cmap='Blues')
    # # for (j,i),label in np.ndenumerate(std_matrix):
    # #     plt.text(i,j,f'{label:.4f}',ha='center',va='center')
    # # plt.yticks(range(len(alpha_all)), alpha_all)
    # # plt.xticks(range(len(k_all)), k_all)
    # # plt.xlabel('k')
    # # plt.ylabel('scale')
    # # plt.title('std')
    # # plt.colorbar()
    # # plt.tight_layout()
    # # plt.title(title)
    # # plt.savefig(f'plot_results/{title}.png')
