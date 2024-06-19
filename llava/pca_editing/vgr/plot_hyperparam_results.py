import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


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
        # print(f)
        step = 7
        file = open(f, "r")
        lines = [line.split("\n")[:-1] for line in file.readlines()]
        for i in range(0, len(lines), step):
            try:
                if len(lines[i][0]) > 0:
                    filename = lines[i]
                    performance_yes = lines[i + 1]
                    performance_no = lines[i + 2]
                else:
                    break
                fname, params = parse_filename(lines[i][0].split("/")[-1])

                # print('fname', fname, 'params', params)
                acc_yes = float(performance_yes[0].split("Accuracy: ")[-1][:-1])
                acc_no = float(performance_no[0].split("Accuracy: ")[-1][:-1])
                if params not in perf_dict:
                    perf_dict[params] = {fname: {"acc_yes": acc_yes, "acc_no": acc_no}}
                else:
                    perf_dict[params][fname] = {"acc_yes": acc_yes, "acc_no": acc_no}
            except Exception as e:
                raise e
                # continue
    # print(perf_dict)
    perf_by_hparams = {}
    for key in perf_dict:
        hparams_val = []
        hparams_str = key.split("_")
        for i, name in enumerate(hparams_name):
            hparams_val.append(float(hparams_str[i].split(name)[-1]))
        # print(hparams_val)
        acc_yes = []
        acc_no = []
        for key2 in perf_dict[key]:
            acc_yes.append(perf_dict[key][key2]["acc_yes"])
            acc_no.append(perf_dict[key][key2]["acc_no"])
        # print('acc yes', acc_yes)
        # print('acc no', acc_no)
        std = np.mean((np.std(acc_yes), np.std(acc_no)))
        acc_yes.extend(acc_no)
        mean = np.mean(acc_yes)
        if hparams_val[0] in perf_by_hparams:
            perf_by_hparams[hparams_val[0]].append({int(hparams_val[1]): {"mean": mean, "std": std}})
        else:
            perf_by_hparams[hparams_val[0]] = [{int(hparams_val[1]): {"mean": mean, "std": std}}]
        # print(key)
        # print((f"mean = {mean:.3f}"))
        # print(f"std = {std:.3f}")
        # print("")
    alpha_all = []
    for k, v in perf_by_hparams.items():
        alpha_all.append(k)
        v.sort(key=lambda x: list(x.keys())[0])
    alpha_all.sort()
    alpha_all = alpha_all[:-2]
    perf_by_hparams = {k: perf_by_hparams[k] for k in alpha_all}

    k_all = [list(obj_.keys())[0] for obj_ in perf_by_hparams[alpha_all[0]]]
    # print(k_all)
    mean_all = np.zeros((len(k_all), len(alpha_all)))
    std_all = np.zeros((len(k_all), len(alpha_all)))

    for i, k in enumerate(k_all):
        # mean_param = []
        # std_param = []
        # print(alpha, perf_by_hparams[alpha])
        for j, alpha in enumerate(alpha_all):
            # print(alpha,k)
            # try:
            mean = perf_by_hparams[alpha][i][k]["mean"]
            std = perf_by_hparams[alpha][i][k]["std"]
            mean_all[i, j] = mean
            std_all[i, j] = std
            # except:
            #     continue
    #         mean_param.append(mean)
    #         std_param.append(std)
    #     mean_all.append(mean_param)
    #     std_all.append(std_param)
    # mean_all = np.array(mean_all).T
    # std_all = np.array(std_all).T
    # mean_all = np.array(mean_all).reshape((len(k_all), len(alpha_all)))
    # std_all = np.array(std_all).reshape((len(k_all), len(alpha_all)))

    orig_map = plt.cm.get_cmap("coolwarm")
    reversed_map = orig_map.reversed()

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axs[0].imshow(mean_all, cmap=reversed_map)
    for (j, i), label in np.ndenumerate(mean_all):
        if label > 70:
            c = "white"
        else:
            c = "black"
        axs[0].text(i, j, f"{label:.1f}", ha="center", va="center", c=c, size="large")
    axs[0].set_title("Mean Acc%")
    axs[0].set_xticks(np.arange(len(alpha_all)))
    axs[0].set_xticklabels([str(a) for a in alpha_all])
    axs[0].set_yticks(np.arange(len(k_all)))
    axs[0].set_yticklabels([str(k) for k in k_all])
    axs[0].set_xlabel(r"Intervention strength $\alpha$")
    axs[0].set_ylabel(r"Number of heads intervened $K$")

    im1 = axs[1].imshow(std_all, cmap=orig_map)
    axs[1].set_title("Std%")
    axs[1].set_xticks(np.arange(len(alpha_all)))
    axs[1].set_xticklabels([str(a) for a in alpha_all])
    axs[1].set_yticks(np.arange(len(k_all)))
    axs[1].set_yticklabels([])
    for (j, i), label in np.ndenumerate(std_all):
        if label < 7:
            c = "white"
        else:
            c = "black"
        axs[1].text(i, j, f"{label:.1f}", ha="center", va="center", c=c, size="large")
    axs[1].set_xlabel(r"Intervention strength $\alpha$")
    # axs[1].set_ylabel(r'Number of heads intervened $K$')
    plt.colorbar(im0, ax=axs[0])
    plt.colorbar(im1, ax=axs[1])
    plt.tight_layout()
    plt.savefig("vgr_hyperparams.png")
