import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_filename(str):
    if "noyes" in str:
        return "no/yes", str.split(f"noyes_")[-1]
    else:
        return "yes/no", str.split(f"yesno_")[-1]


def get_nth_exp_perf(result_dir):
    txt_files_to_parse = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if ".txt" in f]
    # print(result_dir)
    perf_dict = {}
    for f in txt_files_to_parse:
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
        hparams_str = key.split("_")[1:]
        for i, name in enumerate(hparams_name):
            hparams_val.append(int(hparams_str[i].split(name)[-1]))
        acc_yes = []
        acc_no = []
        for key2 in perf_dict[key]:
            acc_yes.append(perf_dict[key][key2]["acc_yes"])
            acc_no.append(perf_dict[key][key2]["acc_no"])
        std = np.mean((np.std(acc_yes), np.std(acc_no)))
        acc_yes.extend(acc_no)
        mean = np.mean(acc_yes)
        if hparams_val[0] in perf_by_hparams:
            perf_by_hparams[hparams_val[0]].append({"mean": mean, "std": std})
        else:
            perf_by_hparams[hparams_val[0]] = [{"mean": mean, "std": std}]
    return perf_by_hparams


def avg_nth_perf(perf_by_hparams):
    n_all = []
    for k, v in perf_by_hparams.items():
        n_all.append(k)
    n_all.sort()
    n_all = n_all[1:]
    perf_by_hparams = {k: perf_by_hparams[k] for k in n_all}
    mean_all = []
    std_all = []
    for i, n in enumerate(n_all):
        mean = perf_by_hparams[n][0]["mean"]
        std = perf_by_hparams[n][0]["std"]
        mean_all.append(mean)
        std_all.append(std)
    return n_all, mean_all, std_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    result_dir = args.result_dir
    hparams_name = ["n"]
    mean_all = []
    std_all = []
    for n in range(1, 11, 1):
        nth_exp_dir = os.path.join(result_dir, str(n))
        perf_dict = get_nth_exp_perf(nth_exp_dir)
        n_all, mean_n, std_n = avg_nth_perf(perf_dict)
        mean_all.append(mean_n)
        std_all.append(std_n)

    baseline_mean = [71.04 for i in range(len(n_all))]
    baseline_std = [12.6 for i in range(len(n_all))]
    iti_100_mean = [65.91 for i in range(len(n_all))]
    iti_100_std = [7.9 for i in range(len(n_all))]
    iti_500_mean = [71.63 for i in range(len(n_all))]
    iti_500_std = [9.1 for i in range(len(n_all))]

    mean_all = np.array(mean_all)
    error_mean = np.std(mean_all, axis=0) / np.sqrt(len(mean_all))
    std_all = np.array(std_all)
    error_std = np.std(std_all, axis=0) / np.sqrt(len(std_all))
    linewidth = 4
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    mean_plot = np.mean(mean_all, axis=0)
    alpha_supp = 1
    linewidth_supp = 3
    print(len(n_all), len(mean_plot))
    axs[0].plot(mean_plot, label="SteerFair", linewidth=linewidth, marker="o", markersize=6)
    axs[0].plot(baseline_mean, label="Vanilla", linewidth=linewidth_supp, linestyle="dashed", alpha=alpha_supp)
    axs[0].plot(iti_100_mean, label="ITI(100)", linewidth=linewidth_supp, linestyle="dotted", alpha=alpha_supp)
    axs[0].plot(iti_500_mean, label="ITI(500)", linewidth=linewidth_supp, linestyle="dashdot", alpha=alpha_supp)
    axs[0].fill_between(
        np.arange(len(n_all)),
        mean_plot - error_mean,
        mean_plot + error_mean,
        alpha=0.2,
    )
    axs[0].set_title(r"Avg% ($\uparrow$)", fontsize=18)

    std_plot = np.mean(std_all, axis=0)
    axs[1].plot(std_plot, linewidth=linewidth, label="SteerFair", marker="o", markersize=5)
    axs[1].fill_between(
        np.arange(len(n_all)),
        std_plot - error_std,
        std_plot + error_std,
        alpha=0.2,
    )
    axs[1].plot(baseline_std, label="Vanilla", linewidth=linewidth_supp, linestyle="dashed", alpha=alpha_supp)
    axs[1].plot(iti_100_std, label="ITI(100)", linewidth=linewidth_supp, linestyle="dotted", alpha=alpha_supp)
    axs[1].plot(iti_500_std, label="ITI(500)", linewidth=linewidth_supp, linestyle="dashdot", alpha=alpha_supp)
    axs[1].set_title(r"Std ($\downarrow$)", fontsize=20)

    plt.xticks(np.arange(len(n_all)), [n for i, n in enumerate(n_all)])
    axs[0].tick_params(axis="x", labelsize=10)
    axs[1].tick_params(axis="x", labelsize=10)
    # axs[0].set_xlabel('# of unlabeled sample', fontsize=15)
    axs[1].set_xlabel("# of unlabeled sample", fontsize=15)
    axs[0].set_ylabel("Score", fontsize=12)
    axs[1].set_ylabel("Score", fontsize=12)
    axs[1].grid(linewidth=0.2)
    axs[0].grid(linewidth=0.2)

    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    axs[1].legend(
        lines,
        labels,
        ncol=4,
        fontsize=10,
        bbox_to_anchor=(0.93, -0.23),
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig("ablations_REPEAT.png")
