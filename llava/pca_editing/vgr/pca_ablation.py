import os
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


def get_per_head_component_value(pca_values_all):
    values_mean = np.mean(pca_values_all, axis=-1)
    return values_mean


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def get_top_heads(per_head_component, num_to_intervene):
    top_component_val = np.argsort(per_head_component.reshape(40 * 40))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, 40) for idx in top_component_val]
    return top_heads


if __name__ == "__main__":
    var_all_yes = []
    var_all_no = []
    n_all = [50, 100, 300, 500, 1000, 2000, 5000, 10000]
    for i in range(1, 11, 1):
        nth_pca_dir_all = [f"pca_ablation/pca_i{i}_n{n}" for n in n_all]
        var_nth_exp_yes = []
        var_nth_exp_no = []
        for i, dir_ in enumerate(nth_pca_dir_all):
            yes_dir_val = np.load(os.path.join(dir_, "yes_1", "exp_var_ratio.npy"))[60, 0]
            no_dir_val = np.load(os.path.join(dir_, "no_1", "exp_var_ratio.npy"))[60, 0]
            var_nth_exp_yes.append(yes_dir_val)
            var_nth_exp_no.append(no_dir_val)
        var_all_yes.append(var_nth_exp_yes)
        var_all_no.append(var_nth_exp_no)
    var_all_yes = np.array(var_all_yes).squeeze()
    yes_error = np.std(var_all_yes, axis=0)
    var_all_yes = np.mean(var_all_yes, axis=0)

    var_all_no = np.array(var_all_no).squeeze()
    no_error = np.std(var_all_no, axis=0)
    var_all_no = np.mean(var_all_no, axis=0)
    plt.figure(figsize=(5, 4))
    plt.plot(var_all_yes * 100, label="last", marker="o", linewidth=5, markersize=10)
    plt.fill_between(
        np.arange(len(n_all)),
        (var_all_yes * 100) - (yes_error * 100),
        (var_all_yes * 100) + (yes_error * 100),
        alpha=0.2,
    )
    plt.plot(var_all_no * 100, label="first", marker="o", linewidth=5, markersize=10)
    plt.fill_between(
        np.arange(len(n_all)),
        (var_all_no * 100) - (no_error * 100),
        (var_all_no * 100) + (no_error * 100),
        alpha=0.2,
    )
    plt.plot(
        np.mean(np.vstack([var_all_no * 100, var_all_yes * 100]), axis=0),
        label="mean",
        marker="D",
        linewidth=4,
        markersize=7,
        linestyle="--",
    )
    plt.title("Explained variance (%) in the 1st PC", fontsize=15)
    plt.xticks(np.arange(len(n_all)), [n for i, n in enumerate(n_all)])
    plt.legend()
    plt.tight_layout()
    plt.grid()

    plt.savefig("pca.png")
