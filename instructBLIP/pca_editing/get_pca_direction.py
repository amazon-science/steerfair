from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import numpy as np

import argparse
from tqdm import tqdm
import os
from einops import rearrange
from scipy.linalg import norm


def get_pca_direction(head_values, num_layers, num_heads, save=True, save_dir=None):
    pca = PCA(n_components=1)
    head_values = rearrange(head_values, "b l (h d) -> b l h d", h=num_heads)
    head_pca_directions = []
    pca_values = []
    for l in tqdm(range(num_layers)):
        for h in range(num_heads):
            try:
                X_head_orig = np.asarray(head_values[:, l, h, :])
                # print(len(np.argwhere(X_head_orig == np.nan)))
                if len(np.argwhere(np.isnan(X_head_orig))) > 0:
                    nan_idx = np.argwhere(np.isnan(X_head_orig))[0, :][0]
                    print("NAN!", len(nan_idx))
                    X_head_orig = np.delete(X_head_orig, nan_idx, axis=0)
                    # print(nan_idx)
                    # exit()
                norm_ = np.linalg.norm(X_head_orig, axis=0)
                # .astype(np.float16)
                if len(np.argwhere(norm_ == 0)) == 0:
                    X_head = X_head_orig / norm_
                # print(X_head)
                # exit()
                pca.fit(X_head)
                X_pca = pca.transform(X_head)
                head_pca_directions.append(pca.components_)
                pca_values.append(X_pca)
            except Exception as e:
                raise e
                # print(np.array(head_values[:, l, h, :], dtype=np.float16))
                # print(np.sum(X_head_orig, axis=1))
                # print(np.argwhere(X_head_orig== np.nan))
                # print(np.argwhere(X_head_orig== np.inf))
                # print(np.sum(X_head_orig), np.amax(X_head_orig), np.amin(X_head_orig))
                # print(norm_)
                # print(X_head)
                # exit()
    if save:
        assert save_dir != None
        save_dir = os.path.join(save_dir, subdir[:-2], subdir[-1:])
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"pca_direction_{args.n_samples}.npy"), head_pca_directions)
        np.save(os.path.join(save_dir, f"pca_values_{args.n_samples}.npy"), pca_values)
    return head_pca_directions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-values-dir", type=str, required=True)
    parser.add_argument("--n-samples", default=1000)
    args = parser.parse_args()

    save_dir = f"sqa/pca_by_category"
    os.makedirs(save_dir, exist_ok=True)

    n_heads = 32
    n_layers = 32

    head_values_dir = args.head_values_dir
    head_values_all = []
    for subdir in os.listdir(head_values_dir):
        load_dir = os.path.join(head_values_dir, subdir)
        head_values = np.load(os.path.join(load_dir, f"head_{str(args.n_samples)}.npy"))
        pca_head_directions = get_pca_direction(head_values, n_layers, n_heads, save=True, save_dir=save_dir)
