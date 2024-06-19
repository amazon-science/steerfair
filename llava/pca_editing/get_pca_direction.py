from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import numpy as np

import argparse
from tqdm import tqdm
import os
from einops import rearrange

def get_pca_direction(head_values, num_layers, num_heads, save=True, save_dir=None):
    pca = PCA(n_components=10)
    head_values = rearrange(head_values, 'b l (h d) -> b l h d', h = num_heads)
    head_pca_directions = []
    pca_values = []
    exp_var_all = []
    exp_var_ratio_all = []
    singular_values_all = []
    noise_all = []
    for l in tqdm(range(num_layers)):
        for h in range(num_heads):
            X_head = np.array(head_values[:, l, h, :])
            X_head = X_head/np.linalg.norm(X_head, axis=0)
            pca.fit(X_head)
            X_pca = pca.transform(X_head)
            head_pca_directions.append(pca.components_)
            pca_values.append(X_pca)
            exp_var_ratio = pca.explained_variance_ratio_
            exp_var_ratio_all.append(exp_var_ratio)
            
            exp_var = pca.explained_variance_
            exp_var_all.append(exp_var)

            singular_values = pca.singular_values_
            singular_values_all.append(singular_values)

            noise = pca.noise_variance_
            noise_all.append(noise)
    if save:
        assert save_dir != None
        save_dir = os.path.join(save_dir,subdir[:-2],subdir[-1])
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"pca_direction.npy"), head_pca_directions)
        np.save(os.path.join(save_dir, f"pca_values.npy"), pca_values)
        np.save(os.path.join(save_dir, f"exp_var_ratio.npy"), exp_var_ratio_all)
        np.save(os.path.join(save_dir, f"exp_var.npy"), exp_var_all)
        np.save(os.path.join(save_dir, f"singular_values.npy"), singular_values_all)
        np.save(os.path.join(save_dir, f"noise.npy"), noise_all)
    return head_pca_directions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-values-dir", type=str, required=True)
    # parser.add_argument("--options", default=["no", "yes"])
    parser.add_argument("--n-samples", default=1000)
    parser.add_argument("--save-dir", type=str)
    args = parser.parse_args()

<<<<<<< HEAD
    save_dir = f"scienceqa/pca_by_category"
=======
    save_dir = f"sqa/pca_by_category"
>>>>>>> f948403329f039d45a6aeab3360083ee554bb8d8
    os.makedirs(save_dir, exist_ok=True)

    n_heads = 40
    n_layers = 40 

    head_values_dir = args.head_values_dir 
    head_values_all = []
    for subdir in os.listdir(head_values_dir):
        load_dir = os.path.join(head_values_dir, subdir)
        head_values = np.load(os.path.join(load_dir, f"head.npy"))
        print(subdir, head_values.shape)
        pca_head_directions = get_pca_direction(head_values, n_layers, n_heads, save=True, save_dir=save_dir)
        # head_values_all.append(head_values)
    # head_values_all = np.vstack(head_values_all)
    # pca_head_directions = get_pca_direction(head_values_all, n_layers, n_heads, save=False)
    # np.save(os.path.join(save_dir, "combined", f"pca_direction_{args.n_samples}.npy"), pca_head_directions)
    # print(h)
    