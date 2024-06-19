import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from baukit import TraceDict


def get_llama_activations_bau(model, prompt, device):
    model.eval()

    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS + MLPS) as ret:
            output = model(prompt, output_hidden_states=True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def train_single_prob(X_all, y_all, val_size):
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=val_size)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    return clf, val_acc


def convert_labels(labels):
    labels_new = np.zeros(len(labels), dtype=np.int64)
    class_idx = 1
    for i in range(1, labels.shape[1]):
        one_idxs = np.argwhere(labels[:, i] == 1)
        labels_new[one_idxs] = class_idx
        class_idx += 1
    return labels_new
