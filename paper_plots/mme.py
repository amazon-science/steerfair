import matplotlib.pyplot as plt
import os
import numpy as np

model_name = "instructBLIP"

models = [model_name for i in range(4)]
methods = ["vanilla", "ITI 100", "ITI 500", "FairSteer"]
titles = [f"{model} {method}" for model, method in zip(models, methods)]

X = ["yes/no", "no/yes"]
X_axis = np.arange(len(X))

scores = {
    "vanilla": [1186.519977193308, 1165.185058098809],
    "ITI 100": [1180.3569743597136, 1188.245259038165],
    "ITI 500": [1180.903989, 1179.843541],
    "FairSteer": [1175.2916913714257, 1196.558509618626],
}

save_dir = os.path.join("mme_plots")
if not os.path.isdir(model_name):
    os.makedirs(save_dir, exist_ok=True)
# plt.figure(figsize=(4,3))

fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 3))
for i in range(len(axs)):
    barlist = axs[i].plot(X_axis, scores[methods[i]], marker="X", markersize=10, linestyle="dotted")
    # barlist[0].set_color('tab:blue')
    # barlist[1].set_color('tab:orange')
    axs[i].set_title(titles[i])

plt.xticks(X_axis, X)
axs[0].set_ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{model_name}.png"))
