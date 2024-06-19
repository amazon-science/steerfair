import matplotlib.pyplot as plt
import os


model_name = "instructBLIP"
method_name = "FairSteer"

acc_2 = [63.51, 63.73, 61.18]
acc_3 = [44.49, 61.28, 22.45, 48.09]
acc_4 = [56.08, 89.94, 41.73, 48.80, 51.69]
acc_5 = [21.05, 86.84, 2.63, 2.63, 5.26, 2.63]

acc_all = [acc_2, acc_3, acc_4, acc_5]
labels = ["# options 2", "# options 3", "# options 4", " # options 5"]

save_dir = os.path.join("plots_sqa", model_name)
if not os.path.isdir(model_name):
    os.makedirs(save_dir, exist_ok=True)
bplot = plt.boxplot(acc_all, vert=True, patch_artist=True, labels=labels)  # vertical box alignment

colors = ["pink", "lightblue", "lightgreen", "orange", "dimgray"]

for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)
plt.ylabel("Acc %")
plt.grid(axis="y")
plt.title(f"ScienceQA {model_name} {method_name}")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{method_name}.png"))
