import matplotlib.pyplot as plt
import os
import numpy as np

model_name = "idefics"
method_name = "FairSteer"

X = ["Correct captions", "Wrong Captions"]
X_axis = np.arange(len(X))

yesno_acc = [52.53, 51.09]
noyes_acc = [61.55, 43.11]

save_dir = os.path.join("vgr_plots", model_name)
if not os.path.isdir(model_name):
    os.makedirs(save_dir, exist_ok=True)
plt.figure(figsize=(4, 3))
plt.bar(X_axis - 0.1, yesno_acc, 0.2, label="yes/no", align="center")
plt.bar(X_axis + 0.1, noyes_acc, 0.2, label="no/yes", align="center")

plt.xticks(X_axis, X)
plt.ylabel("Acc %")
plt.title(f"VGR {model_name} {method_name}")
plt.legend()
plt.savefig(os.path.join(save_dir, f"{method_name}.png"))
