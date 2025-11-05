# code produced by chatgpt
import os
import matplotlib.pyplot as plt
from PIL import Image

# Base directorypp
base_dir = "data/MMNIST/train"

# 5 classes (m0 to m4), 10 examples (0.0.png to 0.9.png)
num_classes = 5
num_examples = 10

fig, axes = plt.subplots(num_classes, num_examples, figsize=(10, 5))

for i in range(num_classes):
    class_dir = os.path.join(base_dir, f"m{i}")
    for j in range(num_examples):
        filename = f"{j/10:.1f}.png"  # creates 0.0, 0.1, …, 0.9
        img_path = os.path.join(class_dir, filename)
        
        # Open image
        img = Image.open(img_path) # grayscale
        # img = Image.open(img_path).convert("L")  # grayscale
        
        ax = axes[i, j]
        ax.imshow(img, aspect="auto")
        ax.axis("off")
        
        # Optional: label rows by class
        if j == 0:
            ax.set_ylabel(f"m{i}", rotation=0, labelpad=20, fontsize=12, va='center')


plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
# plt.tight_layout()
plt.show()
plt.savefig("data/MMNIST/example.png")
