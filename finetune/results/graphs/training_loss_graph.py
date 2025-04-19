import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../training_loss.csv")

plt.figure(figsize=(12, 6))

for epoch, group in df.groupby("Epoch"):
    plt.plot(group["Step"], group["Loss"], label=f"Epoch {epoch}")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss per Step")
plt.legend(title="Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_graph.png", dpi=600, bbox_inches='tight', transparent=True)

plt.show()