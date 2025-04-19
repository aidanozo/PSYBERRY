import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb

def darken_color(hex_color, factor=0.85):
    r, g, b = to_rgb(hex_color)
    return mcolors.to_hex((r * factor, g * factor, b * factor))

df = pd.read_csv("../testing_metrics.csv")

metric_means = df.groupby("metric")["metric_value"].mean().sort_values(ascending=False)

turquoise_cmap = cm.get_cmap("PuBuGn", len(metric_means))

colors = {}
for i, metric in zip(reversed(range(len(metric_means))), metric_means.index):
    base_color = turquoise_cmap(i)
    hex_color = mcolors.rgb2hex(base_color)
    if metric == "BLEU":
        hex_color = darken_color(hex_color, factor=0.7)
    colors[metric] = hex_color

plt.figure(figsize=(12, 6))

for metric in metric_means.index:
    data = df[df["metric"] == metric]
    plt.plot(data["test_pair"], data["metric_value"], label=metric, color=colors[metric], linewidth=2)

plt.xlabel("Test Pair")
plt.ylabel("Value")
plt.title("Evaluation Metrics per Test Pair")
plt.legend(title="Metric")
plt.grid(True)
plt.tight_layout()

plt.savefig("testing_metrics_graph.png", dpi=600, bbox_inches='tight', transparent=True)

plt.show()