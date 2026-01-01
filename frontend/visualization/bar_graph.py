import io
import matplotlib.pyplot as plt

def plot_bar(values: dict, title="Metric Comparison"):
    labels = list(values.keys())
    nums = list(values.values())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(labels, nums)
    ax.set_title(title)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer.getvalue()
