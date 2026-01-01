import io
import matplotlib.pyplot as plt

def plot_pie(label_counts, title="Class Distribution"):

    labels = list(label_counts.keys())
    sizes = list(label_counts.values())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title(title)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer.getvalue()
