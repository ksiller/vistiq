import networkx as nx
import matplotlib.pyplot as plt


def draw_dag(edges, positions, title, filename):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    plt.figure(figsize=(10, 4))
    nx.draw(
        G,
        pos=positions,
        with_labels=True,
        node_size=3200,
        node_color="lightblue",
        font_size=10,
        arrows=True,
        arrowsize=20,
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()


# -----------------------
# 1. Chain DAG
# -----------------------
chain_edges = [
    ("image", "preprocess"),
    ("preprocess", "segment"),
    ("segment", "analyze"),
]

chain_pos = {
    "image": (0, 0),
    "preprocess": (1, 0),
    "segment": (2, 0),
    "analyze": (3, 0),
}

draw_dag(chain_edges, chain_pos, "Chain Pipeline", "chain_dag.png")


# -----------------------
# 2. Branch DAG
# -----------------------
branch_edges = [
    ("image", "preprocess"),
    ("preprocess", "segment"),
    ("segment", "analyze"),
    ("segment", "classify"),
]

branch_pos = {
    "image": (0, 0),
    "preprocess": (1, 0),
    "segment": (2, 0),
    "analyze": (3, 1),
    "classify": (3, -1),
}

draw_dag(branch_edges, branch_pos, "Branch Pipeline", "branch_dag.png")


# -----------------------
# 3. Merge DAG
# -----------------------
merge_edges = [
    ("channel_1", "preprocess_1"),
    ("channel_2", "preprocess_2"),
    ("preprocess_1", "segment_1"),
    ("preprocess_2", "segment_2"),
    ("segment_1", "coincidence"),
    ("segment_2", "coincidence"),
]

merge_pos = {
    "channel_1": (0, 1),
    "preprocess_1": (1, 1),
    "segment_1": (2, 1),
    "channel_2": (0, -1),
    "preprocess_2": (1, -1),
    "segment_2": (2, -1),
    "coincidence": (3, 0),
}

draw_dag(merge_edges, merge_pos, "Merge Pipeline (Coincidence)", "merge_dag.png")


# -----------------------
# 4. Training DAG
# -----------------------
train_edges = [
    ("raw_images", "train"),
    ("ground_truth_labels", "train"),
    ("train", "fine_tuned_model"),
    ("fine_tuned_model", "segment"),
]

train_pos = {
    "raw_images": (0, 1),
    "ground_truth_labels": (0, -1),
    "train": (1.5, 0),
    "fine_tuned_model": (3, 0),
    "segment": (4.5, 0),
}

draw_dag(train_edges, train_pos, "Training Pipeline", "train_dag.png")


print("Saved: chain_dag.png, branch_dag.png, merge_dag.png, train_dag.png")