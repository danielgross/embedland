"""Tools to visualize embeddings."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def cluster(embeddings, labels, cluster_count=10):
    """Cluster the embeddings and return embeddings and cluster labels."""
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d, labels


def plot_matplotlib(embeddings_2d, labels, labels_dict):
    """Embeddings 2D is the output of the TSNE function. 
    Labels is the output of the KMeans function. 
    Labels dict is a dictionary mapping the cluster number to the label.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
               c=labels, cmap="rainbow")
    for cluster, label in labels_dict.items():
        ax.plot([], [], label=label, marker="o", c=f"C{cluster}")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=14)
    plt.show()


def plot_ploty(embeddings_2d, labels, labels_dict):
    """Same as above, but with Plotly."""
    df = pd.DataFrame(
        {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": labels})
    df["label"] = df["label"].map(labels_dict)
    fig = px.scatter(df, x="x", y="y", color="label")
    fig.show()


def plot_3d_embeddings(embeddings, labels, labels_dict):
    """Project into three dimensions.
    Unlike prior functions, this one takes the embeddings straight up.
    """
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings)
    df = pd.DataFrame(
        {"x": embeddings_3d[:, 0], "y": embeddings_3d[:, 1], "z": embeddings_3d[:, 2], "label": labels})
    df["label"] = df["label"].map(labels_dict)
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="label")
    fig.show()
