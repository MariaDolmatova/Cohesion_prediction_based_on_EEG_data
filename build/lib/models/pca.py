import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..utils.logger import get_logger

def pca(csv_file, n_features=15):

    logger = get_logger()
    data = pd.read_csv(csv_file)
    data.drop(columns=["Pair"], inplace=True, errors="ignore")
    data.dropna(inplace=True)

    data_scaled = StandardScaler().fit_transform(data)

    pca = PCA(2)  # we only keep 2 components
    X_pca = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_
    logger.info(f"Explained Variance Ratio (PC1 and PC2) = {explained_variance}")
    logger.info(f"Cumulative Explained Variance (PC1 and PC2) = {np.sum(explained_variance)}")

    components = pca.components_
    features = data.columns

    for i in range(len(components)):
        component = components[i]
        top_indices = np.argsort(np.abs(component))[-n_features:]  # Top features by absolute weight
        top_features = features[top_indices]
        top_weights = component[top_indices]

        # Create a horizontal bar chart using Plotly
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=top_weights,
                y=top_features,
                orientation="h",
                marker=dict(color="yellowgreen", opacity=0.8),
                name=f"PC{i + 1}",
            )
        )

        # Customize the layout
        fig.update_layout(
            title=f"Top {n_features} Features in PC{i + 1}",
            xaxis_title="Weight",
            yaxis_title="Features",
            yaxis=dict(autorange="reversed"),  # Reverse the y-axis for barh-like behavior
            template="plotly_white",
            height=500,
        )

        fig.show()
