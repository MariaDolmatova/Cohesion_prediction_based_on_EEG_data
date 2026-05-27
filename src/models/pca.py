import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..utils.logger import get_logger

def pca(csv_file, n_features=15):
    "Performs principle component analysis"
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

    return components, features
