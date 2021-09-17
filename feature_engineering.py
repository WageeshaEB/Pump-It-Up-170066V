import pandas as pd
from sklearn.decomposition import PCA


def pca_feature_creation(x, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(x)
    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=x.columns,
    )
    return pca, X_pca, loadings
