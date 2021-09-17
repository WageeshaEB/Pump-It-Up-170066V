from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector


def SequentialForwardFeatureSelection(x, y, n_features=10):
    print("\n[Sequential Forward Feature Selection]")
    sfs_selector = SequentialFeatureSelector(estimator=RandomForestClassifier(n_estimators=100),
                                             cv=5,
                                             direction='forward',
                                             n_jobs=-1)
    sfs_selector.fit(x, y.values.ravel())
    features = x.columns[sfs_selector.get_support()]
    print(features.tolist())
    return features.tolist()
