import matplotlib.pyplot as plt
import shap
from sklearn import ensemble


def shap_values_plot(x_train, y_train):
    model = ensemble.RandomForestClassifier(n_estimators=700, min_samples_split=10, criterion='gini', verbose=1)
    model.fit(x_train, y_train.values.ravel())

    shap_values = shap.TreeExplainer(model).shap_values(x_train)
    shap.summary_plot(shap_values, x_train, plot_type="bar")
    plt.show()

    shap.summary_plot(shap_values, x_train)
    plt.show()


