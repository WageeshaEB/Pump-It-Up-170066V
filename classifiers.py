import numpy as np
import pandas as pd
import xgboost as xgb
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn import ensemble
from sklearn.svm import SVC

rf_clf = ensemble.RandomForestClassifier(n_estimators=700, min_samples_split=10, criterion='gini')

gb_clf = ensemble.GradientBoostingClassifier(n_estimators=700)

xg_clf = xgb.XGBClassifier(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                           alpha=10, n_estimators=1000, use_label_encoder=False, verbosity=0)


def RandomForest(x_train, y_train, x_test):
    print("\n[Random forest]")
    rf_clf.fit(x_train, y_train.values.ravel())
    return rf_clf.predict(x_test)


def GradientBoost(x_train, y_train, x_test):
    print("\n[Gradient Boosting]")
    gb_clf.fit(x_train, y_train.values.ravel())
    return gb_clf.predict(x_test)


def XGBoost(x_train, y_train, x_test):
    print("\n[XGBoost]")
    xg_clf.fit(x_train, y_train.values.ravel())
    return xg_clf.predict(x_test)


def AdaBoost(x_train, y_train, x_test):
    print("\n[AdaBoost]")
    ab_clf = ensemble.AdaBoostClassifier(n_estimators=700, learning_rate=1)
    ab_clf.fit(x_train, y_train.values.ravel())
    return ab_clf.predict(x_test)


def NaiveBayes(x_train, y_train, x_test):
    print("\n[Naive Bayes]")
    nb_clf = ensemble.GaussianNB()
    nb_clf.fit(x_train, y_train.values.ravel())
    return nb_clf.predict(x_test)


def SVM(x_train, y_train, x_test):
    print("\n[SVM]")
    sv_clf = SVC(probability=True)
    sv_clf.fit(x_train, y_train.values.ravel())
    return sv_clf.predict(x_test)


def NN(x_train, y_train, x_test):
    print("\nNN")
    model = Sequential()
    # add layers
    model.add(Dense(128, activation='relu', input_shape=(x_train.shape)))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    # compiling
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # training
    model.fit(x_train, y_train, epochs=10)
    return model.predict_classes(x_test, verbose=0)[:, 0]


def ensemble_model(x_train, y_train, x_test):
    results = pd.DataFrame()
    results['rf'] = RandomForest(x_train, y_train, x_test)
    results['gb'] = GradientBoost(x_train, y_train, x_test)
    results['xg'] = XGBoost(x_train, y_train, x_test)
    # print(results)

    pred_list = []
    for index, row in results.iterrows():
        rf, gb, xg = row['rf'], row['gb'], row['xg']

        if rf == gb:
            pred_list.append(rf)
        elif rf == xg:
            pred_list.append(rf)
        elif gb == xg:
            pred_list.append(gb)
        else:
            pred_list.append(rf)

    pred_np_arr = np.array(pred_list)
    return pred_np_arr
