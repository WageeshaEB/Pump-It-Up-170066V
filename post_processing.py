import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.inspection import plot_partial_dependence

import classifiers

rf_clf = classifiers.rf_clf
gb_clf = classifiers.gb_clf
xg_clf = classifiers.xg_clf


def heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, vmax=.3, square=True, cmap="YlGnBu", linewidth=0.3, annot=True, fmt=".2f", ax=ax)
    plt.show()


def histogram(df, column):
    plt.hist(df[column], color="skyblue")
    plt.show()


def stacked_bar_plot(x, y, y_col, column):
    print("[Stacked bar plot]", column)
    df = x
    df[y_col] = y[y_col]

    # top bar - (func need repair)
    total = df.groupby(column)[y_col].size().reset_index()
    bar1 = sns.barplot(x=column, y=y_col, data=total, color='darkblue')

    # middle bar - (non functional)
    non_functional = df[(df[y_col] == 'non functional') | (df[y_col] == 'functional')].groupby(column)[
        y_col].size().reset_index()
    bar2 = sns.barplot(x=column, y=y_col, data=non_functional, color='red')

    # bottom bar - (functional)
    functional = df[df[y_col] == 'functional'].groupby(column)[y_col].size().reset_index()
    bar3 = sns.barplot(x=column, y=y_col, data=functional, color='lightblue')

    # functional_repair = df[df[y_col] == 'functional needs repair'].groupby(column)[y_col].size().reset_index()

    # df_plot = functional
    # df_plot = df_plot.rename(columns={y_col: 'func'}, inplace=False)
    # df_plot['non-func'] = non_functional[y_col]
    # df_plot['func-need-repair'] = functional_repair[y_col]
    # print("\n[Plot]\n", df_plot)
    #
    # s1 = sns.barplot(x=column, y='func', data=df_plot, color='red')
    # s2 = sns.barplot(x=column, y='non-func', data=df_plot, color='blue')
    # s3 = sns.barplot(x=column, y='func-need-repair', data=df_plot, color='green')

    # plt.gcf().subplots_adjust(bottom=0.2)

    top_bar = mpatches.Patch(color='darkblue', label='functional needs repair')
    middle_bar = mpatches.Patch(color='red', label='non functional')
    bottom_bar = mpatches.Patch(color='lightblue', label='functional')
    plt.legend(handles=[top_bar, middle_bar, bottom_bar])

    plt.xticks(rotation=45)
    plt.xlabel(column)
    plt.title("Bar Plot - " + column)
    plt.show()


def percentage_stack_bar_plot(x, y, y_col, column):
    print("[percentage stacked bar plot]", column)
    df = x.copy()
    df[y_col] = y[y_col]
    # total = df.groupby(column)[y_col].count().reset_index()
    # print("\ntotal\n", total)

    # print("\nfunctional")
    functional = df[df[y_col] == 'functional'].groupby(column)[y_col].size().reset_index()
    # print(functional)
    # print("functional- len = ", len(functional.index))

    # print("\nnon-functional")
    non_functional = df[df[y_col] == 'non functional'].groupby(column)[y_col].size().reset_index()
    # print(non_functional)
    # print("non functional- len = ", len(non_functional.index))

    # print("\nfunctional needs repair")
    functional_repair = df[df[y_col] == 'functional needs repair'].groupby(column)[y_col].size().reset_index()
    # print(functional_repair)
    # print("functional needs repair- len = ", len(functional_repair.index))

    df_plot = functional
    df_plot = df_plot.rename(columns={y_col: 'func'}, inplace=False)
    df_plot['non-func'] = non_functional[y_col]
    df_plot['func-need-repair'] = functional_repair[y_col]
    # df_plot['total'] = total[y_col]

    # print(df_plot)
    df_plot = df_plot.replace(np.nan, 0)

    cols = ['func', 'non-func', 'func-need-repair']
    for index, row in df_plot.iterrows():
        df_plot.loc[index, cols[0]] = row[cols[0]] / (row[cols[0]] + row[cols[1]] + row[cols[2]]) * 100
        df_plot.loc[index, cols[1]] = row[cols[1]] / (row[cols[0]] + row[cols[1]] + row[cols[2]]) * 100
        df_plot.loc[index, cols[2]] = row[cols[2]] / (row[cols[0]] + row[cols[1]] + row[cols[2]]) * 100

    # print("\n", df_plot)
    # df_plot.drop('total', axis=1, inplace=True)

    df_plot.plot(x=column,
                 kind='barh',
                 stacked=True,
                 title='Percentage Plot - ' + column,
                 mark_right=True,
                 color=sns.color_palette("Set2"))
    plt.show()


def plot_all_features(x, y, x_cols, y_col):
    for col in x_cols:
        stacked_bar_plot(x=x, y=y, y_col=y_col, column=col)
        percentage_stack_bar_plot(x=x, y=y, y_col=y_col, column=col)


def statistical_analysis_pearson(df):
    df = df.copy()
    df = df.dropna()
    df = df.drop('id', axis=1)
    results = df.corr(method='pearson')
    heatmap(results)


# feature importance

def plot_rf_feature_importance(df):
    print("\n[PLOT - RF Feature Importance]")
    plt.figure(figsize=(10, 6))
    df.drop(df[df['Feature'] == 'id'].index, inplace=True)
    df_sorted = df.sort_values('Importance', ascending=False)
    # print(df_sorted)
    plt.bar('Feature',
            'Importance',
            data=df_sorted,
            color=sns.color_palette("crest", len(df.index)))
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Score')
    plt.show()


def plot_ensemble_feature_importance(df):
    print("\n[PLOT - Ensemble-method Feature Importance]")
    df.drop(df[df['Feature'] == 'id'].index, inplace=True)
    df_sorted = df.sort_values('RF-Importance', ascending=False)
    # print(df_sorted)
    df_sorted.plot(x='Feature', y=['RF-Importance', 'GB-Importance', 'XGB-Importance'], kind='bar', figsize=(20, 10))

    plt.gcf().subplots_adjust(bottom=0.4)
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Score')
    plt.show()


def get_rf_feature_importance(x_cols):
    print("\n[Feature Importance]")
    return pd.DataFrame({"Feature": x_cols,
                         "Importance": rf_clf.feature_importances_})


def get_ensemble_clf_feature_importance(x_cols):
    print("\n[Feature Importance]")
    df = pd.DataFrame({"Feature": x_cols,
                       "RF-Importance": rf_clf.feature_importances_,
                       "GB-Importance": gb_clf.feature_importances_,
                       "XGB-Importance": xg_clf.feature_importances_
                       })
    return df


def get_top_n_features(fi_df, n):
    df_sorted = fi_df.sort_values('Importance', ascending=False)
    print(df_sorted)
    n_df = df_sorted.head(n)
    print(n_df)
    return n_df['Feature'].tolist()


# Partial dependence plots


def partial_dependency_plots(x):
    print("\n[Partial dependence plots]")
    x_cols = list(x.columns.values)
    x_cols.remove('id')
    for i in range(len(x_cols)):
        print(x_cols[i])
        plot_partial_dependence(rf_clf,
                                features=[i],
                                X=x,
                                feature_names=x_cols,
                                grid_resolution=10,
                                target=0,
                                percentiles=(0, 1))

        plt.show()
