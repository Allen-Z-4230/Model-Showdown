import numpy as np
import scipy as sp
from scipy.stats import rankdata
import pandas as pd
import multiprocessing
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

base_dir = './../Data'


def get_files(base_dir):
    files = []
    for r, d, f in os.walk(base_dir):
        for file in f:
            if file.endswith('.csv'):
                files.append(os.path.join(r, file))
    return files


def get_dataframes(f_list):
    dfs = []
    for file in f_list:
        try:
            dfs.append(pd.read_csv(file))
        except:
            dfs.append(pd.read_csv(file, encoding='latin-1'))
    return dfs


def proc_label_1(text):
    if text.startswith('I'):
        return 0
    else:
        return 1


def proc_label_2(text):
    if text == 'ham':
        return 0
    else:
        return 1


class Vectorizer():  # sklearn-style wrapper class for gensim word2vec
    def __init__(self, size):
        self.model = None
        self.len = size

    def fit(self, X):
        print("Fitting Word2Vec Model")
        setattr(self, 'model', Word2Vec(X, min_count=1, size=self.len,
                                        workers=multiprocessing.cpu_count(), sg=1))

    def transform(self, X):
        return np.array([
            np.mean([self.model[w] for w in words if w in self.model.wv]
                    or [np.zeros(self.size)], axis=0) for words in X])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def data_sampler(data, size=200):  # subsamples data for faster code testing
    rand_ints = np.random.randint(low=0, high=data.shape[0], size=size)
    return data.iloc[rand_ints]


def draw_heatmap(errors, param_grid, title):  # slightly modified from hw2
    plt.figure(figsize=(2, 4))
    ax = sns.heatmap(errors, annot=True, fmt='.3f',
                     yticklabels=list(param_grid.values())[0], xticklabels=[])
    ax.collections[0].colorbar.set_label('error')
    ax.set(ylabel=list(param_grid.keys())[0])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(title)
    plt.show()


def cv_model(model_name, model, param_grid, data, split, trial, d_ind, cv=2):  # trains model using a variety of inputs
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=split)

    gs_params = dict(scoring='accuracy', n_jobs=multiprocessing.cpu_count(),
                     cv=cv, verbose=0, refit=True)
    grid_search = GridSearchCV(model, param_grid, **gs_params)
    grid_search.fit(X_train, y=y_train)

    if trial == 2 and d_ind == 0:  # to avoid too many plots, only plot 3 heatmaps per model
        cross_val_errors = np.array(1 - grid_search.cv_results_['mean_test_score']).reshape(-1, 1)
        draw_heatmap(cross_val_errors, param_grid, f'{model_name} Classifier, {split} Split,')

    test_error = 1 - grid_search.best_estimator_.score(X_test, y_test)
    return test_error


def comp_results(result):
    ranks = np.array([rankdata(np.array([v[i] for v in result.values()])) for i in range(3)]).T
    ranks = np.array([[np.sum(1*(ranks[j] == i))/3 for i in range(1, 4)] for j in range(3)])
    return ranks
