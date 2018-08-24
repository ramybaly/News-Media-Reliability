#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import codecs
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')


def CalculatePerformance(y_true, y_pred, task_main):
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    acc = accuracy_score(y_true, y_pred) * 100
    num_classes = 7 if task == 'bias' else 3
    # calculate MAE_m
    err = 0
    num_samples = len(y_true)
    for c in range(num_classes):
        err_c = 0
        num_c = 0
        for i in range(num_samples):
            if y_true[i] == c:
                err_c += abs(y_true[i] - y_pred[i])
                num_c += 1
        err += err_c / num_c
    mae_m = err / num_classes
    # calculate MAE
    err = 0
    for i in range(num_samples):
        err += abs(y_true[i] - y_pred[i])
    mae = err / num_samples
    return [f1, acc, mae, mae_m]


def GetFeaturesAndLabels(corpus, features, task):
    data = pd.read_csv('data/corpus.csv')
    sources = data.source_url_processed
    X = np.empty(data.shape[0]).reshape(-1, 1)
    for file in [f for f in os.listdir('data/features/') if '.npy' in f]:
        if file.replace('.npy', '') in features:
            feats = pd.DataFrame(np.load('data/features/' + file))
            feats = feats[feats.iloc[:, 0].isin(sources)].as_matrix()
            feats = np.delete(feats, 0, axis=1).astype(float)
            X = np.hstack([X, feats[:, :-2]])
    X = np.delete(X, 0, axis=1)
    X = np.hstack([np.asarray(sources).reshape(-1, 1), X])
    X = pd.DataFrame(X)

    labels = {}
    labels['fact'] = {'low': 0, 'mixed': 1, 'high': 2}
    labels['bias'] = {'extreme-right': 0, 'right': 1, 'right-center': 2, 'center': 3, 'left-center': 4, 'left': 5, 'extreme-left': 6}
    data = pd.read_csv('data/corpus.csv')
    if task in labels.keys():
        y = data[task]
        y = [labels[task][L.lower()] for L in y]
    elif task == 'bias3way':
        y = data['bias']
        y = [labels['bias'][L.lower()] for L in y]
        y = [0 if L in [0, 1] else 1 if L in [2, 3, 4] else 2 for L in y]
    y = pd.DataFrame(np.asarray(y).reshape(-1, 1))
    return X, y


def Classification(corpus, features, task):
    X, y = GetFeaturesAndLabels(corpus, features, task)
    with codecs.open('data/splits.json', 'r') as f:
        splits = json.load(f)
    # placeholders to accumulate true and predicted labels
    y_true = []
    y_pred = []
    # start cross-validation
    for i in range(5):
        print('fold ' + str(i))
        # select training instances for current fold
        ids = splits[i]['train-{0}'.format(i)].split('\n')
        Xtr = np.delete(X[X.iloc[:, 0].isin(ids)].values, 0, axis=1).astype(float)
        ytr = np.asarray(y[X.iloc[:, 0].isin(ids)]).reshape(-1, y.shape[1])
        # select testing instances for current fold
        ids = splits[i]['test-{0}'.format(i)].split('\n')
        Xts = np.delete(X[X.iloc[:, 0].isin(ids)].values, 0, axis=1).astype(float)
        yts = np.asarray(y[X.iloc[:, 0].isin(ids)]).reshape(-1, y.shape[1])
        # min-max normalization of the data
        scaler = MinMaxScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xts = scaler.transform(Xts)
        # fine-tune SVM hyperparameters using the training data of current fold
        CLF = GridSearchCV(SVC(), cv=5, scoring='f1_macro', param_grid=[{'kernel': ['rbf'], 'gamma': np.logspace(-6, 1, 8), 'C': np.logspace(-2, 2, 5)}])
        CLF.fit(Xtr, ytr.reshape(-1,))
        # build final model for current fold using best parameters
        CLF_fold = SVC(kernel=CLF.best_estimator_.kernel, gamma=CLF.best_estimator_.gamma, C=CLF.best_estimator_.C, probability=True)
        CLF_fold.fit(Xtr, ytr.reshape(-1,))
        # preform prediction on test data
        yhat = CLF_fold.predict(Xts)
        y_true.extend(yts)
        y_pred.extend(yhat.reshape(-1, 1))
    # transform actual and predicted labels from numpy arrays into lists
    y_true = [yt[0] for yt in y_true]
    y_pred = [yp[0] for yp in y_pred]
    return CalculatePerformance(y_true, y_pred, task)



def parse_params():
    """
    Summary of the different tasks:
    -------------------------------
    fact:       {low, mixed, high}
    bias:       {extreme-right, right, center-right, center, center-left, left, extreme-left}
    bias3way:   {{extreme-right, right}, {center-right, center, center-left}, {left, extreme-left}}
    ===============================================================================================
    Summary of features from the different sources:
    -----------------------------------------------
    traffic:    alexa
    url:        handcrafted_url
    twitter:    has_twitter, created_at, verified, location, url_match, counts, description
    wikipedia:  has_wiki, wikicontent, wikisummary, wikitoc, wikicategories
    articles:   body, title
    =======================================================================================
    """
    parser = argparse.ArgumentParser(description='Source Reliability')
    parser.add_argument('--corpus',             type=str, default='MBFC_v2')
    parser.add_argument('--task',               type=str, default='bias')
    parser.add_argument('--features',           type=str, default='body+title') # list of features must be separated by "+" sign
    params = parser.parse_args()
    return params



if __name__ == '__main__':
    user_params = parse_params()
    corpus = user_params.corpus
    task = user_params.task
    features = user_params.features.split('+')

    print('task:'     + task)
    print('features:' + ', '.join(features))
    results = Classification(corpus, features, task)
    print('Results:')
    print('F1\t{}'.format(results[0]))
    print('Acc.\t{}'.format(results[1]))
    print('MAE\t{}'.format(results[2]))
    print('MAE_u\t{}'.format(results[3]))
