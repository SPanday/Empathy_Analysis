# -*- coding: utf-8 -*-
"""
Created on Wed May  2 23:57:17 2018

@author: sakshi
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso, RandomizedLasso
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import GridSearchCV


def getBestParam(base_model,param_grid, X, y):
    CV_rfc = GridSearchCV(estimator=base_model, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X, y.reshape(606,))
    return CV_rfc.best_params_
    

def tuneSVC(X, y):
    param_grid = {
    'kernel': [ 'linear', 'poly', 'rbf', 'sigmoid' ],
    'C':list(range(1, 4, 1))
    }
    bestParam = getBestParam(SVC(),param_grid, X, y)
    return bestParam


def tuneRFC(X, y):
    param_grid = {
    'n_estimators': list(range(648, 658, 1)),
    'max_features': ['auto', 'sqrt', 'log2']
    }
    bestParam = getBestParam(RandomForestClassifier(n_jobs=-1, oob_score = True),param_grid, X, y)
    return bestParam

def tuneDCT(X, y):
    param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth':list(range(1, 30, 1))
    }
    bestParam = getBestParam(tree.DecisionTreeClassifier(),param_grid, X, y)
    return bestParam

    