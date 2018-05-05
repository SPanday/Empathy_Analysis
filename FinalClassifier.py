# -*- coding: utf-8 -*-
"""
Created on Thu May  3 00:44:27 2018

@author: sakshi
"""
from BasicTries import getCleanedData, train_validate_test_split, getXY, saveFile, loadFile
from ParameterTuning import tuneSVC, tuneDCT, tuneRFC
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso, RandomizedLasso
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle

print('##starting final classifier...')
print('\n  loading dataset...')
cleanedData, finalAttr = getCleanedData()
print('  split data into train(60%), validation(20%) and test(20%) sets...')
#splitting cleaned data into train, validatation and test sets 
train, validate, test = train_validate_test_split(cleanedData)
X, y = getXY(cleanedData, finalAttr)
print('  selecting relevant features...')

# feature selection
from sklearn.feature_selection import RFECV
logreg = LogisticRegression()
rfe = RFECV(logreg, 10)
rfe = rfe.fit(X, y )
ranking = rfe.ranking_
selectedAttr = []
for i, rank in enumerate(ranking):
    if rank < 3 or i == 92:
        selectedAttr.append(finalAttr[i])
X_train, y_train = getXY(train, selectedAttr)
X_val, y_val = getXY(validate, selectedAttr)
X_test, y_test = getXY(test, selectedAttr)

print('  tuning hyperparameters for base classifiers...')
"""
Comment out below chunk for parameter tuning
"""
#bestSVC = tuneSVC(X_train,y_train)
#bestRFC = tuneRFC(X_train,y_train)
#bestDCT = tuneDCT(X_train,y_train)
#saveFile(bestSVC, 'bestSVC' )
#saveFile(bestRFC, 'bestRFC' )
#saveFile(bestDCT, 'bestDCT' )
    
bestSVC = loadFile('bestSVC' )
bestRFC = loadFile('bestRFC' )
bestDCT = loadFile('bestDCT' )
print('  preparing the voting classifier...')

"""
Comment out below chunk for reinitialize the model
"""
#
#clf1 = LogisticRegression(random_state=1)
#
#rfc = RandomForestClassifier(n_jobs=-1, max_features= bestRFC['max_features'], n_estimators= bestRFC['n_estimators'], oob_score = True) #
#clf2 = BaggingClassifier(base_estimator=rfc)
#
#cart = tree.DecisionTreeClassifier(criterion = bestDCT['criterion'], max_depth=bestDCT['max_depth'], max_features=bestDCT['max_features'])
#clf3 = BaggingClassifier(base_estimator=cart)
#
#svc = SVC(kernel = bestSVC['kernel'], C = bestSVC['C'])
#clf4 = BaggingClassifier(base_estimator=svc)
#kmeans = KNeighborsClassifier(5)
#nn = MLPClassifier(alpha=1)
#
#qda = QuadraticDiscriminantAnalysis()
#
#eclf1 = VotingClassifier(estimators=[
#      ('lr', clf1),   ('rf', clf2), ('cart',clf3),('svm',clf4), ('nn', nn), ('qda',qda), ('km', kmeans)], voting='hard') #
#    
#import pickle
#filename = 'ensemble.sav'
#pickle.dump(eclf1, open(filename, 'wb'))

eclf1 = pickle.load(open('ensemble.sav', 'rb'))
validation = []
testing = []

"""
change the value in range to run multiple iterations for average accuracy
"""
for i in range(1):   
    eclf1 = eclf1.fit(X_train, y_train)
    validation.append(np.mean(y_val == eclf1.predict(X_val)))
    validationPred = eclf1.predict(X_val)

    eclf1 = eclf1.fit(np.concatenate((X_train,X_val)),np.concatenate((y_train,y_val)))
    testing.append(np.mean(y_test == eclf1.predict(X_test)))

print('\nFinal accuracy on validation set:',np.mean(validation))
print('Final accuracy on testing set:',np.mean(testing))

print('\n...ending##')
      