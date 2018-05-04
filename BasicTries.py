# -*- coding: utf-8 -*-
"""
Spyder Editor
author: Sakshi
This is a script file for all base classifiers.
"""
import pandas as pd
import numpy as np
import random
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
import pickle

# Data preprocessing
def getCleanedData():
    data = pd.read_csv('responses.csv', header = 0)
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    
    print('  converting categorical features to indicator features...')
    categoricalAttr=['Smoking', 'Alcohol', 'Punctuality', 'Lying', 'Internet usage', 'Gender', 'Left - right handed', 'Education', 'Only child', 'Village - town', 'House - block of flats']
    for var in categoricalAttr:
        categoryList='var'+'_'+var
        categoryList = pd.get_dummies(data[var], prefix=var)
        data1=data.join(categoryList)
        data=data1
    categoricalAttr=['Smoking', 'Alcohol', 'Punctuality', 'Lying', 'Internet usage', 'Gender', 'Left - right handed', 'Education', 'Only child', 'Village - town', 'House - block of flats']
    varsList=data.columns.values.tolist()
    finalAttr=[i for i in varsList if i not in categoricalAttr]
    cleanedData=data[finalAttr]
    return cleanedData, finalAttr


#splitting test data
def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    validate = df.loc[perm[train_end:validate_end]]
    test = df.loc[perm[validate_end:]]
    return train, validate, test

#get respective X and y from entire dataset
def getXY(data, finalAttr):
    finalAttrList=data[finalAttr].columns.values.tolist()
    yattr=['Empathy']
    Xattr=[i for i in finalAttrList if i not in yattr]
    X = np.array(data[Xattr])
    y = np.array(data[yattr])
    return X,y

def assignRadnomValue(size):
    predicted = []
    for x in range(size):
      predicted.append(random.randint(1,6))
    return predicted

def saveFile(obj, name ):
    with open( 'saved_pickles/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadFile(name ):
    with open('saved_pickles/'+ name + '.pkl', 'rb') as f:
        return pickle.load(f)


