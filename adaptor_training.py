#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

feat_dir = 'features/'
fig_dir = 'figures/'

def load_data(fdata):
    df = pd.read_csv(os.path.join(feat_dir, fdata), header=None)
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    
    return X, y

X_trn_AAC, y_trn_AAC = load_data('AP.AAC.csv')
X_trn_CKSAAGP, y_trn_CKSAAGP = load_data('AP.CKSAAGP.csv')
X_trn_CKSAAP, y_trn_CKSAAP = load_data('AP.CKSAAP.csv')
X_trn_CTriad, y_trn_CTriad = load_data('AP.CTriad.csv')
X_trn_DDE, y_trn_DDE = load_data('AP.DDE.csv')
X_trn_DPC, y_trn_DPC = load_data('AP.DPC.csv')
# X_trn_EAAC, y_trn_EAAC = load_data('AP.EAAC.csv')
# X_trn_EGAAC, y_trn_EGAAC = load_data('AP.EGAAC.csv')
X_trn_GAAC, y_trn_GAAC = load_data('AP.GAAC.csv')
X_trn_GDPC, y_trn_GDPC = load_data('AP.GDPC.csv')
X_trn_KSCTriad, y_trn_AAC = load_data('AP.KSCTriad.csv')

y_trn = y_trn_AAC

feat_dict = {'AAC':X_trn_AAC, 'CKSAAP':X_trn_CKSAAP, 'DPC':X_trn_DPC, 'DDE':X_trn_DDE, 'GAAC':X_trn_GAAC,
             'CKSAAGP':X_trn_CKSAAGP, 'GDPC':X_trn_GDPC, 'CTriad':X_trn_CTriad, 'KSCTriad':X_trn_KSCTriad}

import scipy
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = []
names = []
for key, value in feat_dict.items():
    cv_results = cross_val_score(MLPClassifier(), value, y_trn, cv=kfold)
    results.append(cv_results)
    names.append(key)
    print('{}: {} (std={})'.format(key, np.mean(cv_results), np.std(cv_results)))

kfold = StratifiedKFold(n_splits=10, shuffle=True)
cm = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
for train, test in kfold.split(X_trn_CKSAAP, y_trn):
    svm_model = MLPClassifier() 
    ## evaluate the model
    svm_model.fit(X_trn_CKSAAP.iloc[train], y_trn[train])
    # evaluate the model
    true_labels = np.asarray(y_trn[test])
    predictions = svm_model.predict(X_trn_CKSAAP.iloc[test])
#     print(confusion_matrix(true_labels, predictions))
    cm = np.add(cm, confusion_matrix(true_labels, predictions))
print(cm)

# ## SMOTE

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
ros = RandomOverSampler()


# In[170]:


cm = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
for train, test in kfold.split(X_trn_CKSAAP, y_trn):
    X_ros, y_ros = ros.fit_resample(X_trn_CKSAAP.iloc[train], y_trn[train])
    clf = MLPClassifier() 
    ## evaluate the model
    clf.fit(X_ros, y_ros)
    # evaluate the model
    true_labels = np.asarray(y_trn[test])
    predictions = clf.predict(X_trn_CKSAAP.iloc[test])
    cm = np.add(cm, confusion_matrix(true_labels, predictions))
print(cm)

# ### Calculate metrics

cnf_matrix = cm
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)


# In[174]:


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

# MCC
MCC = ((TN*TP)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))


# In[175]:


df_result = pd.DataFrame(columns=['Method', 'TPR', 'TNR', 'PPV', 'NPV', 'ACC', 'MCC'])
df_result['Method'] = ['C1', 'C2', 'C3', 'C4', 'C5']
df_result['TPR'] = TPR
df_result['TNR'] = TNR
df_result['PPV'] = PPV
df_result['NPV'] = NPV
df_result['ACC'] = ACC
df_result['MCC'] = MCC


# In[176]:


df_result.to_csv('results/MLP.OverSampling.cv.metrics.csv', index=None)


