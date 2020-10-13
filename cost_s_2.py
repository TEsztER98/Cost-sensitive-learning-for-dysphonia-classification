# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:11:48 2020

@author: Eszter
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.models import BayesMinimumRiskClassifier
from costcla.metrics import savings_score
import pandas as pd
import numpy as np

#Cost Sensitive Classification 
#Saját adatokkal

#Adataok betöltése

data1 = pd.read_excel (r'D:\BME\SZAKDOLGOZAT\Selected_Features.xlsx')
data2 = pd.DataFrame(data1.drop("Csoport", axis =1))
data_target = data1["Csoport"]



#Number of features
features = pd.DataFrame(data2.columns)
feature_number = len(features)
print(feature_number)

# cost_mat[C_FP,C_FN,C_TP,C_TN]
row = np.array([0.6, 0.4, 0, 0])
cost_mat = np.array([0.6, 0.4, 0, 0])
i = 0

while i < 469:
	cost_mat = np.vstack ((cost_mat, row) )
	i += 1

print (cost_mat)

#Osztályozás
from costcla.models import CostSensitiveRandomForestClassifier

X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = train_test_split(data2, data_target, cost_mat)
y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
f =  CostSensitiveRandomForestClassifier()
y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)


print(savings_score(y_test, y_pred_test_rf, cost_mat_test))

print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))