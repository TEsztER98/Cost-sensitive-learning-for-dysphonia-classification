# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:38:06 2020

@author: Eszter
"""
#Cost Sensitive Classification
#mintakóddal, minta adatokkal

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.models import CostSensitiveRandomForestClassifier
from costcla.metrics import savings_score
data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
f = CostSensitiveRandomForestClassifier()
y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
# Savings using only RandomForest
print(savings_score(y_test, y_pred_test_rf, cost_mat_test))

# Savings using CostSensitiveRandomForestClassifier
print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
