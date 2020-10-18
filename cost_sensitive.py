

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:15:12 2020

@author: Eszter
"""

import pandas as pd
import numpy as np


#Adataok betöltése

data1 = pd.read_excel (r'D:\BME\SZAKDOLGOZAT\Selected_Features.xlsx')
data2 = pd.DataFrame(data1.drop("Csoport", axis =1))
data_target = data1["Csoport"]

# Class label
x = pd.Series(data_target).value_counts()
target = pd.DataFrame()
target['Frequency'] = x
target['Percentage'] = target['Frequency'] / target['Frequency'].sum()
print (target)

#Number of features
features = pd.DataFrame(data2.columns)
feature_number = len(features)
print(feature_number)


# cost_mat[C_FP,C_FN,C_TP,C_TN]
row = np.array([0.3, 0.7, 0, 0])
cost_mat = np.array([0.3, 0.7, 0, 0])
i = 0

while i < 469:
	cost_mat = np.vstack ((cost_mat, row) )
	i += 1

#print (cost_mat)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = \
train_test_split(data2, data_target, cost_mat, test_size=0.3)

#Változók konvertálása a költség-érzékeny modellnek megfelelhez szükséges formába
X_train = X_train.to_numpy(dtype='float')
X_test = X_test.to_numpy(dtype='float')
y_train = y_train.to_numpy(dtype='float')
y_test = y_test.to_numpy(dtype='float')

#Nem költség-érzékeny ossztályozók importálása
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

classifiers = {"RF": {"f": RandomForestClassifier()},
               "DT": {"f": DecisionTreeClassifier()},
               "LR": {"f": LogisticRegression()},
               }


#Az osztályozó modellek illesztése a tanító halmaz felszhasználásával
for model in classifiers.keys():
    classifiers[model]["f"].fit(X_train, y_train)
    classifiers[model]["c"] = classifiers[model]["f"].predict(X_test)
    classifiers[model]["p"] = classifiers[model]["f"].predict_proba(X_test)
    classifiers[model]["p_train"] = classifiers[model]["f"].predict_proba(X_train)

#Pontosság és egyéb értékek kiíratása
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
measures = {"f1": f1_score, "pre": precision_score, 
            "rec": recall_score, "acc": accuracy_score}
results = pd.DataFrame(columns=measures.keys())

for model in classifiers.keys():
    results.loc[model] = [measures[measure](y_test, classifiers[model]["c"]) for measure in measures.keys()]

from costcla.metrics import savings_score, cost_loss 

# Evaluate the savings for each model
results["sav"] = np.zeros(results.shape[0])
for model in classifiers.keys():
    results["sav"].loc[model] = savings_score(y_test, classifiers[model]["c"], cost_mat_test)

#print (results)


# Eredmények ábrázolása
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import seaborn as sns

figsize(10, 5)
ax = plt.subplot(111)

ind = np.arange(results.shape[0])
width = 0.2
l = ax.plot(ind, results, "-o")
plt.legend(iter(l), results.columns.tolist(), loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.25, ind[-1]+.25])
ax.set_xticks(ind)
ax.set_xticklabels(results.index)
plt.show()


#Költség-érzékeny modellek használata

#Bayes Minimum Risk
from costcla.models import BayesMinimumRiskClassifier
ci_models = classifiers.keys()


for model in list(ci_models):
    classifiers[model+"-BMR"] = {"f": BayesMinimumRiskClassifier()}
    # Fit
    classifiers[model+"-BMR"]["f"].fit(y_test, classifiers[model]["p"])  
    # Calibration must be made in a validation set
    # Predict
    classifiers[model+"-BMR"]["c"] = classifiers[model+"-BMR"]["f"].predict(classifiers[model]["p"], cost_mat_test)
    # Evaluate
    results.loc[model+"-BMR"] = 0
    results.loc[model+"-BMR", measures.keys()] = \
    [measures[measure](y_test, classifiers[model+"-BMR"]["c"]) for measure in measures.keys()]
    results["sav"].loc[model+"-BMR"] = savings_score(y_test, classifiers[model+"-BMR"]["c"], cost_mat_test)
    


#Random Patches Classifier
from costcla.models import CostSensitiveRandomPatchesClassifier

classifiers["CSRP"] = {"f": CostSensitiveRandomPatchesClassifier(combination ='majority_voting')}
# Fit
classifiers["CSRP"]["f"].fit(X_train, y_train, cost_mat_train)
# Predict
classifiers["CSRP"]["c"] = classifiers["CSRP"]["f"].predict(X_test)
# Evaluate
results.loc["CSRP"] = 0
results.loc["CSRP", measures.keys()] = \
[measures[measure](y_test, classifiers["CSRP"]["c"]) for measure in measures.keys()]
results["sav"].loc["CSRP"] = savings_score(y_test, classifiers["CSRP"]["c"], cost_mat_test)
    



# Desision Tree Classifier
from costcla.models import CostSensitiveDecisionTreeClassifier

classifiers["CSDT"] = {"f": CostSensitiveDecisionTreeClassifier(criterion = 'entropy_cost', pruned = True)}
# Fit
classifiers["CSDT"]["f"].fit(X_train, y_train, cost_mat_train)
# Predict
classifiers["CSDT"]["c"] = classifiers["CSDT"]["f"].predict(X_test)
# Evaluate
results.loc["CSDT"] = 0
results.loc["CSDT", measures.keys()] = \
[measures[measure](y_test, classifiers["CSDT"]["c"]) for measure in measures.keys()]
results["sav"].loc["CSDT"] = savings_score(y_test, classifiers["CSDT"]["c"], cost_mat_test)




from costcla.models import CostSensitiveRandomForestClassifier

classifiers["CSRFC"] = {"f": CostSensitiveRandomForestClassifier()}
# Fit
classifiers["CSRFC"]["f"].fit(X_train, y_train, cost_mat_train)
# Predict
classifiers["CSRFC"]["c"] = classifiers["CSRFC"]["f"].predict(X_test)
# Evaluate
results.loc["CSRFC"] = 0
results.loc["CSRFC", measures.keys()] = \
[measures[measure](y_test, classifiers["CSRFC"]["c"]) for measure in measures.keys()]
results["sav"].loc["CSRFC"] = savings_score(y_test, classifiers["CSRFC"]["c"], cost_mat_test)



#Bagging Classifier
from costcla.models import CostSensitiveBaggingClassifier


classifiers["CSBC"] = {"f": CostSensitiveBaggingClassifier()}
# Fit
classifiers["CSBC"]["f"].fit(X_train, y_train, cost_mat_train)
# Predict
classifiers["CSBC"]["c"] = classifiers["CSBC"]["f"].predict(X_test)
# Evaluate
results.loc["CSBC"] = 0
results.loc["CSBC", measures.keys()] = \
[measures[measure](y_test, classifiers["CSBC"]["c"]) for measure in measures.keys()]
results["sav"].loc["CSBC"] = savings_score(y_test, classifiers["CSBC"]["c"], cost_mat_test)


#Pasting Classifier
from costcla.models import CostSensitivePastingClassifier

classifiers["CSPC"] = {"f": CostSensitivePastingClassifier()}
# Fit
classifiers["CSPC"]["f"].fit(X_train, y_train, cost_mat_train)
# Predict
classifiers["CSPC"]["c"] = classifiers["CSPC"]["f"].predict(X_test)
# Evaluate
results.loc["CSPC"] = 0
results.loc["CSPC", measures.keys()] = \
[measures[measure](y_test, classifiers["CSPC"]["c"]) for measure in measures.keys()]
results["sav"].loc["CSPC"] = savings_score(y_test, classifiers["CSPC"]["c"], cost_mat_test)

print (results)


