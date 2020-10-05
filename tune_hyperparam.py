# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:53:36 2020

@author: Eszter
"""


#Optimális paraméterbeállítás keresése


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Adatok betöltése Excelből
data = pd.read_excel (r'D:\BME\SZAKDOLGOZAT\470.xlsx')

# Táblázat ellenőrzése
# print (data)

# A minta nevének és a nem kivágása a táblázatból, mert ezekre nem lesz szükség
data = data.drop(["Minta ", "Nem"], axis=1)


# A diagnózis kivágása a táblázatból, hogy aztán a targetbe tegyük
data2 = data.drop("Csoport", axis=1)


# A diagnózist, hogy az adott minta a HC vagy a PA csoportba tartozik-e, az y tartalmazza
y = data["Csoport"] #Diagnózis
# print (y)

# A jellemzőket és a hozzájuk tartozó értékeket mintánként az X táblázat tartalmazza
X = pd.DataFrame(data2) 
# print (X)


# Hiányzó elemek keresése
# Elvileg 2 van, ezeket átlaggal helyettesítem
X.isnull().sum()
column_means = X.mean()
X = X.fillna(column_means)



#Keresett paraméterek
model_params = {
    'n_estimators': [50, 60, 70, 75, 80, 85, 90, 100, 150, 200, 250],
	'max_features' : [10, 20, 30, 40],
}

#Random Forest Classifier modell létrehozása
rf_model = RandomForestClassifier(random_state=1)

#grid search meta-estimator
clf = GridSearchCV(rf_model, model_params, cv=5)

# train the grid search meta-estimator a legjobb paraméterbeállítások megtalálásához
model = clf.fit(X, y)

#A nyertes beállítások kiíratása
from pprint import pprint
pprint(model.best_estimator_.get_params())