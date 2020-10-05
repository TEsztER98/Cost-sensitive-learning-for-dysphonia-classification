# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:55:37 2020

@author: Eszter
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score as acc

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

# Az adatok felosztása tanuló és tesztelő halmazra
# Az adatok 20%-a kerül a tesztelő halmazba, 80% pedig a tanuló halmazba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape


# Forward Feature Selection modell építése
	#n_estimators: az erdőben lévő fák száma
	#random_state: Random választásokat konrollálja, mivel 0, ezért mindig ugyanaz az eredmény jön ki
	#n_jobs: megmondja a számítógépnek, hogy mennyi processzort használhat (-1: nincs korlátozás)
	#verbose: A fa építésről tudunk meg több információt, ezzel a beállítással

sfs = SFS(RandomForestClassifier(n_estimators=60, random_state=0, n_jobs = -1),
         k_features = (1, 49),
          forward= True,
          floating = False,
          verbose= 2, #a faépítésről több infót tudunk
          scoring= 'accuracy',
          cv = 3,
          n_jobs= -1
         ).fit(X_train, y_train)

# A kiválasztott jellemzők nevének megadása
Feature_Names = sfs.k_feature_names_

# A kiválasztott jellemzőket, a hozzájuk tartozó mintánkénti értékekkel együtt selected DataFramebe pakolom
selected = pd.DataFrame()
selected = pd.concat([y, (X.loc[:, Feature_Names]).copy()], axis = 1, sort = False)

#Kiválasztott jellemzők száma
feature_number = len(selected.columns)
print (feature_number)

# Az elért pontosság
accuracy = sfs.k_score_
print ('Accuracy on selected features: %f' % accuracy)

# Modell építés a meghatározott jellemzőkkel
clf = RandomForestClassifier(n_estimators=1000, random_state=0, max_depth=4)
clf.fit(X_train.loc[:, Feature_Names], y_train)

# A tanulás pontossága a kiválasztott jellemzőkkel
y_train_pred = clf.predict(X_train.loc[:, Feature_Names])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

# A tesztelés pontossága a kiválasztott jellemzőkkel
y_test_pred = clf.predict(X_test.loc[:, Feature_Names])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))

# A kiválasztottjellemzőket oszlopostól kimásolom egy Excel fájlba, hogy a Cost Sensitive Classificationnál felhasználhassam
selected.to_excel(r'D:\BME\SZAKDOLGOZAT\Selected_Features.xlsx', sheet_name='Selected', index = False)



