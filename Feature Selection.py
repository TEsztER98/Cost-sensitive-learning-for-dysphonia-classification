# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:21:37 2020

@author: Eszter
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:55:37 2020

@author: Eszter
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import accuracy_score as acc

#Adatbázis formázása Excelből való betöltés után
def dataProcessing(data):
	# A minta azonosítója és a nem kivágása a táblázatból, mert ezekre nem lesz szükség
	data = data.drop(["Minta ", "Nem"], axis=1)


	# A diagnózis (PA vagy HC) kivágása
	#ezek másik változóba fognak kerülni (y-ba)
	data2 = data.drop("Csoport", axis=1)


	# A diagnózist, hogy az adott minta a HC vagy a PA csoportba tartozik-e, az y tartalmazza
	y = data["Csoport"] #Diagnózis


	# Az ajusztikai paramétereket és a hozzájuk tartozó értékeket mintánként az X tartalmazza
	#Ez esetben az akusztikai paraméterek a jellemzők
	X = pd.DataFrame(data2) 
	
	return X, y


#Hiányzó adatok lekezelése
# A hiányzó elemeket átlaggal helyettesítem
def missingValues(X):
	X.isnull().sum()
	column_means = X.mean()
	X = X.fillna(column_means)
	
	return X


# Az adatok felosztása tanuló és tesztelő halmazra
# Az adatok 20%-a kerül a tesztelő halmazba, 80% pedig a tanuló halmazba
def split(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	X_train.shape, X_test.shape
	
	return X_train, X_test, y_train, y_test


# Forward Feature Selection modell építése
	#n_estimators: az erdőben lévő fák száma
	#random_state: Random választásokat konrollálja, mivel 0, ezért mindig ugyanaz az eredmény jön ki
	#n_jobs: megmondja a számítógépnek, hogy mennyi processzort használhat (-1: nincs korlátozás)
	#verbose: A fa építésről tudunk meg több információt, ezzel a beállítással

#Jellemző kiválasztás: Step Forward Feature Selection
def SFFS(X_train, y_train, C_GS, gamma_GS, kernel_GS):
	sfs = SFS(SVC(kernel = kernel_GS, C = C_GS, gamma = gamma_GS),
         k_features = (1, 49),
          forward= True,
          floating = False,
          verbose= 2, #a faépítésről több infót tudunk
          scoring= 'accuracy',
          cv = 10,
          n_jobs= -1
         ).fit(X_train, y_train)

	return sfs


#Kiválasztott jellemzők neve
def featureNames(sfs):
	Feature_Names = sfs.k_feature_names_

	return Feature_Names


#Kiválasztott jellemzők és a mintákhoz tartozó osztály címke (PA vagy HC) selected DataFrambe másolása
def selectedData(names, X, y):
	selected = pd.DataFrame()
	selected = pd.concat([y, (X.loc[:, names]).copy()], axis = 1, sort = False)

	return selected


#Kiválasztott jellemzők száma
def featureNumber(selected):
	number = len(selected.columns)

	return number


#Kiválasztott jellemzőkkel elért pontosság
def accuracy(sfs):
	Acc = sfs.k_score_

	return Acc


#Kiválasztott jellemzők kimásolása (osztály címkével együtt) egy Excel táblázatba
#Az Excel fájl neve Selected_Features, a Munkalap neve Selected
def saveSelectedData(selected):
	selected.to_excel(r'D:\BME\SZAKDOLGOZAT\Selected_Features.xlsx', sheet_name='Selected', index = False)


#Hiperparaméteroptimalizálás
def GridSearch(X_test, y_test):
	param_grid = {'C': [0.1, 1, 10, 100, 1000],
	'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
	'kernel': ['rbf', 'linear']}
	grid = GridSearchCV(SVC(), param_grid, cv = 5, refit = True, verbose =0)
	grid.fit(X_test, y_test)
	#print(grid.best_params_)
	C_GS = grid.best_params_['C']
	gamma_GS = grid.best_params_['gamma']
	kernel_GS = grid.best_params_['kernel']
	
	return C_GS, gamma_GS, kernel_GS

#Normalizálás MinMaxScaler segítségével -1 éa 1 közé
def normalize(X):
	
	scaler = MinMaxScaler()
	scaler.fit(X)
	new_X = scaler.transform(X)
	new_X = pd.DataFrame(new_X)
	
	return new_X

def main():
# Adatok betöltése Excelből
#A táblázat 470 mintára vonatkozóan tartalmaz 49 akusztikai jellemzőt
#Továbbá az egyes mintákhoz tartozó csoportot (PA=0, HC=1)
#Minden mintának van egy azonosítója, illetve a beszélő neme is fel van tüntetve
#X az akusztikai paramétereket tartalmazza
#y a mintákhoz tartozó osztályokat
	data = pd.read_excel (r'D:\BME\SZAKDOLGOZAT\470.xlsx')
	X, y = dataProcessing(data)


# Hiányzó elemek lekezelése
	X = missingValues(X)


#Normalizálás -1 és 1 közé
	X = normalize(X)


# Az adatok felosztása tanuló és tesztelő halmazra
	X_train, X_test, y_train, Y_test = split(X, y)


#Optimális paraméterek keresése
	C_GS, gamma_GS, kernel_GS = GridSearch(X_test, Y_test)

#Jellemző kiválasztás Step Forward Feature Selectionnel
	sfs = SFFS(X_train, y_train, C_GS, gamma_GS, kernel_GS)
	
# A kiválasztott jellemzők nevének lekérdezése, majd kiíratása
	names = featureNames(sfs)
	print (names)

# A kiválasztott jellemzőket, 
#a hozzájuk tartozó mintánkénti értékekkel együtt a selected DataFramebe másolom
	selected = selectedData(names, X, y)

#Kiválasztott jellemzők számának lekérdezése, majd kiíratása
	number = featureNumber(selected)-1
	print ('Kiválasztott jellemzők száma: %f' % number)

# Az elért pontosság kiszámolása a kiválasztott jellemzőkkel, majd annak kiíratása
	Acc = accuracy(sfs)
	print ('A pontosság a kiválasztott jellemzőkön: %f' % Acc)

# A kiválasztottjellemzőket oszlopostól kimásolom egy Excel fájlba,
#hogy a Cost Sensitive Classificationnál felhasználhassam őket
	saveSelectedData(selected)

if __name__ == "__main__":
    main()


