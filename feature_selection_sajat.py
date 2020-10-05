# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 22:07:28 2020

@author: Eszter
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


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
C = X

#j egy olyan változó, amely egy jellemző kiválasztási cikusban mindig az adott jellemzőre, azaz oszlopra "mutat
j = 0

#Mindig az aktuális, j-nek megfelelő oszlopot tartalmazza
x = pd.DataFrame()
#Azokat a jellemzőket és a hozzájuk tartozó oszlopokat tartalmazza,
#amelyekkel az adott körben tesztelni fogunk
#Tehát a már kiválasztott jellemzők és az aktuálisan kiválasztott jellemző

tester = pd.DataFrame()

#Ha egy új jellemző növeli az accuracy értéket, akkor a testerről másolatot készítünk
new_table = pd.DataFrame()

#Az eddig összesen kiválasztott jellemző
features = pd.DataFrame()

#A kiválasztott jellemzőkkel elért pontosság
acc =0 
#Kiválasztott jellemzők száma
f = 0

while f < 20:
	while True:
		if j < 49-f:
			x = C.iloc[:, j] #Az aktuális oszlop
			tester = pd.concat([features, x], axis = 1, sort = False) #A már kiválasztott jellemzők halmaza az aktuális oszloppal kibővítve

			#Osztályozó
			#from sklearn.datasets import make_classification
			X_train, X_test, y_train, y_test = train_test_split(tester, y, test_size=0.3)
			clf=RandomForestClassifier(n_estimators=100, random_state = 1, n_jobs = -1)
			clf.fit(X_train,y_train)
			y_pred=clf.predict(X_test)
		
			from sklearn import metrics
			act_acc = metrics.accuracy_score(y_test, y_pred) #Az aktuális accuracy, a bevett új jellemzővel együtt
			
			#Ha az aktuális accuracy nagyobb, mint amit eddig sikerült elérni,
			#akkor ez lesz az új accuracy és bevesszük a jellemzőt
			#A tester kiürítjük, hogy az újabb ciklus elkezdésekor üres legyen
			if act_acc > acc:
				new_table = tester.copy()
				column_name = C.columns[j]
				acc = act_acc
				tester.drop (tester.iloc[:, 0:len(tester.columns)], inplace=True, axis=1)
			
			#Különben a jellemző nem kerül kiválasztásra, "eldobjuk"
			else:
				tester.drop (tester.iloc[:, 0:len(tester.columns)], inplace=True, axis=1)
				#tester = tester.drop(X.columns[j], axis = 1)
			
			#Következő oszlopra/jellemzőre ugrás
			j += 1
		
		else:
			break
	
	#A már vizsgált jellemzőt kivesszük az eredeti halmazból
	C = X.drop(column_name, axis = 1)
	#Kiválasztott jellemzők számának növelése eggyel
	f += 1
	j = 0
	features = new_table.copy()
	

print (len(features.columns))
print (features.columns)
print (acc)



