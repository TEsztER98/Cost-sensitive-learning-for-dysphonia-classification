# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:09:07 2020
@author: Eszter
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from costcla.models import BayesMinimumRiskClassifier
from costcla.models import CostSensitiveRandomForestClassifier



#Adatbázis formázása Excelből való betöltés után (Akusztikus paraméterek részhalmazát tartalmazó adatbázisra)
def dataProcessing_FS(data):
	# A diagnózis (PA vagy HC) kivágása
	#ezek másik változóba fognak kerülni (y-ba)
	X = pd.DataFrame(data.drop("Csoport", axis =1))
	y = data["Csoport"]

	#X és y konvertálása olyan formába,
	#amely megfelel a később használt költség-érzékeny osztályozók által várt adattípusnak
	X = X.to_numpy(dtype='float')
	y = y.to_numpy(dtype='float')
	return X, y


#Adatbázis formázása Excelből való betöltés után (Összes akusztikus paramétert tartalmazó adatbázisra)
def dataProcessing_A(data):
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

#Adatok eloszlásának leírása: Mennyi az egészségesként és mennyi a betegként diagnosztizáltak száma
def dataDistribution(y):
	x = pd.Series(y).value_counts()
	target = pd.DataFrame()
	target['Frequency'] = x
	target['Percentage'] = target['Frequency'] / target['Frequency'].sum()
	print (target)

#Költség mátrix létrehozása
#FP: False Positive, FN: False Negative
#TP: True Positive = 0, TN: True Negative = 0
#A kölstég-érzékeny osztályozáshoz minden mintához külön mátrix kell
#Mivel 470 minta van összesen, 470 ugyanolyan mátrixra lesz szükség
def cost_matrix(FP, FN):
	row = np.array([FP, FN, 0, 0])
	cost_mat = np.array([FP, FN, 0, 0])
	i = 0

	while i < 469:
		cost_mat = np.vstack ((cost_mat, row) )
		i += 1
	
	return cost_mat


#Adatok felosztása tesztelő és tanító halmazra
# Az adatok 30%-a kerül a tesztelő halmazba, 70% pedig a tanuló halmazba
def train_test(X, y, cost_mat):
	
	X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = \
	train_test_split(X, y, cost_mat, test_size=0.3, random_state = 4)
	
	return X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test


#Normalizálás MinMaxScaler segítségével -1 és 1 közé
def normalize(X_train, X_test):
	
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	new_X_train = scaler.transform(X_train)

	scaler.fit(X_test)
	new_X_test = scaler.transform(X_test)
	
	return new_X_train, new_X_test

#Konvertálás (a cost sensitive modellehez szükséges)
def convert(X_train, X_test, y_train, y_test):
	
	X_train = X_train.to_numpy(dtype='float')
	X_test = X_test.to_numpy(dtype='float')
	y_train = y_train.to_numpy(dtype='float')
	y_test = y_test.to_numpy(dtype='float')
	
	return X_train, X_test, y_train, y_test

#Hiperparaméteroptimalizálás rbf kernelhez
def rbfGridSearch(X_test, y_test):
	param_grid = {'C': [0.1, 1, 10, 100, 1000],
	'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
	'kernel': ['rbf']}
	grid = GridSearchCV(SVC(), param_grid, cv = 5, refit = True, verbose =0)
	grid.fit(X_test, y_test)
	#print(grid.best_params_)
	C_GS = grid.best_params_['C']
	gamma_GS = grid.best_params_['gamma']
	kernel_GS = grid.best_params_['kernel']
	
	return C_GS, gamma_GS, kernel_GS

#Hiperparaméteroptimalizálás lineáris kernelhez
def linearGridSearch(X_test, y_test):
	param_grid = {'C': [0.1, 1, 10, 100, 1000],
	'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
	'kernel': ['linear']}
	grid = GridSearchCV(SVC(), param_grid, cv = 5, refit = True, verbose =0)
	grid.fit(X_test, y_test)
	#print(grid.best_params_)
	C_GS = grid.best_params_['C']
	gamma_GS = grid.best_params_['gamma']
	kernel_GS = grid.best_params_['kernel']
	
	return C_GS, gamma_GS, kernel_GS


#Osztályozás költségek nélkül 
#(A modell ebben az esetben SVM lesz, csak esetenként különböző beállítások mellett)
def classification(model, X_train, X_test, y_train, y_test):
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	
	return acc, y_pred


#Tévesztési mátrix
#Tartalmazza: True Positive, False Positive, True Negative, False Negative
#Továbbá recall, precision, accuracy
def matrix(y_pred, y_test, acc):
	
	cm = confusion_matrix(y_test, y_pred)
	TP = cm[0][0]
	FP = cm[0][1]
	FN = cm[1][0]
	TN = cm[1][1]

	recall_PA = TP/(TP+FN)
	recall_HC = TN/(TN+FP)
	newrow = [recall_PA, recall_HC]
	accuracy = [acc, ""]
	cm = np.vstack([cm, newrow])
	cm = np.vstack([cm, accuracy])

	prec_PA = TP/(TP+FP)
	prec_HC = TN/(TN+FN)

	df1 = pd.DataFrame(data=cm, index=["Pred. PA", "Pred. HC", "Recall", "Accuracy"], columns=["True PA", "True HC"]) 
	precision = [prec_PA, prec_HC, '', '']
	df1['Precision'] = precision

	return df1


#Kölstég-érzékeny osztályozás
#(A modell itt SVM lesz csak esetenként különböző beállítások mellett)
def cost_sensitive_classification(model, X_train, X_test, y_train, y_test, cost_mat_test):

	c_model = BayesMinimumRiskClassifier()
	y_prob_test = model.predict_proba(X_test)
	y_pred_test_model = model.predict(X_test)
	c_model.fit(y_test, y_prob_test)
	y_pred_test_c_model = c_model.predict(y_prob_test, cost_mat_test)
	c_accuracy = accuracy_score(y_test, y_pred_test_c_model)
	
	return c_accuracy, y_pred_test_c_model


#Költség-érzékeny véletlen erdő szintén a CostCla csomagból
#Cost Sensitive Random Forest
def cost_sensitive_RF(X_train, X_test, y_train, y_test, cost_mat_test, cost_mat_train):
	model_RF = CostSensitiveRandomForestClassifier()
	y_pred_test_csdt = model_RF.fit(X_train, y_train, cost_mat_train).predict(X_test)
	c_accuracy_RF = accuracy_score(y_test, y_pred_test_csdt)
	
	return c_accuracy_RF, y_pred_test_csdt


#A jelölések a következőképpen értelmezendők:
	#_FS utótag: Jellemző kiválasztást tartalmazó 
	#_A utótag: Összes jellemző bevételével
	#lin előtag: lineáris kernel alkalmazása
	#rbf előtag: rbf kernel alkalmazása
	#SVM: Support Vector Machine
	#BMR: Bayes Minimum Risk
	#RF: Random Forest

def main():
#Feature Selection által meghatározott jellemk beolvasása az Excel fájlból
	data_FS = pd.read_excel (r'D:\BME\SZAKDOLGOZAT\Selected_Features.xlsx')

#A táblázat 470 mintára vonatkozóan tartalmazza a kiválasztott akusztikai jellemzőket
#Továbbá az egyes mintákhoz tartozó csoportot (PA=0, HC=1)
#X az akusztikai paramétereket tartalmazza
#y a mintákhoz tartozó osztályokat
	X_FS, y_FS = dataProcessing_FS(data_FS)

#Adatok eloszlásának leírása: Mennyi az egészségesként és mennyi a betegként diagnosztizáltak száma
	dataDistribution(y_FS)
	print ()


#Költség-mátrix létrehozása
	cost_mat = cost_matrix(0.4, 0.6)
	print (cost_mat)
	print()


#Jellemző kiválasztással meghatározott adatok felhasználásával tesztelő és tanító halmaz kialakítása
	X_train_FS, X_test_FS, y_train_FS, y_test_FS, cost_mat_train, cost_mat_test = train_test(X_FS, y_FS, cost_mat)


#Normalizálás
	X_train_FS, X_test_FS = normalize(X_train_FS, X_test_FS)


#Paraméter optimalizálás lineáris kernellel(Jellemző kiválasztás esetére)
	linearC_GS_FS, linearGamma_GS_FS, linearKernel_GS_FS = linearGridSearch(X_test_FS, y_test_FS)
	print (linearC_GS_FS, linearGamma_GS_FS, linearKernel_GS_FS)


#Paraméter optimalizálás rbf kernellel (Jellemző kiválasztás esetére)
	rbfC_GS_FS, rbfGamma_GS_FS, rbfKernel_GS_FS = rbfGridSearch(X_test_FS, y_test_FS)
	print (rbfC_GS_FS, rbfGamma_GS_FS, rbfKernel_GS_FS)


#SVM lineáris kernellel jellemző kiválasztás után
	linSVM_FS = SVC(C = linearC_GS_FS, kernel = linearKernel_GS_FS, gamma = linearGamma_GS_FS, probability= True)
	linSVM_FS_acc, linSVM_FS_pred = classification(linSVM_FS, X_train_FS, X_test_FS, y_train_FS, y_test_FS)
	linSVM_FS_MC = matthews_corrcoef(y_test_FS, linSVM_FS_pred)


#Eredmények kiíratása: Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print('SVM osztályozó lineáris kernellel (Csak a kiválasztott jellemzőkkel)')
	conf_matrix_linSVM_FS = matrix(linSVM_FS_pred, y_test_FS, linSVM_FS_acc)
	print (conf_matrix_linSVM_FS)
	print ()
	print(linSVM_FS_MC)
	print ()



#SVM rbf kernellel jellemző kiválasztás után
	rbfSVM_FS = SVC(C = rbfC_GS_FS, kernel = rbfKernel_GS_FS, gamma = rbfGamma_GS_FS, probability= True)
	rbfSVM_FS_acc, rbfSVM_FS_pred = classification(rbfSVM_FS, X_train_FS, X_test_FS, y_train_FS, y_test_FS)
	rbfSVM_FS_MC = matthews_corrcoef(y_test_FS, rbfSVM_FS_pred)


#Eredmények kiíratása:  Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print('SVM osztályozó rbf kernellel (Csak a kiválasztott jellemzőkkel)')
	conf_matrix_rbfSVM_FS = matrix(rbfSVM_FS_pred, y_test_FS, rbfSVM_FS_acc)
	print (conf_matrix_rbfSVM_FS)
	print ()
	print(rbfSVM_FS_MC)
	print ()



#A lineáris SVM felhasználásával költség-érzékeny osztályozás (Bayes Minimum Risk)
	linBMR_FS_acc, linBMR_FS_pred = cost_sensitive_classification(linSVM_FS, X_train_FS, X_test_FS, y_train_FS, y_test_FS, cost_mat_test)
	linBMR_FS_MC = matthews_corrcoef(y_test_FS, linBMR_FS_pred)


#Eredmények kiíratása: Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print ("Bayes Minimum Risk lineáris kernelű SVM osztályozóval (Csak a kiválasztott jellemzőkkel)")
	conf_matrix_linBMR_FS = matrix(linBMR_FS_pred, y_test_FS, linBMR_FS_acc)
	print (conf_matrix_linBMR_FS)
	print ()
	print(linBMR_FS_MC)
	print ()


#Az rbf SVM felhasználásával költség-érzékeny osztályozás (Bayes Minimum Risk)
	rbfBMR_FS_acc, rbfBMR_FS_pred = cost_sensitive_classification(rbfSVM_FS, X_train_FS, X_test_FS, y_train_FS, y_test_FS, cost_mat_test)
	rbfBMR_FS_MC = matthews_corrcoef(y_test_FS, rbfBMR_FS_pred)


#Eredmények kiíratása: Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print ("Bayes Minimum Risk rbf kernelű SVM osztályozóval (Csak a kiválasztott jellemzőkkel)")
	conf_matrix_rbfBMR_FS = matrix(rbfBMR_FS_pred, y_test_FS, rbfBMR_FS_acc)
	print (conf_matrix_rbfBMR_FS)
	print ()
	print(rbfBMR_FS_MC)
	print ()


#Cost Sensitive Random Forest jellemző kiválasztással
	RF_FS_acc, RF_FS_pred = cost_sensitive_RF(X_train_FS, X_test_FS, y_train_FS, y_test_FS, cost_mat_test, cost_mat_train)
	RF_FS_MC = matthews_corrcoef(y_test_FS, RF_FS_pred)


#Eredmények kiíratása: Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print("Jellemző kiválasztás után RF")
	conf_matrix_RF_FS = matrix(RF_FS_pred, y_test_FS, RF_FS_acc)
	print (conf_matrix_RF_FS)
	print ()
	print (RF_FS_MC)
	print ()


#
#
#
#


#Osztályozások az összes jellemző (akusztikus paraméter) bevételével

#Az összes jellemző beolvasása az Excel fájlból
	data_all = pd.read_excel (r'D:\BME\SZAKDOLGOZAT\470.xlsx')
	X_A, y_A = dataProcessing_A(data_all)
	X_A = missingValues(X_A)


#Tesztelő és tanító halmaz kialakítása
	X_train_A, X_test_A, y_train_A, y_test_A, cost_mat_train, cost_mat_test = train_test(X_A, y_A, cost_mat)

#Konvertálás
	X_train_A, X_test_A, y_train_A, y_test_A = convert(X_train_A, X_test_A, y_train_A, y_test_A)

	#Normalizálás
	X_train_A, X_test_A = normalize(X_train_A, X_test_A)



#Paraméter optimalizálás lineáris kernellel(Összes jellemzőre)
	linearC_GS_A, linearGamma_GS_A, linearKernel_GS_A = linearGridSearch(X_test_A, y_test_A)
	print (linearC_GS_A, linearGamma_GS_A, linearKernel_GS_A)

#Paraméter optimalizálás rbf kernellel (Összes jellemzőre)
	rbfC_GS_A, rbfGamma_GS_A, rbfKernel_GS_A = rbfGridSearch(X_test_A, y_test_A)
	print (rbfC_GS_A, rbfGamma_GS_A, rbfKernel_GS_A)



#SVM lineáris kernellel az összes jellemző bevételével
	linSVM_A = SVC(C = linearC_GS_A, gamma = linearGamma_GS_A, kernel = linearKernel_GS_A, probability=True)
	linSVM_A_acc, linSVM_A_pred = classification(linSVM_A, X_train_A, X_test_A, y_train_A, y_test_A)
	linSVM_A_MC = matthews_corrcoef(y_test_A, linSVM_A_pred)

#Eredmények kiíratása: Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print('SVM osztályozó lineáris kernellel (Összes jellemzővel)')
	conf_matrix_linSVM_A = matrix(linSVM_A_pred, y_test_A, linSVM_A_acc)
	print (conf_matrix_linSVM_A)
	print ()
	print (linSVM_A_MC)
	print()



#SVM rbf kernellel az összes jellemző bevételével
	rbfSVM_A = SVC(C = rbfC_GS_A, gamma = rbfGamma_GS_A, kernel = rbfKernel_GS_A, probability=True)
	rbfSVM_A_acc, rbfSVM_A_pred = classification(rbfSVM_A, X_train_A, X_test_A, y_train_A, y_test_A)
	rbfSVM_A_MC = matthews_corrcoef(y_test_A, rbfSVM_A_pred)

#Eredmények kiíratása: Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print('SVM osztályozó rbf kernellel (Összes jellemzővel)')
	conf_matrix_rbfSVM_A = matrix(rbfSVM_A_pred, y_test_A, rbfSVM_A_acc)
	print (conf_matrix_rbfSVM_A)
	print ()
	print (rbfSVM_A_MC)
	print()



#Költség-érzékeny osztályozás a lineáris kernelű SVM felhasználásával és az összes jellemzővel (Bayes Minimum Risk)
	linBMR_A_acc, linBMR_A_pred = cost_sensitive_classification(linSVM_A, X_train_A, X_test_A, y_train_A, y_test_A, cost_mat_test)
	linBMR_A_MC = matthews_corrcoef(y_test_A, linBMR_A_pred)

#Eredmények kiíratása: Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print ("Bayes Minimum Risk lineáris kernelű SVM osztályozóval (Összes jellemzővel)")
	conf_matrix_linBMR_A = matrix(linBMR_A_pred, y_test_A, linBMR_A_acc)
	print (conf_matrix_linBMR_A)
	print ()
	print (linBMR_A_MC)
	print ()



#Költség-érzékeny osztályozás az rbf kernelű SVM felhasználásával és az összes jellemzővel (Bayes Minimum Risk)
	rbfBMR_A_acc, rbfBMR_A_pred = cost_sensitive_classification(rbfSVM_A, X_train_A, X_test_A, y_train_A, y_test_A, cost_mat_test)
	rbfBMR_A_MC = matthews_corrcoef(y_test_A, rbfBMR_A_pred)

#Eredmények kiíratása: Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print ("Bayes Minimum Risk rbf kernelű SVM osztályozóval (Összes jellemzővel)")
	conf_matrix_rbfBMR_A = matrix(rbfBMR_A_pred, y_test_A, rbfBMR_A_acc)
	print (conf_matrix_rbfBMR_A)
	print ()
	print (rbfBMR_A_MC)
	print ()



#Cost Sensitive Random Forest az összes jellemzőt bevéve
	RF_A_acc, RF_A_pred = cost_sensitive_RF(X_train_A, X_test_A, y_train_A, y_test_A, cost_mat_test, cost_mat_train)
	RF_A_MC = matthews_corrcoef(y_test_A, RF_A_pred)

#Eredmények kiíratása: Tévesztési mátrix, recall, precision, accuracy, Mátyás korreláció
	print("Összes jellemző bevételével RF")
	conf_matrix_RF_A = matrix(RF_A_pred, y_test_A, RF_A_acc)
	print (conf_matrix_RF_A)
	print ()
	print (RF_A_MC)
	print ()


if __name__ == "__main__":
    main()