# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:09:07 2020

@author: Eszter
"""
import pandas as pd
import numpy as np
#from sklearn import preprocessing
#from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from costcla.models import BayesMinimumRiskClassifier
from costcla.models import CostSensitiveRandomForestClassifier

def data_processing(data1):
	data2 = pd.DataFrame(data1.drop("Csoport", axis =1))
	data_target = data1["Csoport"]

	data2 = data2.to_numpy(dtype='float')
	data_target = data_target.to_numpy(dtype='float')
	return data2, data_target

def cost_matrix(FP, FN):
	row = np.array([FP, FN, 0, 0])
	cost_mat = np.array([FP, FN, 0, 0])
	i = 0

	while i < 469:
		cost_mat = np.vstack ((cost_mat, row) )
		i += 1
	
	return cost_mat

def train_test(X, y, cost_mat):
	
	X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = \
	train_test_split(X, y, cost_mat, test_size=0.3)
	
	return X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test

def normalize(X_train, X_test):
	
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	new_X_train = scaler.transform(X_train)

	scaler.fit(X_test)
	new_X_test = scaler.transform(X_test)
	
	return new_X_train, new_X_test


def convert(X_train, X_test, y_train, y_test):
	
	X_train = X_train.to_numpy(dtype='float')
	X_test = X_test.to_numpy(dtype='float')
	y_train = y_train.to_numpy(dtype='float')
	y_test = y_test.to_numpy(dtype='float')
	
	return X_train, X_test, y_train, y_test

def classification(model, X_train, X_test, y_train, y_test):
	
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	
	return acc, y_pred

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

def cost_sensitive_classification_Bayes(model, X_train, X_test, y_train, y_test, cost_mat_test):

	c_model = BayesMinimumRiskClassifier()
	y_prob_test = model.predict_proba(X_test)
	y_pred_test_model = model.predict(X_test)
	c_model.fit(y_test, y_prob_test)
	y_pred_test_c_model = c_model.predict(y_prob_test, cost_mat_test)
	c_accuracy = accuracy_score(y_test, y_pred_test_c_model)
	
	return c_accuracy, y_pred_test_c_model

def cost_sensitive_classification(model, X_train, X_test, y_train, y_test, cost_mat_train):
	
	y_pred_test = model.fit(X_train, y_train, cost_mat_train).predict(X_test)
	accuracy = accuracy_score(y_test, y_pred_test)

	return accuracy, y_pred_test


def data_preparation(data):
	
		data = data.drop(["Minta ", "Nem"], axis=1)
		data.isnull().sum()
		column_means = data.mean()
		data = data.fillna(column_means)
		
		return data

def main():
	
	#Az összes jellemző beolvasása
	data_all = pd.read_excel (r'D:\BME\SZAKDOLGOZAT\470.xlsx')
	data_all = data_preparation(data_all)
	X_all, y_all = data_processing(data_all)
	
	#Feature Selection által meghatározott jellemk beolvasása 
	#A továbbiakban _s jelöli a változók végén, hogy a jellemzőkiválasztás esetéről van szó
	data_selected = pd.read_excel (r'D:\BME\SZAKDOLGOZAT\Selected_Features.xlsx')
	X_selected, y_selected = data_processing(data_selected)
	#print (X_selected)
	#print (y_selected)
	
	#Költség-mátrix létrehozása
	cost_mat = cost_matrix(0.4, 0.7)
	print (cost_mat)
	
	#Adatok felosztása tanító és tesztelő halmazra
	X_train_s, X_test_s, y_train_s, y_test_s, cost_mat_train, cost_mat_test = train_test(X_selected, y_selected, cost_mat)
	print (X_train_s)
	
	X_train_s, X_test_s = normalize(X_train_s, X_test_s)
	print (X_train_s)

	model1_s = SVC(kernel = 'linear', probability=True)
	acc1_s, y_pred1_s = classification(model1_s, X_train_s, X_test_s, y_train_s, y_test_s)
	#print (acc1_s)
	print ()
	
	print ("Jellemző kiválasztás és SVM lineáris kernellel:")
	conf_matrix_SVM_lin_s = matrix(y_pred1_s, y_test_s, acc1_s)
	print (conf_matrix_SVM_lin_s)
	print ()
	
	model2_s = SVC(kernel = 'rbf',C = 1, gamma = 0.0204, probability=True)
	acc2_s, y_pred2_s = classification(model2_s, X_train_s, X_test_s, y_train_s, y_test_s)
	#print (acc2_s)
	print ()
	
	print("Jellemző kiválasztás és SVM rbf kernellel:")
	conf_matrix_SVM_rbf_s = matrix(y_pred2_s, y_test_s, acc2_s)
	print (conf_matrix_SVM_rbf_s)
	print ()
	
	c_accuracy_s, y_pred_test_c_model_s = cost_sensitive_classification_Bayes(model1_s, X_train_s, X_test_s, y_train_s, y_test_s, cost_mat_test)
	#print (c_accuracy_s)
	print ()
	
	print ("Jellemző kiválasztás költség-érzékeny lineáris SVM")
	conf_matrix_cost_sensitive_s = matrix(y_pred_test_c_model_s, y_test_s, c_accuracy_s)
	print (conf_matrix_cost_sensitive_s)
	print ()
	
	c_accuracy2_s, y_pred_test_c_model2_s = cost_sensitive_classification_Bayes(model2_s, X_train_s, X_test_s, y_train_s, y_test_s, cost_mat_test)
	#print (c_accuracy_s)
	print ()
	
	print ("Jellemző kiválasztás költség-érzékeny rbf SVM")
	conf_matrix_cost_sensitive2_s = matrix(y_pred_test_c_model2_s, y_test_s, c_accuracy2_s)
	print (conf_matrix_cost_sensitive2_s)
	print ()


#Adatok felosztása tanító és tesztelő halmazra
#Továbbiakban _a jelöli a változók végén, hogy a jellemzőkiválasztás esetéről van szó
	X_train_a, X_test_a, y_train_a, y_test_a, cost_mat_train, cost_mat_test = train_test(X_all, y_all, cost_mat)
	#print (X_train_a)
	
	X_train_a, X_test_a = normalize(X_train_a, X_test_a)
	#print (X_train_a)


	model1_a = SVC(kernel = 'linear', probability=True)
	acc1_a, y_pred1_a = classification(model1_a, X_train_a, X_test_a, y_train_a, y_test_a)
	#print (acc1_a)
	print ()
	
	
	print ("Összes jellemző és SVM lineáris kernellel:")
	conf_matrix_SVM_lin_a = matrix(y_pred1_a, y_test_a, acc1_a)
	print (conf_matrix_SVM_lin_a)
	print ()
	
	model2_a = SVC(kernel = 'rbf',C = 1, gamma = 0.0204, probability=True)
	acc2_a, y_pred2_a = classification(model2_a, X_train_a, X_test_a, y_train_a, y_test_a)
	#print (acc2_a)
	print ()
	
	print ("Összes jellemző és SVM rbf kernellel:")
	conf_matrix_SVM_rbf_a = matrix(y_pred2_a, y_test_a, acc2_a)
	print (conf_matrix_SVM_rbf_a)
	print ()
	
	c_accuracy1_a, y_pred_test_c_model_a = cost_sensitive_classification_Bayes(model1_a, X_train_a, X_test_a, y_train_a, y_test_a, cost_mat_test)
	#print (c_accuracy_a)
	print ()
	
	print ("Jellemző kiválasztás költség-érzékeny lineáris SVM")
	conf_matrix_cost_sensitive1_a = matrix(y_pred_test_c_model_a, y_test_a, c_accuracy1_a)
	print (conf_matrix_cost_sensitive1_a)
	print ()
	
	c_accuracy2_a, y_pred_test_c_model2_a = cost_sensitive_classification_Bayes(model2_a, X_train_a, X_test_a, y_train_a, y_test_a, cost_mat_test)
	#print (c_accuracy2_a)
	print ()
	
	print ("Jellemző kiválasztás költség-érzékeny rbf SVM")
	conf_matrix_cost_sensitive2_a = matrix(y_pred_test_c_model2_a, y_test_a, c_accuracy2_a)
	print (conf_matrix_cost_sensitive2_a)
	print ()


if __name__ == "__main__":
    main()

