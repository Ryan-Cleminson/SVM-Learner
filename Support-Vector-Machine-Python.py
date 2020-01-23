#############################################################################
# Author: Jayden Lee (Jayden.Lee@student.uts.edu.au)
# Author: Ryan Cleminson (Ryan.Cleminson@student.uts.edu.au)
# Date: 6/10/19
# Purpose: Using the SVM learner one must predict the outcome of whether a 
# 		   clients quote will default.
#############################################################################

#############################################################################

from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import time
#############################################################################
# Input: filename
# This function DOES
# Output: save file of predicted quote default outcomes
#############################################################################

def SVMRegression(filename):

	#Read in File
	data = pd.read_csv("Inputs/"+ filename + '.csv', sep=',', low_memory=False)
	# Drop the first 2 columns from X
	X = data.drop(["QuoteConversion_Flag", "Quote_ID"], axis=1)
	# Y Contains the Quote Conversion Column
	y = data['QuoteConversion_Flag']

	#print(X)
	
	# Pass in the features->X, Pass in the Target->Y, Takes 30% of the data for testing and the other 70% is for training, Randomly selects data"
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)
	#print(X_train)
	#print(y_train)
	# Allocates the start time
	start_time = time.time()

	#Create a svm Classifier using a linear kernal which could be replaced with a polynomial kernal or radial kernal
	svclassifier = svm.SVC(kernel='linear', degree=4, gamma='auto', max_iter = 100000, tol=0.001)

	#Train the model using the training sets
	print("Fitting the data.")
	print("##################################################")
	svclassifier.fit(X_train, y_train) 
	print("--- %s seconds to complete ---" % (time.time() - start_time))
	print("Fitment Complete. Moving onto the Prediciton.")
	print("##################################################")

	#Predict the response for test dataset
	y_pred = svclassifier.predict(X_test)
	start_time = time.time()
	print("--- %s seconds to complete ---" % (time.time() - start_time))
	#y_pred.to_csv(y_pred,"Prediction" sep=',', low_memory=False)

	print("##################################################")
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	print("Precision:",metrics.precision_score(y_test, y_pred))
	print("Recall:",metrics.recall_score(y_test, y_pred))
	print("F Score:",metrics.f1_score(y_test, y_pred))


	dmp = pickle.dump(svclassifier, open("Outputs/" + filename + '.sav','wb'))
	np.savetxt("Outputs/" + filename + ".csv", y_pred, delimiter=",")

#############################################################################
# 								Main Code. 
#############################################################################

filename = "Ryan2"

SVMRegression(filename)