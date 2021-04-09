#!/usr/bin/env python3

from MultiD_LinearRegression import MultiD_LinearRegression

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from time import time

start = time()

def sklearn_to_df(data_loader):
	X_data = data_loader.data
	X_columns = data_loader.feature_names
	x = pd.DataFrame(X_data, columns = X_columns)
	y_data = data_loader.target
	y = pd.Series(y_data, name = 'target')
	return x, y

#############################################
###    Load data from an external file    ###
#############################################

if(__name__ == "__main__"):		
	x, y = sklearn_to_df(load_boston())

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	## Vizualize the data
	print(x_train.describe)

	# Creating the class object
	regressor = MultiD_LinearRegression(x_train, x_test, y_train, y_test)

	#############################
	###    Train the model    ###
	#############################

	regressor.fit_multi()

	###########################
	###    Use the model    ###
	###########################

	# Prediciting the values
	y_pred = regressor.predict(x_test)

	# Evaluate the model by calculating its coefficient of determination
	r2 = regressor.coef_det(y_pred)

	#
	print("The coefficient of determination is equal to: {}".format(r2))
	print("The code ran in: {} seconds".format(time()-start))

	## We can't vizualise the data since it's multidimensional