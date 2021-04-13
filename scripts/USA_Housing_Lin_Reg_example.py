#!/usr/bin/env python3

from MultiD_LinearRegression import MultiD_LinearRegression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from time import time


start = time()

def explore_data():
	print(df.head)
	print(df.describe())
	print(df.columns)
	## ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms'
	##  ,'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']
	print(df.info())

if(__name__ == "__main__"):		

	#############################################
	###    Load data from an external file    ###
	#############################################

	df = pd.read_csv("USA_Housing.csv")

	##############################
	###    Explore the data    ###
	##############################

	#explore_data()
	
	##############################
	###    Prepare the data    ###
	##############################
	
	## We want to predict the price of a house based on this available data, so our target will be the house price
	y = df["Price"]

	## The rest of the data is our features (excluding Y = Price), and , 'Address' 
	## "Address" must be enumerated in order to be used by our model, but each
	## house has a unique address (check with df["Address"].nunique if you're not convinced)
	## so we can't use one-hot-encoding .. but maybe we could do some string manipulation
	## magic to determine the houses on the same street and maybe see a correlation ...
	## Some EDA (Exploratory Data Analysis) is needed anyway ... 
	x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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

	##############################################
	###    Scikit-Learn's Linear Regression    ###
	##############################################

	sk_regressor = LinearRegression()
	sk_regressor.fit(x_train, y_train)
	predict = sk_regressor.predict(x_test)

	##############################
	###    Model evaluation    ###
	##############################
	
	"""
	## Sklearn's performance
	print('MAE ',mean_absolute_error(y_test,predict))
	print('MSE ',mean_squared_error(y_test,predict))
	print('R² Score ',r2_score(y_test,predict))
	print('RMSE ',np.sqrt(mean_squared_error(y_test,predict)))
	
	## Our model's performance
	print('OUR__MAE ',mean_absolute_error(y_test,y_pred))
	print('OUR__MSE ',mean_squared_error(y_test,y_pred))
	print('OUR__R² Score ',r2_score(y_test,y_pred))
	print('OUR__RMSE ',np.sqrt(mean_squared_error(y_test,y_pred)))

	"""

	print("Our Final R^2 score: {}".format(r2))
	print("Scikit-Learn\'s Final R^2 score: {}".format(r2_score(y_test,predict)))

	print("The code ran in: {} seconds".format(time()-start))

	## We can't vizualise the data since it's multidimensional (in a single graph I mean)
