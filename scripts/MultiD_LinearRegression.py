#!/usr/bin/env python3

import numpy as np
import random 
import copy

# Definition of the class
class MultiD_LinearRegression:
	def __init__(self, x, x_t, y, y_t):
		self.data  		  = x
		self.label 		  = y
		self.data_test    = x_t
		self.label_test   = y_t
		self.coefficients = None
		self.intercept 	  = None

	## Multi linear regression in an instantaneous matter (non-iterative method)
	## Theory:
	## We have a real function f(x) that we don't know, but we're trying to approximate with h(x),
	## f(x) is linear so h(x) is of the form "h(x) = W.X + B".
	## We have a sample data to help us approximate it, and we a have an error measure that helps
	## us determine how good we're approximating f(x) through h(x) BUT that's the in-sample error,
	## we don't know anything about the out-of-sample error yet.
	## ---
	## We approximating f(x) as best as we could, implies we minimize the in-sample error (E_in).
	## E_in = (1/N).Sum[(h(x)-f(x)_in)²] (f(x)_in are the labels Y of the sample_data)
	## min(E_in) <=> grad(E_in) = 0 (when the derivative of E_in is null, we have our min)
	## --> E_in = (1/N)*||x.w - y||² (mathematically, this is how we get E_in, ||.|| is the norm)
	## After a call to linear algebra, we get the solution to grad(E_in) = 0 is W = (X^t.X)^-1.Y
	def fit_multi(self):
		## We add a column called "ones" to the X df at index 0 which will be the intercept vector
		x = self._transform_x(self.data)
		## We're basically just preserving the original df here with a deep_copy
		y = self._transform_y(self.label)
		## Calculate the least square estimate of the coefficient values
		omegas = self._estimate_coeff(x, y)
		self.intercept    = omegas[0] 
		self.coefficients = omegas[1:]

	## Coefficient of determination: 
	## r² = 1 - SE(Ŷ)   # Ŷ is equivalent to Y
	##          _____
	##          SE(/Y)
	def coef_det(self, predicted_ys):
		# True values
		y_val = self.label_test.values
		## Y mean line
		y_mean_line = np.average(y_val)
		
		squeared_error_reg = 0
		squeared_error_y_mean = 0
		for i in range(len(y_val)):
			## SE(Ŷ) # Squared error of the Y line
			squeared_error_reg += (y_val[i] - predicted_ys[i])**2
			## SE(/Y) # Squared error of the mean line
			squeared_error_y_mean += (y_val[i] - y_mean_line)**2
		# R² :: Accuracy of the multiple linear regression model
		return 1 - (squeared_error_reg / squeared_error_y_mean)

	## We have the model is in the form of:
	## Y = X.W + B
	##  _  _     _          _     _  _ 
	## | Y1 |   | - - X1 - - |   | W1 |  
	## | Y2 |   | - - X2 - - |   | W2 |   
	## | .  | = | . . . . .  | . | .  |   
	## | .  |   | . . . . .  |   | .  |   
	## | Yn |   | - - Xn - - |   | Wn |  
	##  _  _     _          _     _  _ 
	## We get the W from the fit_multi call and we apply the previous
	def predict(self, input_ ):
 		predictions = []
 		## input_ is a dataFrame
 		for _, row in input_.iterrows():
 			val  = row.values
 			pred = np.multiply(val, self.coefficients)
 			pred = sum(pred)
 			pred += self.intercept
 			predictions.append(pred)
 		return predictions


	def _estimate_coeff(self, x, y):
		## Pseudo-inverse of X: X* = (X^t.X)^-1.X^t
		## Coefficients: 		w  = X*.Y
		x_tran = x.transpose()
		x_inv  = np.linalg.inv(x_tran.dot(x))
		return x_inv.dot(x_tran).dot(y)
		
	def _transform_x(self, x):
		x = copy.deepcopy(x)
		x.insert(0, 'ones', np.ones((x.shape[0], 1)))
		return x.values

	def _transform_y(self, y):
		y = copy.deepcopy(y)
		return y.values


