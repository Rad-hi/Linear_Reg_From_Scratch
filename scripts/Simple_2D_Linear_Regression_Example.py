#!/usr/bin/env python3

from TwoD_LinearRegression import TwoD_LinearRegression

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from time import time

start = time()

style.use('seaborn-whitegrid')

Test_X = [x for x in range(80) ]

Test_Y = [ -15.0,    7.0,  -18.0,   -4.0,  -29.0,    4.0,    2.0,  -16.0,  -32.0,  -35.0,  -59.0,  -59.0
		  ,-21.0,  -30.0,  -66.0,  -33.0,  -66.0,  -70.0,  -54.0,  -50.0,  -91.0, -75.0,  -98.0,  -77.0
		  ,-92.0,  -85.0, -120.0, -137.0,  -98.0,  -91.0, -149.0, -128.0, -100.0, -135.0, -156.0, -161.0
		  ,-129.0, -146.0, -145.0, -147.0, -132.0, -163.0, -172.0, -201.0, -194.0, -163.0, -205.0, -198.0
		  ,-179.0, -213.0, -191.0, -177.0, -206.0, -198.0, -201.0, -237.0, -221.0, -213.0, -221.0, -257.0
		  ,-232.0, -225.0, -251.0, -242.0, -254.0, -275.0, -267.0, -267.0, -300.0, -297.0, -305.0, -297.0
		  ,-294.0, -303.0, -268.0, -311.0, -311.0, -293.0, -338.0, -333.0]

# Creating the class object
regressor = TwoD_LinearRegression()

#x, y = regressor.create_data(80, 30, 4, correlation = "negative")
x, y = np.array(Test_X, dtype = np.float64),np.array(Test_Y, dtype = np.float64)
regressor.assign_data(x, y)

#############################################
###    Load data from an external file    ###
#############################################

"""
# Don't forget to import pandas
#import pandas as pd

# Loading the data from an external file
df = pd.read_csv('DATA_FILE_NAME.csv')

# Prepare the data
x = np.array(df.iloc[:,0])
y = np.array(df.iloc[:,1])

# Load the data into the class instance
regressor.assign_data(x, y)
"""

#############################
###    Train the model    ###
#############################

# Training the model: 
# --> 1st mehod is by using gradient descent to optimize the weights
# --> 2nd method is by calculating the slope and intercept
# Uncomment the desired method

#regressor.fit_grad(100, 0.0001) # epochs-1000 , learning_rate - 0.0001
regressor.best_fit_slope_intercept()

###########################
###    Use the model    ###
###########################

# Prediciting the values
y_pred = regressor.predict(x)

# Evaluate the model by calculating its coefficient of determination
r2 = regressor.coef_det(y_pred)

#
print("The coefficient of determination is equal to: {}".format(r2))
print("The code ran in: {} seconds".format(time()-start))
 
# Plotting the results
plt.scatter(x,y)
plt.plot(x , y_pred, color = "black")
plt.xlabel('x' , size = 10)
plt.ylabel('y', size = 10)
plt.show()
