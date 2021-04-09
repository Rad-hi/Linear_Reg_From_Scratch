#!/usr/bin/env python3

from TwoD_LinearRegression import TwoD_LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style
from time import time

start = time()

style.use('seaborn-whitegrid')

# Creating the class object
regressor = TwoD_LinearRegression()

x, y = regressor.create_data(80, 30, 4, correlation = "negative")

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

print(1235)

preformance_report = {}

for i in range (1, 1, 10):
	lr = 1/i

	regressor.fit_grad(1000, lr) # epochs-1000 , learning_rate - 0.0001
	#regressor.best_fit_slope_intercept()

	###########################
	###    Use the model    ###
	###########################

	# Prediciting the values
	y_pred = regressor.predict(x)

	# Evaluate the model by calculating its coefficient of determination
	r2 = regressor.coef_det(y_pred)

	print(i)
	#preformance_report[lr] = r2

print(1235)

#print(preformance_report)
"""
#
print("The coefficient of determination is equal to: {}".format(r2))
print("The code ran in: {} seconds".format(time()-start))
 
# Plotting the results
plt.scatter(x,y)
plt.plot(x , y_pred, color = "black")
plt.xlabel('x' , size = 10)
plt.ylabel('y', size = 10)
plt.show()
"""