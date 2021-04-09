from statistics import mean
import numpy as np
import random 

# Definition of the class
class TwoD_LinearRegression:
    def __init__(self):
        self.data  = 0
        self.label = 0
        self.m     = 0
        self.b     = 0
        self.n     = 0

    # Input external data
    def assign_data(self, x, y):
        self.data  = x
        self.label = y
        self.n     = len(x)

    ## Create random data distributed around a single line 
    def create_data(self, num_points, distribution_interval, step = 2, correlation = False) :
        value = 1 # Initial values
        Ys = [] # Final values container
        for i in range(num_points) :
            y = value + random.randrange(-distribution_interval, distribution_interval)
            Ys.append(y)
            if correlation and correlation == 'positive' :
                value += step
            elif correlation and correlation == 'negative' :
                value -= step
        Xs = [i for i in range(len(Ys))]
        self.data  = np.array(Xs, dtype = np.float64) 
        self.label = np.array(Ys, dtype = np.float64)
        self.n     = len(Xs)
        return self.data, self.label

    ## Calculate the best coeffecients for the regression line 
    def best_fit_slope_intercept(self) :
        ## Slope : m = /X./Y - /(X.Y)  # /X : Mean of all elements of X
        ##             ______________
        ##             (/X)² - /(X²)
        self.m = ((mean(self.data)*mean(self.label) - mean(self.data*self.label))
                 /(mean(self.data)*mean(self.data) - mean(self.data*self.data)))

        ## Intercept : Y = m.X + b --> b = Y - m.X (for a single point)
        ##             b = /Y - m./X  
        self.b = mean(self.label) - self.m*mean(self.data)

    ## Squared error
    def squared_error(self, first, second):
        return sum((second - first) ** 2)

    ## Coefficient of determination: 
    ## r² = 1 - SE(Ŷ)   # Ŷ is equivalent to Y
    ##          _____
    ##          SE(/Y)
    def coef_det(self, predicted_ys):
        ## Y mean line
        y_mean_line = [mean(self.label) for _ in self.label ]
        ## SE(Ŷ) # Squared error of the Y line
        squeared_error_reg = self.squared_error(self.label, predicted_ys)
        ## SE(/Y) # Squared error of the mean line
        squeared_error_y_mean = self.squared_error(self.label, y_mean_line)
        return 1 - (squeared_error_reg / squeared_error_y_mean)
         
    def fit_grad(self , epochs , learning_rate):
        #Implementing Gradient Descent
        for i in range(epochs):
            y_pred = self.m * self.data + self.b
            # Calculating derivatives with respect to the prameters
            D_m = (-2/self.n)*sum(self.data * (self.label - y_pred))
            D_b = (-1/self.n)*sum(self.label-y_pred)
            # Updating Parameters of the linear approximation
            self.m = self.m - learning_rate * D_m
            self.c = self.b - learning_rate * D_b


    def predict(self , input_):
        y_pred = self.m * input_ + self.b 
        return y_pred

