import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MyLassoRealDataset:
    """
    This is a demo of implementation of LASSO Regression with L1-Regularization using
    Cyclic Coordinate Descent and Random Coordinate Descent optimization algorithms.
    
    """
    
    def __init__(self):
        """
        This is a demo of implementation of LASSO Regression with L1-Regularization using
        Cyclic Coordinate Descent and Random Coordinate Descent optimization algorithms.
        
        No Parameters to be passed.
        
        Dataset used is publicly available Hitters Dataset. It contains Major League 
        Baseball Data from the 1986 and 1987 seasons.
        Link: https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv
        
        The L1-Regularization parameter lambda is set to optimal value 0.11069 found by
        cross validation.
        
        Maximum iterations is set to 1000.
        
        """
        self.dataset = 'https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv'
        self.lambda_ = 0.11069
        self.max_iter = 1000
        
    def computeobj(self, beta_v, X_train, y_train):
        """
        Computes objective value for vector of beta values provided.
        
        Parameters
        ----------
        beta_v: numpy array
        This is the array of all beta coefficient values.
        
        X_train: numpy array
        This is the multidimensional array of all the features of training data.
        
        y_train: numpy array
        This is the array of all labels of training data.
        
        Returns:
        --------
        objective: float
        It returns the computed objective value including the regularization parameter.
        
        """
        objective = 1/len(y_train) * np.linalg.norm(y_train-X_train.T.dot(beta_v))**2 +\
                    self.lambda_*np.linalg.norm(beta_v, ord = 1)
        return objective
    
    def compute_threshold(self, j, beta_v, X_train, y_train):
        """
        Computes the Threshold Term for Coordinate Descent Algorithm.
        
        Parameters:
        -----------
        j: int
        This is the pointer to the specific observation in the training dataset.
        
        beta_v: numpy array
        This is the array of all beta coefficient values.
        
        X_train: numpy array
        This is the multidimensional array of all the features of training data.
        
        y_train: numpy array
        This is the array of all labels of training data.
        
        Returns:
        --------
        threshold: float
        It returns the computed threshold value for Coordinate Descent Algorithm.
        
        """
        X_rem_j = np.delete(X_train,j,0)
        beta_rem_j = np.delete(beta_v,j,0)
        R_j = y_train - X_rem_j.T.dot(beta_rem_j)
        n = len(y_train)
        threshold = 2/n * X_train[j,:].dot(R_j)
        return threshold
    
    def check_threshold(self, threshold):
        """
        Checks the threshold value against lambda_ for Coordinate Descent Algorithm.
        
        Parameters:
        -----------
        threshold: float
        This is the threshold value computed for Coordinate Descent Algorithm.
        
        Returns:
        --------
        value: int
        This value is used as sign in computation of Coordinate Descent Algorithm.
        
        """
        if threshold < -self.lambda_:
            return 1
        elif threshold > self.lambda_:
            return -1
        else:
            return 0
    
    def cycliccoorddescent(self, X_train, y_train):
        """
        Implements the Cyclic Coordinate Descent Algorithm.
        
        Parameters:
        -----------
        X_train: numpy array
        This is the multidimensional array of all the features of training data.
        
        y_train: numpy array
        This is the array of all labels of training data.
        
        Returns:
        --------
        beta_vals: numpy array
        It returns multidimensional array with beta coefficient values for each iteration
        of the Coordinate Descent Algorithm.
        
        """
        iters = 0
        n = len(y_train)
        beta_vals = []
        np.random.seed(9)
        beta_v = np.random.normal(size = X_train.shape[0])
        while iters < self.max_iter:
            for d in range(len(beta_v)):
                j = d
                threshold = self.compute_threshold(j, beta_v, X_train, y_train)
                sign = self.check_threshold(threshold)
                if sign == 0:
                    beta_j = 0
                else:
                    z_j = np.linalg.norm(X_train[j,:])**2
                    beta_j = (sign * self.lambda_ + threshold) / (2/n * z_j)
                beta_v[j] = beta_j
            beta_vals.append(beta_v.copy())
            iters += 1
        return np.array(beta_vals)

    def pickcoord(self, beta_v):
        """
        Randomly picks a coordinate index from the provided beta coefficient values.
        
        Parameters:
        -----------
        beta_v: numpy array
        This is the array of all beta coefficient values.
        
        Returns:
        --------
        coordinate: float
        This is index of one of the coordinates from the provided beta coefficient values.
        
        """
        return np.random.randint(low=0,high=len(beta_v),size=1)

    def randcoorddescent(self, X_train, y_train):
        """
        Implements the Random Coordinate Descent Algorithm.
        
        Parameters:
        -----------
        X_train: numpy array
        This is the multidimensional array of all the features of training data.
        
        y_train: numpy array
        This is the array of all labels of training data.
        
        Returns:
        --------
        beta_vals: numpy array
        It returns multidimensional array with beta coefficient values for each iteration
        of the Coordinate Descent Algorithm.
        
        """
        iters = 0
        n = len(y_train)
        beta_vals = []
        np.random.seed(9)
        beta_v = np.random.normal(size = X_train.shape[0])
        while iters < self.max_iter:
            for d in range(len(beta_v)):
                j = self.pickcoord(beta_v)
                threshold = self.compute_threshold(j, beta_v, X_train, y_train)
                sign = self.check_threshold(threshold)
                if sign == 0:
                    beta_j = 0
                else:
                    z_j = np.linalg.norm(X_train[j,:])**2
                    beta_j = (sign * self.lambda_ + threshold) / (2/n * z_j)
                beta_v[j] = beta_j
            beta_vals.append(beta_v.copy())
            iters += 1
        return np.array(beta_vals)
    
    def preprocess_real_data(self):
        """
        Preprocesses the Hitter's dataset. Data is randomly split into Training and 
        Test data split using the ration 75%:25%. Data is then standardized suing 
        StandardScaler().
        
        No parameters to be passed.
        
        Returns:
        --------
        X_train, y_train, X_test, y_test: numpy arrays
        Returns four numpy arrays corresponding to Train Set features, Train Set labels,
        Test Set features and Test Set labels.
        
        """
        hitters = pd.read_csv(self.dataset, sep = ',', header = 0)
        hitters = hitters.dropna()
        
        # Create our X matrix with the predictors and y vector with the response
        X = hitters.drop('Salary', axis = 1) # Axis denotes either the rows (0) or the columns (1)
        X = pd.get_dummies(X, drop_first = True)
        y = hitters.Salary

        # Divide the data into training and test sets. By default, 25% goes into the test set.
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

        # Standardize the data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler = preprocessing.StandardScaler().fit(y_train.values.reshape(-1, 1))
        y_train = scaler.transform(y_train.values.reshape(-1, 1)).reshape((-1))
        y_test = scaler.transform(y_test.values.reshape(-1, 1)).reshape((-1))
        
        X_train = X_train.T
        X_test = X_test.T
        return X_train, y_train, X_test, y_test
           
    def plot_objectives(self,betas_cyc,betas_rnd,X_train,y_train):
        """
        Plots the Objectives values against the Interations for both Cyclic and Coordinate
        Descent Algorithms. Useful to visuliaze the Training Process.
        
        No parameters to be passed.
        
        Returns:
        --------
        Plots a graph using Matplotlib.
        """
        plt.figure(figsize = (10, 5))
        plt.plot([self.computeobj(betas_cyc[::20][x], X_train, y_train) for x in range(50)], label="Cyclic Coordinate Descent")
        plt.plot([self.computeobj(betas_rnd[::20][x], X_train, y_train) for x in range(50)], label="Random Coordinate Descent")
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.title("Objective Value vs. Iteration (Lambda = 0.11069)")
        plt.legend()        

if __name__=='__main__':
    """
    Demo execution is automatically initiated once this file is called.
    
    Example:
    --------
    To explicity run the demo, use the following code:
    
    myalgo = MyLassoRealDataset()
    X_train, y_train, X_test, y_test = myalgo.preprocess_data()
    betas_cyc = myalgo.cycliccoorddescent(X_train,y_train)
    betas_rnd = myalgo.randcoorddescent(X_train,y_train)
    myalgo.plot_objectives(betas_cyc,betas_rnd,X_train,y_train)
    
    """
    print("Applying MyLasso algorithm on public Hitters Dataset:\n")
    print("Link to the dataset: https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv\n")
    print("Both Cylic and Coordinate Descent algorithms are used.\n")
    print("Training process started ...", end = '')
    myalgo = MyLassoRealDataset()
    X_train, y_train, X_test, y_test = myalgo.preprocess_real_data()
    betas_cyc = myalgo.cycliccoorddescent(X_train, y_train)
    betas_rnd = myalgo.randcoorddescent(X_train, y_train)
    print("Complete!!\n")
    print("Here is the visualization of the Training Process:\n")
    myalgo.plot_objectives(betas_cyc, betas_rnd, X_train, y_train)