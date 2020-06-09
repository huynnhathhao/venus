from __future__ import print_function
import numpy as np
class NormalEquation():
    def __init__(self):
        pass
    def fit(self, X, y):
        train_ones = np.ones((X_train.shape[0], 1))
        X_train_plus_ones = np.concatenate((train_ones, X_train), axis =1)
        XtX = np.dot(X_train_plus_ones.T, X_train_plus_ones )
        XtXpinv = np.linalg.pinv(XtX)
        Xty = np.dot(X_train_plus_ones.T, y_train)
        self.w = np.dot(XtXpinv, Xty)
    def predict(self,X):
        test_ones = np.ones((X_test.shape[0], 1))
        X_test_plus_ones = np.concatenate((test_ones, X_test), axis = 1)
        predictions = np.dot(X_test_plus_ones, self.w)
        return predictions
    def mean_squared_error(self, y_true, y_predict):
        return np.sum(np.power(y_true - y_predict, 2))/len(y_true)
    
