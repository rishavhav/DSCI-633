from turtle import distance
import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="euclidean", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y
        return

    def dist(self,x):
        # Calculate distances of training data to a single input data point (distances from self.X to x)
        # Output np.array([distances to x])
        if self.metric == "minkowski":
            distances = [np.power(np.sum((x-x2)**self.p),1/self.p) for x2 in self.X.to_numpy()]

        elif self.metric == "euclidean":
            distances = [np.sqrt(np.sum((x-x2)**2)) for x2 in self.X.to_numpy()]

        elif self.metric == "manhattan":
            distances = [np.sum(np.abs(x-x2)) for x2 in self.X.to_numpy()]

        elif self.metric == "cosine":
            distances =  [1 - (np.dot(x,x2)/(np.sqrt(np.sum(x**2))*np.sqrt(np.sum(x2**2)))) for x2 in self.X.to_numpy()]

        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        distances = self.dist(x)
        return Counter([self.y[i] for i in np.argsort(distances)[0:5]])
       

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        
        probs = self.predict_proba(X)        
        probs = probs.fillna(0)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():        
            prob = {}            
            temp = dict(self.k_neighbors(x))
            for i in temp:
              print(temp)
              temp[i] = temp[i] / sum(temp.values())
              prob[max(temp, key=temp.get)] = max(temp.values())
            probs.append(prob)
        probs = pd.DataFrame(probs, columns=self.classes_)
        
        return probs