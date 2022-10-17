import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from collections import Counter
from pdb import set_trace


class my_normalizer:
    def __init__(self, norm="Min-Max", axis = 1):
        #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
        #     axis = 0: normalize rows
        #     axis = 1: normalize columns
        self.norm = norm
        self.axis = axis

    def fit(self, X):
        #     X: input matrix
        #     Calculate offsets and scalers which are used in transform()
        X_array  = np.asarray(X)
        m, n = X_array.shape
        self.offsets = []
        self.scalers = []
        if self.axis == 1:
            for col in range(n):
                offset, scaler = self.vector_norm(X_array[:, col])
                self.offsets.append(offset)
                self.scalers.append(scaler)
        elif self.axis == 0:
            for row in range(m):
                offset, scaler = self.vector_norm(X_array[row])
                self.offsets.append(offset)
                self.scalers.append(scaler)
        else:
            raise Exception("Unknown axis.")

    def transform(self, X):
        X_norm = deepcopy(np.asarray(X))
        m, n = X_norm.shape
        if self.axis == 1:
            for col in range(n):
                X_norm[:, col] = (X_norm[:, col]-self.offsets[col])/self.scalers[col]
        elif self.axis == 0:
            for row in range(m):
                X_norm[row] = (X_norm[row]-self.offsets[row])/self.scalers[row]
        else:
            raise Exception("Unknown axis.")
        return X_norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def vector_norm(self, x):
        # Calculate the offset and scaler for input vector x
        if self.norm == "Min-Max":
            offset = x.min()
            scaler = x.max()-x.min()

        elif self.norm == "L1":
            offset = 0
            scaler = np.sqrt(np.sum(x**2))

        elif self.norm == "L2":
            offset = 0
            scaler = np.sum(x)

        elif self.norm == "Standard_Score":
            offset = np.average(x)
            scaler = np.std(x)

        else:
            raise Exception("Unknown normlization.")
        return offset, scaler

class my_pca:
    def __init__(self, n_components = 5):
        #     n_components: number of principal components to keep
        self.n_components = n_components

    def fit(self, X):
        #  Use svd to perform PCA on X
        #  Inputs:
        #     X: input matrix
        #  Calculates:
        #     self.principal_components: the top n_components principal_components
        U, s, Vh = svd(X)
        self.principal_components = Vh.T[:,:self.n_components]



    def transform(self, X):
        #     X_pca = X.dot(self.principal_components)
        X_array = np.asarray(X)
        return X_array.dot(self.principal_components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: a 1-d array of class labels
    #     0 < ratio < 1: number of samples = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    Cv, Ccounts=np.unique(y_array, return_counts=True)
    CcountsL=list(np.ceil(Ccounts*ratio).astype(int))
    Cvs=list(Cv)
    Cvs_len = len(Cvs)
    classCount=[0]*Cvs_len
    sampleList=[]
    while classCount[0]<CcountsL[0] or classCount[1]<CcountsL[1] or classCount[2]<CcountsL[2]:
        for i in range(len(y_array)):
            if np.random.random()>ratio and i not in sampleList:  
                index=Cvs.index(y_array[i])
                if classCount[index]<CcountsL[index]:
                    classCount[index]=classCount[index]+1
                    sampleList.append(i)
            if classCount==CcountsL:
                break  
    return sampleList
