import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.stats import multivariate_normal, norm


class MultiVariateBayes(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        
        means = []
        covs = []
        for i in sorted(set(y)):
            X_class = X[y == i]
            means.append(X_class.mean(axis=0).tolist())
            covs.append(np.cov(X_class, rowvar=False))
            
        self.means = means
        self.covs = covs
        
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        pred = []
        for x in X:
            pdfs = []
            for i in range(3):
                pdfs.append(multivariate_normal.pdf(x, self.means[i], self.covs[i], allow_singular=True))
            pred.append(np.argmax(pdfs))
        
        return pred
    
    
class GaussianBayes(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        
        means = []
        vars_ = []
        for i in sorted(set(y)):
            X_class = X[y == i]
            means.append(X_class.mean(axis=0).tolist())
            vars_.append(X_class.var(axis=0).tolist())
            
        self.means = means
        self.vars_ = vars_
        
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        pred = []
        for x in X:
            pdfs = []
            for i in range(3):
                class_pdf = norm.pdf(x, self.means[i], self.vars_[i])
                class_pdf = class_pdf[~np.isnan(class_pdf)]
                pdfs.append(np.prod(class_pdf))
            pred.append(np.argmax(pdfs))
        
        return pred
    
    
class ParzenWindow(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        
        self.X_train = X
        self.y_train = y
            
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        pred = []
        for row in X:
            ps = []
            for i in sorted(set(self.y_train)):
                X_class = self.X_train[self.y_train == i]
                p = 0
                for xi in X_class:
                    p += multivariate_normal.pdf(row-xi, [0]*len(xi))
                ps.append(p/len(X_class))
            pred.append(np.argmax(ps))
        self.pred = pred
        
        return self.pred