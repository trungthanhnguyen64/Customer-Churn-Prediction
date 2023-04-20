from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class OutlierHandling(BaseEstimator, TransformerMixin):
    outliers = {}
    q2 = {}
    lwr_bound = {}
    upr_bound = {}
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.outliers[col] = []
            q1 = np.percentile(X[col], 25)
            q3 = np.percentile(X[col], 75)
            IQR = q3 - q1
            self.lwr_bound[col] = q1-(1.5*IQR)
            self.upr_bound[col] = q3+(1.5*IQR)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)   
        for col in X.columns:
            X[col] = X[col].map(lambda x: self.lwr_bound[col] if x<self.lwr_bound[col] 
                                        else(self.upr_bound[col] if x>self.upr_bound[col]
                                        else x))
        return X