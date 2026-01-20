"""
Models:
- LinearRegressionOLS
- RidgeRegression (L2)
- LassoRegression (L1, proximal)
- ElasticNetRegression (L1 + L2)
"""

class ModelPlaceholder:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
