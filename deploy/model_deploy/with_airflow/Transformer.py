import pandas as pd
import joblib
import numpy as np
import json
class Transformer(object):
    def __init__(self):
        self.ordinalencoder = joblib.load('ordinalencoder.pkl')
        self.onehotencoder = joblib.load('onehotencoder.pkl')
        
    def transform_input(self, X, feature_names, meta):
        # print(request)
        # X = request.get("data", {}).get("ndarray")
        # feature_names = request.get("data", {}).get("names")
        print(X)
        df = pd.DataFrame(X, columns=feature_names)
        print(df)
        df = self.ordinalencoder.transform(df)
        print(df)
        df = self.onehotencoder.transform(df)
        print(df)

        #df = df.drop(['customerID'], axis=1)
        return df.to_numpy()