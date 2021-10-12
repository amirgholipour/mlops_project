import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import constant

import json

class Transformer(object):
    def __init__(self):
        self.tokenizer = joblib.load('tokenizer.pkl')
        
    def transform_input(self, X, feature_names, meta):
        # print(request)
        X = X.get("data", {}).get("ndarray")
        print(X)
        output = self.tokenizer.texts_to_sequences(X)
        print(X)
        
        print(output)
        output = pad_sequences(output, maxlen=348,padding='post')
        print(output)
        output = constant(output)
        print(output)
        return output