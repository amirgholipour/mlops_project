import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

class Transformer(object):
    def __init__(self):
        self.tokenizer = joblib.load('tokenizer.pkl')
        
    def transform_input(self, X, feature_names, meta):
        print(X)
#         X = X[0]
        print(X[0])
        output = self.tokenizer.texts_to_sequences(X[0])
        print(X)
        
        print(output)
        output = pad_sequences(output, maxlen=348,padding='post')
        print(output)
        output = tf.constant(output)
        print(output)
        return output