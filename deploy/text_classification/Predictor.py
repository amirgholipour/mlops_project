import tensorflow as tf
import joblib
import numpy as np
import json

class Predictor(object):

    def __init__(self):
        self.model = tf.keras.models.load_model('model.h5')
        self.labelencoder = joblib.load('labelencoder.pkl')



    def predict(self, X,features_names):
        # data = request.get("data", {}).get("ndarray")
        # mult_types_array = np.array(data, dtype=object)
        print(X)
#         result = self.model.predict(X)
        result = tf.math.argmax(tf.sigmoid(self.model(X)),axis=1)
        print(result)
        print(result.shape)
        print(self.labelencoder.inverse_transform(result))

        return json.dumps(result, cls=JsonSerializer)

class JsonSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (
        np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)