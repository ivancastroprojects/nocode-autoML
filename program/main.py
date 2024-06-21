# main.py
from training import Training
import mqtt
import global_data
import json

DEBUG = True

class FakeMsg:
    def __init__(self, message):
        self.payload = message

def init():
    global_data.training = Training()

    """Initialize MQTT connection"""
    if not DEBUG:
        mqtt.init()
    else:
        train = {
            "command": "train",
            "params": {
                "dataset": "http://tokii.datasets.tips", 
                "target": "tip",
                "features": None,  # Este campo puede ser "null" o una lista de nombres de columnas
                "preprocessing": ["impute_numeric", "scale_numeric", "impute_categorical", "encode_categorical"],
                "recommendations": True,
                "algorithms": [
                    {"name": "KNeighborsRegressor", "params": {"n_neighbors": 3, "weights": "distance"}}, 
                    {"name": "SVR", "params": {"C": 3, "degree": 87}}
                ],
                "crossvalidation": 80
            }
        }
        
        predict = {
            "command": "predict",
            "params": {
                "model": "SVR.pkl",
                "features": {'total_bill': 18.53, 'sex': "Male", 'smoker':"No", 'day':"Sun", 'time':"Dinner", 'size':3},
                "preprocessing": ["impute_numeric", "scale_numeric", "impute_categorical", "encode_categorical"]
                }
        }
        mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(train)))

if __name__ == "__main__":
    init()
