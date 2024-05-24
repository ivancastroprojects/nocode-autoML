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
        mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(
            {
                "command": "train",
                "params": {
                    "dataset": "http://tokii.datasets.iris", 
                    "target": "Species",  
                    "preprocessing": {}, 
                    "algorithms": [
                        {"name": "KNeighborsRegressor", "params": {"n_neighbors": 3, "weights": "distance"}}, 
                        {"name": "SVC", "params": {"C": 3, "degree": 87}}
                    ],
                    "crossvalidation": 80
                }
            }
        )))
    
def process_data(type: str):
    """Process data based on the given type"""
    trainingInstance: Training = global_data.training

    if type == "dataset":
        trainingInstance.dataset.eda_generico()
        trainingInstance.train_and_evaluate()
    elif type == "model":
        trainingInstance.train_and_evaluate()
    elif type == "predict":
        trainingInstance.predict_and_evaluate()
    else:
        raise ValueError("Invalid type")

if __name__ == "__main__":
    init()
