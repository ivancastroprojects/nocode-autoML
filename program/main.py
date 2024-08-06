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
            "dataset": "http://tokii.datasets.wine",
            "target": "target",
            "features": None,
            "preprocessing": ["impute_numeric", "scale_numeric", "impute_categorical", "encode_categorical"],
            "recommendations": True,
            "algorithms": [
                # Algoritmos de regresion
                {"name": "LinearRegression", "params": {}},
                {"name": "Ridge", "params": {"alpha": 1.0}},
                {"name": "Lasso", "params": {"alpha": 1.0}},
                {"name": "SVR", "params": {"C": 1.0, "kernel": "rbf"}},
                {"name": "KNeighborsRegressor", "params": {"n_neighbors": 5, "weights": "uniform"}},
                {"name": "DecisionTreeRegressor", "params": {"max_depth": 5}},
                {"name": "RandomForestRegressor", "params": {"n_estimators": 100, "max_depth": 5}},
                {"name": "GradientBoostingRegressor", "params": {"n_estimators": 100, "learning_rate": 0.1}},
                {"name": "XGBRegressor", "params": {"n_estimators": 100, "learning_rate": 0.1}},
                
                # Algoritmos de clasificacion
                {"name": "LogisticRegression", "params": {"C": 1.0}},
                {"name": "SVC", "params": {"C": 1.0, "kernel": "rbf"}},
                {"name": "KNeighborsClassifier", "params": {"n_neighbors": 5, "weights": "uniform"}},
                {"name": "DecisionTreeClassifier", "params": {"max_depth": 5}},
                {"name": "RandomForestClassifier", "params": {"n_estimators": 100, "max_depth": 5}},
                {"name": "GradientBoostingClassifier", "params": {"n_estimators": 100, "learning_rate": 0.1}},
                {"name": "XGBClassifier", "params": {"n_estimators": 100, "learning_rate": 0.1}}
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
