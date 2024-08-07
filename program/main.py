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
        # "params": {
        #     "model": "SVC.pkl",
        #     "features": {'total_bill': 18.53, 'sex': "Male", 'smoker':"No", 'day':"Sun", 'time':"Dinner", 'size':3},
        #     "preprocessing": ["impute_numeric", "scale_numeric", "impute_categorical", "encode_categorical"]
        #     }
        "params": {
            "model": "RandomForestClassifier.pkl",  # Usaremos el RandomForestClassifier para la predicci\u00f3n
            "features": {
                "alcohol": 13.2,
                "malic_acid": 1.78,
                "ash": 2.14,
                "alcalinity_of_ash": 11.2,
                "magnesium": 100,
                "total_phenols": 2.65,
                "flavanoids": 2.76,
                "nonflavanoid_phenols": 0.26,
                "proanthocyanins": 1.28,
                "color_intensity": 4.38,
                "hue": 1.05,
                "od280/od315_of_diluted_wines": 3.4,
                "proline": 1050
            }
        }
    }
    
    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(train)))
    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(predict)))

if __name__ == "__main__":
    init()
