# main.py
from training.training import Training
from api import mqtt
from data import global_data
import json
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

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
        dataset = {
            "command": "dataset",
            "data":
            {
                "dataset": {
                    "path": "http://tokii.datasets.breast_cancer",
                    "target": "target",
                    "features": None, #["mean smoothness", "worst fractal dimension", "fractal dimension error", "symmetry error"],
                    "class_labels": None,
                    "categorical_features": None,
                    "numeric_features": None,
                    "datetime_features": None,
                    "text_features": None
                },
                "processing": {
                    "outlier_columns": None,
                    "imputation_strategy": None, #simple, median, knn, iterative
                    "scaling_strategy": None, #standar, robust, minmax
                    "encoding_strategy": None, #onehot, ordinal
                    "handle_outliers_strategy": None
                }
            }
        }
        
        train = {
            "command": "train",
            "data":
            {
                # Check if still we have this dataset processed
                # If at final of name have "_opt" we take optimized df version
                "dataset": "http://tokii.datasets.breast_cancer",
                "crossvalidation": 80,
                "recommendations": True,
                "algorithms": [
                    # Algoritmos de regresion
                    # {"name": "LinearRegression", "params": {}},
                    # {"name": "Ridge", "params": {"alpha": 1.0}},
                    # {"name": "Lasso", "params": {"alpha": 1.0}},
                    # {"name": "SVR", "params": {"C": 1.0, "kernel": "rbf"}},
                    # {"name": "KNeighborsRegressor", "params": {"n_ghneighbors": 5, "weights": "uniform"}},
                    # {"name": "DecisionTreeRegressor", "params": {"max_depth": 5}},
                    # {"name": "RandomForestRegressor", "params": {"n_estimators": 100, "max_depth": 5}},
                    # {"name": "GradientBoostingRegressor", "params": {"n_estimators": 100, "learning_rate": 0.1}},
                    # {"name": "XGBRegressor", "params": {"n_estimators": 100, "learning_rate": 0.1}},
                    
                    # Algoritmos de clasificacion
                    # {"name": "LogisticRegression", "params": {"C": 1.0}},
                    # {"name": "SVC", "params": {"C": 1.0, "kernel": "linear"}},
                    {"name": "KNeighborsClassifier", "params": {"n_neighbors": 5, "weights": "uniform"}},
                    # {"name": "DecisionTreeClassifier", "params": {"max_depth": 5}},
                    # {"name": "RandomForestClassifier", "params": {"n_estimators": 100, "max_depth": 5}},
                    # {"name": "GradientBoostingClassifier", "params": {"n_estimators": 100, "learning_rate": 0.1}},
                    # {"name": "XGBClassifier", "params": {"n_estimators": 100, "learning_rate": 0.1}}
                ]
            }
        }
              
        predict = {
            "command": "predict",
            "data":
            {
                "model": "KNeighborsClassifier_base_95_60_breast_cancer_basic.pkl",
                "features": { #cancer #1.0
                    "mean radius": -0.6780247444904469,
                    "mean texture": -0.7302624354517607,
                    "mean perimeter": -0.647286764643,
                    "mean area": -0.8330137427213203,
                    "mean smoothness": -0.549131852352993,
                    "mean compactness": -0.6127965119160192,
                    "mean concavity": -0.6720046347024191,
                    "mean concave points": -0.6097902605053337,
                    "mean symmetry": -0.9827174429469521,
                    "mean fractal dimension": -0.48277789525302706,
                }
                # "features": { #tips
                #     'total_bill': 18.53, 
                #     'sex': "Male", 
                #     'smoker':"No", 
                #     'day':"Sun", 
                #     'time':"Dinner", 
                #     'size':3},
                # }
                # "features": { #wine
                #     "alcohol": 13.2,
                #     "malic_acid": 1.78,
                #     "ash": 2.14,
                #     "alcalinity_of_ash": 11.2,
                #     "magnesium": 100,
                #     "total_phenols": 2.65,
                #     "flavanoids": 2.76,
                #     "nonflavanoid_phenols": 0.26,
                #     "proanthocyanins": 1.28,
                #     "color_intensity": 4.38,
                #     "hue": 1.05,
                #     "od280/od315_of_diluted_wines": 3.4,
                #     "proline": 1050
                # }
                # "features": { #iris #setosa
                #     "SepalLengthCm": 5.1,
                #     "SepalWidthCm": 3.5,
                #     "PetalLengthCm": 1.4,
                #     "PetalWidthCm": 0.2,
                # }
                # "features": { #versicolor
                #     "SepalLengthCm": 6.7,
                #     "SepalWidthCm": 3.1,
                #     "PetalLengthCm": 4.7,
                #     "PetalWidthCm": 1.5,
                # }
                # "features": { #virginica
                #     "SepalLengthCm": 7.2,
                #     "SepalWidthCm": 3.6,
                #     "PetalLengthCm": 6.1,
                #     "PetalWidthCm": 2.5,
                # }
            }
        }

    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(dataset)))
    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(train)))
    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(predict)))

if __name__ == "__main__":
    init()
