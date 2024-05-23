import sys
import os
import json
import joblib
from typing import List, Dict, Union

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import serializer

# Función para entrenar los modelos
def train_models(X_train, y_train, model_params: List[Dict]):
    trained_models = []
    for model_info in model_params:
        model_name = model_info['name']
        params = model_info['params']
        if model_name in serializer.model_classes:
            model_class = serializer.model_classes[model_name]
            model = model_class(**params)  # Instantiate model with provided hyperparameters
            model.fit(X_train, y_train)  # Train the model
            
            serializer.to_pickle(model, model_name)
            trained_models.append(model)
        else:
            print(f"Model '{model_name}' not found. Skipping...")
    return trained_models

# Función para evaluar los modelos
def evaluate_models(trained_models: List[serializer.ScikitModel], X_test, y_test):
    evaluation_results = {}
    for model in trained_models:
        plt.clf()
        y_pred = model.predict(X_test)        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
        
        evaluation_results[str(model)] = {"accuracy": accuracy, "f1_score": f1, "confusion_matrix": cm}
    return evaluation_results

# Funcion para comparar los modelos entrenados
def compare_models(dirpath: Union[os.PathLike, str], save_report: bool = False) -> Dict:
    """Load and evaluate models"""
    model_filenames = os.listdir(dirpath)
    models = []
    for filename in model_filenames:
        if filename.endswith(".pkl"):
            model = joblib.load(open(os.path.join(dirpath, filename), "rb"))
            models.append(model)
        elif filename.endswith(".h5"):
            model = load_model(os.path.join(dirpath, filename))
            models.append(model)

    if not models:
        print("No trained models found.")
        sys.exit(0)

    for model in models:
        print(f"Loaded model {model}")

    # load evaluation data
    X_train, X_test, y_train, y_test = load_split_dataset(os.path.join("data", "covtype.data"))

    # generate report
    report = {}
    for model in models:
        report[str(model)] = evaluate_models([model], X_train, X_test)
    if save_report:
        json.dump(report, open("report.json", "a"))
    return report