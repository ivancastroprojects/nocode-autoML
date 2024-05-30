# train.py
import sys
import os
import json
import joblib
from typing import List, Dict, Union
import serializer

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_squared_error, r2_score

# Función para entrenar los modelos
def train_models(X_train, y_train, model_params: List[Dict]):
    trained_models = []
    for model_info in model_params:
        model_name = model_info['name']
        params = model_info['params']
        if model_name in serializer.model_classes:
            model_class = serializer.model_classes[model_name]
            model = model_class(**params)  # Instanciar modelo con hiperparámetros proporcionados
            model.fit(X_train, y_train)  # Entrenar el modelo
            
            serializer.to_pickle(model, model_name)
            trained_models.append(model)
        else:
            print(f"Model '{model_name}' not found. Skipping...")
    return trained_models


# Función para evaluar modelos de clasificación
def evaluate_classification_models(trained_models: List[serializer.ScikitModel], X_test, y_test):
    evaluation_results = {}
    for model in trained_models:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
        
        evaluation_results[str(model)] = {"accuracy": accuracy, "f1_score": f1, "confusion_matrix": cm, "classification_report": report}
    return evaluation_results

# Función para evaluar modelos de regresión
def evaluate_regression_models(trained_models: List[serializer.ScikitModel], X_test, y_test):
    evaluation_results = {}
    for model in trained_models:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        plt.show()
        
        evaluation_results[str(model)] = {"mean_squared_error": mse, "r2_score": r2}
    return evaluation_results

# Función para comparar los modelos entrenados
def compare_models(dirpath: Union[os.PathLike, str], save_report: bool = False) -> Dict:
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
        if isinstance(model, (serializer.classification_models)):
            report[str(model)] = evaluate_classification_models([model], X_test, y_test)
        else:
            report[str(model)] = evaluate_regression_models([model], X_test, y_test)
    
    if save_report:
        json.dump(report, open("report.json", "a"))
    return report