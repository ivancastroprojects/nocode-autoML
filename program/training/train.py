#train.py
import random
import sys
import os
import joblib
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from training import serializer

from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, median_absolute_error, r2_score, mean_absolute_error, explained_variance_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# Función para entrenar los modelos
def train_models(X_train, y_train, model_params: List[Dict], problem_type, dataset_name: str):
    trained_models = []
    for model_info in model_params:
        model_name = model_info['name']
        params = model_info['params']
        if model_name in serializer.model_classes:
            model_class = serializer.model_classes[model_name]
            model = model_class(**params)  # Instanciar modelo con hiperparámetros proporcionados
            # Comprobar que este algoritmo encaja con el tipo de problema según el tipo de dataset
            if (problem_type == 'classification' and isinstance(model,tuple(serializer.classification_models)) or (problem_type == 'regression' and isinstance(model,tuple(serializer.regression_models)))):
                model.fit(X_train, y_train)  # Entrenar el modelo
                serializer.to_pickle(model, model_name, dataset_name)
                trained_models.append(model)
        else:
            print(f"Model '{model_name}' not found. Skipping...")
    return trained_models

# Funci\u00f3n para evaluar modelos de clasificaci\u00f3n
def evaluate_classification_models(models: List, X_test, y_test, target, features):
    evaluation_results = {}

    for model in models:
        X_test_array = np.array(X_test)
        y_pred = model.predict(X_test)
        
        print(f"Tres muestras aleatorias para el modelo {model.__class__.__name__}:\n")
        
        # Seleccionar 3 \u00edndices aleatorios
        random_indices = random.sample(range(len(X_test_array)), 3)
        
        for i in random_indices:
            print(f"Index: {i}")
            print(f"Actual target: {y_test[i]}")
            print(f"Prediction: {y_pred[i]}")
            print("Feature values:")
            for j, feature in enumerate(features):
                if j < X_test_array.shape[1]:
                    print(f"  {feature}: {X_test_array[i][j]}")
                else:
                    print(f"  {feature}: Feature index out of bounds")
            print("\n")
        
        # Calcular m\u00e9tricas de evaluaci\u00f3n
        cv_score = np.mean(cross_val_score(model, X_test, y_test, cv=5))
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Imprimir el reporte de clasificaci\u00f3n
        print(classification_report(y_test, y_pred, target_names=[str(t) for t in np.unique(y_test)]))
        
        # Matriz de confusi\u00f3n
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f'Matriz de Confusi\u00f3n - {model.__class__.__name__}')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        cm_path = f'confusion_matrix_{model.__class__.__name__}.png'
        plt.savefig(cm_path)
        plt.close()
        
        evaluation_results[model.__class__.__name__] = {
            "accuracy": accuracy,
            "cross_validation_score": cv_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix_path": cm_path
        }
    
    return evaluation_results

def evaluate_regression_models(trained_models: List[serializer.ScikitModel], X_test, y_test, target, features):
    evaluation_results = {}
    for model in trained_models:
        y_pred = model.predict(X_test)
        
        print(f"Primeras tres predicciones para el modelo {model.__class__.__name__}:\n")
        for i in range(3):
            # Crear un DataFrame con las características y sus valores
            sample_df = pd.DataFrame([X_test[i]], columns=features)
            
            print(f"Muestra {i+1}:")
            print(sample_df.to_string(index=False))
            print(f"\nValor real: {y_test[i]:.4f}")
            print(f"Valor predicho: {y_pred[i]:.4f}")
            print(f"Diferencia: {abs(y_test[i] - y_pred[i]):.4f}")
            print("-" * 50)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        
        cv_scores_r2 = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
        cv_scores_neg_mse = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
        
        # Crear gráfico de dispersión
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title(f'Valores Reales vs Predicciones - {model.__class__.__name__}')
        scatter_plot_path = f'scatter_plot_{model.__class__.__name__}.png'
        plt.savefig(scatter_plot_path)
        plt.close()
        
        evaluation_results[model.__class__.__name__] = {
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "mean_absolute_error": mae,
            "median_absolute_error": medae,
            "r2_score": r2,
            "explained_variance_score": evs,
            "cv_r2_score_mean": np.mean(cv_scores_r2),
            "cv_r2_score_std": np.std(cv_scores_r2),
            "cv_scores_neg_mse": np.std(cv_scores_neg_mse),
            "scatter_plot_path": scatter_plot_path
        }
    return evaluation_results

# Función para seleccionar automáticamente las características más relevantes
def select_features(X, y, model, k=10):
    if isinstance(model, tuple(serializer.classification_models)):
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)
    
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features

# Función para comparar los modelos entrenados
def compare_models(dirpath: Union[os.PathLike, str], save_report: bool = False) -> Dict:
    model_filenames = os.listdir(dirpath)
    models = []
    for filename in model_filenames:
        if filename.endswith(".pkl"):
            model = joblib.load(open(os.path.join(dirpath, filename), "rb"))
        elif filename.endswith(".h5"):
            model = joblib.load(os.path.join(dirpath, filename))
        
        models.append(model)

    if not models:
        print("No trained models found.")
        sys.exit(0)

    for model in models:
        print(f"Loaded model {model}")