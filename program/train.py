#train.py
import sys
import os
import joblib
from typing import List, Dict, Union
import serializer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, median_absolute_error, r2_score, mean_absolute_error, explained_variance_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# Función para entrenar los modelos
def train_models(X_train, y_train, model_params: List[Dict], problem_type):
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
                serializer.to_pickle(model, model_name)
                trained_models.append(model)
        else:
            print(f"Model '{model_name}' not found. Skipping...")
    return trained_models

# Función para evaluar modelos de clasificación
def evaluate_classification_models(models: List[serializer.ScikitModel], X_test, y_test):
    evaluation_results = {}
    for model in models:
        y_pred = model.predict(X_test)
        
        # Common metrics
        cv_score = np.mean(cross_val_score(model, X_test, y_test, cv=5))
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {str(model)}')
        plt.ylabel('Valores Reales')
        plt.xlabel('Predicciones')
        cm_path = f'confusion_matrix_{str(model)}.png'
        plt.savefig(cm_path)
        plt.close()
        
        evaluation_results[str(model)] = {
            "cross_validation_score": cv_score,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix_path": cm_path
        }    
    return evaluation_results

# Función para evaluar modelos de regresión
def evaluate_regression_models(trained_models: List[serializer.ScikitModel], X_test, y_test):
    evaluation_results = {}
    for model in trained_models:
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        
        cv_scores_r2 = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
        cv_scores_neg_mse = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
        
        # Crear gr\u00e1fico de dispersi\u00f3n
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Valores Reales vs Predicciones')
        scatter_plot_path = f'scatter_plot_{str(model.__class__.__name__)}.png'
        plt.savefig(scatter_plot_path)
        plt.close()
        
        evaluation_results[str(model)] = {
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "mean_absolute_error": mae,
            "median_absolute_error": medae,
            "r2_score": r2,
            "explained_variance_score": evs,
            "cv_r2_score_mean": np.mean(cv_scores_r2),
            "cv_r2_score_std": np.std(cv_scores_r2),
            "cv_scores_neg_mse": np.std(cv_scores_neg_mse),
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