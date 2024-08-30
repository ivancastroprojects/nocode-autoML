#train.py
import sys
import os
import joblib
from typing import Dict, Union

from training.scikitdb import serializer
from training.trainoptimizations import train_with_important_features

# Función para entrenar los modelos
def train_model(X_train, y_train, algorithm, problem_type, dataset_name, feature_names, recommendations=False):
    model_class = serializer.model_classes[algorithm['name']]
    base_model = model_class(**algorithm.get('params', {}))
    base_model.fit(X_train, y_train)
    
    optimized_model = None
    if recommendations:
        # Buscar los mejores parámetros para el modelo
        param_grid = algorithm.get('param_grid', {})
        if param_grid:
            grid_search = GridSearchCV(model_class(), param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            optimized_model = grid_search.best_estimator_
        else:
            optimized_model = base_model
    
    return base_model, optimized_model

# Función para comparar los modelos entrenados
def compare_models(base_model, optimized_model, X_test, y_test):
    base_score = base_model.score(X_test, y_test)
    optimized_score = optimized_model.score(X_test, y_test)
    
    print(f"Comparación de modelos:")
    print(f"Base model score: {base_score}")
    print(f"Optimized model score: {optimized_score}")
    print(f"Improvement: {(optimized_score - base_score) / base_score * 100:.2f}%")

def recommend_best_model(X_train, y_train, X_test, y_test, problem_type, feature_names):
    # Seleccionar las mejores características
    selector = SelectKBest(k=10)
    X_new = selector.fit_transform(X_train, y_train)
    selected_features = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_features]
    
    # Probar diferentes algoritmos
    algorithms = serializer.classification_models if problem_type == 'classification' else serializer.regression_models
    best_model = None
    best_score = -np.inf
    
    for algorithm in algorithms:
        model = algorithm()
        model.fit(X_new, y_train)
        score = model.score(selector.transform(X_test), y_test)
        if score > best_score:
            best_score = score
            best_model = model
    
    # Optimizar los parámetros del mejor modelo
    param_grid = get_param_grid(best_model)
    grid_search = GridSearchCV(best_model.__class__(), param_grid, cv=5)
    grid_search.fit(X_new, y_train)
    
    final_model = grid_search.best_estimator_
    
    print(f"Mejor modelo recomendado: {final_model.__class__.__name__}")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor puntuación: {grid_search.best_score_}")
    
    return final_model