#train.py

import numpy as np
import pandas as pd

from data.datasetprocessing import select_best_features
from training.paramoptimization import get_optimized_params
import training.scikitdb.serializer as serializer

def train_simple_model(X, y, model_name, params=None):
    """
    Entrena un modelo simple con los parámetros dados.
    
    Args:
    X (array-like): Características de entrenamiento.
    y (array-like): Etiquetas de entrenamiento.
    model_name (str): Nombre del modelo a entrenar.
    params (dict): Parámetros para el modelo. Si es None, se usarán los parámetros por defecto.
    
    Returns:
    object: Modelo entrenado.
    """
    if model_name not in serializer.model_classes:
        raise ValueError(f"Modelo {model_name} no soportado.")
    
    model_class = serializer.model_classes[model_name]
    
    if params is None:
        model = model_class()
    else:
        model = model_class(**params)
    
    model.fit(X, y)
    return model

# Función para entrenar los modelos
def train_custom_models(X_train, y_train, algorithm, problem_type, dataset_name, feature_names, recommendations=False):
    model_name = algorithm['name']
    if model_name not in serializer.model_classes:
        print(f"Modelo '{model_name}' no encontrado. Omitiendo...")
        return None, None, None

    model_class = serializer.model_classes[model_name]
    
    # Entrenamiento del modelo base con todas las características
    base_model = train_simple_model(X_train, y_train, model_name, algorithm.get('params', {}))
    print(f"Parámetros seleccionados por el usuario para {algorithm['name']}: {algorithm.get('params', {})}")

    # Entrenamiento del modelo optimizado con características seleccionadas
    if recommendations:
        optimized_model, X_new, selected_feature_names = train_optimization(X_train, y_train, model_class, problem_type, feature_names)

    return base_model, optimized_model, selected_feature_names

def recommend_best_model(X_train, y_train, X_test, y_test, problem_type, feature_names):
    algorithms = serializer.classification_models if problem_type == 'classification' else serializer.regression_models
    best_model = None
    best_score = -np.inf
    
    for algorithm in algorithms:
        model_class = algorithm  # Asumiendo que algorithms contiene las clases de modelo, no instancias
        optimized_model, X_new, selected_feature_names = train_optimization(X_train, y_train, model_class, problem_type, feature_names)
        
        # Seleccionar las características correctas para X_test
        if isinstance(X_test, pd.DataFrame):
            X_test_selected = X_test[selected_feature_names]
        else:
            feature_indices = [feature_names.index(feature) for feature in selected_feature_names]
            X_test_selected = X_test[:, feature_indices]
        
        score = optimized_model.score(X_test_selected, y_test)
        if score > best_score:
            best_score = score
            best_model = optimized_model
    
    if best_model is not None:
        print(f"Mejor modelo recomendado: {best_model.__class__.__name__}")
        print(f"Mejor puntuación: {best_score}")
    else:
        print("No se pudo recomendar ningún modelo.")
    
    return best_model

def train_optimization(X_train, y_train, model, problem_type, feature_names, k=10, optimization_method='random'):
    """
    Entrena un modelo utilizando las características más importantes y parámetros optimizados.
    """
    X_new, selected_feature_names = select_best_features(X_train, y_train, problem_type, feature_names, k=k)
    
    if isinstance(model, type):
        model_name = model.__name__
        model = model()
    else:
        model_name = model.__class__.__name__
    
    best_params = get_optimized_params(model, X_new, y_train, optimization_method)
    
    from training.train import train_simple_model
    optimized_model = train_simple_model(X_new, y_train, model_name, best_params)
    
    return optimized_model, X_new, selected_feature_names

# Función para comparar los modelos entrenados
def compare_models(base_model, optimized_model, X_test, y_test):
    base_score = base_model.score(X_test, y_test)
    optimized_score = optimized_model.score(X_test, y_test)
    
    print(f"Comparación de modelos:")
    print(f"Base model score: {base_score}")
    print(f"Optimized model score: {optimized_score}")
    print(f"Improvement: {(optimized_score - base_score) / base_score * 100:.2f}%")