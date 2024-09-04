#train.py

import numpy as np
import pandas as pd
from training.paramoptimization import get_optimized_params, get_param_grid
import training.scikitdb.serializer as serializer
from sklearn.base import BaseEstimator
import inspect
from sklearn.model_selection import GridSearchCV
from utils.logger import logger

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
        print(f"Modelo {model_name} no soportado. Omitiendo...")
        return None
    
    model_class = serializer.model_classes[model_name]
    
    if params is None:
        model = model_class()
    else:
        # Filtrar los parámetros válidos para este modelo
        valid_params = {}
        for param, value in params.items():
            if param in inspect.signature(model_class.__init__).parameters:
                valid_params[param] = value
            else:
                print(f"Advertencia: El parámetro '{param}' no es válido para {model_name}. Se ignorará.")
        
        model = model_class(**valid_params)
    
    if isinstance(model, BaseEstimator):
        model.fit(X, y)
        return model
    else:
        print(f"Advertencia: {model_name} no es un estimador válido de scikit-learn. No se pudo entrenar.")
        return None

# Función para entrenar los modelos
def train_custom_models(X_train, y_train, algorithm, problem_type, dataset_path, feature_names, recommendations=True):
    model_name = algorithm["name"]
    params = algorithm["params"]
    
    try:
        # Asegurarse de que X_train es un DataFrame con las columnas correctas
        X_train = pd.DataFrame(X_train, columns=feature_names)
        
        base_model = train_simple_model(X_train, y_train, model_name, params)
        
        if recommendations:
            param_grid = get_param_grid(base_model)
            optimized_model = optimize_model(X_train, y_train, base_model, param_grid)
        else:
            optimized_model = base_model

        selected_features = feature_names  # Por ahora, usamos todas las características

        return base_model, optimized_model, selected_features
    except Exception as e:
        logger.error(f"Error en train_custom_models para {model_name}: {str(e)}")
        return None, None, None

def optimize_model(X, y, model, param_grid, cv=5):
    try:
        # Asegurarse de que X es un DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Verificar que las columnas existen
        if not all(col in X.columns for col in X.columns):
            raise ValueError(f"Columnas no encontradas en X: {X.columns}")

        grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=0)
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error durante la optimización del modelo: {str(e)}")
        return model  # Devolver el modelo original si hay un error

def train_optimization(X_train, y_train, model_class, problem_type, selected_features, params=None, optimize_params=True):
    """
    Entrena un modelo optimizado o con parámetros específicos.
    
    Args:
    X_train (array-like): Características de entrenamiento.
    y_train (array-like): Etiquetas de entrenamiento.
    model_class (class): Clase del modelo a entrenar.
    problem_type (str): Tipo de problema ('classification' o 'regression').
    selected_features (list): Lista de características seleccionadas.
    params (dict): Parámetros específicos del modelo (opcional).
    optimize_params (bool): Si se deben optimizar los parámetros o usar los proporcionados.

    Returns:
    tuple: (modelo optimizado, X_train seleccionado, características seleccionadas)
    """
    # Eliminar 'target' de selected_features si está presente
    selected_features = [f for f in selected_features if f != 'target']

    if isinstance(X_train, pd.DataFrame):
        X_new = X_train[selected_features]
    elif isinstance(X_train, np.ndarray):
        # Asumimos que las características están en el mismo orden que en selected_features
        X_new = X_train
    else:
        raise ValueError("X_train debe ser un DataFrame de pandas o un array de numpy")
    
    if optimize_params:
        best_params = get_optimized_params(model_class(), X_new, y_train, 'random')
    else:
        best_params = params if params is not None else {}
    
    optimized_model = model_class(**best_params)
    optimized_model.fit(X_new, y_train)
    
    return optimized_model, X_new, selected_features