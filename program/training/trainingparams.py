#trainingparams.py
# from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.estimator_checks import parametrize_with_checks

import training.serializer as serializer
from typing import List

def train_recommendedparams(X_train, y_train, algorithms: List[dict]):
    recommended_params = []
    n_samples, n_features = X_train.shape
    if n_samples > 10000 or n_features > 50:
        n_iter = 25
        param_factor = 0.5
    else:
        n_iter = 50
        param_factor = 1.0
    
    for algorithm in algorithms:
        if isinstance(algorithm, dict) and 'name' in algorithm:
            algorithm_name = algorithm['name']
            if algorithm_name in serializer.model_classes:
                model_class = serializer.model_classes[algorithm_name]
                
                # Estimaci\u00f3n de par\u00e1metros iniciales
                estimator, _ = parametrize_with_checks(model_class)
                estimator.fit(X_train, y_train)
                initial_params = estimator.get_params()

                # Definir el espacio de b\u00fasqueda para BayesSearchCV
                param_distributions = get_default_params(model_class, X_train, y_train, param_factor)

                # Realizar la b\u00fasqueda bayesiana de hiperpar\u00e1metros
                opt = BayesSearchCV(model_class(), param_distributions, n_iter=n_iter, cv=StratifiedKFold(n_splits=5), random_state=42)
                opt.fit(X_train, y_train)
                best_params = opt.best_params_
                recommended_params.append({'name': algorithm_name, 'initial_params': initial_params, 'params': best_params})
            else:
                print(f"El algoritmo {algorithm_name} no est\u00e1 en serializer.model_classes")
        else:
            print(f"Formato de algoritmo incorrecto: {algorithm}")
    
    return recommended_params

def get_default_params(model_class, X_train, y_train, param_factor):
    default_params = {}
    # Utilizamos parametrize_with_checks para estimar los valores de los parámetros iniciales
    estimator, _ = parametrize_with_checks(model_class)
    estimator.fit(X_train, y_train)
    params_dict = estimator.get_params()
    for param, value in params_dict.items():
        if isinstance(value, int) or isinstance(value, float):
            # Si el parámetro es un número, definimos un rango de búsqueda en función del valor estimado y el factor de reducción
            default_params[param] = (value * param_factor / 10, value * param_factor * 10)  # Definimos un rango de búsqueda
        elif isinstance(value, str):
            # Si el parámetro es una cadena, dejamos que el modelo elija entre algunos valores predefinidos
            default_params[param] = [value]  # Definimos una lista de valores posibles
        else:
            # Otros tipos de parámetros (como booleanos) pueden no necesitar optimización
            pass
    return default_params
