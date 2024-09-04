#modeloptimization.py
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from training.train import train_optimization
from data.datasetprocessing import select_best_features
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from utils.logger import logger
import time

def recommend_best_model(X_train, y_train, X_test, y_test, problem_type, trained_models=None, feature_names=None):
    """
    Recomienda el mejor modelo basado en los datos de entrenamiento y prueba.
    
    Args:
    X_train, y_train: Datos de entrenamiento
    X_test, y_test: Datos de prueba
    problem_type: Tipo de problema ('classification' o 'regression')
    trained_models: Lista de modelos ya entrenados (opcional)
    feature_names: Nombres de las características (opcional)
    
    Returns:
    tuple: Mejor modelo, características seleccionadas, X_train y X_test con las mejores características
    """
    logger.info("Iniciando recomendación del mejor modelo...")
    start_time = time.time()

    best_algorithms = select_best_algorithm(X_train, y_train, problem_type, trained_models, feature_names)
    
    best_model = None
    best_score = -np.inf
    best_selected_features = None
    
    logger.info("Seleccionando las mejores características...")
    X_train_selected, X_test_selected, selected_features = select_best_features(X_train, X_test, y_train, feature_names)
    
    logger.info(f"Características seleccionadas: {selected_features}")

    for name, score, model in best_algorithms:
        logger.info(f"Optimizando {name}...")
        optimized_model, X_train_opt, _ = train_optimization(X_train_selected, y_train, model.__class__, problem_type, selected_features, optimize_params=True)

        logger.info(f"Evaluando {name}...")
        if problem_type == 'classification':
            score = accuracy_score(y_test, optimized_model.predict(X_test_selected))
        else:
            score = r2_score(y_test, optimized_model.predict(X_test_selected))
        
        logger.info(f"Puntuación para {name}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = optimized_model
            best_selected_features = selected_features

    logger.info(f"\nMejor modelo recomendado: {best_model.__class__.__name__}")
    logger.info(f"Mejor puntuación: {best_score:.4f}")
    logger.info(f"Características seleccionadas: {best_selected_features}")

    end_time = time.time()
    logger.info(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")

    return best_model, best_selected_features, X_train_selected, X_test_selected, None, None

def select_best_algorithm(X, y, problem_type, trained_models=None, feature_names=None, n_algorithms=3):
    """
    Realiza una selección inteligente de los mejores algoritmos potenciales.
    
    Args:
    X (array-like): Características del dataset.
    y (array-like): Variable objetivo.
    problem_type (str): Tipo de problema ('classification' o 'regression').
    trained_models (list): Lista de modelos ya entrenados (opcional).
    feature_names (list): Nombres de las características (opcional).
    n_algorithms (int): Número de algoritmos a seleccionar.
    
    Returns:
    list: Lista de los mejores algoritmos seleccionados.
    """
    # Análisis preliminar de los datos
    n_samples, n_features = X.shape
    class_balance = np.unique(y, return_counts=True)[1] / len(y) if problem_type == 'classification' else None
    
    logger.info(f"Análisis preliminar del dataset:")
    logger.info(f"Número de muestras: {n_samples}")
    logger.info(f"Número de características: {n_features}")
    if class_balance is not None:
        logger.info(f"Balance de clases: {class_balance}")
    
    # Normalizar los datos para una comparación justa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Definir los algoritmos a probar basados en el tipo de problema
    algorithms = get_popular_algorithms(problem_type)
    scoring = make_scorer(accuracy_score if problem_type == 'classification' else r2_score)
    
    # Evaluar modelos ya entrenados
    results = []
    if trained_models:
        for model in trained_models:
            model_name = model.__class__.__name__
            try:
                score = scoring(model, X, y)
                results.append((model_name, score, model))
                logger.info(f"Puntuación para modelo pre-entrenado {model_name}: {score:.4f}")
            except Exception as e:
                logger.error(f"Error al evaluar el modelo pre-entrenado {model_name}: {str(e)}")
    
    # Realizar una validación cruzada rápida para los algoritmos restantes
    for name, model in algorithms:
        if name not in [r[0] for r in results]:
            try:
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring=scoring, n_jobs=-1)
                mean_score = np.mean(scores)
                results.append((name, mean_score, model))
                logger.info(f"Puntuación media para {name}: {mean_score:.4f}")
            except Exception as e:
                logger.error(f"Error al evaluar {name}: {str(e)}")
    
    # Ordenar los resultados y seleccionar los mejores
    results.sort(key=lambda x: x[1], reverse=True)
    best_algorithms = results[:n_algorithms]
    
    logger.info(f"\nLos mejores algoritmos seleccionados son:")
    for name, score, _ in best_algorithms:
        logger.info(f"{name}: {score:.4f}")
    
    return best_algorithms

def get_popular_algorithms(problem_type):
    """
    Retorna una lista de algoritmos populares basados en el tipo de problema.

    Args:
    problem_type (str): Tipo de problema de machine learning ('Clasificación' o 'Regresión')

    Returns:
    list: Lista de tuplas (nombre_algoritmo, clase_algoritmo)
    """
    problem_type = problem_type.lower()  # Convertir a minúsculas para hacer la comparación más robusta
    
    if problem_type == 'clasificación' or problem_type == 'clasificacion':
        return [
            ('RandomForestClassifier', RandomForestClassifier),
            ('GradientBoostingClassifier', GradientBoostingClassifier),
            ('LogisticRegression', LogisticRegression),
            ('SVC', SVC),
            ('DecisionTreeClassifier', DecisionTreeClassifier),
            ('KNeighborsClassifier', KNeighborsClassifier)
        ]
    elif problem_type == 'regresión' or problem_type == 'regresion':
        return [
            ('RandomForestRegressor', RandomForestRegressor),
            ('GradientBoostingRegressor', GradientBoostingRegressor),
            ('LinearRegression', LinearRegression),
            ('SVR', SVR),
            ('DecisionTreeRegressor', DecisionTreeRegressor),
            ('KNeighborsRegressor', KNeighborsRegressor)
        ]
    else:
        raise ValueError(f"Tipo de problema no reconocido: {problem_type}. Debe ser 'Clasificación' o 'Regresión'.")
        
def apply_transfer_learning(best_model, trained_models, X_train, y_train, X_test, feature_names, selected_feature_names):
    if not trained_models:
        return best_model, X_train, X_test, selected_feature_names

    def preprocess_features(X, features):
        if isinstance(X, pd.DataFrame):
            return X[features]
        elif isinstance(X, np.ndarray):
            return X[:, [feature_names.index(f) for f in features]]
        else:
            raise ValueError("X debe ser un DataFrame de pandas o un array de numpy")

    X_train_selected = preprocess_features(X_train, selected_feature_names)
    X_test_selected = preprocess_features(X_test, selected_feature_names)

    # Crear un ensamble de modelos
    train_predictions = []
    test_predictions = []
    for model in trained_models:
        try:
            model_features = [f for f in model.feature_names_ if f in selected_feature_names]
            train_pred = model.predict(preprocess_features(X_train, model_features))
            test_pred = model.predict(preprocess_features(X_test, model_features))
            train_predictions.append(train_pred)
            test_predictions.append(test_pred)
        except Exception as e:
            print(f"Error al predecir con el modelo {model.__class__.__name__}: {str(e)}")
    
    if not train_predictions or not test_predictions:
        print("No se pudieron hacer predicciones con los modelos entrenados. Retornando el mejor modelo sin cambios.")
        return best_model, X_train_selected, X_test_selected, selected_feature_names

    train_ensemble_predictions = np.mean(train_predictions, axis=0)
    test_ensemble_predictions = np.mean(test_predictions, axis=0)

    # Combinar las predicciones del ensamble con las características seleccionadas
    X_train_enhanced = np.column_stack((X_train_selected, train_ensemble_predictions.reshape(-1, 1)))
    X_test_enhanced = np.column_stack((X_test_selected, test_ensemble_predictions.reshape(-1, 1)))

    # Crear nuevos nombres de características
    enhanced_feature_names = selected_feature_names + ['ensemble_prediction']

    print(f"Características mejoradas: {enhanced_feature_names}")
    print(f"Forma de X_train_enhanced: {X_train_enhanced.shape}")
    print(f"Forma de X_test_enhanced: {X_test_enhanced.shape}")

    # Entrenar el mejor modelo con los datos mejorados
    best_model_enhanced = clone(best_model)
    best_model_enhanced.fit(X_train_enhanced, y_train)