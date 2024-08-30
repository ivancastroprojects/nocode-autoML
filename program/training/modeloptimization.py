#modeloptimization.py

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from training.train import train_optimization

def recommend_best_model(X_train, y_train, X_test, y_test, problem_type, trained_models=None, feature_names=None):
    """
    Recomienda el mejor modelo basado en una selección inteligente, optimización y transfer learning.
    
    Args:
    X_train, y_train, X_test, y_test: Datos de entrenamiento y prueba.
    problem_type (str): Tipo de problema ('classification' o 'regression').
    trained_models (list): Lista de modelos ya entrenados.
    feature_names (list): Nombres de las características.
    
    Returns:
    object: El mejor modelo recomendado.
    """
    best_algorithms = select_best_algorithm(X_train, y_train, problem_type, trained_models, feature_names)
    
    best_model = None
    best_score = -np.inf
    
    for name, score, model in best_algorithms:
        optimized_model, X_new, selected_feature_names = train_optimization(X_train, y_train, model.__class__, problem_type, feature_names)
        
        # Seleccionar las características correctas para X_test
        if isinstance(X_test, pd.DataFrame):
            X_test_selected = X_test[selected_feature_names]
        else:
            feature_indices = [feature_names.index(feature) for feature in selected_feature_names]
            X_test_selected = X_test[:, feature_indices]
        
        # Evaluar el modelo optimizado
        if problem_type == 'classification':
            score = accuracy_score(y_test, optimized_model.predict(X_test_selected))
        else:
            score = r2_score(y_test, optimized_model.predict(X_test_selected))
        
        if score > best_score:
            best_score = score
            best_model = optimized_model
    
    if best_model is not None:
        print(f"\nMejor modelo recomendado: {best_model.__class__.__name__}")
        print(f"Mejor puntuación: {best_score:.4f}")
        
        # Filtrar 'target' de feature_names si está presente
        feature_names_without_target = [f for f in feature_names if f != 'target']
        
        # Aplicar transfer learning
        best_model_enhanced, X_train_enhanced, X_test_enhanced = apply_transfer_learning(
            best_model, trained_models, X_train, y_train, X_test, feature_names_without_target
        )
        
        # Evaluar el modelo mejorado
        if problem_type == 'classification':
            enhanced_score = accuracy_score(y_test, best_model_enhanced.predict(X_test_enhanced))
        else:
            enhanced_score = r2_score(y_test, best_model_enhanced.predict(X_test_enhanced))
        
        print(f"Puntuación después de transfer learning: {enhanced_score:.4f}")
        
        if enhanced_score > best_score:
            best_model = best_model_enhanced
            best_score = enhanced_score
            print("Se ha adoptado el modelo mejorado con transfer learning.")
        else:
            print("El modelo original superó al modelo con transfer learning.")
    else:
        print("No se pudo recomendar ningún modelo.")
    
    return best_model

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
    
    print(f"Análisis preliminar del dataset:")
    print(f"Número de muestras: {n_samples}")
    print(f"Número de características: {n_features}")
    if class_balance is not None:
        print(f"Balance de clases: {class_balance}")
    
    # Normalizar los datos para una comparación justa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Definir los algoritmos a probar basados en el tipo de problema
    if problem_type == 'classification':
        algorithms = [
            ('RandomForestClassifier', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('LogisticRegression', LogisticRegression(random_state=42)),
            ('SVC', SVC(random_state=42)),
            ('KNeighborsClassifier', KNeighborsClassifier()),
            ('GaussianNB', GaussianNB()),
            ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=42))
        ]
        scoring = make_scorer(accuracy_score)
    else:  # regression
        algorithms = [
            ('RandomForestRegressor', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('LinearRegression', LinearRegression()),
            ('SVR', SVR()),
            ('KNeighborsRegressor', KNeighborsRegressor()),
            ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42))
        ]
        scoring = make_scorer(r2_score)
    
    # Evaluar modelos ya entrenados
    results = []
    if trained_models:
        for model in trained_models:
            model_name = model.__class__.__name__
            try:
                score = scoring(model, X, y)
                results.append((model_name, score, model))
                print(f"Puntuación para modelo pre-entrenado {model_name}: {score:.4f}")
            except Exception as e:
                print(f"Error al evaluar el modelo pre-entrenado {model_name}: {str(e)}")
    
    # Realizar una validación cruzada rápida para los algoritmos restantes
    for name, model in algorithms:
        if name not in [r[0] for r in results]:
            try:
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring=scoring, n_jobs=-1)
                mean_score = np.mean(scores)
                results.append((name, mean_score, model))
                print(f"Puntuación media para {name}: {mean_score:.4f}")
            except Exception as e:
                print(f"Error al evaluar {name}: {str(e)}")
    
    # Ordenar los resultados y seleccionar los mejores
    results.sort(key=lambda x: x[1], reverse=True)
    best_algorithms = results[:n_algorithms]
    
    print(f"\nLos mejores algoritmos seleccionados son:")
    for name, score, _ in best_algorithms:
        print(f"{name}: {score:.4f}")
    
    return best_algorithms

def apply_transfer_learning(best_model, trained_models, X_train, y_train, X_test, original_feature_names):
    """
    Aplica transfer learning al mejor modelo seleccionado utilizando los modelos previamente entrenados.
    
    Args:
    best_model: El mejor modelo seleccionado.
    trained_models: Lista de modelos previamente entrenados.
    X_train (array-like): Características del dataset de entrenamiento.
    X_test (array-like): Características del dataset de prueba.
    original_feature_names (list): Nombres de las características originales.
    
    Returns:
    tuple: (Modelo mejorado con transfer learning, X_train_enhanced, X_test_enhanced)
    """
    if not trained_models:
        return best_model, X_train, X_test

    def preprocess_features(X, feature_names):
        if isinstance(X, pd.DataFrame):
            return X[feature_names]
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError("X debe ser un DataFrame de pandas o un array de numpy")

    X_train = preprocess_features(X_train, original_feature_names)
    X_test = preprocess_features(X_test, original_feature_names)

    # Crear un ensamble de modelos
    train_predictions = []
    test_predictions = []
    for model in trained_models:
        try:
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_predictions.append(train_pred)
            test_predictions.append(test_pred)
        except Exception as e:
            print(f"Error al predecir con el modelo {model.__class__.__name__}: {str(e)}")
    
    if not train_predictions or not test_predictions:
        print("No se pudieron hacer predicciones con los modelos entrenados. Retornando el mejor modelo sin cambios.")
        return best_model, X_train, X_test

    train_ensemble_predictions = np.mean(train_predictions, axis=0)
    test_ensemble_predictions = np.mean(test_predictions, axis=0)

    # Combinar las predicciones del ensamble con las características originales
    X_train_enhanced = np.column_stack((X_train, train_ensemble_predictions.reshape(-1, 1)))
    X_test_enhanced = np.column_stack((X_test, test_ensemble_predictions.reshape(-1, 1)))

    # Entrenar el mejor modelo con los datos mejorados
    best_model_enhanced = clone(best_model)
    best_model_enhanced.fit(X_train_enhanced, y_train)

    return best_model_enhanced, X_train_enhanced, X_test_enhanced