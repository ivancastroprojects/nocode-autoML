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
from training.paramoptimization import get_optimized_params
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
import time


def recommend_best_model(X_train, y_train, X_test, y_test, problem_type, trained_models=None, feature_names=None, apply_transfer_learning=False):
    print("Iniciando recomendación del mejor modelo...")
    start_time = time.time()

    best_algorithms = select_best_algorithm(X_train, y_train, problem_type, trained_models, feature_names)
    
    best_model = None
    best_score = -np.inf
    best_selected_features = None
    
    print("Creando características polinomiales...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    if feature_names is None:
        feature_names = [f'x{i}' for i in range(X_train.shape[1])]
    else:
        feature_names = [f for f in feature_names if f != 'target']
    
    try:
        poly_feature_names = poly.get_feature_names_out(feature_names)
    except ValueError:
        print("Error al obtener los nombres de las características polinomiales. Usando nombres genéricos.")
        poly_feature_names = [f'poly_{i}' for i in range(X_train_poly.shape[1])]

    print("Seleccionando características...")
    selector = RFE(estimator=RandomForestRegressor(n_estimators=10), n_features_to_select=10, step=1)
    selector = selector.fit(X_train_poly, y_train)
    
    X_train_selected = selector.transform(X_train_poly)
    X_test_selected = selector.transform(X_test_poly)
    
    selected_features = poly_feature_names[selector.support_]
    print(f"Características seleccionadas: {selected_features}")

    for name, score, model in best_algorithms:
        print(f"Optimizando {name}...")
        optimized_model, X_train_opt, _ = train_optimization(X_train_selected, y_train, model.__class__, problem_type, selected_features, optimize_params=True)

        print(f"Evaluando {name}...")
        if problem_type == 'classification':
            score = accuracy_score(y_test, optimized_model.predict(X_test_selected))
        else:
            score = r2_score(y_test, optimized_model.predict(X_test_selected))
        
        print(f"Puntuación para {name}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = optimized_model
            best_selected_features = selected_features

    if apply_transfer_learning:
        print("Aplicando transfer learning...")
        best_model, X_train_selected, X_test_selected, best_selected_features = apply_transfer_learning(best_model, trained_models, X_train_selected, y_train, X_test_selected, poly_feature_names, best_selected_features)

    print(f"\nMejor modelo recomendado: {best_model.__class__.__name__}")
    print(f"Mejor puntuación: {best_score:.4f}")
    print(f"Características seleccionadas: {best_selected_features}")

    end_time = time.time()
    print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")

    return best_model, best_selected_features, X_train_selected, X_test_selected, poly, poly_feature_names

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

    return best_model_enhanced, X_train_enhanced, X_test_enhanced, enhanced_feature_names