#modeloptimization.py
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from training.train import get_param_grid, optimize_model
from data.datasetprocessing import select_best_features
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from utils.logger import logger
import time
import inspect

def recommend_best_model(X_train, y_train, X_test, y_test, problem_type, trained_models=None, feature_names=None, optimization_strategy='random', n_iter_random=20, n_trials_optuna=30):
    """
    Recomienda el mejor modelo basado en los datos de entrenamiento y prueba.
    
    Args:
    X_train, y_train: Datos de entrenamiento
    X_test, y_test: Datos de prueba
    problem_type: Tipo de problema ('classification' o 'regression')
    trained_models: Lista de modelos ya entrenados (opcional)
    feature_names: Nombres de las características (opcional)
    optimization_strategy: Estrategia de optimización para la fase final.
    n_iter_random: Iteraciones para Random Search en fase final.
    n_trials_optuna: Trials para Optuna en fase final.
    
    Returns:
    tuple: Mejor modelo, características seleccionadas, X_train y X_test con las mejores características
    """
    logger.info("Iniciando recomendación del mejor modelo...")
    start_time = time.time()

    # Estandarizar problem_type para uso interno
    pt_lower = problem_type.lower()
    if 'clasificacion' in pt_lower or 'clasificación' in pt_lower:
        current_problem_type = 'clasificacion'
    elif 'regresion' in pt_lower or 'regresión' in pt_lower:
        current_problem_type = 'regresion'
    else:
        logger.error(f"Tipo de problema no reconocido en recommend_best_model: {problem_type}")
        return None, None, X_train, X_test, None, None

    best_algorithms = select_best_algorithm(X_train, y_train, current_problem_type, trained_models, feature_names)
    
    best_model = None
    best_score = -np.inf
    best_selected_features = None
    
    logger.info("Seleccionando las mejores características (llamada a select_best_features)...")
    selected_feature_names, X_train_sel = select_best_features(
        X_train, y_train, current_problem_type, feature_names=feature_names
    )
    
    if isinstance(X_test, pd.DataFrame):
        valid_test_cols = [col for col in selected_feature_names if col in X_test.columns]
        X_test_sel = X_test[valid_test_cols]
    elif isinstance(X_test, np.ndarray) and isinstance(feature_names, list):
        try:
            original_indices = [feature_names.index(name) for name in selected_feature_names]
            X_test_sel = X_test[:, original_indices]
        except ValueError as e:
            logger.error(f"Error al obtener índices para X_test: {e}. Nombres en selected_feature_names: {selected_feature_names}. Nombres en feature_names (original): {feature_names}")
            X_test_sel = X_test
    else:
        logger.warning("X_test no es DataFrame o feature_names no es lista; no se pudo aplicar selección de características a X_test de forma segura.")
        X_test_sel = X_test

    logger.info(f"Características finalmente usadas para el entrenamiento post-selección: {selected_feature_names}")

    for name, model_class_candidate in best_algorithms:
        logger.info(f"Procesando candidato para mejor modelo global: {name} con características seleccionadas ({selected_feature_names})...")
        
        # Instanciar el modelo candidato
        try:
            model_instance = model_class_candidate()
        except Exception as e:
            logger.error(f"No se pudo instanciar {name}: {e}. Saltando este candidato.")
            continue

        # Asegurarse de que X_train_sel es un DataFrame con los nombres de columna correctos
        if not isinstance(X_train_sel, pd.DataFrame):
            X_train_sel_df = pd.DataFrame(X_train_sel, columns=selected_feature_names)
        else:
            # Si ya es DataFrame, asegurar que solo tenga las selected_feature_names
            X_train_sel_df = X_train_sel[selected_feature_names]

        # Obtener param_grid para la instancia del modelo candidato
        param_grid_for_final_opt = get_param_grid(model_instance) # model_instance es el modelo base del algoritmo candidato

        if not param_grid_for_final_opt:
            logger.warning(f"No se pudo generar param_grid para la optimización final de {name}. Se usará la instancia base.")
            final_optimized_model = clone(model_instance)
            final_optimized_model.fit(X_train_sel_df, y_train)
        else:
            final_optimized_model = optimize_model(
                X_train_sel_df, 
                y_train,      
                model_instance, # Pasar la instancia del modelo base del candidato
                param_grid_for_final_opt,
                strategy=optimization_strategy, # Usar la estrategia pasada a recommend_best_model
                n_iter_random=n_iter_random,    # Usar los parámetros pasados
                n_trials_optuna=n_trials_optuna # Usar los parámetros pasados
            )

        logger.info(f"Evaluando {name} con características seleccionadas ({selected_feature_names})...")
        # Asegurarse que X_test_sel también es DataFrame con las columnas correctas
        if not isinstance(X_test_sel, pd.DataFrame):
            X_test_sel_df = pd.DataFrame(X_test_sel, columns=selected_feature_names)
        else:
            X_test_sel_df = X_test_sel[selected_feature_names]

        if current_problem_type == 'clasificacion':
            score = accuracy_score(y_test, final_optimized_model.predict(X_test_sel_df))
        elif current_problem_type == 'regresion':
            score = r2_score(y_test, final_optimized_model.predict(X_test_sel_df))
        else:
            logger.error(f"Tipo de problema no reconocido para la puntuación (post estandarización): {current_problem_type}")
            score = -np.inf
        
        logger.info(f"Puntuación para {name} (post-selección y optimización final): {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = final_optimized_model
            best_selected_features = selected_feature_names

    if best_model is not None:
        logger.info(f"Mejor modelo recomendado: {best_model.__class__.__name__}")
    else:
        logger.warning("No se pudo recomendar ningún modelo.")

    logger.info(f"Mejor puntuación (post-selección): {best_score:.4f}")
    logger.info(f"Características seleccionadas finales: {best_selected_features}")

    end_time = time.time()
    logger.info(f"Tiempo total de ejecución de recommend_best_model: {end_time - start_time:.2f} segundos")

    # Devolver X_train_sel y X_test_sel como DataFrames si es posible
    if not isinstance(X_train_sel, pd.DataFrame) and selected_feature_names is not None:
        X_train_sel_df_final = pd.DataFrame(X_train_sel, columns=selected_feature_names)
    elif isinstance(X_train_sel, pd.DataFrame):
        X_train_sel_df_final = X_train_sel
    else:
        X_train_sel_df_final = pd.DataFrame(X_train_sel) # Puede fallar si no hay columnas

    if not isinstance(X_test_sel, pd.DataFrame) and selected_feature_names is not None and hasattr(X_test_sel, 'shape') and X_test_sel.shape[1] == len(selected_feature_names):
        X_test_sel_df_final = pd.DataFrame(X_test_sel, columns=selected_feature_names)
    elif isinstance(X_test_sel, pd.DataFrame):
        X_test_sel_df_final = X_test_sel
    else:
        X_test_sel_df_final = pd.DataFrame(X_test_sel) # Puede fallar

    return best_model, best_selected_features, X_train_sel_df_final, X_test_sel_df_final, None, None

def select_best_algorithm(X, y, problem_type, trained_models_input=None, feature_names_input=None, n_algorithms=3):
    """
    Realiza una selección inteligente de los mejores algoritmos potenciales (clases de modelo).
    
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
    # La estandarización ya se hace aquí, así que la entrada debe ser 'clasificacion' o 'regresion'
    # Si se llama desde fuera, asegurarse de que se pase así, o estandarizar en el punto de llamada.
    # Por consistencia, volvemos a estandarizar aquí por si se llama desde otro sitio.
    pt_lower_internal = problem_type.lower()
    if not ('clasificacion' in pt_lower_internal or 'regresion' in pt_lower_internal):
        logger.warning(f"get_popular_algorithms llamado con problem_type inesperado: {problem_type}. Se devolverá una lista vacía.")
        return []

    if 'clasificacion' in pt_lower_internal: # Buscamos la subcadena normalizada
        return [
            ('RandomForestClassifier', RandomForestClassifier),
            ('GradientBoostingClassifier', GradientBoostingClassifier),
            ('LogisticRegression', LogisticRegression),
            ('SVC', SVC),
            ('DecisionTreeClassifier', DecisionTreeClassifier),
            ('KNeighborsClassifier', KNeighborsClassifier)
        ]
    elif 'regresion' in pt_lower_internal: # Buscamos la subcadena normalizada
        return [
            ('RandomForestRegressor', RandomForestRegressor),
            ('GradientBoostingRegressor', GradientBoostingRegressor),
            ('LinearRegression', LinearRegression),
            ('SVR', SVR),
            ('DecisionTreeRegressor', DecisionTreeRegressor),
            ('KNeighborsRegressor', KNeighborsRegressor)
        ]
    else:
        # Esto no debería ocurrir si la estandarización inicial funcionó
        logger.error(f"Tipo de problema no reconocido en select_best_algorithm: {problem_type}")
        # Devolver una lista vacía o manejar el error como sea apropiado
        return []

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
    if trained_models_input:
        for model_obj in trained_models_input:
            if model_obj is None:
                continue
            model_name = model_obj.__class__.__name__
            try:
                if not callable(getattr(model_obj, "predict", None)):
                    logger.error(f"El objeto {model_name} en trained_models no es un estimador válido con método predict. Saltando.")
                    continue
                score = scoring(model_obj, X, y)
                results.append((model_name, score, model_obj))
                logger.info(f"Puntuación para modelo pre-entrenado {model_name}: {score:.4f}")
            except Exception as e:
                logger.error(f"Error al evaluar el modelo pre-entrenado {model_name}: {str(e)}")
    
    # Realizar una validación cruzada rápida para los algoritmos restantes
    for name, model_class_ref in algorithms:
        if name not in [r[0] for r in results]:
            try:
                model_instance = model_class_ref()
                if hasattr(model_instance, 'verbose') and isinstance(getattr(model_instance, 'verbose', None), int):
                    try:
                        model_instance.set_params(verbose=0)
                    except Exception:
                        pass
                elif 'verbose' in inspect.signature(model_class_ref.__init__).parameters:
                    try:
                        model_instance = model_class_ref(verbose=0)
                    except Exception:
                        model_instance = model_class_ref()

                scores = cross_val_score(model_instance, X_scaled, y, cv=3, scoring=scoring, n_jobs=-1)
                mean_score = np.mean(scores)
                results.append((name, mean_score, model_instance))
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
    # La estandarización ya se hace aquí, así que la entrada debe ser 'clasificacion' o 'regresion'
    # Si se llama desde fuera, asegurarse de que se pase así, o estandarizar en el punto de llamada.
    # Por consistencia, volvemos a estandarizar aquí por si se llama desde otro sitio.
    pt_lower_internal = problem_type.lower()
    if not ('clasificacion' in pt_lower_internal or 'regresion' in pt_lower_internal):
        logger.warning(f"get_popular_algorithms llamado con problem_type inesperado: {problem_type}. Se devolverá una lista vacía.")
        return []

    if 'clasificacion' in pt_lower_internal: # Buscamos la subcadena normalizada
        return [
            ('RandomForestClassifier', RandomForestClassifier),
            ('GradientBoostingClassifier', GradientBoostingClassifier),
            ('LogisticRegression', LogisticRegression),
            ('SVC', SVC),
            ('DecisionTreeClassifier', DecisionTreeClassifier),
            ('KNeighborsClassifier', KNeighborsClassifier)
        ]
    elif 'regresion' in pt_lower_internal: # Buscamos la subcadena normalizada
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
        
# La función apply_transfer_learning puede mantenerse si se considera útil en el futuro,
# pero no forma parte del flujo principal de optimización que estamos refactorizando ahora.
# Por ahora, la comentaremos para enfocarnos en el flujo principal.

# def apply_transfer_learning(best_model, trained_models, X_train, y_train, X_test, feature_names, selected_feature_names):
#     if not trained_models:
#         return best_model, X_train, X_test, selected_feature_names
# 
#         def preprocess_features(X, features):
#             if isinstance(X, pd.DataFrame):
#                 return X[features]
#             elif isinstance(X, np.ndarray):
#                 return X[:, [feature_names.index(f) for f in features]]
#             else:
#                 raise ValueError("X debe ser un DataFrame de pandas o un array de numpy")
# 
#         X_train_selected = preprocess_features(X_train, selected_feature_names)
#         X_test_selected = preprocess_features(X_test, selected_feature_names)
# 
#         # Crear un ensamble de modelos
#     train_predictions = []
#     test_predictions = []
#         for model in trained_models:
#             try:
#                 model_features = [f for f in model.feature_names_ if f in selected_feature_names]
#                 train_pred = model.predict(preprocess_features(X_train, model_features))
#                 test_pred = model.predict(preprocess_features(X_test, model_features))
#                 train_predictions.append(train_pred)
#                 test_predictions.append(test_pred)
#             except Exception as e:
#                 print(f"Error al predecir con el modelo {model.__class__.__name__}: {str(e)}")
#         
#         if not train_predictions or not test_predictions:
#             print("No se pudieron hacer predicciones con los modelos entrenados. Retornando el mejor modelo sin cambios.")
#             return best_model, X_train_selected, X_test_selected, selected_feature_names
# 
#     train_ensemble_predictions = np.mean(train_predictions, axis=0)
#     test_ensemble_predictions = np.mean(test_predictions, axis=0)
# 
#         # Combinar las predicciones del ensamble con las características seleccionadas
#         X_train_enhanced = np.column_stack((X_train_selected, train_ensemble_predictions.reshape(-1, 1)))
#         X_test_enhanced = np.column_stack((X_test_selected, test_ensemble_predictions.reshape(-1, 1)))
# 
#         # Crear nuevos nombres de características
#         enhanced_feature_names = selected_feature_names + ['ensemble_prediction']
# 
#     print(f"Características mejoradas: {enhanced_feature_names}")
#     print(f"Forma de X_train_enhanced: {X_train_enhanced.shape}")
#     print(f"Forma de X_test_enhanced: {X_test_enhanced.shape}")
# 
#         # Entrenar el mejor modelo con los datos mejorados
#         best_model_enhanced = clone(best_model)
#         best_model_enhanced.fit(X_train_enhanced, y_train)
#         return best_model_enhanced, X_train_enhanced, X_test_enhanced, enhanced_feature_names