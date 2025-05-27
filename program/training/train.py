#train.py

import numpy as np
import pandas as pd
import training.scikitdb.serializer as serializer
from sklearn.base import BaseEstimator
import inspect
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from utils.logger import logger
import optuna
from sklearn.base import clone
from data.datasetprocessing import select_best_features

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
    
    model_init_params = inspect.signature(model_class.__init__).parameters
    model_kwargs = params.copy() if params is not None else {}

    if 'verbose' not in model_kwargs and 'verbose' in model_init_params:
        model_kwargs['verbose'] = 0
    
        # Filtrar los parámetros válidos para este modelo
        valid_params = {}
    for param, value in model_kwargs.items():
        if param in model_init_params:
            valid_params[param] = value
        else:
            # Mantener la advertencia original si es útil
            print(f"Advertencia: El parámetro '{param}' no es válido para {model_name}. Se ignorará.")
        
        model = model_class(**valid_params)
    
    if isinstance(model, BaseEstimator):
        model.fit(X, y)
        return model
    else:
        print(f"Advertencia: {model_name} no es un estimador válido de scikit-learn. No se pudo entrenar.")
        return None

# Función para entrenar los modelos
def train_custom_models(X_train, y_train, algorithm, problem_type, dataset_path, feature_names, recommendations=True, optimization_strategy='random', n_iter_random=20, n_trials_optuna=30):
    model_name = algorithm["name"]
    params = algorithm["params"]
    
    try:
        # Asegurarse de que X_train es un DataFrame con las columnas correctas
        # Log para verificar X_train ANTES de cualquier modificación en train_custom_models
        logger.info(f"[train_custom_models] Inicio. Recibido X_train con columnas: {feature_names if feature_names else '(feature_names es None)'}")
        if isinstance(X_train, pd.DataFrame):
            logger.info(f"[train_custom_models] X_train (DataFrame) primeras filas ANTES de train_simple_model:\n{X_train.head()}")
            logger.info(f"[train_custom_models] Columnas de X_train (DataFrame): {X_train.columns.tolist()}")
            X_train_df = X_train # Ya es un DataFrame
        elif hasattr(X_train, 'shape'): # Podría ser un array numpy
            logger.info(f"[train_custom_models] X_train (array-like) forma: {X_train.shape}")
            if feature_names and len(feature_names) == X_train.shape[1]:
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
                logger.info(f"[train_custom_models] X_train convertido a DataFrame (primeras filas):\n{X_train_df.head()}")
            else:
                logger.error(f"[train_custom_models] X_train es array pero los nombres de características no coinciden o faltan. Shape: {X_train.shape}, Feature names: {feature_names}")
                # Aquí podrías decidir si continuar con un DF sin nombres o fallar.
                # Por ahora, intentamos crear un DF genérico para que train_simple_model no falle inmediatamente al esperar un DF.
                X_train_df = pd.DataFrame(X_train) 
                logger.warning(f"[train_custom_models] X_train convertido a DataFrame con nombres genéricos.")
        else:
            logger.error("[train_custom_models] X_train no es DataFrame ni tiene forma. No se puede proceder.")
            return None, None, None
        
        logger.info(f"[train_custom_models] Llamando a train_simple_model para {model_name} con X_train_df (columnas: {X_train_df.columns.tolist()})")
        base_model = train_simple_model(X_train_df, y_train, model_name, params)
        
        optimized_model = None
        current_selected_features = feature_names # Default a las features originales
        X_train_to_optimize = X_train_df

        if recommendations and base_model: 
            logger.info(f"Iniciando selección de características para {model_name} antes de la optimización de hiperparámetros...")
            try:
                # Usamos X_train (numpy array) y feature_names para que select_best_features construya el DataFrame si lo necesita.
                # O pasamos X_train_df directamente si select_best_features puede manejarlo.
                # Asumiendo que X_train (original) y y_train son numpy arrays o listas compatibles.
                # La función select_best_features espera X como DataFrame o array, y feature_names si X es array.
                temp_X_train_for_fs = pd.DataFrame(X_train, columns=feature_names) # Asegurar DataFrame para select_best_features

                selected_names_for_opt, X_selected_for_opt_df = select_best_features(
                    temp_X_train_for_fs, 
                    y_train, 
                    problem_type, 
                    feature_names=feature_names, # Nombres originales
                    search_level='intermediate' # o 'basic' 
                )

                if selected_names_for_opt and not X_selected_for_opt_df.empty:
                    logger.info(f"Características seleccionadas para {model_name} (pre-optimización): {selected_names_for_opt}")
                    current_selected_features = selected_names_for_opt
                    X_train_to_optimize = X_selected_for_opt_df
                else:
                    logger.warning(f"La selección de características no devolvió resultados para {model_name}. Se usarán las características originales.")
                    # X_train_to_optimize y current_selected_features ya tienen los defaults correctos

            except Exception as fs_exc:
                logger.error(f"Error durante la selección de características para {model_name}: {fs_exc}. Se usarán las características originales.")
                # X_train_to_optimize y current_selected_features ya tienen los defaults correctos

            # Proceder a la optimización de hiperparámetros (con X_train_to_optimize y current_selected_features)
            param_grid_for_opt = get_param_grid(base_model) # Usar la nueva get_param_grid
            if param_grid_for_opt:
                optimized_model = optimize_model(
                    X_train_to_optimize, # Usar los datos con características seleccionadas (o los originales si falló la selección)
                    y_train, 
                    base_model, # Pasar la instancia base_model
                    param_grid_for_opt, 
                    strategy=optimization_strategy, 
                    n_iter_random=n_iter_random,
                    n_trials_optuna=n_trials_optuna
                )
        else:
                logger.warning(f"No se pudo obtener param_grid para {model_name}. Saltando optimización de hiperparámetros.")
                optimized_model = clone(base_model) # Clonar para evitar modificar el base_model original
                optimized_model.fit(X_train_to_optimize, y_train) # Entrenar con las features seleccionadas (o no)
        
        if optimized_model is None: # Si recommendations era False o falló la optimización
            optimized_model = base_model

        return base_model, optimized_model, current_selected_features
    except Exception as e:
        logger.error(f"Error en train_custom_models para {model_name}: {str(e)}")
        # Print traceback for more detailed debugging
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

def optimize_model(X, y, model, param_grid, cv=5, strategy='random', n_iter_random=20, n_trials_optuna=30):
    try:
        # Asegurarse de que X es un DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        logger.info(f"Optimizando modelo {model.__class__.__name__} con estrategia: {strategy}")

        search_cv = None
        best_estimator = None

        if strategy == 'grid':
            search_cv = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=0)
            search_cv.fit(X, y)
            best_estimator = search_cv.best_estimator_
        elif strategy == 'random':
            search_cv = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter_random,
                cv=cv,
                verbose=0,
                random_state=42, 
                n_jobs=-1
            )
            search_cv.fit(X, y)
            best_estimator = search_cv.best_estimator_
        elif strategy == 'optuna':
            optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce Optuna's verbosity

            def objective(trial):
                params = {}
                for param_name, values in param_grid.items():
                    # Optuna requires that choices are serializable (e.g. no None directly if it's not supported by all models for a param)
                    # We assume param_grid provides valid choices for the model.
                    # If a parameter can be None, it should be part of the list of choices.
                    # Example: 'max_depth': [3, 5, None]
                    params[param_name] = trial.suggest_categorical(param_name, values)
                
                current_model = clone(model)
                current_model.set_params(**params)
                
                # Use cross_val_score for evaluation within the trial
                # The default scoring of the model will be used if not specified (e.g., accuracy for classifiers, R2 for regressors)
                # Ensure y is Series for consistency if it comes from DataFrame
                y_series = y if isinstance(y, pd.Series) else pd.Series(y)
                score = cross_val_score(current_model, X, y_series, cv=cv, n_jobs=-1).mean()
                return score

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials_optuna)
            
            best_params = study.best_params
            logger.info(f"Optuna mejores parámetros: {best_params}")
            
            # Retrain the best model on the full data X, y
            best_estimator = clone(model)
            best_estimator.set_params(**best_params)
            best_estimator.fit(X, y)
            
        else:
            logger.warning(f"Estrategia de optimización '{strategy}' no reconocida. Usando Random Search por defecto.")
            # Defaulting to Random Search
            search_cv = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter_random, # Use n_iter_random for the default
                cv=cv,
                verbose=0,
                random_state=42,
                n_jobs=-1
            )
            search_cv.fit(X, y)
            best_estimator = search_cv.best_estimator_
        
        return best_estimator

    except Exception as e:
        logger.error(f"Error durante la optimización del modelo ({strategy}): {str(e)}")
        # Print traceback for more detailed debugging
        import traceback
        logger.error(traceback.format_exc())
        return model  # Devolver el modelo original si hay un error

def get_param_grid(model):
    """
    Genera un grid de parámetros más completo para la optimización.
    Toma un modelo instanciado como entrada.
    """
    estimator_name = model.__class__.__name__
    params = model.get_params()
    grid = {}

    # Common parameters with predefined search spaces
    # Ensemble methods (RandomForest, GradientBoosting, etc.)
    if hasattr(model, 'n_estimators'):
        grid['n_estimators'] = [50, 100, 200, 300]
    if hasattr(model, 'learning_rate'): # For GradientBoosting, XGBoost, LGBM
        grid['learning_rate'] = [0.01, 0.05, 0.1, 0.2]
    if hasattr(model, 'subsample'): # For GradientBoosting
        grid['subsample'] = [0.7, 0.8, 0.9, 1.0]

    # Tree-based methods (DecisionTree, RandomForest, GradientBoosting)
    if hasattr(model, 'max_depth'):
        grid['max_depth'] = [None, 5, 10, 15, 20]
    if hasattr(model, 'min_samples_split'):
        grid['min_samples_split'] = [2, 5, 10, 15]
    if hasattr(model, 'min_samples_leaf'):
        grid['min_samples_leaf'] = [1, 2, 5, 10]
    if hasattr(model, 'max_features'):
        grid['max_features'] = ['sqrt', 'log2', None] # Common for RF and DT
        if estimator_name in ['RandomForestRegressor', 'RandomForestClassifier', 'GradientBoostingClassifier', 'GradientBoostingRegressor']:
            # For ensembles, can also be a float
             grid['max_features'].extend([0.5, 0.7, 1.0])

    # Quitar oob_score del grid general para RandomForest para evitar conflicto con bootstrap=False
    # Si bootstrap es True (default), oob_score puede ser True. Si bootstrap es False, oob_score debe ser False.
    # La forma más simple de evitar el error es no hiperparametrizar oob_score si bootstrap también se varía o podría ser False.
    # Otra opción sería fijar bootstrap=True y luego sí permitir oob_score=[True, False].
    # Por ahora, lo removemos para evitar el warning/error.
    # Si 'oob_score' estaba en el grid generado por la lógica de parámetros booleanos, también se eliminará si 'bootstrap' está presente y es variable.
    if estimator_name in ['RandomForestRegressor', 'RandomForestClassifier']:
        if 'bootstrap' in grid and False in grid['bootstrap']: # Si bootstrap puede ser False
            if 'oob_score' in grid:
                del grid['oob_score'] # No permitir oob_score=True si bootstrap puede ser False
        elif 'bootstrap' not in grid and not getattr(model, 'bootstrap', True): # Si bootstrap es False por defecto en el modelo base y no está en el grid
             if 'oob_score' in grid and True in grid['oob_score']:
                 grid['oob_score'] = [False] # Forzar oob_score a False
        # Si bootstrap es True (implícita o explícitamente) y oob_score se añadió por la lógica genérica booleana, está bien.

    # SVM specific
    if estimator_name in ['SVC', 'SVR']:
        grid['C'] = [0.1, 1, 10, 100]
        grid['kernel'] = ['linear', 'rbf', 'poly', 'sigmoid']
        # Gamma is only used by rbf, poly, sigmoid. Optuna will handle conditional params if defined in objective.
        # For GridSearchCV/RandomizedSearchCV, this might lead to warnings if gamma is specified for linear.
        # A more complex setup would involve conditional grids or separate grids per kernel.
        # For simplicity, we list them; optimization methods might ignore or warn.
        grid['gamma'] = ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        if 'poly' in grid['kernel']:
             grid['degree'] = [2, 3, 4] # For 'poly' kernel

    # Logistic Regression specific
    if estimator_name == 'LogisticRegression':
        # The previous grid definition is replaced by the one below.
        # Conditional penalties and solvers are tricky with RandomizedSearchCV's single dict requirement.
        # The goal here is to provide a single dictionary for RandomizedSearchCV
        # that is less prone to ValueErrors due to incompatible solver/penalty pairs,
        # even if some warnings about unused parameters (like l1_ratio for non-elasticnet)
        # or suboptimal combinations might still occur.

        grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None], 
            'solver': ['saga', 'liblinear'],            
            'l1_ratio': np.linspace(0.1, 0.9, 5).tolist(), 
            'fit_intercept': [True, False],
            'warm_start': [True, False]
            # 'dual' parameter is removed as it was causing FitFailedWarnings with incompatible solvers/penalties.
            # It's often solver-specific (e.g. liblinear with L2) and its default is generally fine.
        }
        # This revised grid aims to reduce critical errors by guiding RandomizedSearchCV towards more compatible pairings,
        # though it's not a perfect solution for conditional hyperparameter spaces without a more advanced search strategy
        # or modifying how RandomizedSearchCV is called (e.g., multiple separate searches).

    # K-Neighbors specific
    if estimator_name in ['KNeighborsClassifier', 'KNeighborsRegressor']:
        grid['n_neighbors'] = [3, 5, 7, 10, 15]
        grid['weights'] = ['uniform', 'distance']
        grid['metric'] = ['euclidean', 'manhattan', 'minkowski']

    # Regularization parameters for linear models like Lasso, Ridge, ElasticNet
    if hasattr(model, 'alpha'): # Common in Lasso, Ridge, ElasticNet
        grid['alpha'] = [0.001, 0.01, 0.1, 1, 10, 100]
    if hasattr(model, 'l1_ratio'): # For ElasticNet
        grid['l1_ratio'] = [0.1, 0.3, 0.5, 0.7, 0.9]

    # For parameters that are boolean and present in the model, but not covered above
    for param_name in params:
        if param_name not in grid and isinstance(params[param_name], bool):
            if estimator_name == 'LogisticRegression' and param_name == 'dual':
                continue  # Skip adding 'dual' for LogisticRegression via this generic loop
            grid[param_name] = [True, False]
        elif param_name not in grid and param_name == 'criterion': # Common for tree models
            if estimator_name == 'GradientBoostingClassifier':
                grid['criterion'] = ['friedman_mse', 'squared_error']
            elif estimator_name in ['RandomForestClassifier', 'DecisionTreeClassifier']:
                grid['criterion'] = ['gini', 'entropy'] # DecisionTreeClassifier también soporta 'log_loss' pero es menos común para el árbol base
                                                        # RandomForestClassifier también soporta 'log_loss'
                                                        # Para simplificar, mantenemos los más comunes. Podríamos añadir log_loss si es necesario.
            elif estimator_name == 'GradientBoostingRegressor':
                grid['criterion'] = ['friedman_mse', 'squared_error', 'absolute_error', 'poisson']
            elif estimator_name in ['RandomForestRegressor', 'DecisionTreeRegressor']:
                grid['criterion'] = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']

    # Limpieza final específica para oob_score y bootstrap si ambos están presentes
    if 'bootstrap' in grid and 'oob_score' in grid and False in grid['bootstrap'] and True in grid['oob_score']:
        # Si bootstrap puede ser False y oob_score puede ser True, es un problema.
        # Opción 1: Quitar oob_score del grid
        logger.info(f"Ajustando grid para {estimator_name}: eliminando 'oob_score' debido a posible conflicto con 'bootstrap=False'.")
        del grid['oob_score']
        # Opción 2: (más compleja) modificar los valores para que no haya conflicto, o usar Optuna con lógica condicional.

    # Remove parameters that might have been added but are not in the original model's params
    # (e.g. 'degree' if kernel is not 'poly' by default)
    final_grid = {k: v for k, v in grid.items() if k in params}
    
    if not final_grid:
        logger.warning(f"Could not generate a hyperparameter grid for {estimator_name}. Returning empty grid.")
        return {}

    logger.info(f"Generated param grid for {estimator_name}: {final_grid}")
    return final_grid

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
    # selected_features aquí es la lista de nombres de características para X_train_sel
    # o podría ser None si algo salió muy mal en select_best_features
    if selected_features is None:
        logger.error("selected_features es None en train_optimization. Usando todas las columnas de X_train si es DataFrame.")
        if isinstance(X_train, pd.DataFrame):
            selected_features = X_train.columns.tolist()
        else:
            # No se pueden determinar características para un array NumPy sin nombres, esto probablemente fallará
            # o el modelo usará todas las características. Devolver las originales para evitar el error de iteración.
            logger.warning("X_train no es DataFrame y selected_features es None. Se pasarán todas las características (índices) al modelo.")
            # Crear una lista de índices si X_train es numpy y selected_features es None
            # Esto es una suposición arriesgada, es mejor que selected_features nunca sea None.
            # En este punto, es mejor dejar que falle o usar todas las columnas originales si se conocen.
            # Por ahora, para evitar el error de iteración, si es None y X_train no es DF, crear lista vacía
            # para que el filtrado de 'target' no falle, aunque el modelo probablemente falle después.
            selected_features = [] 

    # Eliminar 'target' de selected_features si está presente y selected_features es una lista
    if isinstance(selected_features, list):
        processed_selected_features = [f for f in selected_features if f != 'target']
    else:
        # Si selected_features no es una lista (ej. None, aunque ya lo manejamos arriba), usar una lista vacía
        logger.warning(f"selected_features no era una lista en train_optimization (era {type(selected_features)}). Usando lista vacía.")
        processed_selected_features = []

    if isinstance(X_train, pd.DataFrame):
        # Usar processed_selected_features que ya no tiene 'target'
        # Asegurarse de que todas las processed_selected_features estén en X_train.columns
        valid_cols_for_X_new = [col for col in processed_selected_features if col in X_train.columns]
        if len(valid_cols_for_X_new) != len(processed_selected_features):
            logger.warning(f"Algunas características seleccionadas no estaban en X_train: {set(processed_selected_features) - set(valid_cols_for_X_new)}")
        X_new = X_train[valid_cols_for_X_new]
    elif isinstance(X_train, np.ndarray):
        # Esto es complicado si selected_features era None y se convirtió en []
        # Si processed_selected_features está vacío, X_new será un array vacío en la dim de características.
        # Si X_train es numpy, necesitamos los índices originales de las características.
        # Esta parte necesita que selected_features sea una lista de nombres que existían en el X_train original.
        # Por ahora, asumimos que si X_train es numpy, selected_features eran nombres válidos.
        # Si processed_selected_features está vacío y X_train es numpy, esto fallará o dará resultados inesperados.
        # La lógica de indexación para numpy arrays basada en nombres de características no está aquí.
        # X_new = X_train # Temporalmente, si es numpy y no hay forma clara de seleccionar, usar todo X_train.
        # La línea original era: X_new = X_train. Esto implica que para numpy, no se hacía subselección aquí.
        # Esto debe ser consistente con cómo se seleccionaron las características originalmente.
        # Si X_train ya es X_train_sel (ya subseleccionado), y selected_features son los nombres de X_train_sel,
        # entonces X_new debería ser X_train (que es X_train_sel).
        # Y processed_selected_features son los nombres de las columnas de X_new.
        X_new = X_train # Asumiendo que X_train ya está pre-seleccionado si es numpy.
    else:
        raise ValueError("X_train debe ser un DataFrame de pandas o un array de numpy")
    
    current_params = params.copy() if params is not None else {}
    if optimize_params:
        # Considerar si get_optimized_params debe también intentar setear verbose=0 internamente o si lo controlamos aquí
        current_params = get_optimized_params(model_class(), X_new, y_train, 'random')
    
    # Intentar añadir verbose=0 si el modelo lo acepta y no está ya en current_params
    model_init_params = inspect.signature(model_class.__init__).parameters
    if 'verbose' not in current_params and 'verbose' in model_init_params:
        current_params['verbose'] = 0
    
    optimized_model = model_class(**current_params)
    # Aquí X_new debe tener las columnas correctas y el orden que espera el modelo.
    # Si X_new es un DataFrame, el modelo lo manejará. Si es NumPy, el orden importa.
    optimized_model.fit(X_new, y_train)
    
    return optimized_model, X_new, processed_selected_features # Devolver processed_selected_features