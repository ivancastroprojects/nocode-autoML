# training.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, roc_auc_score
from training.train import train_custom_models
import training.scikitdb.serializer as serializer
from data.datasetprocessing import determine_problem_type_from_dataset, basic_dfpreprocess
from data.dataset import Dataset
from training.evaluation import Evaluation
from utils.logger import logger
from data.visualizer import Visualizer
from training.modeloptimization import recommend_best_model
import numpy as np
import traceback

class Training:
    """
    Clase principal para gestionar el entrenamiento y evaluación de modelos de machine learning.
    """

    def __init__(self, dataset=None):
        """
        Inicializa la clase Training con los parámetros necesarios para el entrenamiento.
        """
        self.dataset = dataset # Almacena el dataset si se proporciona
        self.problem_type = None
        self.target = None  # Debe ser establecida por el código que instancia Training según la selección del usuario.
        self.dataset_name = None
        self.features = None
        self.algorithms = []
        self.crossvalidation = 80  # Porcentaje de datos para entrenamiento
        self.recommendations = False  # Si se deben hacer recomendaciones de modelos
        self.preprocessing = False  # Si se debe aplicar preprocesamiento a los datos
        self.trained_models = []  # Lista para almacenar los modelos entrenados
        self.poly_transform = None
        self.selected_features = None
        self.optimization_strategy = 'random'  # Default optimization strategy
        self.n_iter_random = 20  # Default iterations for RandomSearch
        self.n_trials_optuna = 30  # Default trials for Optuna
        self.preprocessor = None  # Added for the new predict method

    def determine_problem_type(self, dataset):
        """
        Determina el tipo de problema basado en el dataset.

        Args:
        dataset (Dataset): Objeto Dataset con los datos cargados.
        """
        try:
            df = dataset.get_dataframe()
            if self.target not in df.columns:
                raise ValueError(f"La columna objetivo '{self.target}' no está presente en el dataset.")
            
            self.problem_type = determine_problem_type_from_dataset(df, self.target)
            logger.info(f"Tipo de problema determinado: {self.problem_type}")
        except Exception as e:
            logger.error(f"Error al determinar el tipo de problema: {str(e)}")
            self.problem_type = None
        
        if self.problem_type is None:
            logger.warning("No se pudo determinar el tipo de problema automáticamente.")
            self._manual_problem_type_selection()

    def _manual_problem_type_selection(self):
        """
        Permite al usuario seleccionar manualmente el tipo de problema.
        """
        while self.problem_type is None:
            user_input = input("Por favor, seleccione el tipo de problema (1: Clasificación, 2: Regresión): ").strip()
            if user_input == '1':
                self.problem_type = 'classification'
            elif user_input == '2':
                self.problem_type = 'regression'
            else:
                print("Entrada no válida. Por favor, seleccione 1 o 2.")
        
        logger.info(f"Tipo de problema seleccionado manualmente: {self.problem_type}")

    def split_and_train(self, dataset, dataset_path):
        """
        Divide el dataset, entrena los modelos y evalúa su rendimiento.

        Args:
        dataset (Dataset): Objeto Dataset con los datos cargados.
        dataset_path (str): Ruta del dataset.

        Returns:
        tuple: (X_test, y_test, trained_models, evaluation_results)
        """
        try:
            # Cargar y preprocesar el dataset si es necesario
            if self.dataset is None:
                logger.error("No hay dataset cargado para procesar.")
                return None, None, [], {}

            df = self.dataset.get_dataframe()
            if df is None or df.empty:
                logger.error("El DataFrame obtenido del dataset está vacío o es None.")
                return None, None, [], {}

            logger.info(f"DataFrame original para preprocesar (primeras filas):\n{df.head()}")
            logger.info(f"Columnas del DataFrame original: {df.columns.tolist()}")
            logger.info(f"Target column para basic_dfpreprocess: {self.target}")

            # Aplicar preprocesamiento básico
            # Asegúrate de que 'target' sea el nombre correcto de la columna objetivo.
            df_processed, preprocessor = basic_dfpreprocess(df, target_column=self.target)
            
            if df_processed is None:
                logger.error("basic_dfpreprocess devolvió None. No se puede continuar.")
                return None, None, [], {}

            logger.info(f"DataFrame después de basic_dfpreprocess (primeras filas):\n{df_processed.head()}")
            logger.info(f"Columnas después de basic_dfpreprocess: {df_processed.columns.tolist()}")


            # Separar características (X) y objetivo (y) del DF procesado
            if self.target not in df_processed.columns:
                logger.error(f"La columna objetivo '{self.target}' no se encontró en el DataFrame procesado.")
                # Intentar recuperarse si solo es un problema de capitalización o espacios
                target_candidates = [col for col in df_processed.columns if col.lower().strip() == self.target.lower().strip()]
                if target_candidates:
                    self.target = target_candidates[0]
                    logger.warning(f"Se encontró una columna similar: '{self.target}'. Usando esta como objetivo.")
                else:
                    logger.error(f"No se pudo encontrar una columna objetivo válida. Columnas disponibles: {df_processed.columns.tolist()}")
                    return None, None, [], {}
            
            y = df_processed[self.target]
            X = df_processed.drop(columns=[self.target])
            
            # Guardar las características utilizadas para el entrenamiento
            self.features = X.columns.tolist()
            logger.info(f"Características finales para entrenamiento (después de drop target): {self.features}")


            # Dividir los datos en conjuntos de entrenamiento y prueba
            # Convertir X e y a NumPy arrays antes de pasarlos a train_test_split
            # ya que esto es lo que esperan muchos modelos de scikit-learn y evita problemas de índice.
            X_np = X.values
            y_np = y.values
            
            test_size_float = self.crossvalidation / 100.0
            X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
                X_np, y_np, test_size=test_size_float, random_state=42
            )

            # Convertir de nuevo a DataFrames de Pandas con los nombres de columna correctos
            # Esto es crucial para que las funciones posteriores (como train_custom_models)
            # puedan trabajar con nombres de columnas y para la interpretabilidad.
            X_train_processed = pd.DataFrame(X_train_np, columns=self.features)
            X_test_processed = pd.DataFrame(X_test_np, columns=self.features)
            y_train_series = pd.Series(y_train_np, name=self.target)
            y_test_series = pd.Series(y_test_np, name=self.target)

            logger.info(f"Columnas de X_train_processed INMEDIATAMENTE después de split_and_train (antes de _train_models): {X_train_processed.columns.tolist()}")
            logger.info(f"Primeras filas de X_train_processed:\n{X_train_processed.head()}")
            logger.info(f"Columnas de X_test_processed INMEDIATAMENTE después de split_and_train: {X_test_processed.columns.tolist()}")


            # Guardar los conjuntos de datos divididos (opcional, para depuración o análisis posterior)
            # Esto debería ocurrir después de asegurar que df_processed y self.target son válidos
            self.preprocessor = preprocessor
            self.dataset_name = dataset_path
            trained_models, evaluation_results = self._train_models(X_train_processed, y_train_series, X_test_processed, y_test_series, dataset_path, self.features, self.preprocessor)

            return X_test_processed, y_test_series, trained_models, evaluation_results

        except Exception as e:
            logger.error(f"Error en split_and_train: {str(e)}")
            logger.error(traceback.format_exc())
            # Devolver valores por defecto en caso de error
            return None, None, [], {}

    def _train_models(self, X_train, y_train, X_test, y_test, dataset_path, feature_names, fitted_preprocessor):
        """
        Entrena los modelos seleccionados por el usuario y los optimiza si se solicita.

        Args:
        X_train (DataFrame): Características de entrenamiento.
        y_train (Series): Variable objetivo de entrenamiento.
        X_test (DataFrame): Características de prueba.
        y_test (Series): Variable objetivo de prueba.
        dataset_path (str): Ruta del dataset.
        feature_names (list): Nombres de las características.
        fitted_preprocessor (object): El preprocesador ajustado en el X_train original (puede ser None).

        Returns:
        tuple: (trained_models, evaluation_results)
        """
        trained_models = []
        evaluation_results = {}
        evaluation = Evaluation(self.problem_type, self.dataset_name)
        visualizer = Visualizer()
        visualizer.set_dataset_name(self.dataset_name)

        logger.info(f"Algoritmos a procesar en _train_models: {self.algorithms}")

        for algorithm in self.algorithms:
            try:
                algorithm_name = algorithm if isinstance(algorithm, str) else algorithm.get('name')
                algorithm_params = {} if isinstance(algorithm, str) else algorithm.get('params', {})

                if not self._is_appropriate_model(algorithm_name, self.problem_type):
                    logger.warning(f"{algorithm_name} no es apropiado para problemas de {self.problem_type}. Saltando...")
                    continue
                
                base_model_instance, optimized_model_instance, selected_features_for_optimized = train_custom_models(
                    X_train, y_train, {'name': algorithm_name, 'params': algorithm_params}, 
                    self.problem_type, dataset_path, feature_names,
                    True, 
                    optimization_strategy=self.optimization_strategy,
                    n_iter_random=self.n_iter_random,
                    n_trials_optuna=self.n_trials_optuna
                )
                
                current_model_eval_results = {}
                final_model_dict_to_store = None

                if base_model_instance is not None:
                    base_model_dict = {'model': base_model_instance, 'features': feature_names, 'name': f"{algorithm_name}_Base"}
                    base_results = self._evaluate_and_visualize_model(
                        model_to_eval=base_model_instance,
                        X_test_data=X_test,
                        y_test_data=y_test,
                        features_used_for_this_model=feature_names,
                        evaluation_obj=evaluation,
                        visualizer_obj=visualizer,
                        dataset_name_str=self.dataset_name,
                        model_unique_name=f"{algorithm_name}_Base",
                        X_train_original_for_stats=X_train,
                        fitted_preprocessor=fitted_preprocessor
                    )
                    current_model_eval_results['base'] = base_results
                    final_model_dict_to_store = base_model_dict

                if optimized_model_instance is not None and optimized_model_instance != base_model_instance:
                    optimized_model_dict = {'model': optimized_model_instance, 'features': selected_features_for_optimized, 'name': f"{algorithm_name}_Optimized"}
                    optimized_results = self._evaluate_and_visualize_model(
                        model_to_eval=optimized_model_instance,
                        X_test_data=X_test,
                        y_test_data=y_test,
                        features_used_for_this_model=selected_features_for_optimized,
                        evaluation_obj=evaluation,
                        visualizer_obj=visualizer,
                        dataset_name_str=self.dataset_name,
                        model_unique_name=f"{algorithm_name}_Optimized",
                        X_train_original_for_stats=X_train,
                        fitted_preprocessor=fitted_preprocessor
                    )
                    current_model_eval_results['optimized'] = optimized_results

                if base_results and optimized_results:
                    is_classification = 'classification' in self.problem_type.lower() or 'clasificación' in self.problem_type.lower()
                    primary_metric = 'accuracy' if is_classification else 'r2_score'
                    default_value_if_missing = -float('inf')
                    base_score = base_results.get(primary_metric, default_value_if_missing)
                    optimized_score = optimized_results.get(primary_metric, default_value_if_missing)
                    
                    logger.info(f"Comparing models for {algorithm_name}: Base ({primary_metric}={base_score:.4f}) vs Optimized ({primary_metric}={optimized_score:.4f})")
                    if optimized_score > base_score:
                        logger.info(f"Optimized model is better for {algorithm_name}. Selecting optimized.")
                        final_model_dict_to_store = optimized_model_dict
                        evaluation_results[algorithm_name] = {'selected_optimized': optimized_results, 'base_raw': base_results} 
                    else:
                        logger.info(f"Base model is better or equal for {algorithm_name}. Selecting base.")
                        evaluation_results[algorithm_name] = {'selected_base': base_results, 'optimized_raw': optimized_results}
                elif optimized_results:
                    logger.info(f"Only optimized model has results for {algorithm_name}. Selecting optimized.")
                    final_model_dict_to_store = optimized_model_dict
                    evaluation_results[algorithm_name] = {'selected_optimized': optimized_results}
                
                if final_model_dict_to_store:
                    self.trained_models.append(final_model_dict_to_store)

                if algorithm_name not in evaluation_results and current_model_eval_results:
                    if 'optimized' in current_model_eval_results and 'base' in current_model_eval_results:
                         evaluation_results[algorithm_name] = {'selected_base_default': current_model_eval_results['base'], 'optimized_raw': current_model_eval_results['optimized']}
                    elif 'base' in current_model_eval_results:
                        evaluation_results[algorithm_name] = {'selected_base_default': current_model_eval_results['base']}
                    elif 'optimized' in current_model_eval_results:
                        evaluation_results[algorithm_name] = {'selected_optimized_default': current_model_eval_results['optimized']}
                elif not current_model_eval_results:
                    logger.warning(f"No evaluation results generated for {algorithm_name}. Omitiendo de self.trained_models y evaluation_results.")

            except Exception as e:
                logger.error(f"Error al procesar el algoritmo {algorithm_name} en _train_models: {str(e)}")
                logger.error(traceback.format_exc())

        if self.recommendations:
            self._find_and_evaluate_best_model(X_train, y_train, X_test, y_test, feature_names, 
                                            self.dataset_name, self.trained_models, evaluation, visualizer, fitted_preprocessor)

        return self.trained_models, evaluation_results

    def _is_appropriate_model(self, model_name, problem_type):
        """
        Verifica si un modelo es apropiado para el tipo de problema actual.
        """
        if problem_type is None:
            logger.warning("El tipo de problema no ha sido determinado. No se puede verificar la idoneidad del modelo.")
            return False
        
        # Obtener los nombres de las clases de los modelos
        classification_model_names = {cls.__name__ for cls in (serializer.classification_models if hasattr(serializer, 'classification_models') else [])}
        regression_model_names = {cls.__name__ for cls in (serializer.regression_models if hasattr(serializer, 'regression_models') else [])}
        
        # El problem_type puede ser 'Clasificación Binaria', 'Clasificación Multiclase' o 'Regresión'
        # Usamos las raíces en español para la comprobación.
        problem_type_lower = problem_type.lower()
        is_classification_problem = 'clasificación' in problem_type_lower or 'clasificacion' in problem_type_lower
        is_regression_problem = 'regresión' in problem_type_lower or 'regresion' in problem_type_lower

        if is_classification_problem:
            return model_name in classification_model_names
        elif is_regression_problem:
            return model_name in regression_model_names
        
        return False # Si el tipo de problema no coincide con ninguno

    def _evaluate_and_visualize_model(self, model_to_eval,
                                      X_test_data, y_test_data, 
                                      features_used_for_this_model, 
                                      evaluation_obj, visualizer_obj, 
                                      dataset_name_str, model_unique_name,
                                      X_train_original_for_stats, fitted_preprocessor,
                                      _optimized_model_placeholder=None):
        """
        Evalúa un modelo específico (ya sea base u optimizado) y genera visualizaciones.
        model_unique_name es el nombre completo del modelo (ej. RandomForest_Base).
        features_used_for_this_model son las características que este model_to_eval específico espera.
        X_train_original_for_stats es el DataFrame X_train original para calcular min/max.
        fitted_preprocessor es el preprocesador ajustado en X_train_original_for_stats.
        """
        results = {}
        try:
            if not isinstance(X_test_data, pd.DataFrame):
                logger.error(f"X_test_data no es un DataFrame en _evaluate_and_visualize_model para {model_unique_name}.")
                if X_test_data.shape[1] != len(features_used_for_this_model):
                    logger.error(f"Discrepancia en el número de features para {model_unique_name}. X_test_data tiene {X_test_data.shape[1]}, se esperaban {len(features_used_for_this_model)}")
                    return {}
                X_test_subset = X_test_data
            else:
                missing_features = [f for f in features_used_for_this_model if f not in X_test_data.columns]
                if missing_features:
                    logger.error(f"Faltan características en X_test_data para evaluar {model_unique_name}: {missing_features}. No se puede evaluar.")
                    return {}
                X_test_subset = X_test_data[features_used_for_this_model]

            current_metrics = self._evaluate_model(model_to_eval, X_test_subset, y_test_data, evaluation_obj)
            results.update(current_metrics)

            if current_metrics:
                visualizer_obj.set_model_name(model_unique_name)
                y_pred = model_to_eval.predict(X_test_subset)
                
                problem_type_lower = evaluation_obj.problem_type.lower()
                if problem_type_lower.startswith('regres'):
                    visualizer_obj.plot_residuals(y_test_data, y_pred, X_test_subset.columns.tolist())
                elif problem_type_lower.startswith('clasificacion') or problem_type_lower.startswith('clasificación'):
                    if "Error" not in current_metrics:
                        visualizer_obj.plot_confusion_matrix(y_test_data, y_pred, class_names=model_to_eval.classes_ if hasattr(model_to_eval, 'classes_') else np.unique(y_test_data))
                        if hasattr(model_to_eval, "predict_proba"):
                            y_pred_proba = model_to_eval.predict_proba(X_test_subset)
                            visualizer_obj.plot_roc_curve(y_test_data, y_pred_proba)
                            visualizer_obj.plot_precision_recall_curve(y_test_data, y_pred_proba)
                        else:
                            logger.warning(f"El modelo {model_unique_name} no tiene predict_proba. Omitiendo ROC y Precision-Recall.")
                        
                        if hasattr(model_to_eval, 'feature_importances_') or (hasattr(model_to_eval, 'coef_') and model_to_eval.coef_.ndim == 1):
                            visualizer_obj.plot_feature_importance(model_to_eval, features_used_for_this_model)
                        elif hasattr(model_to_eval, 'coef_') and model_to_eval.coef_.ndim > 1 and len(features_used_for_this_model) == model_to_eval.coef_.shape[1]:
                            importances = np.linalg.norm(model_to_eval.coef_, axis=0)
                            class DummyModel: pass
                            dummy_model = DummyModel()
                            dummy_model.feature_importances_ = importances
                            visualizer_obj.plot_feature_importance(dummy_model, features_used_for_this_model)

                    logger.info(f"Guardando modelo: {model_unique_name} para el dataset {dataset_name_str}")
                    
                    is_classification_prob = 'classification' in self.problem_type.lower() or 'clasificación' in self.problem_type.lower()
                    primary_metric_key = 'accuracy' if is_classification_prob else 'r2_score'
                    metric_for_filename = current_metrics.get(primary_metric_key, 0.0)

                    serializer.to_pickle(
                        model=model_to_eval,
                        model_name=model_unique_name,
                        dataset_path=dataset_name_str, 
                        accuracy=metric_for_filename,
                        X_train=X_train_original_for_stats,
                        actual_feature_names=features_used_for_this_model,
                        preprocessor=fitted_preprocessor
                    )
                    logger.info(f"Modelo {model_unique_name} guardado exitosamente.")

        except Exception as e:
            logger.error(f"Error en _evaluate_and_visualize_model para {model_unique_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
        return results

    def _evaluate_model(self, model, X_test_subset, y_test, evaluation):
        """
        Evalúa un modelo utilizando la clase Evaluation. 
        X_test_subset YA DEBE TENER las características correctas para el modelo.
        """
        results = {}
        # model_name_for_eval = model.__class__.__name__ # El nombre específico se maneja en la función llamadora

        # current_feature_names ya no se necesita aquí, se asume que X_test_subset es correcto
        
        try:
            # Realizar la predicción aquí
            y_pred = model.predict(X_test_subset)

            problem_type_lower = self.problem_type.lower()
            if problem_type_lower.startswith('clasificacion') or problem_type_lower.startswith('clasificación'):
                # Llamar con y_test e y_pred
                metrics = evaluation.evaluate_classification_metrics(y_test, y_pred)
                results.update(metrics)
                # Añadir ROC AUC si es posible (requiere predict_proba)
                if hasattr(model, "predict_proba"):
                    try:
                        y_pred_proba = model.predict_proba(X_test_subset)
                        if y_pred_proba.shape[1] == 2: # Binaria
                            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else: # Multiclase
                            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    except Exception as roc_e:
                        logger.warning(f"No se pudo calcular ROC AUC para {model.__class__.__name__}: {roc_e}")
                        results['roc_auc'] = np.nan
                else:
                    results['roc_auc'] = np.nan # No se puede calcular

            elif problem_type_lower.startswith('regres'):
                # Llamar con y_test e y_pred
                metrics = evaluation.evaluate_regression_metrics(y_test, y_pred)
                results.update(metrics)
            else:
                logger.error(f"Tipo de problema desconocido en _evaluate_model: {self.problem_type}")
        except Exception as e:
            logger.error(f"Error durante la evaluación del modelo ({model.__class__.__name__}): {str(e)}")
            logger.error(traceback.format_exc())
            return {} # Retornar dict vacío si hay error

        return results

    def _find_and_evaluate_best_model(self, X_train, y_train, X_test, y_test, feature_names, 
                                      dataset_name, current_trained_model_dicts, 
                                      evaluation, visualizer, fitted_preprocessor):
        """
        Encuentra, evalúa y visualiza el mejor modelo para el problema actual.
        current_trained_model_dicts es una lista de diccionarios: {'model': model_instance, 'features': feature_list, 'name': str_name}
        fitted_preprocessor es el preprocesador ajustado en el X_train original.
        """
        logger.info("\nIniciando recomendación del mejor modelo...")
        
        model_instances_for_recommendation = [d['model'] for d in current_trained_model_dicts]

        try:
            recommended_model_instance, recommended_features_list, \
            X_train_recommended_df, X_test_recommended_df, \
            _poly_transform_placeholder, _poly_feature_names_placeholder = self._recommend_best_model(
                X_train, y_train, X_test, y_test, 
                self.problem_type, 
                feature_names,    
                model_instances_for_recommendation 
            )

            if recommended_model_instance is not None and recommended_features_list and not X_test_recommended_df.empty:
                logger.info(f"Mejor modelo recomendado globalmente: {recommended_model_instance.__class__.__name__}")
                logger.info(f"Características usadas por el mejor modelo recomendado: {recommended_features_list}")
                
                unique_recommended_name = f"GlobalBest_{recommended_model_instance.__class__.__name__}"
                self._evaluate_and_visualize_model(
                    model_to_eval=recommended_model_instance,
                    X_test_data=X_test_recommended_df, 
                    y_test_data=y_test, 
                    features_used_for_this_model=recommended_features_list, 
                    evaluation_obj=evaluation, 
                    visualizer_obj=visualizer, 
                    dataset_name_str=dataset_name,
                    model_unique_name=unique_recommended_name,
                    X_train_original_for_stats=X_train,
                    fitted_preprocessor=fitted_preprocessor
                )
            else:
                logger.warning("No se pudo obtener un modelo recomendado globalmente o faltan datos/características.")
        except Exception as e:
            logger.error(f"Error durante la búsqueda del mejor modelo global: {str(e)}")
            logger.error(traceback.format_exc())

    def _print_evaluation_results(self, results):
        """
        Imprime los resultados de la evaluación de un modelo.
        """
        for key, value in results.items():
            logger.info(f"{key}: {value}")
        logger.info("-" * 50)

    def _calculate_accuracy(self, model, X_test, y_test):
        """
        Calcula la precisión o R2 score del modelo.
        """
        problem_type_lower = self.problem_type.lower()
        if 'clasificación' in problem_type_lower or 'clasificacion' in problem_type_lower:
            return accuracy_score(y_test, model.predict(X_test))
        elif 'regresión' in problem_type_lower or 'regresion' in problem_type_lower:
            return r2_score(y_test, model.predict(X_test))
        else:
            logger.error(f"Tipo de problema no reconocido para calcular accuracy/R2: {self.problem_type}")
            return 0.0 # O algún valor por defecto / None

    def _recommend_best_model(self, X_train, y_train, X_test, y_test, problem_type, feature_names, current_trained_models_list):
        # from training.modeloptimization import recommend_best_model # Removed from here
        # Usar current_trained_models_list en lugar de self.trained_models (que está vacío)
        # Asegurarse de que problem_type aquí es el string, no self.problem_type que podría ser None si no se ha determinado
        best_model, best_selected_features, X_train_selected, X_test_selected, poly, poly_feature_names = recommend_best_model(
            X_train, y_train, X_test, y_test, problem_type, current_trained_models_list, feature_names
        )
        return best_model, best_selected_features, X_train_selected, X_test_selected, poly, poly_feature_names

    def predict(self, model_name_selected, dataset_group_name):
        """
        Realiza una predicción usando un modelo entrenado.
        model_name_selected es el nombre del archivo .pkl (e.g., GlobalBest_LogisticRegression.pkl)
        dataset_group_name es el nombre del directorio del dataset (e.g., breast_cancer)
        """
        try:
            # Cargar el diccionario del modelo usando la función de carga de serializer que devuelve el dict
            model_data_dict = serializer.load_model_from_group(dataset_group_name, model_name_selected)

            if not model_data_dict or 'model' not in model_data_dict:
                logger.error(f"No se pudo cargar la información del modelo para {dataset_group_name}/{model_name_selected}")
                return None

            model = model_data_dict['model']
            model_features = model_data_dict.get('feature_names') # Estas son las features que el MODELO espera
            feature_mins = model_data_dict.get('feature_mins')
            feature_maxs = model_data_dict.get('feature_maxs')
            loaded_preprocessor = model_data_dict.get('preprocessor')

            if not model_features:
                logger.error(f"No se encontraron nombres de características para el modelo {model_name_selected}.")
                # Tratar de obtener de model.feature_names_in_ si existe como fallback
                if hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                    logger.info(f"Fallback: Usando feature_names_in_ del modelo: {model_features}")
                else:
                    logger.error("No se pueden determinar las características esperadas por el modelo.")
                    return None
            
            logger.info(f"\nPredicción con: {dataset_group_name}/{model_name_selected}")
            logger.info(f"Características esperadas por el modelo (en orden): {model_features}")

            features_predict_input = {}
            for feature in model_features:
                prompt_text = f"Ingrese el valor para '{feature}'"
                min_val_str = ""
                max_val_str = ""
                # Mostrar min/max si están disponibles para esta característica específica
                if feature_mins and feature in feature_mins and feature_maxs and feature in feature_maxs:
                    min_val = feature_mins[feature]
                    max_val = feature_maxs[feature]
                    # Formatear solo si no son None
                    min_val_str = f"{min_val:.2f}" if min_val is not None else "N/A"
                    max_val_str = f"{max_val:.2f}" if max_val is not None else "N/A"
                    prompt_text += f" (min: {min_val_str}, max: {max_val_str})"
                
                prompt_text += ": "
                
                while True:
                    try:
                        value_str = input(prompt_text)
                        value_float = float(value_str)
                        features_predict_input[feature] = value_float
                        break
                    except ValueError:
                        logger.error("Entrada no válida. Por favor, ingrese un número.")
            
            # Preparar las características usando el preprocesador cargado
            # _prepare_features ahora necesitará el preprocesador y las features que el modelo espera
            features_df = self._prepare_features(features_predict_input, loaded_preprocessor, model_features)
            
            if features_df is None:
                logger.error("No se pudieron preparar las características para la predicción.")
                return None

            # Asegurarse de que las columnas estén en el orden correcto que el modelo espera
            # Esto es crucial si el preprocesador cambia el orden o el número de columnas
            # Si loaded_preprocessor es None, model_features son las columnas originales.
            # Si loaded_preprocessor no es None, las columnas de features_df son las transformadas.
            # El modelo fue entrenado con las columnas transformadas (si hubo preprocesador).
            # Entonces, model_features debería ser las features *después* de la transformación si preprocessor no es None.
            # Esto necesita ser consistente. El `model_features` guardado DEBE ser el que el modelo serializado espera.

            # Si hay un preprocesador, las `model_features` guardadas deberían ser las post-transformación.
            # Si no hay preprocesador, `model_features` son las originales.
            
            # El `serializer.to_pickle` guarda `actual_feature_names` que son las que el modelo usó.
            # Si hubo preprocesador, estas `actual_feature_names` son las transformadas.
            
            # Reordenar `features_df` para que coincida con `model_features` (que son las que el modelo espera)
            try:
                features_df_ordered = features_df[model_features]
            except KeyError as e:
                logger.error(f"Error al ordenar características para predicción. Faltan columnas en DataFrame preparado: {e}")
                logger.error(f"Columnas en DataFrame preparado: {features_df.columns.tolist()}")
                logger.error(f"Columnas esperadas por el modelo: {model_features}")
                return None


            prediction = model.predict(features_df_ordered) # Usar .values podría no ser necesario si es DF
            class_label = self._get_class_label(prediction) # Asume que self.problem_type está seteado
            
            # Para _print_prediction, pasamos las features originales ingresadas por el usuario
            self._print_prediction(features_predict_input, prediction, class_label, model_name_selected)
            return prediction

        except FileNotFoundError:
            logger.error(f"No se pudo encontrar el archivo del modelo: {dataset_group_name}/{model_name_selected}")
            return None
        except Exception as e:
            logger.error(f"Error durante la predicción con {dataset_group_name}/{model_name_selected}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _prepare_features(self, features_predict_input: dict, preprocessor, model_expected_features: list):
        """
        Prepara las características para la predicción usando el preprocesador cargado.
        features_predict_input: dict de características ingresadas por el usuario.
        preprocessor: el preprocesador ColumnTransformer cargado (o None).
        model_expected_features: lista de nombres de características que el modelo espera (post-transformación si preprocessor existe).
        """
        # Convertir el diccionario de entrada a DataFrame
        # Las claves del diccionario son los nombres de las características *originales*
        features_df = pd.DataFrame([features_predict_input])

        if preprocessor:
            try:
                logger.info("Aplicando transformaciones del preprocesador cargado...")
                # El preprocesador espera las columnas en el orden en que fue ajustado.
                # Necesitamos asegurar que features_df tenga las columnas que el preprocesador espera.
                # El preprocesador fue ajustado en X_train, que tenía todas las características originales.
                
                # Obtener las características en las que se ajustó el preprocesador
                # Esto es un poco complicado porque preprocessor.feature_names_in_ no siempre está
                # y get_feature_names_out() da los nombres de salida.
                # Necesitamos las features de ENTRADA del preprocesador.
                # Asumimos que las claves de features_predict_input son las originales.
                # Si hay un desajuste, el transform fallará o dará resultados incorrectos.
                
                # La forma más segura es que `preprocessor.feature_names_in_` exista y se use.
                # O, si `model_data_dict` guardara las features originales en las que se ajustó el preprocesador.
                # Por ahora, asumimos que features_df (con columnas de features_predict_input) es lo que el preprocessor espera.

                features_transformed_array = preprocessor.transform(features_df)
                
                # Obtener los nombres de las características transformadas del preprocesador
                # Estos DEBERÍAN coincidir con `model_expected_features` si todo se guardó y cargó correctamente.
                try:
                    transformed_feature_names = preprocessor.get_feature_names_out()
                except Exception as e_fn:
                    logger.warning(f"No se pudieron obtener nombres de preprocesador con get_feature_names_out: {e_fn}. Usando model_expected_features.")
                    # Si falla, y el número de columnas coincide, asumimos que model_expected_features es el orden correcto.
                    if features_transformed_array.shape[1] == len(model_expected_features):
                        transformed_feature_names = model_expected_features
                    else:
                        logger.error("Discrepancia de columnas después de transformar y model_expected_features. No se pueden asignar nombres.")
                        return None

                features_df_processed = pd.DataFrame(features_transformed_array, columns=transformed_feature_names, index=features_df.index)
                logger.info(f"Características transformadas: {features_df_processed.columns.tolist()}")

            except Exception as e:
                logger.error(f"Error al aplicar el preprocesador durante la preparación de características: {e}")
                logger.error(traceback.format_exc())
                # Si la transformación falla, no podemos continuar de forma fiable
                return None
        else:
            # No hay preprocesador, las características son las originales
            features_df_processed = features_df
            logger.info("No se aplicó preprocesador (no fue cargado con el modelo).")

        # Asegurarse de que todas las características esperadas por el modelo existan en el DataFrame procesado,
        # rellenando con 0 si faltan (esto es un fallback, idealmente no debería suceder si el preprocesador y model_expected_features son coherentes)
        for feature in model_expected_features:
            if feature not in features_df_processed.columns:
                logger.warning(f"La característica esperada por el modelo '{feature}' no está en el DataFrame procesado después de la transformación (o no hubo). Añadiendo como 0.")
                features_df_processed[feature] = 0 
        
        # Devolver solo las características que el modelo espera, en el orden correcto.
        try:
            return features_df_processed[model_expected_features]
        except KeyError as e:
            logger.error(f"Error final al seleccionar características para el modelo. Faltan columnas: {e}")
            logger.error(f"Columnas disponibles: {features_df_processed.columns.tolist()}")
            logger.error(f"Columnas esperadas: {model_expected_features}")
            return None

    def _get_class_label(self, prediction):
        """
        Obtiene la etiqueta de clase para una predicción (solo para clasificación).
        """
        if self.problem_type == 'classification' and hasattr(self, 'class_labels'):
            return Dataset.class_labels.get(int(prediction[0]), prediction[0])
        return prediction[0]

    def _print_prediction(self, featuresPredict, prediction, class_label, model_name):
        """
        Imprime el resultado de una predicción.
        """
        logger.info(f"\nPredicción realizada con el modelo: {model_name}")
        logger.info(f"La {self.target} predicha para las características proporcionadas es {prediction[0]:.2f} ({class_label})")
        logger.info("\nCaracterísticas utilizadas para la predicción:")
        for feature, value in featuresPredict.items():
            logger.info(f"  {feature}: {value:.4f}")
        logger.info("\nNota: Si algunas características esperadas por el modelo no estaban presentes,")
        logger.info("se utilizaron valores predeterminados (0 o la media del conjunto de entrenamiento).")