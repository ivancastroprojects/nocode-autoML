# training.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, roc_auc_score, confusion_matrix, classification_report
from program.training.train import train_custom_models
import program.training.scikitdb.serializer as serializer
from program.data.datasetprocessing import determine_problem_type, basic_dfpreprocess
from program.data.dataset import Dataset
from program.training.evaluation import Evaluation
from program.utils.logger import logger
from program.data.visualizer import Visualizer
from program.training.modeloptimization import recommend_best_model
import numpy as np
import traceback
import logging
from program.utils.mqtt_handler import MQTTLogHandler

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
        self.trained_models = {}  # DE LISTA A DICCIONARIO: para almacenar los modelos entrenados
        self.poly_transform = None
        self.selected_features = None
        self.optimization_strategy = 'random'  # Default optimization strategy
        self.n_iter_random = 20  # Default iterations for RandomSearch
        self.n_trials_optuna = 30  # Default trials for Optuna
        self.preprocessor = None  # Added for the new predict method
        self.best_model_details = None # To store info about the best saved model
        self.params = {}

    def set_training_parameters(self, target_column, algorithms, cross_validation_split, recommendations, params):
        """Asigna los parámetros de entrenamiento a la instancia."""
        self.target = target_column
        self.algorithms = algorithms
        self.cross_validation_split = cross_validation_split
        self.recommendations = recommendations
        self.params = params

    def determine_problem_type(self, df, target_column=None):
        """
        Determina el tipo de problema basado en el dataset.

        Args:
        df (DataFrame): Objeto DataFrame con los datos cargados.
        target_column (str): Nombre de la columna objetivo.
        """
        try:
            if target_column is None:
                if self.target not in df.columns:
                    raise ValueError(f"La columna objetivo '{self.target}' no está presente en el dataset.")
                target_column = self.target
            
            self.problem_type = determine_problem_type(df, target_column)
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

    def preprocess_data_for_training(self, X, y):
        """
        Toma X e y, los combina, aplica el preprocesamiento básico y los vuelve a separar.
        """
        logger.info("Iniciando preprocesamiento de datos para entrenamiento...")
        
        # Combinar X e y en un solo DataFrame para el preprocesador
        # self.target ya contiene el nombre de la columna y
        temp_df = pd.concat([X, y], axis=1)
        
        # Llamar a la función de preprocesamiento global
        df_processed, preprocessor = basic_dfpreprocess(temp_df, target_column=self.target)
        
        # Guardar el preprocesador para uso futuro (ej. en predicciones)
        self.preprocessor = preprocessor
        
        # Volver a separar X e y del dataframe procesado
        if self.target not in df_processed.columns:
            logger.error(f"La columna objetivo '{self.target}' se perdió durante el preprocesamiento.")
            raise ValueError(f"Target column '{self.target}' not found in processed data.")
            
        y_processed = df_processed[self.target]
        X_processed = df_processed.drop(columns=[self.target])
        
        logger.info("Preprocesamiento de datos para entrenamiento completado.")
        return X_processed, y_processed, X_processed.columns.tolist()

    def split_and_train(self, X, y, dataset_name):
        """
        Preprocesa, divide los datos y entrena los modelos.
        """
        X_processed, y_series, feature_names = self.preprocess_data_for_training(X, y)
        
        test_split = 1 - (self.cross_validation_split / 100)
        
        if self.problem_type in ['Clasificación Binaria', 'Clasificación Multiclase']:
            # Usar StratifiedShuffleSplit para mantener la proporción de clases
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
            train_index, test_index = next(sss.split(X_processed, y_series))
            X_train, X_test = X_processed.iloc[train_index], X_processed.iloc[test_index]
            y_train, y_test = y_series.iloc[train_index], y_series.iloc[test_index]
        else: # Regresión
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y_series, test_size=test_split, random_state=42)
        
        logger.info(f"Datos divididos. Entrenamiento: {len(X_train)} filas, Prueba: {len(X_test)} filas.")

        trained_models_details, evaluation_results = self._train_models(
            X_train, y_train, X_test, y_test, self.algorithms, feature_names, dataset_name
        )
        return trained_models_details, evaluation_results

    def _train_models(self, X_train, y_train, X_test, y_test, algorithms, features, dataset_name):
        """
        Entrena una lista de modelos y devuelve los resultados.
        """
        trained_models_details = {}
        evaluation_results = {}
        evaluation = Evaluation(self.problem_type, dataset_name)

        for model_name in algorithms:
            try:
                original_algorithm_name = model_name if isinstance(model_name, str) else model_name.get('name')
                
                # Corregir el nombre del modelo si no es apropiado
                corrected_algorithm_name = self._find_and_correct_model_name(original_algorithm_name)

                if not corrected_algorithm_name:
                    continue # Saltar al siguiente modelo

                logger.info(f"Entrenando modelo: {corrected_algorithm_name}")
                
                # Usar el nombre original para obtener los params, pero el corregido para entrenar
                model, train_score, test_score, metrics = evaluation.train_and_evaluate_model(
                    corrected_algorithm_name, X_train, y_train, X_test, y_test, features, self.params.get(original_algorithm_name, {})
                )
                
                trained_models_details[corrected_algorithm_name] = model
                evaluation_results[corrected_algorithm_name] = {
                    'Train Score': train_score,
                    'Test Score': test_score,
                    'Metrics': metrics
                }
            except Exception as e:
                logger.error(f"Fallo al entrenar el modelo {original_algorithm_name}: {e}", exc_info=True)
        
        return trained_models_details, evaluation_results

    def _find_and_correct_model_name(self, model_name):
        """
        Verifica si el modelo es apropiado para el problema y, si no, intenta encontrar su homólogo.
        """
        if self._is_appropriate_model(model_name, self.problem_type):
            return model_name  # El modelo es correcto

        logger.warning(f"El modelo '{model_name}' no es apropiado para un problema de '{self.problem_type}'.")
        
        counterpart = None
        problem_type_lower = self.problem_type.lower()

        # Intentar encontrar el homólogo
        if "Classifier" in model_name and ('regresion' in problem_type_lower or 'regresión' in problem_type_lower):
            counterpart = model_name.replace("Classifier", "Regressor")
        elif "Regressor" in model_name and ('clasificacion' in problem_type_lower or 'clasificación' in problem_type_lower):
            counterpart = model_name.replace("Regressor", "Classifier")
        elif model_name == 'SVC' and ('regresion' in problem_type_lower or 'regresión' in problem_type_lower):
            counterpart = 'SVR'
        elif model_name == 'SVR' and ('clasificacion' in problem_type_lower or 'clasificación' in problem_type_lower):
            counterpart = 'SVC'
        
        # Verificar si el homólogo existe y es apropiado
        if counterpart and self._is_appropriate_model(counterpart, self.problem_type):
            logger.warning(f"Se cambiará automáticamente a su homólogo: '{counterpart}'.")
            return counterpart
        else:
            logger.error(f"No se pudo encontrar un homólogo apropiado para '{model_name}'. Se saltará este modelo.")
            return None

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
                    visualizer_obj.plot_residuals(y_test_data, y_pred)
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

                    saved_model_filename = serializer.to_pickle(
                        model=model_to_eval,
                        model_name=model_unique_name,
                        dataset_path=dataset_name_str, 
                        accuracy=metric_for_filename,
                        X_train=X_train_original_for_stats,
                        actual_feature_names=features_used_for_this_model,
                        preprocessor=fitted_preprocessor
                    )
                    logger.info(f"Modelo {model_unique_name} guardado exitosamente como {saved_model_filename}.")

                    if model_unique_name.startswith("GlobalBest_") and saved_model_filename:
                        self.best_model_details = {
                            "dataset_name": dataset_name_str,
                            "model_filename": saved_model_filename,
                            "model_unique_name": model_unique_name,
                            "metric_value": metric_for_filename,
                            "metric_name": primary_metric_key
                        }
                        logger.info(f"Detalles del mejor modelo global guardados: {self.best_model_details}")

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

    def _find_and_evaluate_best_model(self, X_train, y_train, X_test, y_test, problem_type, current_trained_models_list, feature_names):
        print("Seleccionando las mejores características (llamada a select_best_features)... ")
        
        best_model, best_selected_features, _, _, _, _ = recommend_best_model(
            X_train, y_train, X_test, y_test, problem_type, current_trained_models_list, feature_names
        )
        
        if best_model is None:
            print("No se pudo recomendar un mejor modelo global.")
            return None

        # Si se encuentra un modelo, se evalúa y guarda, pero la lógica principal está en recommend_best_model
        # Esta función principalmente orquesta y devuelve los detalles para ser añadidos
        
        # Crear un nombre único para el modelo global
        model_name = f"GlobalBest_{type(best_model).__name__}"
        
        # Evaluar el modelo
        evaluation = Evaluation(self.problem_type, self.dataset_name)
        metrics = evaluation.evaluate_model(best_model, X_test, y_test, best_selected_features)

        # Guardar el modelo
        model_path = serializer.save_model(best_model, self.dataset_name, model_name, metrics, feature_names, self.preprocessor)
        
        print(f"Mejor modelo global ({model_name}) guardado en {model_path}.")

        # Devolver los detalles del modelo para que se puedan añadir a la lista general
        return {
            'model': best_model,
            'name': model_name,
            'metrics': metrics,
            'features': best_selected_features,
            'path': model_path
        }

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

    def predict(self, model_name_selected: str, dataset_group_name: str, features_input: dict):
        """
        Realiza una predicción usando un modelo entrenado y un conjunto de características de entrada.
        model_name_selected: El nombre del archivo .pkl del modelo (ej. GlobalBest_LogisticRegression.pkl)
        dataset_group_name: El nombre del grupo del dataset (ej. breast_cancer)
        features_input: Un diccionario con los nombres de las características y sus valores.
        """
        try:
            # Cargar el diccionario del modelo usando la función de carga de serializer que devuelve el dict
            model_data_dict = serializer.load_model_from_group(dataset_group_name, model_name_selected)

            if not model_data_dict or 'model' not in model_data_dict:
                logger.error(f"No se pudo cargar la información del modelo para {dataset_group_name}/{model_name_selected}")
                return None

            model = model_data_dict['model']
            model_features = model_data_dict.get('feature_names') # Estas son las features que el MODELO espera
            loaded_preprocessor = model_data_dict.get('preprocessor')

            if not model_features:
                logger.error(f"No se encontraron nombres de características para el modelo {model_name_selected}.")
                if hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                    logger.info(f"Fallback: Usando feature_names_in_ del modelo: {model_features}")
                if not model_features:
                    logger.error("No se pueden determinar las características esperadas por el modelo.")
                    return None

            logger.info(f"\nPredicción con: {dataset_group_name}/{model_name_selected}")
            logger.info(f"Características esperadas por el modelo (en orden): {model_features}")
            logger.info(f"Características recibidas para la predicción: {features_input}")

            # Preparar las características usando el preprocesador cargado
            features_df = self._prepare_features(features_input, loaded_preprocessor, model_features)
            
            if features_df is None:
                logger.error("No se pudieron preparar las características para la predicción.")
                return None

            # Asegurarse de que las columnas estén en el orden correcto que el modelo espera
            try:
                features_df_ordered = features_df[model_features]
            except KeyError as e:
                logger.error(f"Error al ordenar características para predicción. Faltan columnas en DataFrame preparado: {e}")
                logger.error(f"Columnas en DataFrame preparado: {features_df.columns.tolist()}")
                logger.error(f"Columnas esperadas por el modelo: {model_features}")
                return None

            prediction = model.predict(features_df_ordered)
            
            # Obtener probabilidad si está disponible
            prediction_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    prediction_proba = model.predict_proba(features_df_ordered)
                except Exception as e:
                    logger.warning(f"No se pudo obtener la probabilidad de la predicción: {e}")

            class_label = self._get_class_label(prediction) # Asume que self.problem_type está seteado
            
            # Para _print_prediction, pasamos las features originales ingresadas por el usuario
            self._print_prediction(features_input, prediction, class_label, model_name_selected)
            
            # Devolver un diccionario con más detalles
            return {
                "prediction": prediction.tolist()[0] if isinstance(prediction, np.ndarray) else prediction,
                "class_label": class_label,
                "prediction_probability": prediction_proba.tolist()[0] if prediction_proba is not None else None
            }
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
        print(f"\n--- Prediction for {model_name} ---")
        print(f"Features: {featuresPredict}")
        print(f"Prediction: {prediction}")
        if class_label:
            print(f"Class Label: {class_label}")

    def start_training(self):
        """
        Función principal que orquesta todo el proceso de entrenamiento.
        """
        # CORRECCIÓN: Inicializar feature_names aquí para que esté disponible en todo el método.
        feature_names = None
        log_handler = MQTTLogHandler()
        logger.addHandler(log_handler)
        
        try:
            logger.info("Iniciando el proceso de entrenamiento...")
            if self.dataset is None or self.dataset.df is None:
                logger.error("El dataset no está cargado. Abortando el entrenamiento.")
                return

            df = self.dataset.df
            dataset_name = self.dataset_name or "default_dataset"
            
            if self.target is None:
                logger.error("La columna objetivo (target) no ha sido especificada. Abortando.")
                return
            
            # Asegurarse de que 'algorithms' es una lista no vacía.
            if not self.algorithms:
                logger.error("No se han especificado algoritmos para el entrenamiento. Abortando.")
                return

            try:
                if self.target not in df.columns:
                    prefixed_target = f"num__{self.target}"
                    if prefixed_target in df.columns:
                        self.target = prefixed_target
                        logger.info(f"Target actualizado a su versión procesada: {self.target}")
                    else:
                        logger.error(f"Columnas disponibles: {df.columns.tolist()}")
                        raise ValueError(f"La columna objetivo '{self.target}' no se encontró.")

                # Determina el tipo de problema basado en la columna objetivo del dataframe.
                self.problem_type = determine_problem_type(df[self.target])
                logger.info(f"Tipo de problema detectado: {self.problem_type}")

                y = df[self.target]
                X = df.drop(columns=[self.target])
                
                self.trained_models, evaluation_results = self.split_and_train(X, y, dataset_name)
                
                if not self.trained_models:
                    logger.warning("El entrenamiento no produjo ningún modelo.")
                    return None
                
                # Encontrar el mejor modelo entre los entrenados
                best_model_name, best_score_metrics, best_model_params = self.find_best_model(evaluation_results)

                # Obtenemos el objeto del modelo usando el nombre que encontramos.
                best_model = self.trained_models.get(best_model_name)
                
                # Lógica para guardar el mejor modelo
                if best_model and best_model_name:
                    final_score_value = best_score_metrics.get('Test Score', 0.0) if isinstance(best_score_metrics, dict) else best_score_metrics
                    logger.info(f"Mejor modelo seleccionado: {best_model_name} con score de {final_score_value:.4f}")
                    
                    try:
                        metrics_for_saving = evaluation_results.get(best_model_name, {})

                        # Ahora 'feature_names' está disponible aquí
                        model_path = serializer.save_model(
                            model=best_model,
                            dataset_name=self.dataset_name,
                            model_name=best_model_name,
                            metrics=metrics_for_saving,
                            feature_names=feature_names,
                            preprocessor=self.preprocessor,
                            model_params=best_model_params,
                            target_encoder=self.target_encoder if hasattr(self, 'target_encoder') else None
                        )
                        self.best_model_details = model_path
                        logger.info(f"Mejor modelo '{best_model_name}' guardado en: {model_path}")
                    except Exception as e:
                        logger.error(f"Error al guardar el mejor modelo '{best_model_name}': {e}", exc_info=True)
                else:
                    logger.warning("No se encontró un mejor modelo para guardar.")
                    self.best_model_details = None

                # FINALIZACIÓN
                logger.info("Proceso de entrenamiento finalizado.")
                return self.best_model_details

            except ValueError as ve:
                logger.error(f"Error de valor durante el entrenamiento: {ve}", exc_info=True)
                # No relanzar para permitir que el bloque finally se ejecute limpiamente.
            except Exception as e:
                logger.error(f"Error general durante el proceso de entrenamiento: {e}", exc_info=True)
                # No relanzar, el error ya ha sido logueado.
        
        finally:
            logger.info("Iniciando fase final de guardado de modelo...")
            
            # Encontrar el mejor modelo entre los entrenados
            if not evaluation_results:
                logger.error("No hay resultados de evaluación para determinar el mejor modelo.")
                self.best_model_details = None
            else:
                best_model_name, best_score_metrics, best_model_params = self.find_best_model(evaluation_results)

                # Obtenemos el objeto del modelo usando el nombre que encontramos.
                best_model = self.trained_models.get(best_model_name)
                
                # Lógica para guardar el mejor modelo
                if best_model and best_model_name:
                    final_score_value = best_score_metrics.get('Test Score', 0.0) if isinstance(best_score_metrics, dict) else best_score_metrics
                    logger.info(f"Mejor modelo seleccionado: {best_model_name} con score de {final_score_value:.4f}")
                    
                    try:
                        metrics_for_saving = evaluation_results.get(best_model_name, {})

                        # Ahora 'feature_names' está disponible aquí
                        model_path = serializer.save_model(
                            model=best_model,
                            dataset_name=self.dataset_name,
                            model_name=best_model_name,
                            metrics=metrics_for_saving,
                            feature_names=feature_names,
                            preprocessor=self.preprocessor,
                            model_params=best_model_params,
                            target_encoder=self.target_encoder if hasattr(self, 'target_encoder') else None
                        )
                        self.best_model_details = model_path
                        logger.info(f"Mejor modelo '{best_model_name}' guardado en: {model_path}")
                    except Exception as e:
                        logger.error(f"Error al guardar el mejor modelo '{best_model_name}': {e}", exc_info=True)
                else:
                    logger.warning("No se encontró un mejor modelo para guardar.")
                    self.best_model_details = None

            logger.info("Proceso de entrenamiento finalizado.")
            logger.removeHandler(log_handler)

    def find_best_model(self, evaluation_results):
        """
        Encuentra el mejor modelo basado en los resultados de la evaluación.
        """
        best_model_name = None
        best_score = float('-inf')

        for model_name, results in evaluation_results.items():
            score = results.get('Test Score', 0)
            if score > best_score:
                best_score = score
                best_model_name = model_name

        return best_model_name, evaluation_results[best_model_name], best_score
