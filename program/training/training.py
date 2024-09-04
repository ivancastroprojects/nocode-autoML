# training.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from training.train import train_custom_models
import training.scikitdb.serializer as serializer
from data.datasetprocessing import determine_problem_type_from_dataset, basic_dfpreprocess
from data.dataset import Dataset
from training.evaluation import Evaluation
from utils.logger import logger
from data.visualizer import Visualizer

class Training:
    """
    Clase principal para gestionar el entrenamiento y evaluación de modelos de machine learning.
    """

    def __init__(self):
        """
        Inicializa la clase Training con los parámetros necesarios para el entrenamiento.
        """
        self.problem_type = None
        self.target = 'target'  # Asumimos que la columna objetivo se llama 'target' por defecto
        self.dataset_name = None
        self.features = None
        self.algorithms = []
        self.crossvalidation = 80  # Porcentaje de datos para entrenamiento
        self.recommendations = False  # Si se deben hacer recomendaciones de modelos
        self.preprocessing = False  # Si se debe aplicar preprocesamiento a los datos
        self.trained_models = []  # Lista para almacenar los modelos entrenados
        self.poly_transform = None
        self.selected_features = None

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
            df = dataset.get_dataframe()
            if self.problem_type is None:
                self.determine_problem_type(dataset)
            
            if self.problem_type is None:
                raise ValueError("No se pudo determinar el tipo de problema.")

            if self.target not in df.columns:
                raise ValueError(f"La columna objetivo '{self.target}' no está presente en el dataset.")

            X = df.drop(columns=[self.target])
            y = df[self.target]
            feature_names = X.columns.tolist()

            # Calcular el tamaño del conjunto de prueba basado en crossvalidation
            test_size = (100 - self.crossvalidation) / 100

            # Dividir el dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Preparar las rutas para los archivos de entrenamiento y prueba
            base_name = os.path.splitext(os.path.basename(dataset_path))[0]
            train_path = os.path.join(os.path.dirname(dataset_path), f"{base_name}_train.csv")
            test_path = os.path.join(os.path.dirname(dataset_path), f"{base_name}_test.csv")

            # Guardar los conjuntos de entrenamiento y prueba, sobrescribiendo si ya existen
            pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
            pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

            logger.info(f"Conjuntos de entrenamiento y prueba guardados en {train_path} y {test_path}")
            logger.info(f"Tamaño del conjunto de entrenamiento: {len(X_train)}, Tamaño del conjunto de prueba: {len(X_test)}")

            logger.info(f"Preparando para entrenar modelos con dataset: {dataset_path}")
            logger.info(f"Características: {feature_names}")
            logger.info(f"Tipo de problema: {self.problem_type}")
            logger.info(f"Algoritmos seleccionados: {self.algorithms}")

            trained_models, evaluation_results = self._train_models(X_train, y_train, X_test, y_test, dataset_path, feature_names)

            return X_test, y_test, trained_models, evaluation_results

        except Exception as e:
            logger.error(f"Error en split_and_train: {str(e)}")
            # Devolver valores por defecto en caso de error
            return None, None, [], {}

    def _train_models(self, X_train, y_train, X_test, y_test, dataset_path, feature_names):
        """
        Entrena los modelos seleccionados por el usuario y los optimiza si se solicita.

        Args:
        X_train (DataFrame): Características de entrenamiento.
        y_train (Series): Variable objetivo de entrenamiento.
        X_test (DataFrame): Características de prueba.
        y_test (Series): Variable objetivo de prueba.
        dataset_path (str): Ruta del dataset.
        feature_names (list): Nombres de las características.

        Returns:
        tuple: (trained_models, evaluation_results)
        """
        trained_models = []
        evaluation_results = {}
        evaluation = Evaluation(self.problem_type, dataset_path)
        visualizer = Visualizer(dataset_path)

        for algorithm in self.algorithms:
            try:
                algorithm_name = algorithm if isinstance(algorithm, str) else algorithm.get('name')
                algorithm_params = {} if isinstance(algorithm, str) else algorithm.get('params', {})

                if not self._is_appropriate_model(algorithm_name):
                    logger.warning(f"{algorithm_name} no es apropiado para problemas de {self.problem_type}. Saltando...")
                    continue
                
                base_model, optimized_model, selected_features = train_custom_models(
                    X_train, y_train, {'name': algorithm_name, 'params': algorithm_params}, 
                    self.problem_type, dataset_path, feature_names, self.recommendations
                )
                
                if base_model is not None:
                    results = self._evaluate_and_visualize_model(base_model, optimized_model, X_test, y_test, selected_features, 
                                                                evaluation, visualizer, dataset_path, algorithm_name)
                    evaluation_results[algorithm_name] = results
                    trained_models.extend([base_model, optimized_model] if optimized_model else [base_model])
                else:
                    logger.warning(f"No se pudo entrenar el modelo {algorithm_name}. Omitiendo...")
            except Exception as e:
                logger.error(f"Error al entrenar el modelo {algorithm_name}: {str(e)}")

        if self.recommendations:
            self._find_and_evaluate_best_model(X_train, y_train, X_test, y_test, feature_names, 
                                            dataset_path, trained_models, evaluation, visualizer)

        return trained_models, evaluation_results

    def _is_appropriate_model(self, model_name):
        """
        Verifica si un modelo es apropiado para el tipo de problema actual.
        """
        if self.problem_type is None:
            logger.warning("El tipo de problema no ha sido determinado. No se puede verificar la idoneidad del modelo.")
            return False
        
        classification_models = serializer.classification_models if hasattr(serializer, 'classification_models') else []
        regression_models = serializer.regression_models if hasattr(serializer, 'regression_models') else []
        
        return ((self.problem_type == 'classification' and model_name in classification_models) or 
                (self.problem_type == 'regression' and model_name in regression_models))

    def _evaluate_and_visualize_model(self, base_model, optimized_model, X_test, y_test, selected_features, 
                                      evaluation, visualizer, dataset_path, model_name):
        """
        Evalúa, compara y visualiza los modelos base y optimizado.
        """
        logger.info(f"\nEvaluación y visualización del modelo {model_name}:")
        
        # Evaluar y visualizar modelo base
        base_results = self._evaluate_model(base_model, X_test, y_test, evaluation)
        visualizer.plot_model_performance(base_model, X_test, y_test, f"{model_name}_base")
        
        if optimized_model:
            # Evaluar y visualizar modelo optimizado
            X_test_optimized = X_test[selected_features]
            optimized_results = self._evaluate_model(optimized_model, X_test_optimized, y_test, evaluation)
            visualizer.plot_model_performance(optimized_model, X_test_optimized, y_test, f"{model_name}_optimized")
            
            # Comparar modelos
            logger.info("\nComparación de modelos:")
            evaluation.compare_models(base_model, optimized_model, X_test, y_test)
            visualizer.plot_model_comparison(base_model, optimized_model, X_test, y_test, model_name)
            
            return {'base': base_results, 'optimized': optimized_results}
        
        return {'base': base_results}

    def _evaluate_model(self, model, X_test, y_test, evaluation):
        """
        Evalúa un modelo utilizando la clase Evaluation.
        """
        if self.problem_type == 'classification':
            results = evaluation.evaluate_classification_models([model], X_test, y_test, self.target)
        else:
            results = evaluation.evaluate_regression_models([model], X_test, y_test, self.target)
        self._print_evaluation_results(results)
        return results

    def _find_and_evaluate_best_model(self, X_train, y_train, X_test, y_test, feature_names, 
                                      dataset_path, all_trained_models, evaluation, visualizer):
        """
        Encuentra, evalúa y visualiza el mejor modelo para el problema actual.
        """
        best_model, best_selected_features, X_train_selected, X_test_selected, poly, poly_feature_names = self._recommend_best_model(
            X_train, y_train, X_test, y_test, self.problem_type, all_trained_models, feature_names
        )
        
        if best_model is not None:
            logger.info("\nEvaluación y visualización del mejor modelo recomendado:")
            best_results = self._evaluate_model(best_model, X_test_selected, y_test, evaluation)
            visualizer.plot_model_performance(best_model, X_test_selected, y_test, "best_model")
            
            self.poly_transform = poly
            self.selected_features = best_selected_features
            
            accuracy = self._calculate_accuracy(best_model, X_test_selected, y_test)
            model_name = best_model.__class__.__name__
            serializer.to_pickle(best_model, f"{model_name}_best", dataset_path, round(accuracy * 100, 1))

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
        if self.problem_type == 'classification':
            return accuracy_score(y_test, model.predict(X_test))
        else:
            return r2_score(y_test, model.predict(X_test))

    def _recommend_best_model(self, X_train, y_train, X_test, y_test, problem_type, feature_names):
        from training.modeloptimization import recommend_best_model
        best_model, best_selected_features, X_train_selected, X_test_selected, poly, poly_feature_names = recommend_best_model(
            X_train, y_train, X_test, y_test, problem_type, self.trained_models, feature_names
        )
        return best_model, best_selected_features, X_train_selected, X_test_selected, poly, poly_feature_names

    def predict(self, model_name, featuresPredict):
        """
        Realiza una predicción usando un modelo entrenado.
        """
        model_data = serializer.load_model(model_name)
        
        # Verificar si model_data es un diccionario y extraer el modelo
        if isinstance(model_data, dict):
            if 'model' in model_data:
                model = model_data['model']
                model_features = model_data.get('features', [])
            else:
                raise ValueError(f"El modelo cargado '{model_name}' no contiene un objeto de modelo válido.")
        else:
            model = model_data
            model_features = getattr(model, 'feature_names_', [])

        # Verificar si las características proporcionadas existen en el modelo
        missing_features = set(featuresPredict.keys()) - set(model_features)
        if missing_features:
            raise ValueError(f"Las siguientes características no existen en el modelo: {missing_features}")

        features_df = self._prepare_features(featuresPredict)

        # Asegurarse de que features_df tenga todas las características necesarias
        for feature in model_features:
            if feature not in features_df.columns:
                features_df[feature] = 0  # O cualquier otro valor por defecto apropiado

        # Seleccionar solo las características utilizadas por el modelo
        features_df = features_df[model_features]

        prediction = model.predict(features_df.values)
        class_label = self._get_class_label(prediction)
        self._print_prediction(featuresPredict, prediction, class_label, model_name)
        return prediction
    
    def _prepare_features(self, featuresPredict):
        """
        Prepara las características para la predicción.
        """
        features_df = pd.DataFrame([featuresPredict])
        features_df.columns = [Dataset.clean_filename(col) for col in features_df.columns]
        if self.preprocessing:
            features_df = basic_dfpreprocess(features_df, target_column=None)
        return features_df

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