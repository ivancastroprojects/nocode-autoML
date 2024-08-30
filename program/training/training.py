# training.py
import pandas as pd
import numpy as np

import os
from sklearn.model_selection import train_test_split
from training.train import train_custom_models
from training.train import recommend_best_model
import training.scikitdb.serializer as serializer
from data.datasetprocessing import basic_dfpreprocess
from data.dataset import Dataset
from training.evaluation import evaluate_classification_models, evaluate_regression_models

class Training:
    """
    Clase principal para gestionar el entrenamiento y evaluación de modelos de machine learning.
    """

    def __init__(self, problem_type=None, target=None, features=None, algorithms=None, crossvalidation=80, recommendations=False):
        """
        Inicializa la clase Training con los parámetros necesarios para el entrenamiento.
        """
        self.problem_type = problem_type  # Tipo de problema: 'classification' o 'regression'
        self.target = target  # Variable objetivo
        self.features = features  # Características a utilizar
        self.algorithms = algorithms if algorithms is not None else []  # Algoritmos a entrenar
        self.crossvalidation = crossvalidation  # Porcentaje de datos para entrenamiento
        self.recommendations = recommendations  # Si se deben hacer recomendaciones de modelos
        self.preprocessing = False  # Si se debe aplicar preprocesamiento a los datos
        self.trained_models = []  # Lista para almacenar los modelos entrenados

    def split_and_train(self, dataset, dataset_name, feature_names):
        """
        Divide el dataset, entrena los modelos y opcionalmente recomienda el mejor modelo.
        """
        print("\n------------------- ENTRENAMIENTOS --------------------")    
        X, y = self._prepare_data(dataset)
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        all_trained_models = self._train_models(X_train, y_train, X_test, y_test, dataset_name, feature_names)
        
        # Actualizar la lista de modelos entrenados
        self.trained_models = all_trained_models
        
        if self.recommendations:
            best_model = self._recommend_best_model(X_train, y_train, X_test, y_test, feature_names)
            all_trained_models.append(best_model)
            serializer.to_pickle(best_model, "model_bestsolution", dataset_name)
        
        return X_test, y_test, all_trained_models

    def _prepare_data(self, df):
        """
        Prepara los datos para el entrenamiento, separando features y target.
        """
        if self.target is not None and self.target in df.columns:
            df.target_names = self.target
        else:
            self.target = df.target_names
        
        y = df[self.target].values
        X = df.drop(columns=self.target)
        
        if self.features:
            X = X[self.features]
        else:
            self.features = X.columns.tolist()

        return X, y  # Devuelve X como DataFrame y y como array

    def _split_data(self, X, y):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        """
        return train_test_split(X, y, test_size=(100 - self.crossvalidation) / 100, random_state=42)

    def _train_models(self, X_train, y_train, X_test, y_test, dataset_name, feature_names):
        """
        Entrena todos los modelos especificados en self.algorithms.
        """
        all_trained_models = []
        for algorithm in self.algorithms:
            if algorithm['name'] in serializer.model_classes:
                model_class = serializer.model_classes[algorithm['name']]
                if self._is_appropriate_model(model_class):
                    base_model, optimized_model, selected_features = train_custom_models(X_train, y_train, algorithm, self.problem_type, dataset_name, feature_names, self.recommendations)
                    if base_model is not None and optimized_model is not None:
                        all_trained_models.extend([base_model, optimized_model])
                        
                        # Seleccionar las características para X_test_optimized
                        if isinstance(X_test, pd.DataFrame):
                            X_test_optimized = X_test.loc[:, selected_features]
                        elif isinstance(X_test, np.ndarray):
                            feature_indices = [feature_names.index(feature) for feature in selected_features]
                            X_test_optimized = X_test[:, feature_indices]
                        else:
                            raise TypeError("X_test debe ser un DataFrame de pandas o un array de NumPy")
                        
                        # Comparar modelos usando los conjuntos de datos apropiados
                        print(f"\nComparación para {algorithm['name']}:")
                        print("Modelo base (todas las características):")
                        self._evaluate_model(base_model, X_test, y_test)
                        print("\nModelo optimizado (características seleccionadas):")
                        self._evaluate_model(optimized_model, X_test_optimized, y_test)
                    else:
                        print(f"No se pudo entrenar el modelo {algorithm['name']}. Omitiendo...")
                else:
                    print(f"El algoritmo '{algorithm['name']}' no es apropiado para este caso. Omitiendo...")
            else:
                print(f"Modelo '{algorithm['name']}' no encontrado. Omitiendo...")
        return all_trained_models

    def _is_appropriate_model(self, model_class):
        """
        Verifica si un modelo es apropiado para el tipo de problema actual.
        """
        return ((self.problem_type == 'classification' and model_class in serializer.classification_models) or 
                (self.problem_type == 'regression' and model_class in serializer.regression_models))

    def _recommend_best_model(self, X_train, y_train, X_test, y_test, feature_names):
        """
        Recomienda el mejor modelo basado en una selección inteligente y optimización.
        """
        from training.modeloptimization import recommend_best_model
        return recommend_best_model(X_train, y_train, X_test, y_test, self.problem_type, self.trained_models, feature_names)

    def evaluate(self, X_test, y_test, all_trained_models):
        """
        Evalúa todos los modelos entrenados en el conjunto de prueba.
        """
        print("\n------------------- EVALUACIONES --------------------")
        eval_results = {}
        for model in all_trained_models:
            print(f"\nEvaluación del modelo {model.__class__.__name__}:")
            model_results = self._evaluate_model(model, X_test, y_test)
            self._print_evaluation_results(model_results)
            eval_results[model.__class__.__name__] = model_results
        return eval_results

    def _evaluate_model(self, model, X_test, y_test):
        """
        Evalúa un modelo específico usando las métricas apropiadas para el tipo de problema.
        """
        if self.problem_type == 'classification':
            return evaluate_classification_models([model], X_test, y_test, self.target, self.features)
        else:
            return evaluate_regression_models([model], X_test, y_test, self.target, self.features)

    def _print_evaluation_results(self, results):
        """
        Imprime los resultados de la evaluación de un modelo.
        """
        for key, value in results.items():
            print(f"{key}: {value}")
        print("-"*50)

    def predict(self, model_name, featuresPredict):
        """
        Realiza una predicción usando un modelo entrenado.
        """
        model = self._load_model(model_name)
        features_df = self._prepare_features(featuresPredict)
        prediction = model.predict(features_df.values)
        class_label = self._get_class_label(prediction)
        self._print_prediction(featuresPredict, prediction, class_label)
        return prediction

    def _load_model(self, model_name):
        """
        Carga un modelo entrenado desde el disco.
        """
        model_path = self._find_model_path(model_name)
        if not model_path:
            raise FileNotFoundError(f"No se encontró el modelo {model_name}")
        return serializer.from_pickle(model_path)

    def _find_model_path(self, model_name):
        """
        Busca la ruta de un modelo guardado.
        """
        for root, dirs, files in os.walk('models'):
            for file in files:
                if model_name in file:
                    return os.path.join(root, file)
        return None

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

    def _print_prediction(self, featuresPredict, prediction, class_label):
        """
        Imprime el resultado de una predicción.
        """
        print(f"The predicted {self.target} for {featuresPredict} is {prediction[0]:.2f} ({class_label})")