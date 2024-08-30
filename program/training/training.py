# training.py
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from training.train import train_custom_models
import training.scikitdb.serializer as serializer
from data.datasetprocessing import basic_dfpreprocess
from data.dataset import Dataset
from training.evaluation import Evaluation

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
        self.features = features if features is not None else []  # Características a utilizar
        self.algorithms = algorithms if algorithms is not None else []  # Algoritmos a entrenar
        self.crossvalidation = crossvalidation  # Porcentaje de datos para entrenamiento
        self.recommendations = recommendations  # Si se deben hacer recomendaciones de modelos
        self.preprocessing = False  # Si se debe aplicar preprocesamiento a los datos
        self.trained_models = []  # Lista para almacenar los modelos entrenados
        self.dataset_name = None  # Inicializamos el nombre del dataset como None

    def split_and_train(self, dataset, dataset_path, feature_names):
        """
        Divide el dataset, entrena los modelos y opcionalmente recomienda el mejor modelo.
        """
        self.dataset_name = Path(dataset_path).stem  # Extraemos el nombre del dataset del path
        print(f"\n------------------- ENTRENAMIENTOS PARA {self.dataset_name} --------------------")    
        X, y = self._prepare_data(dataset)
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        all_trained_models = self._train_models(X_train, y_train, X_test, y_test, dataset_path, feature_names)
        
        # Actualizar la lista de modelos entrenados
        self.trained_models = all_trained_models
        
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

    def _train_models(self, X_train, y_train, X_test, y_test, dataset_path, feature_names):
        """
        Entrena todos los modelos especificados en self.algorithms.
        """
        all_trained_models = []
        for algorithm in self.algorithms:
            if algorithm['name'] in serializer.model_classes:
                model_class = serializer.model_classes[algorithm['name']]
                if self._is_appropriate_model(model_class):
                    base_model, optimized_model, selected_features = train_custom_models(X_train, y_train, algorithm, self.problem_type, dataset_path, feature_names, self.recommendations)
                    if base_model is not None and optimized_model is not None:
                        # Almacenar las características utilizadas en cada modelo
                        base_model.feature_names_ = X_train.columns.tolist()
                        optimized_model.feature_names_ = selected_features
                        all_trained_models.extend([base_model, optimized_model])
                        
                        # Seleccionar las características para X_test_optimized
                        if isinstance(X_test, pd.DataFrame):
                            X_test_optimized = X_test[selected_features]
                        elif isinstance(X_test, np.ndarray):
                            feature_indices = [feature_names.index(feature) for feature in selected_features]
                            X_test_optimized = X_test[:, feature_indices]
                        else:
                            raise TypeError("X_test debe ser un DataFrame de pandas o un array de NumPy")
                        
                        # Calcular la precisión o mejor métrica
                        if self.problem_type == 'classification':
                            base_accuracy = accuracy_score(y_test, base_model.predict(X_test))
                            optimized_accuracy = accuracy_score(y_test, optimized_model.predict(X_test_optimized))
                        else:  # regression
                            base_accuracy = r2_score(y_test, base_model.predict(X_test))
                            optimized_accuracy = r2_score(y_test, optimized_model.predict(X_test_optimized))
                        
                        # Guardar los modelos con el nombre correcto del dataset
                        serializer.to_pickle(base_model, f"{algorithm['name']}_base", dataset_path, round(base_accuracy * 100, 1))
                        serializer.to_pickle(optimized_model, f"{algorithm['name']}_optimized", dataset_path, round(optimized_accuracy * 100, 1))
                        
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
        
        # Modelo óptimo para el problema actual (mejor modelo y parámetros)
        if self.recommendations:
            best_model, best_selected_features = self._recommend_best_model(X_train, y_train, X_test, y_test, feature_names)
            if best_model is not None:
                all_trained_models.append(best_model)
                
                # Seleccionar las características para X_test_best
                if isinstance(X_test, pd.DataFrame):
                    X_test_best = X_test[best_selected_features]
                elif isinstance(X_test, np.ndarray):
                    feature_indices = [feature_names.index(feature) for feature in best_selected_features]
                    X_test_best = X_test[:, feature_indices]
                else:
                    raise TypeError("X_test debe ser un DataFrame de pandas o un array de NumPy")
                
                # Calcular la precisión del mejor modelo
                if self.problem_type == 'classification':
                    best_accuracy = accuracy_score(y_test, best_model.predict(X_test_best))
                else:
                    best_accuracy = r2_score(y_test, best_model.predict(X_test_best))
                
                # Usar el nombre de la clase del modelo en lugar de intentar acceder a 'name'
                model_name = best_model.__class__.__name__
                serializer.to_pickle(best_model, f"{model_name}_best", dataset_path, round(best_accuracy * 100, 1))
        
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
        # Obtener las características utilizadas durante el entrenamiento
        if hasattr(model, 'feature_names_'):
            model_features = model.feature_names_
        else:
            # Si el modelo no tiene feature_names_, usar todas las características
            model_features = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else list(range(X_test.shape[1]))

        # Filtrar X_test para incluir solo las características utilizadas en el entrenamiento
        if isinstance(X_test, pd.DataFrame):
            X_test_filtered = X_test[model_features]
        else:  # Asumimos que es un array de numpy
            feature_indices = [list(X_test.columns).index(feature) for feature in model_features]
            X_test_filtered = X_test[:, feature_indices]

        evaluation = Evaluation(self.problem_type, self.dataset_name)
        if self.problem_type == 'classification':
            return evaluation.evaluate_classification_models([model], X_test_filtered, y_test, self.target)
        else:
            return evaluation.evaluate_regression_models([model], X_test_filtered, y_test, self.target)

    # ... (otros métodos)

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
        model = serializer.load_model(model_name)
        features_df = self._prepare_features(featuresPredict)

        # Obtener las características utilizadas durante el entrenamiento
        if hasattr(model, 'feature_names_'):
            model_features = model.feature_names_
        else:
            # Si el modelo no tiene feature_names_, asumimos que usa todas las características
            model_features = self.features if self.features else features_df.columns.tolist()

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
        print(f"\nPredicción realizada con el modelo: {model_name}")
        print(f"La {self.target} predicha para las características proporcionadas es {prediction[0]:.2f} ({class_label})")
        print("\nCaracterísticas utilizadas para la predicción:")
        for feature, value in featuresPredict.items():
            print(f"  {feature}: {value:.4f}")
        print("\nNota: Si algunas características esperadas por el modelo no estaban presentes,")
        print("se utilizaron valores predeterminados (0 o la media del conjunto de entrenamiento).")