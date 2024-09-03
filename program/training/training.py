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
        self.poly_transform = None
        self.selected_features = None


    def split_and_train(self, X, y, dataset_path, feature_names):
        """
        Divide el dataset, entrena los modelos y opcionalmente recomienda el mejor modelo.
        """
        self.dataset_name = Path(dataset_path).stem  # Extraemos el nombre del dataset del path
        print(f"\n------------------- ENTRENAMIENTOS PARA {self.dataset_name} --------------------")    
        
        # Asegurarse de que feature_names sea una lista de strings
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                raise ValueError("Se deben proporcionar los nombres de las características cuando X no es un DataFrame.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        all_trained_models = self._train_models(X_train, y_train, X_test, y_test, dataset_path, feature_names)
        
        # Asegurarse de que all_trained_models sea una lista
        if not isinstance(all_trained_models, list):
            all_trained_models = [all_trained_models]
        
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
        all_trained_models = []

        for algorithm in self.algorithms:
            # Verificar si el algoritmo es apropiado para el tipo de problema
            if ((self.problem_type == 'classification' and algorithm['name'] in serializer.regression_models) or 
                (self.problem_type == 'regression' and algorithm['name'] in serializer.classification_models)):
                print(f"Advertencia: {algorithm['name']} no es apropiado para problemas de {self.problem_type}. Saltando...")
                continue
            
            if algorithm['name'] in serializer.model_classes:
                model_class = serializer.model_classes[algorithm['name']]
                if self._is_appropriate_model(model_class):
                    base_model, optimized_model, selected_features = train_custom_models(X_train, y_train, algorithm, self.problem_type, dataset_path, feature_names, self.recommendations)
                    if base_model is not None and optimized_model is not None:
                        # Almacenar las características utilizadas en cada modelo
                        print(f"\nModelo base de {algorithm['name']}")
                        base_model.feature_names_ = X_train.columns.tolist()
                        print(f"Modelo optimizado de {algorithm['name']}")
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
                        
                        # Comparar modelos
                        print("\nComparación de modelos:")
                        evaluation = Evaluation(self.problem_type, dataset_path)
                        evaluation.compare_models(base_model, optimized_model, X_test, y_test)
                        
                        # Generar visualizaciones
                        from data.visualizer import Visualizer
                        visualizer = Visualizer()
                        visualizer.set_model_name(f"Training_{algorithm['name']}")  # Establecemos un nombre para la carpeta de salida
                        visualizer.set_dataset_name(self.dataset_name)  # Asumiendo que tienes un atributo dataset_name en tu clase
    
                        # Generar visualizaciones para el modelo base
                        y_pred_base = base_model.predict(X_test)
                        visualizer.generate_visualizations(base_model, X_test, y_test, y_pred_base)
                        
                        # Generar visualizaciones para el modelo optimizado
                        X_test_optimized = X_test[selected_features]
                        y_pred_optimized = optimized_model.predict(X_test_optimized)
                        visualizer.generate_visualizations(optimized_model, X_test_optimized, y_test, y_pred_optimized)
                        
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
            best_model, best_selected_features, X_train_selected, X_test_selected, poly, poly_feature_names = self._recommend_best_model(X_train, y_train, X_test, y_test, self.problem_type, feature_names)
        
            # Usar X_train_selected y X_test_selected para entrenar y evaluar el modelo
            best_model.fit(X_train_selected, y_train)
            y_pred = best_model.predict(X_test_selected)
        
            # Evaluar el modelo
            evaluation = Evaluation(self.problem_type, dataset_path)
            evaluation.evaluate(best_model, X_test_selected, y_test, y_pred, best_selected_features)
        
            all_trained_models.append((best_model, evaluation, best_selected_features))

            # Evaluar el mejor modelo
            self._evaluate_model(best_model, X_test_selected, y_test)

            # Guardar información sobre la transformación polinomial
            self.poly_transform = poly
            self.selected_features = best_selected_features
            
            # Calcular la precisión del mejor modelo
            if self.problem_type == 'classification':
                best_accuracy = accuracy_score(y_test, best_model.predict(X_test_selected))
            else:
                best_accuracy = r2_score(y_test, best_model.predict(X_test_selected))
            
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

    def _recommend_best_model(self, X_train, y_train, X_test, y_test, problem_type, feature_names):
        from training.modeloptimization import recommend_best_model
        best_model, best_selected_features, X_train_selected, X_test_selected, poly, poly_feature_names = recommend_best_model(
            X_train, y_train, X_test, y_test, problem_type, self.trained_models, feature_names
        )
        return best_model, best_selected_features, X_train_selected, X_test_selected, poly, poly_feature_names

    def evaluate(self, X_test, y_test, all_trained_models):
        print("\n------------------- EVALUACIONES --------------------")
        eval_results = {}
        
        # Comprobar si all_trained_models es una lista o un solo modelo
        if not isinstance(all_trained_models, list):
            all_trained_models = [all_trained_models]
        
        for model in all_trained_models:
            # Si el modelo es una tupla, extraer solo el modelo
            if isinstance(model, tuple):
                model = model[0]
            
            print(f"\nEvaluación del modelo {model.__class__.__name__}:")
            model_results = self._evaluate_model(model, X_test, y_test)
            self._print_evaluation_results(model_results)
            eval_results[model.__class__.__name__] = model_results
        
        return eval_results

    def _evaluate_model(self, model, X_test, y_test):
        print(f"Evaluation class: {Evaluation}")
        print(f"Problem type: {self.problem_type}, Dataset name: {self.dataset_name}")
        
        # Obtener las características utilizadas durante el entrenamiento
        if hasattr(model, 'feature_names_'):
            model_features = model.feature_names_
        else:
            # Si el modelo no tiene feature_names_, usar todas las características disponibles
            model_features = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else list(range(X_test.shape[1]))

        # Convertir X_test a DataFrame si es un array de NumPy, manteniendo los nombres originales
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=model_features)

        # Asegurarse de que X_test tenga las columnas correctas
        X_test_filtered = X_test[model_features]

        try:
            evaluation = Evaluation(self.problem_type, self.dataset_name)
        except TypeError:
            print("Advertencia: La clase Evaluation no acepta argumentos. Usando inicialización por defecto.")
            evaluation = Evaluation()
            evaluation.problem_type = self.problem_type
            evaluation.dataset_name = self.dataset_name

        if self.problem_type == 'classification':
            return evaluation.evaluate_classification_models([model], X_test_filtered, y_test, self.target, model_features)
        else:
            return evaluation.evaluate_regression_models([model], X_test_filtered, y_test, self.target, model_features)

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
        print(f"\nPredicción realizada con el modelo: {model_name}")
        print(f"La {self.target} predicha para las características proporcionadas es {prediction[0]:.2f} ({class_label})")
        print("\nCaracterísticas utilizadas para la predicción:")
        for feature, value in featuresPredict.items():
            print(f"  {feature}: {value:.4f}")
        print("\nNota: Si algunas características esperadas por el modelo no estaban presentes,")
        print("se utilizaron valores predeterminados (0 o la media del conjunto de entrenamiento).")