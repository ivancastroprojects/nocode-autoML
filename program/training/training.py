# training.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from training.train import train_model
from training.evaluation import evaluate_classification_models, evaluate_regression_models
import training.scikitdb.serializer as serializer
from training.trainoptimizations import recommend_best_model
from data.datasetprocessing import basic_dfpreprocess
from data.dataset import Dataset

class Training:
    def __init__(self, problem_type=None, target=None, features=None, algorithms=None, crossvalidation=80, recommendations=False):
        self.problem_type = problem_type
        self.target = target
        self.features = features
        self.algorithms = algorithms if algorithms is not None else []
        self.crossvalidation = crossvalidation
        self.recommendations = recommendations

    # Función para separar datos de entrenamiento y test, y entrenar
    def split_and_train(self, dataset, dataset_name, feature_names):
        print("\n------------------- ENTRENAMIENTOS --------------------")    
        df = dataset
        
        # VARIABLE OBJETIVO DEL DATASET
        if self.target is not None and self.target in df.columns:
            df.target_names = self.target
        else:
            self.target = df.target_names
        
        y = df[self.target].values
        
        # CARACTERÍSTICAS DEL DATASET
        X = df.drop(columns=self.target)
        
        if self.features:
            X = X[self.features]
        else:
            self.features = X.columns.tolist()

        X = X.values

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - self.crossvalidation) / 100, random_state=42)
        
        # Entrenar los modelos pedidos y óptimos para cada algoritmo
        all_trained_models = []
        for algorithm in self.algorithms:
            if algorithm['name'] in serializer.model_classes:
                model_class = serializer.model_classes[algorithm['name']]
                if ((self.problem_type == 'classification' and model_class in serializer.classification_models) or 
                    (self.problem_type == 'regression' and model_class in serializer.regression_models)):
                    trained_models = train_model(X_train, y_train, algorithm, self.problem_type, dataset_name, feature_names, self.recommendations)
                    all_trained_models.extend(trained_models)
                else:
                    print(f"El algoritmo '{algorithm['name']}' no es apropiado para este caso. Omitiendo...")
            else:
                print(f"Modelo '{algorithm['name']}' no encontrado. Omitiendo...")
        
        # Recomendar el mejor algoritmo
        if self.recommendations:
            print("\n------------------- MODELO ÓPTIMO --------------------")
            best_model = recommend_best_model(X_train, y_train, X_test, y_test)
            all_trained_models.append(best_model)
            serializer.to_pickle(best_model, "model_bestsolution", dataset_name)
        
        return X_test, y_test, all_trained_models

    # Evaluar los modelos
    def evaluate(self, X_test, y_test, all_trained_models):
        print("\n------------------- EVALUACIONES --------------------")
        eval_results = {}
        for model in all_trained_models:
            print(f"\nEvaluación del modelo {model.__class__.__name__}:")
            if self.problem_type == 'classification':
                model_results = evaluate_classification_models([model], X_test, y_test, self.target, self.features)
            else:
                model_results = evaluate_regression_models([model], X_test, y_test, self.target, self.features)
            
            for key, value in model_results.items():
                print(f"{key}: {value}")
            print("-"*50)
            
            eval_results[model.__class__.__name__] = model_results
        
        return eval_results    
         
    def predict(self, model_name, featuresPredict):
        # Buscar el modelo en la carpeta models
        model_path = None
        for root, dirs, files in os.walk('models'):
            if model_name in files:
                model_path = os.path.join(root, model_name)
                break
        
        if not model_path:
            raise FileNotFoundError(f"No se encontró el modelo {model_name}")
        
        # Cargar el modelo desde el archivo
        model = serializer.from_pickle(model_path)
        
        # Convertir las características proporcionadas a un DataFrame
        features_df = pd.DataFrame([featuresPredict])

        # Limpiar nombres de columnas
        features_df.columns = [Dataset.clean_filename(col) for col in features_df.columns]

        # Realizar preprocesamiento si se especificó durante el entrenamiento
        if self.preprocessing:
            features_df = basic_dfpreprocess(features_df, target_column=None)
        
        # Convertir el DataFrame a un array numpy
        features_array = features_df.values
        
        # Realizar la predicción
        prediction = model.predict(features_array)
        
        # Determinar el tipo de problema
        if self.problem_type == 'classification' and hasattr(self, 'class_labels'):
            # Obtener la etiqueta de clase correspondiente
            class_label = Dataset.class_labels.get(int(prediction[0]), prediction[0])
        else:
            class_label = prediction[0]
        
        print(f"The predicted {self.target} for {featuresPredict} is {prediction[0]:.2f} ({class_label})")
        return prediction