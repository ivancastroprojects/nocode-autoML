# training.py
from sklearn.model_selection import train_test_split
import api_interface
from utils import auto_preprocess
import train
import trainingparams
import serializer
from dataset import Dataset
import pandas as pd
import utils
import os

class Training:
    def __init__(self, dataset: Dataset = None):
        self.dataset = dataset
        self.dataset_name = dataset.df.Name if dataset else None
        self.test_dataset = None
        self.training = None
        self.crossvalidation = None
        self.algorithms = None
        self.target = None
        self.features = None
        self.preprocessing = None
        self.recommendations = False
    
    def train_and_evaluate(self):
        dataset = self.dataset.as_dataframe()

        X = dataset.drop(columns=self.target)
        y = dataset[self.target]
        
        ######### ENTRENAMIENTO CUSTOM #########
        # Si el usuario especifica las columnas, usarlas
        if self.features:
            X = X[self.features]

        # Convertir a valores numpy
        X = X.values
        y = y.values

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - self.crossvalidation) / 100, random_state=42)

        # Comprobar que los algoritmos pedidos concuerden con el tipo de problema
        problem_type = utils.determine_problem_type(y)
        
        # Entrenar los modelos
        trained_models = train.train_models(X_train, y_train, self.algorithms, problem_type, self.dataset_name)       
         
        # Evaluar los modelos
        evaluation_results = {}
        for model in trained_models:
            if (problem_type == 'classification'):
                eval_results = train.evaluate_classification_models([model], X_test, y_test)
            else:
                eval_results = train.evaluate_regression_models([model], X_test, y_test)
            
            # Imprimir las métricas con espaciado
            print(f"Evaluación del modelo {model.__class__.__name__}:")
            for key, value in evaluation_results.items():
                print(f"\n{key}: {value}")
            print("\n" + "-"*50 + "\n")
            
            evaluation_results.update(eval_results)

            ######### ENTRENAMIENTO AUTOMÁTICO #########
            if self.recommendations:
                # Entrenamiento automático detectando columnas más relevantes
                self.train_with_important_features()
                
                # Entrenar modelos con parámetros recomendados
                recommended_params = trainingparams.train_recommendedparams(X_train, y_train, self.algorithms)   
                evaluation_results.update(recommended_params)
         
        ######### ENVÍO DE DATOS #########
        # Enviar resultados de evaluación y parámetros recomendados a la API
        #api_interface.POST_modeleval(evaluation_results)

    def train_with_important_features(self, k=10):
        dataset = self.dataset.as_dataframe()
        
        # Separar las características (X) de la variable objetivo (y)
        X = dataset.drop(columns=self.target).values
        y = dataset[self.target].values

        # Seleccionar las características más relevantes
        X_new, selected_features = train.select_features(X, y, model, k=k)
        print(f"Selected features: {dataset.columns[selected_features]}")
        
        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=(100 - self.crossvalidation) / 100, random_state=42)
        
        # Entrenar los modelos
        trained_models = train.train_models(X_train, y_train, self.algorithms, self.dataset_name)
        
        # Evaluar los modelos
        evaluation_results = {}
        for model in trained_models:
            if isinstance(model, tuple(serializer.classification_models)):
                eval_results = train.evaluate_classification_models([model], X_test, y_test)
            else:
                eval_results = train.evaluate_regression_models([model], X_test, y_test)
            evaluation_results.update(eval_results)

        # Enviar resultados de evaluación a la API
        api_interface.POST_modeleval(evaluation_results)

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
            features_df = auto_preprocess(features_df, target_column=None)
        
        # Realizar la predicción
        prediction = model.predict(features_df)
        
        print(f"The predicted {self.target} for {featuresPredict} is {prediction[0]:.2f}")
        return prediction