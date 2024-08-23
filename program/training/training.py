# training.py
from sklearn.model_selection import train_test_split
from utils.utils import auto_preprocess
import training.train as train
import training.trainingparams as trainingparams
import training.serializer as serializer
from data.dataset import Dataset
import pandas as pd
import utils.utils as utils
import os

dataset = None

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
        self.problem_type = None
        self.class_labels = None
    
    def train_and_evaluate(self, X_new = None, features = None):
        dataset = self.dataset.as_dataframe()

        # VARIABLE OBJETIVO DEL DATASET
        if self.target is not None and self.target in dataset.columns:
            dataset.target_names = self.target
        else: self.target = dataset.target_names
        
        y = dataset[self.target]
        y = y.values
        
        # CATEGORÍAS DEL DATASET
        if X_new is not None:
            X = X_new
            
        X = dataset.drop(columns=self.target)
        
        if features is not None and len(features) > 0:
            X = X[features]
        else:
            if self.features:
                X = X[self.features]
            else:
                self.features = X

        X = X.values

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - self.crossvalidation) / 100, random_state=42)

        # Comprobar que los algoritmos pedidos concuerden con el tipo de problema
        self.problem_type = utils.determine_problem_type(y)
        
        # Entrenar los modelos
        trained_models = train.train_models(X_train, y_train, self.algorithms, self.problem_type, self.dataset_name)       
        
        # Evaluar los modelos
        print("\n------------------- EVALUACIONES --------------------")
        print(f"\nEjemplos de validación:\n")
        eval_results = {}
        for model in trained_models:
            if (self.problem_type == 'classification'):
                eval_results = train.evaluate_classification_models([model], X_test, y_test, self.target, self.features)
            else:
                eval_results = train.evaluate_regression_models([model], X_test, y_test, self.target, self.features)

            # Imprimir las métricas con espaciado
            print(f"Evaluación del modelo {model.__class__.__name__}:")
            for key, value in eval_results.items():
                print(f"\n{key}: {value}")
            print("\n" + "-"*50 + "\n")
            
            eval_results.update(eval_results)
            
            ######### OPTIMIZACIÓN AUTOMÁTICA DEL ENTRENAMIENTO #########

            if self.recommendations and features is None:
                print(f"\n------- OPTIMIZACIÓN AUTOMÁTICA DEL MODELO {model.__class__.__name__} --------")
                # Detección automática de variables fuertemente dependientes con la target para quitarlas y que no afecte a la inferencia
                # Entrenamiento automático detectando columnas más relevantes
                trainingparams.train_with_important_features(dataset, self.target, model)
                #eval_results.update(recommended_params)
                
        ######### ENVÍO DE DATOS #########
        # Enviar resultados de evaluación y parámetros recomendados a la API
        #api_interface.POST_modeleval(evaluation_results)
         
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
        
        # Convertir el DataFrame a un array numpy
        features_array = features_df.values
        
        # Realizar la predicción
        prediction = model.predict(features_array)
        
        # Determinar el tipo de problema
        if self.problem_type == 'classification' and hasattr(self, 'class_labels'):
            # Obtener la etiqueta de clase correspondiente
            class_label = self.class_labels.get(int(prediction[0]), prediction[0])
        else:
            class_label = prediction[0]
        
        print(f"The predicted {self.target} for {featuresPredict} is {prediction[0]:.2f} ({class_label})")
        return prediction