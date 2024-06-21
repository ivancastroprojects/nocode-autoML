Buenas! estoy realizando el documento de la memoria final de un Sistema de Predicción basado en Aprendizaje Automático. De cara a que en los próximos mensajes que te envíe, vayamos completando el documento apartado a apartado te compartiré información sobre todo el proyecto para que lo conozcas bien. Primero te pasaré el cógido del programa, y después lo que llevo actualmente rellenado del documento. En el archivo adjunto encontrarás el documento de instrucciones para el trabajo. En él viene que apartados debe tener el documento, y cómo debo realizar cada apartado de la memoria final. Código: 

#api_interface.py
import requests

# Función para descargar y cargar el dataset
def GET_dataset(dataset_url):
    header = {
        "accept": "application/json",
        "Authorization": "Token " + token
    }
    response = requests.get(dataset_url, headers=header)
    # Guardar el dataset en el servidor o retornarlo para su uso
    return response.content

# Función para enviar las predicciones por API

def POST_modeleval(evaluation_results):
    # URL de la API donde enviar las métricas de evaluación del modelo
    api_url = "http://tu-api.com/model_evaluation"

    # Payload de la solicitud POST
    payload = {
        "evaluation_results": evaluation_results  # Aquí puedes ajustar el formato del payload según lo requiera tu API
    }

    # Encabezados de la solicitud POST
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Enviamos la solicitud POST a la API
        response = requests.post(api_url, json=payload, headers=headers)
        # Verificamos el código de estado de la respuesta
        if response.status_code == 200:
            print("Métricas de evaluación del modelo enviadas correctamente a la API.")
        else:
            print("Error al enviar las métricas de evaluación del modelo a la API:", response.status_code)
    except Exception as e:
        print("Error al enviar las métricas de evaluación del modelo a la API:", str(e))


def POST_predictions(predictions):
    # URL de la API donde enviar las métricas de la predicción
    api_url = "http://tokii.com/predictions"

    # Payload de la solicitud POST
    payload = {
        "predictions": predictions.tolist()  # Convertimos las predicciones a una lista si es necesario
    }

    # Encabezados de la solicitud POST
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Enviamos la solicitud POST a la API
        response = requests.post(api_url, json=payload, headers=headers)
        # Verificamos el código de estado de la respuesta
        if response.status_code == 200:
            print("Predicciones enviadas correctamente a la API.")
        else:
            print("Error al enviar las predicciones a la API:", response.status_code)
    except Exception as e:
        print("Error al enviar las predicciones a la API:", str(e)) # dataset.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from api_interface import GET_dataset

class Dataset:
    def __init__(self, url: str):
        self.df: pd.DataFrame = None
        if url == "http://tokii.datasets.iris":
            self.df = pd.read_csv("program/Iris.csv")
        elif url == "http://tokii.datasets.tips":
            self.df = pd.read_csv("program/tips.csv")
        else:
            self.df = GET_dataset(url)
    
    def as_dataframe(self) -> pd.DataFrame:
        return self.df
    
    def eda_generico(self):
        df = self.df

        # Análisis descriptivo
        print("\nAnálisis Descriptivo:")
        print(df.head())
        print(df.info())
        print(df.describe())
        
        # Visualización de distribuciones de datos
        # for col in df.columns:
        #     if df[col].dtype == 'object':
        #         sns.countplot(x=col, data=df)
        #         plt.title(f'Distribución de {col}')
        #         plt.show()
        #     else:
        #         sns.histplot(df[col], kde=True)
        #         plt.title(f'Distribución de {col}')
        #         plt.show()
        
        # Identificación y manejo de valores faltantes
        print("\nValores Faltantes:")
        print(df.isnull().sum())
        
        # Correlación entre variables
        print("\nCorrelación entre Variables:")
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlación')
        plt.show()
        
        # Detección de outliers
        print("\nDetección de Outliers:")
        for col in df.columns:
            if df[col].dtype != 'object':
                sns.boxplot(x=df[col])
                plt.title(f'Outliers en {col}')
                plt.show()
        
        # Balance de clases (para problemas de clasificación)
        if 'target' in df.columns and df['target'].dtype == 'object':
            sns.countplot(x='target', data=df)
            plt.title('Balance de Clases')
            plt.show() # main.py
from training import Training
import mqtt
import global_data
import json

DEBUG = True

class FakeMsg:
    def __init__(self, message):
        self.payload = message

def init():
    global_data.training = Training()

    """Initialize MQTT connection"""
    if not DEBUG:
        mqtt.init()
    else:
        train = {
            "command": "train",
            "params": {
                "dataset": "http://tokii.datasets.tips", 
                "target": "tip",
                "features": None,  # Este campo puede ser "null" o una lista de nombres de columnas
                "preprocessing": ["impute_numeric", "scale_numeric", "impute_categorical", "encode_categorical"],
                "recommendations": True,
                "algorithms": [
                    {"name": "KNeighborsRegressor", "params": {"n_neighbors": 3, "weights": "distance"}}, 
                    {"name": "SVR", "params": {"C": 3, "degree": 87}}
                ],
                "crossvalidation": 80
            }
        }
        
        predict = {
            "command": "predict",
            "params": {
                "model": "SVR.pkl",
                "features": {'total_bill': 18.53, 'sex': "Male", 'smoker':"No", 'day':"Sun", 'time':"Dinner", 'size':3},
                "preprocessing": ["impute_numeric", "scale_numeric", "impute_categorical", "encode_categorical"]
                }
        }
        mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(train)))

if __name__ == "__main__":
    init()
#mqtt.py
import paho.mqtt.client as mqtt
import json
import traceback

from utils import pipeline_preprocessing
from training import Training
from dataset import Dataset
import global_data

broker_address = "mqtt-container"
broker_port = 1883
topic = "test/topic"

def init():
    # Create MQTT client instance
    client = mqtt.Client()

    # Assign callback functions
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to MQTT broker
    client.connect(broker_address, broker_port)

    # Start MQTT client loop
    client.loop_forever()

# Callback when connected to the MQTT broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(topic)

# Callback when a message is received from the MQTT broker
def on_message(client, userdata, msg):
    try:
        training: Training = global_data.training
        message = json.loads(msg.payload)
        
        # Realizar analisis del dataset y entrenamiento según input
        if message["command"] == "train":
            dataset_url = message["params"]["dataset"]
            training.algorithms = message["params"]["algorithms"]
            training.crossvalidation = message["params"]["crossvalidation"]
            training.target = message["params"]["target"]
            training.preprocessing = message["params"].get("preprocessing", [])
            training.dataset = Dataset(dataset_url)
            
            # Realizar EDA
            # Preprocesar el dataset si se especifica
            if training.preprocessing:
                training.df = pipeline_preprocessing(training.dataset.as_dataframe(), training.target, training.preprocessing)
            
            training.dataset.eda_generico()
            
            # Entrenar y evaluar el modelo
            training.train_and_evaluate()
        
        # Realizar predicciones con el modelo seleccionado y con la/las columnas seleccionadas
        elif message["command"] == "predict":
            training.predict(message["params"]["model"], message["params"]["features"])
            
    except Exception as err:
        print(traceback.format_exc())
#train.py
import sys
import os
import joblib
from typing import List, Dict, Union
import serializer

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_squared_error, r2_score

# Función para entrenar los modelos
def train_models(X_train, y_train, model_params: List[Dict]):
    trained_models = []
    for model_info in model_params:
        model_name = model_info['name']
        params = model_info['params']
        if model_name in serializer.model_classes:
            model_class = serializer.model_classes[model_name]
            model = model_class(**params)  # Instanciar modelo con hiperparámetros proporcionados
            model.fit(X_train, y_train)  # Entrenar el modelo
            
            serializer.to_pickle(model, model_name)
            trained_models.append(model)
        else:
            print(f"Model '{model_name}' not found. Skipping...")
    return trained_models

# Función para evaluar modelos de clasificación
def evaluate_classification_models(trained_models: List[serializer.ScikitModel], X_test, y_test):
    evaluation_results = {}
    for model in trained_models:
        plt.clf()
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
        
        evaluation_results[str(model)] = {"accuracy": accuracy, "f1_score": f1, "confusion_matrix": cm, "classification_report": report}
    return evaluation_results

# Función para evaluar modelos de regresión
def evaluate_regression_models(trained_models: List[serializer.ScikitModel], X_test, y_test):
    evaluation_results = {}
    for model in trained_models:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("Mean Squared Error:", mse)
        print("R-squared:", r2)
        
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        plt.show()
        
        evaluation_results[str(model)] = {"mean_squared_error": mse, "r2_score": r2}
    return evaluation_results

# Función para seleccionar automáticamente las características más relevantes
def select_features(X, y, model, k=10):
    if isinstance(model, tuple(serializer.classification_models)):
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)
    
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features

# Función para comparar los modelos entrenados
def compare_models(dirpath: Union[os.PathLike, str], save_report: bool = False) -> Dict:
    model_filenames = os.listdir(dirpath)
    models = []
    for filename in model_filenames:
        if filename.endswith(".pkl"):
            model = joblib.load(open(os.path.join(dirpath, filename), "rb"))
        elif filename.endswith(".h5"):
            model = joblib.load(os.path.join(dirpath, filename))
        
        models.append(model)

    if not models:
        print("No trained models found.")
        sys.exit(0)

    for model in models:
        print(f"Loaded model {model}") # training.py
from sklearn.model_selection import train_test_split
import api_interface
from utils import pipeline_preprocessing
import train
import trainingparams
import serializer
from dataset import Dataset
import pandas as pd

class Training:
    def __init__(self, dataset: Dataset = None):
        self.dataset = dataset
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

        ######### PREPROCESSING ##########
        # Realizar preprocesamiento si se especifica
        if self.preprocessing:
            dataset = pipeline_preprocessing(dataset, self.target, self.preprocessing)

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

        # Entrenar los modelos
        trained_models = train.train_models(X_train, y_train, self.algorithms)
        
        # Evaluar los modelos
        evaluation_results = {}
        for model in trained_models:
            if isinstance(model, tuple(serializer.classification_models)):
                eval_results = train.evaluate_classification_models([model], X_test, y_test)
            else:
                eval_results = train.evaluate_regression_models([model], X_test, y_test)
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
        api_interface.POST_modeleval(evaluation_results)

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
        trained_models = train.train_models(X_train, y_train, self.algorithms)
        
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

    def predict(self, model_path, featuresPredict):
        # Cargar el modelo desde el archivo
        model: serializer.ScikitModel = serializer.from_pickle(model_path)
        
        # Convertir las características proporcionadas a un DataFrame
        features_df = pd.DataFrame([featuresPredict])
        
        # Realizar preprocesamiento si se especificó durante el entrenamiento
        if self.preprocessing:
            features_df = pipeline_preprocessing(features_df, self.target, self.preprocessing)
        
        # Realizar la predicción
        prediction = model.predict(features_df)
        
        print(f"The predicted {self.target} for {featuresPredict} is {prediction[0]:.2f}")
        return prediction #trainingparams.py
#from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.estimator_checks import parametrize_with_checks

import serializer
from typing import List

def train_recommendedparams(X_train, y_train, algorithms: List[str]):
    recommended_params = []
    # Comprobación del tamaño del conjunto de datos
    n_samples, n_features = X_train.shape
    if n_samples > 10000 or n_features > 50:
        # Reducción del espacio de búsqueda y número de iteraciones
        n_iter = 25
        param_factor = 0.5
    else:
        n_iter = 50
        param_factor = 1.0
    
    for algorithm in algorithms:
        model_class = serializer.model_classes[algorithm]
        # Estimación de parámetros iniciales
        estimator, _ = parametrize_with_checks(model_class)
        estimator.fit(X_train, y_train)
        initial_params = estimator.get_params()

        # Definir el espacio de búsqueda para BayesSearchCV
        param_distributions = get_default_params(model_class, X_train, y_train, param_factor)

        # Realizar la búsqueda bayesiana de hiperparámetros
        opt = BayesSearchCV(model_class(), param_distributions, n_iter=n_iter, cv=StratifiedKFold(n_splits=5), random_state=42)
        opt.fit(X_train, y_train)
        best_params = opt.best_params_
        recommended_params.append({'name': algorithm, 'initial_params': initial_params, 'params': best_params})
    return recommended_params

def get_default_params(model_class, X_train, y_train, param_factor):
    default_params = {}
    # Utilizamos parametrize_with_checks para estimar los valores de los parámetros iniciales
    estimator, _ = parametrize_with_checks(model_class)
    estimator.fit(X_train, y_train)
    params_dict = estimator.get_params()
    for param, value in params_dict.items():
        if isinstance(value, int) or isinstance(value, float):
            # Si el parámetro es un número, definimos un rango de búsqueda en función del valor estimado y el factor de reducción
            default_params[param] = (value * param_factor / 10, value * param_factor * 10)  # Definimos un rango de búsqueda
        elif isinstance(value, str):
            # Si el parámetro es una cadena, dejamos que el modelo elija entre algunos valores predefinidos
            default_params[param] = [value]  # Definimos una lista de valores posibles
        else:
            # Otros tipos de parámetros (como booleanos) pueden no necesitar optimización
            pass
    return default_params
#utils.py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def pipeline_preprocessing(df: pd.DataFrame, target: str, preprocessing_steps: list) -> pd.DataFrame:
    """
    Función para realizar el preprocesamiento de un dataset.
    
    Parámetros:
    - target: str, el nombre de la columna objetivo.
    - preprocessing_steps: list, lista de pasos de preprocesamiento.
    
    Retorna:
    - pipeline: Pipeline, el pipeline de preprocesamiento.
    - df_processed: DataFrame, el dataset procesado.
    """
    X = df.drop(columns=target)
    y = df[target]
    # Definir características numéricas y categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Crear listas de transformadores
    transformers = []

    # Añadir transformadores según los pasos de preprocesamiento especificados
    if 'impute_numeric' in preprocessing_steps:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        transformers.append(('impute_numeric', numeric_transformer, numeric_features))

    if 'scale_numeric' in preprocessing_steps:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformers.append(('scale_numeric', numeric_transformer, numeric_features))

    if 'impute_categorical' in preprocessing_steps:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
        ])
        transformers.append(('impute_categorical', categorical_transformer, categorical_features))

    if 'encode_categorical' in preprocessing_steps:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('encode_categorical', categorical_transformer, categorical_features))

    # ColumnTransformer para aplicar transformaciones adecuadas a cada tipo de característica
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Crear pipeline con PCA opcional
    if 'use_pca' in preprocessing_steps:
        n_components = preprocessing_steps.get('n_components', 2)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=n_components))
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
    
    # Aplicar preprocesamiento al dataset
    X_processed = pipeline.fit_transform(X)
    
    # Convertir a DataFrame para facilitar análisis posteriores
    df_processed = pd.DataFrame(X_processed, columns=[f'feature_{i}' for i in range(X_processed.shape[1])])
    df_processed[target] = y.values
    
    return df_processed
# serializer.py
from typing import Dict, Type
import pickle
from typing import Protocol

import classification as clf
import regression as reg

from sklearn.svm import SVC, SVR
from sklearn import svm, discriminant_analysis, dummy
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...

model_classes: Dict[str, Type[ScikitModel]] = {
    "BernoulliNB": BernoulliNB,
    "GaussianNB": GaussianNB,
    "MultinomialNB": MultinomialNB,
    "ComplementNB": ComplementNB,
    "LinearDiscriminantAnalysis": discriminant_analysis.LinearDiscriminantAnalysis,
    "QuadraticDiscriminantAnalysis": discriminant_analysis.QuadraticDiscriminantAnalysis,
    "Perceptron": Perceptron,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "MLPClassifier": MLPClassifier,
    "LogisticRegression": LogisticRegression,
    "LinearRegression": LinearRegression,
    "Lasso": Lasso,
    "Ridge": Ridge,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "MLPRegressor": MLPRegressor,
    'SVC': SVC,
    "SVR": SVR,
    "KNeighborsClassifier": KNeighborsClassifier,
    "KNeighborsRegressor": KNeighborsRegressor,
    "SGDClassifier": SGDClassifier,
    "SGDRegressor": SGDRegressor,
    "AdaBoostClassifier": AdaBoostClassifier,
    "AdaBoostRegressor": AdaBoostRegressor,
    "BaggingClassifier": BaggingClassifier,
    "BaggingRegressor": BaggingRegressor,
}

classification_models = {
    BernoulliNB, GaussianNB, MultinomialNB, ComplementNB,
    discriminant_analysis.LinearDiscriminantAnalysis,
    discriminant_analysis.QuadraticDiscriminantAnalysis,
    Perceptron, DecisionTreeClassifier, GradientBoostingClassifier,
    RandomForestClassifier, MLPClassifier, LogisticRegression, SVC,
    KNeighborsClassifier, SGDClassifier, AdaBoostClassifier, BaggingClassifier
}

regression_models = {
    LinearRegression, Lasso, Ridge, DecisionTreeRegressor,
    GradientBoostingRegressor, RandomForestRegressor, MLPRegressor,
    SVR, KNeighborsRegressor, SGDRegressor, AdaBoostRegressor, BaggingRegressor
}

def serialize_model(model):
    if isinstance(model, LogisticRegression):
        return clf.serialize_logistic_regression(model)
    elif isinstance(model, BernoulliNB):
        return clf.serialize_bernoulli_nb(model)
    elif isinstance(model, GaussianNB):
        return clf.serialize_gaussian_nb(model)
    elif isinstance(model, MultinomialNB):
        return clf.serialize_multinomial_nb(model)
    elif isinstance(model, ComplementNB):
        return clf.serialize_complement_nb(model)
    elif isinstance(model, discriminant_analysis.LinearDiscriminantAnalysis):
        return clf.serialize_lda(model)
    elif isinstance(model, discriminant_analysis.QuadraticDiscriminantAnalysis):
        return clf.serialize_qda(model)
    elif isinstance(model, svm.SVC):
        return clf.serialize_svm(model)
    elif isinstance(model, Perceptron):
        return clf.serialize_perceptron(model)
    elif isinstance(model, DecisionTreeClassifier):
        return clf.serialize_decision_tree(model)
    elif isinstance(model, GradientBoostingClassifier):
        return clf.serialize_gradient_boosting(model)
    elif isinstance(model, RandomForestClassifier):
        return clf.serialize_random_forest(model)
    elif isinstance(model, MLPClassifier):
        return clf.serialize_mlp(model)

    elif isinstance(model, LinearRegression):
        return reg.serialize_linear_regressor(model)
    elif isinstance(model, Lasso):
        return reg.serialize_lasso_regressor(model)
    elif isinstance(model, Ridge):
        return reg.serialize_ridge_regressor(model)
    elif isinstance(model, SVR):
        return reg.serialize_svr(model)
    elif isinstance(model, DecisionTreeRegressor):
        return reg.serialize_decision_tree_regressor(model)
    elif isinstance(model, GradientBoostingRegressor):
        return reg.serialize_gradient_boosting_regressor(model)
    elif isinstance(model, RandomForestRegressor):
        return reg.serialize_random_forest_regressor(model)
    elif isinstance(model, MLPRegressor):
        return reg.serialize_mlp_regressor(model)
    else:
        raise ModellNotSupported('This model type is not currently supported. Email support@mlrequest.com to request a feature or report a bug.')

def deserialize_model(model_dict):
    if model_dict['meta'] == 'lr':
        return clf.deserialize_logistic_regression(model_dict)
    elif model_dict['meta'] == 'bernoulli-nb':
        return clf.deserialize_bernoulli_nb(model_dict)
    elif model_dict['meta'] == 'gaussian-nb':
        return clf.deserialize_gaussian_nb(model_dict)
    elif model_dict['meta'] == 'multinomial-nb':
        return clf.deserialize_multinomial_nb(model_dict)
    elif model_dict['meta'] == 'complement-nb':
        return clf.deserialize_complement_nb(model_dict)
    elif model_dict['meta'] == 'lda':
        return clf.deserialize_lda(model_dict)
    elif model_dict['meta'] == 'qda':
        return clf.deserialize_qda(model_dict)
    elif model_dict['meta'] == 'svm':
        return clf.deserialize_svm(model_dict)
    elif model_dict['meta'] == 'perceptron':
        return clf.deserialize_perceptron(model_dict)
    elif model_dict['meta'] == 'decision-tree':
        return clf.deserialize_decision_tree(model_dict)
    elif model_dict['meta'] == 'gb':
        return clf.deserialize_gradient_boosting(model_dict)
    elif model_dict['meta'] == 'rf':
        return clf.deserialize_random_forest(model_dict)
    elif model_dict['meta'] == 'mlp':
        return clf.deserialize_mlp(model_dict)

    elif model_dict['meta'] == 'linear-regression':
        return reg.deserialize_linear_regressor(model_dict)
    elif model_dict['meta'] == 'lasso-regression':
        return reg.deserialize_lasso_regressor(model_dict)
    elif model_dict['meta'] == 'ridge-regression':
        return reg.deserialize_ridge_regressor(model_dict)
    elif model_dict['meta'] == 'svr':
        return reg.deserialize_svr(model_dict)
    elif model_dict['meta'] == 'decision-tree-regression':
        return reg.deserialize_decision_tree_regressor(model_dict)
    elif model_dict['meta'] == 'gb-regression':
        return reg.deserialize_gradient_boosting_regressor(model_dict)
    elif model_dict['meta'] == 'rf-regression':
        return reg.deserialize_random_forest_regressor(model_dict)
    elif model_dict['meta'] == 'mlp-regression':
        return reg.deserialize_mlp_regressor(model_dict)
    else:
        raise ModellNotSupported('Model type not supported or corrupt JSON file. Email support@mlrequest.com to request a feature or report a bug.')

def to_dict(model):
    return serialize_model(model)

def from_dict(model_dict):
    return deserialize_model(model_dict)

def to_pickle(model, model_name: str):
    if not model_name.endswith(".pkl"):
        model_name += ".pkl"
    with open(model_name, 'wb') as model_file:
        pickle.dump(model, model_file)

def from_pickle(model_name):
    with open(model_name, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
        return loaded_model

class ModellNotSupported(Exception):
    pass
#classification.py
import numpy as np
import scipy as sp

from sklearn import svm, discriminant_analysis, dummy
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

import regression
import csr


def serialize_logistic_regression(model):
    serialized_model = {
        'meta': 'lr',
        'classes_': model.classes_.tolist(),
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_.tolist(),
        'params': model.get_params()
    }

    return serialized_model


def deserialize_logistic_regression(model_dict):
    model = LogisticRegression(model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.coef_ = np.array(model_dict['coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])
    model.n_iter_ = np.array(model_dict['intercept_'])

    return model


def serialize_bernoulli_nb(model):
    serialized_model = {
        'meta': 'bernoulli-nb',
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_log_prior_': model.class_log_prior_.tolist(),
        'feature_count_': model.feature_count_.tolist(),
        'feature_log_prob_': model.feature_log_prob_.tolist(),
        'params': model.get_params()
    }

    return serialized_model


def deserialize_bernoulli_nb(model_dict):
    model = BernoulliNB(model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_log_prior_ = np.array(model_dict['class_log_prior_'])
    model.feature_count_= np.array(model_dict['feature_count_'])
    model.feature_log_prob_ = np.array(model_dict['feature_log_prob_'])

    return model


def serialize_gaussian_nb(model):
    serialized_model = {
        'meta': 'gaussian-nb',
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_prior_': model.class_prior_.tolist(),
        'theta_': model.theta_.tolist(),
        'sigma_': model.sigma_.tolist(),
        'epsilon_': model.epsilon_,
        'params': model.get_params()
    }

    return serialized_model


def deserialize_gaussian_nb(model_dict):
    model = GaussianNB(model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_prior_ = np.array(model_dict['class_prior_'])
    model.theta_ = np.array(model_dict['theta_'])
    model.sigma_ = np.array(model_dict['sigma_'])
    model.epsilon_ = model_dict['epsilon_']

    return model


def serialize_multinomial_nb(model):
    serialized_model = {
        'meta': 'multinomial-nb',
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_log_prior_': model.class_log_prior_.tolist(),
        'feature_count_': model.feature_count_.tolist(),
        'feature_log_prob_': model.feature_log_prob_.tolist(),
        'params': model.get_params()
    }

    return serialized_model


def deserialize_multinomial_nb(model_dict):
    model = MultinomialNB(model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_log_prior_ = np.array(model_dict['class_log_prior_'])
    model.feature_count_= np.array(model_dict['feature_count_'])
    model.feature_log_prob_ = np.array(model_dict['feature_log_prob_'])

    return model


def serialize_complement_nb(model):
    serialized_model = {
        'meta': 'complement-nb',
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_log_prior_': model.class_log_prior_.tolist(),
        'feature_count_': model.feature_count_.tolist(),
        'feature_log_prob_': model.feature_log_prob_.tolist(),
        'feature_all_': model.feature_all_.tolist(),
        'params': model.get_params()
    }

    return serialized_model


def deserialize_complement_nb(model_dict):
    model = ComplementNB(model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_log_prior_ = np.array(model_dict['class_log_prior_'])
    model.feature_count_= np.array(model_dict['feature_count_'])
    model.feature_log_prob_ = np.array(model_dict['feature_log_prob_'])
    model.feature_all_ = np.array(model_dict['feature_all_'])

    return model


def serialize_lda(model):
    serialized_model = {
        'meta': 'lda',
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'explained_variance_ratio_': model.explained_variance_ratio_.tolist(),
        'means_': model.means_.tolist(),
        'priors_': model.priors_.tolist(),
        'scalings_': model.scalings_.tolist(),
        'xbar_': model.xbar_.tolist(),
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }
    if 'covariance_' in model.__dict__:
        serialized_model['covariance_'] = model.covariance_.tolist()

    return serialized_model


def deserialize_lda(model_dict):
    model = discriminant_analysis.LinearDiscriminantAnalysis(**model_dict['params'])

    model.coef_ = np.array(model_dict['coef_']).astype(np.float64)
    model.intercept_ = np.array(model_dict['intercept_']).astype(np.float64)
    model.explained_variance_ratio_ = np.array(model_dict['explained_variance_ratio_']).astype(np.float64)
    model.means_ = np.array(model_dict['means_']).astype(np.float64)
    model.priors_ = np.array(model_dict['priors_']).astype(np.float64)
    model.scalings_ = np.array(model_dict['scalings_']).astype(np.float64)
    model.xbar_ = np.array(model_dict['xbar_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_']).astype(np.int64)

    return model


def serialize_qda(model):
    serialized_model = {
        'meta': 'qda',
        'means_': model.means_.tolist(),
        'priors_': model.priors_.tolist(),
        'scalings_': [array.tolist() for array in model.scalings_],
        'rotations_': [array.tolist() for array in model.rotations_],
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }
    if 'covariance_' in model.__dict__:
        serialized_model['covariance_'] = model.covariance_.tolist()

    return serialized_model


def deserialize_qda(model_dict):
    model = discriminant_analysis.QuadraticDiscriminantAnalysis(**model_dict['params'])

    model.means_ = np.array(model_dict['means_']).astype(np.float64)
    model.priors_ = np.array(model_dict['priors_']).astype(np.float64)
    model.scalings_ = np.array(model_dict['scalings_']).astype(np.float64)
    model.rotations_ = np.array(model_dict['rotations_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_']).astype(np.int64)

    return model


def serialize_svm(model):
    serialized_model = {
        'meta': 'svm',
        'class_weight_': model.class_weight_.tolist(),
        'classes_': model.classes_.tolist(),
        'support_': model.support_.tolist(),
        'n_support_': model.n_support_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'probA_': model.probA_.tolist(),
        'probB_': model.probB_.tolist(),
        '_intercept_': model._intercept_.tolist(),
        'shape_fit_': model.shape_fit_,
        '_gamma': model._gamma,
        'params': model.get_params()
    }

    if isinstance(model.support_vectors_, sp.sparse.csr_matrix):
        serialized_model['support_vectors_'] = csr.serialize_csr_matrix(model.support_vectors_)
    elif isinstance(model.support_vectors_, np.ndarray):
        serialized_model['support_vectors_'] = model.support_vectors_.tolist()

    if isinstance(model.dual_coef_, sp.sparse.csr_matrix):
        serialized_model['dual_coef_'] = csr.serialize_csr_matrix(model.dual_coef_)
    elif isinstance(model.dual_coef_, np.ndarray):
        serialized_model['dual_coef_'] = model.dual_coef_.tolist()

    if isinstance(model._dual_coef_, sp.sparse.csr_matrix):
        serialized_model['_dual_coef_'] = csr.serialize_csr_matrix(model._dual_coef_)
    elif isinstance(model._dual_coef_, np.ndarray):
        serialized_model['_dual_coef_'] = model._dual_coef_.tolist()

    return serialized_model


def deserialize_svm(model_dict):
    model = svm.SVC(**model_dict['params'])
    model.shape_fit_ = model_dict['shape_fit_']
    model._gamma = model_dict['_gamma']

    model.class_weight_ = np.array(model_dict['class_weight_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_'])
    model.support_ = np.array(model_dict['support_']).astype(np.int32)
    model.n_support_ = np.array(model_dict['n_support_']).astype(np.int32)
    model.intercept_ = np.array(model_dict['intercept_']).astype(np.float64)
    model.probA_ = np.array(model_dict['probA_']).astype(np.float64)
    model.probB_ = np.array(model_dict['probB_']).astype(np.float64)
    model._intercept_ = np.array(model_dict['_intercept_']).astype(np.float64)

    if 'meta' in model_dict['support_vectors_'] and model_dict['support_vectors_']['meta'] == 'csr':
        model.support_vectors_ = csr.deserialize_csr_matrix(model_dict['support_vectors_'])
        model._sparse = True
    else:
        model.support_vectors_ = np.array(model_dict['support_vectors_']).astype(np.float64)
        model._sparse = False

    if 'meta' in model_dict['dual_coef_'] and model_dict['dual_coef_']['meta'] == 'csr':
        model.dual_coef_ = csr.deserialize_csr_matrix(model_dict['dual_coef_'])
    else:
        model.dual_coef_ = np.array(model_dict['dual_coef_']).astype(np.float64)

    if 'meta' in model_dict['_dual_coef_'] and model_dict['_dual_coef_']['meta'] == 'csr':
        model._dual_coef_ = csr.deserialize_csr_matrix(model_dict['_dual_coef_'])
    else:
        model._dual_coef_ = np.array(model_dict['_dual_coef_']).astype(np.float64)

    return model


def serialize_dummy_classifier(model):
    model.classes_ = model.classes_.tolist()
    model.class_prior_ = model.class_prior_.tolist()
    return model.__dict__


def serialize_tree(tree):
    serialized_tree = tree.__getstate__()

    dtypes = serialized_tree['nodes'].dtype
    serialized_tree['nodes'] = serialized_tree['nodes'].tolist()
    serialized_tree['values'] = serialized_tree['values'].tolist()

    return serialized_tree, dtypes


def deserialize_tree(tree_dict, n_features, n_classes, n_outputs):
    tree_dict['nodes'] = [tuple(lst) for lst in tree_dict['nodes']]

    names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples']
    tree_dict['nodes'] = np.array(tree_dict['nodes'], dtype=np.dtype({'names': names, 'formats': tree_dict['nodes_dtype']}))
    tree_dict['values'] = np.array(tree_dict['values'])

    tree = Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
    tree.__setstate__(tree_dict)

    return tree


def serialize_decision_tree(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        'meta': 'decision-tree',
        'feature_importances_': model.feature_importances_.tolist(),
        'max_features_': model.max_features_,
        'n_classes_': int(model.n_classes_),
        'n_features_': model.n_features_,
        'n_outputs_': model.n_outputs_,
        'tree_': tree,
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }


    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)

    serialized_model['tree_']['nodes_dtype'] = tree_dtypes

    return serialized_model


def deserialize_decision_tree(model_dict):
    deserialized_model = DecisionTreeClassifier(**model_dict['params'])

    deserialized_model.classes_ = np.array(model_dict['classes_'])
    deserialized_model.max_features_ = model_dict['max_features_']
    deserialized_model.n_classes_ = model_dict['n_classes_']
    deserialized_model.n_features_ = model_dict['n_features_']
    deserialized_model.n_outputs_ = model_dict['n_outputs_']

    tree = deserialize_tree(model_dict['tree_'], model_dict['n_features_'], model_dict['n_classes_'], model_dict['n_outputs_'])
    deserialized_model.tree_ = tree

    return deserialized_model


def serialize_gradient_boosting(model):
    serialized_model = {
        'meta': 'gb',
        'classes_': model.classes_.tolist(),
        'max_features_': model.max_features_,
        'n_classes_': model.n_classes_,
        'n_features_': model.n_features_,
        'train_score_': model.train_score_.tolist(),
        'params': model.get_params(),
        'estimators_shape': list(model.estimators_.shape),
        'estimators_': []
    }

    if  isinstance(model.init_, dummy.DummyClassifier):
        serialized_model['init_'] = serialize_dummy_classifier(model.init_)
        serialized_model['init_']['meta'] = 'dummy'
    elif isinstance(model.init_, str):
        serialized_model['init_'] = model.init_

    if isinstance(model.loss_, _gb_losses.BinomialDeviance):
        serialized_model['loss_'] = 'deviance'
    elif isinstance(model.loss_, _gb_losses.ExponentialLoss):
        serialized_model['loss_'] = 'exponential'
    elif isinstance(model.loss_, _gb_losses.MultinomialDeviance):
        serialized_model['loss_'] = 'multinomial'

    if 'priors' in model.init_.__dict__:
        serialized_model['priors'] = model.init_.priors.tolist()

    serialized_model['estimators_'] = [regression.serialize_decision_tree_regressor(regression_tree) for regression_tree in model.estimators_.reshape(-1, )]

    return serialized_model


def deserialize_gradient_boosting(model_dict):
    model = GradientBoostingClassifier(**model_dict['params'])
    estimators = [regression.deserialize_decision_tree_regressor(tree) for tree in model_dict['estimators_']]
    model.estimators_ = np.array(estimators).reshape(model_dict['estimators_shape'])
    if 'init_' in model_dict and model_dict['init_']['meta'] == 'dummy':
        model.init_ = dummy.DummyClassifier()
        model.init_.__dict__ = model_dict['init_']
        model.init_.__dict__.pop('meta')

    model.classes_ = np.array(model_dict['classes_'])
    model.train_score_ = np.array(model_dict['train_score_'])
    model.max_features_ = model_dict['max_features_']
    model.n_classes_ = model_dict['n_classes_']
    model.n_features_ = model_dict['n_features_']
    if model_dict['loss_'] == 'deviance':
        model.loss_ = _gb_losses.BinomialDeviance(model.n_classes_)
    elif model_dict['loss_'] == 'exponential':
        model.loss_ = _gb_losses.ExponentialLoss(model.n_classes_)
    elif model_dict['loss_'] == 'multinomial':
        model.loss_ = _gb_losses.MultinomialDeviance(model.n_classes_)

    if 'priors' in model_dict:
        model.init_.priors = np.array(model_dict['priors'])
    return model


def serialize_random_forest(model):
    serialized_model = {
        'meta': 'rf',
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf': model.min_samples_leaf,
        'min_weight_fraction_leaf': model.min_weight_fraction_leaf,
        'max_features': model.max_features,
        'max_leaf_nodes': model.max_leaf_nodes,
        'min_impurity_decrease': model.min_impurity_decrease,
        'min_impurity_split': model.min_impurity_split,
        'n_features_': model.n_features_,
        'n_outputs_': model.n_outputs_,
        'classes_': model.classes_.tolist(),
        'estimators_': [serialize_decision_tree(decision_tree) for decision_tree in model.estimators_],
        'params': model.get_params()
    }

    if 'oob_score_' in model.__dict__:
        serialized_model['oob_score_'] = model.oob_score_
    if 'oob_decision_function_' in model.__dict__:
        serialized_model['oob_decision_function_'] = model.oob_decision_function_.tolist()

    if isinstance(model.n_classes_, int):
        serialized_model['n_classes_'] = model.n_classes_
    else:
        serialized_model['n_classes_'] = model.n_classes_.tolist()

    return serialized_model


def deserialize_random_forest(model_dict):
    model = RandomForestClassifier(**model_dict['params'])
    estimators = [deserialize_decision_tree(decision_tree) for decision_tree in model_dict['estimators_']]
    model.estimators_ = np.array(estimators)

    model.classes_ = np.array(model_dict['classes_'])
    model.n_features_ = model_dict['n_features_']
    model.n_outputs_ = model_dict['n_outputs_']
    model.max_depth = model_dict['max_depth']
    model.min_samples_split = model_dict['min_samples_split']
    model.min_samples_leaf = model_dict['min_samples_leaf']
    model.min_weight_fraction_leaf = model_dict['min_weight_fraction_leaf']
    model.max_features = model_dict['max_features']
    model.max_leaf_nodes = model_dict['max_leaf_nodes']
    model.min_impurity_decrease = model_dict['min_impurity_decrease']
    model.min_impurity_split = model_dict['min_impurity_split']

    if 'oob_score_' in model_dict:
        model.oob_score_ = model_dict['oob_score_']
    if 'oob_decision_function_' in model_dict:
        model.oob_decision_function_ = model_dict['oob_decision_function_']

    if isinstance(model_dict['n_classes_'], list):
        model.n_classes_ = np.array(model_dict['n_classes_'])
    else:
        model.n_classes_ = model_dict['n_classes_']

    return model


def serialize_perceptron(model):
    serialized_model = {
        'meta': 'perceptron',
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_,
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }
    if 'covariance_' in model.__dict__:
        serialized_model['covariance_'] = model.covariance_.tolist()

    return serialized_model


def deserialize_perceptron(model_dict):
    model = Perceptron(**model_dict['params'])

    model.coef_ = np.array(model_dict['coef_']).astype(np.float64)
    model.intercept_ = np.array(model_dict['intercept_']).astype(np.float64)
    model.n_iter_ = np.array(model_dict['n_iter_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_']).astype(np.int64)

    return model


def serialize_label_binarizer(label_binarizer):
    serialized_label_binarizer = {
        'neg_label': label_binarizer.neg_label,
        'pos_label': label_binarizer.pos_label,
        'sparse_output': label_binarizer.sparse_output,
        'y_type_': label_binarizer.y_type_,
        'sparse_input_': label_binarizer.sparse_input_,
        'classes_': label_binarizer.classes_.tolist()
    }

    return serialized_label_binarizer


def deserialize_label_binarizer(label_binarizer_dict):
    label_binarizer = LabelBinarizer()
    label_binarizer.neg_label = label_binarizer_dict['neg_label']
    label_binarizer.pos_label = label_binarizer_dict['pos_label']
    label_binarizer.sparse_output = label_binarizer_dict['sparse_output']
    label_binarizer.y_type_ = label_binarizer_dict['y_type_']
    label_binarizer.sparse_input_ = label_binarizer_dict['sparse_input_']
    label_binarizer.classes_ = np.array(label_binarizer_dict['classes_'])

    return label_binarizer


def serialize_mlp(model):
    serialized_model = {
        'meta': 'mlp',
        'coefs_': [array.tolist() for array in model.coefs_],
        'loss_': model.loss_,
        'intercepts_': [array.tolist() for array in model.intercepts_],
        'n_iter_': model.n_iter_,
        'n_layers_': model.n_layers_,
        'n_outputs_': model.n_outputs_,
        'out_activation_': model.out_activation_,
        '_label_binarizer': serialize_label_binarizer(model._label_binarizer),
        'params': model.get_params()
    }

    if isinstance(model.classes_, list):
        serialized_model['classes_'] = [array.tolist() for array in model.classes_]
    else:
        serialized_model['classes_'] = model.classes_.tolist()

    return serialized_model


def deserialize_mlp(model_dict):
    model = MLPClassifier(**model_dict['params'])

    model.coefs_ = np.array(model_dict['coefs_'])
    model.loss_ = model_dict['loss_']
    model.intercepts_ = np.array(model_dict['intercepts_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_layers_ = model_dict['n_layers_']
    model.n_outputs_ = model_dict['n_outputs_']
    model.out_activation_ = model_dict['out_activation_']
    model._label_binarizer = deserialize_label_binarizer(model_dict['_label_binarizer'])

    model.classes_ = np.array(model_dict['classes_'])

    return model
#regression.py
import numpy as np
import scipy as sp

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree._tree import Tree
from sklearn.svm import SVR
from sklearn import dummy

import csr


def serialize_linear_regressor(model):
    serialized_model = {
        'meta': 'linear-regression',
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'params': model.get_params()
    }

    return serialized_model


def deserialize_linear_regressor(model_dict):
    model = LinearRegression(model_dict['params'])

    model.coef_ = np.array(model_dict['coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])

    return model


def serialize_lasso_regressor(model):
    serialized_model = {
        'meta': 'lasso-regression',
        'coef_': model.coef_.tolist(),
        'params': model.get_params()
    }

    if isinstance(model.n_iter_, int):
        serialized_model['n_iter_'] = model.n_iter_
    else:
        serialized_model['n_iter_'] = model.n_iter_.tolist()

    if isinstance(model.n_iter_, float):
        serialized_model['intercept_'] = model.intercept_
    else:
        serialized_model['intercept_'] = model.intercept_.tolist()

    return serialized_model


def deserialize_lasso_regressor(model_dict):
    model = Lasso(model_dict['params'])

    model.coef_ = np.array(model_dict['coef_'])

    if isinstance(model_dict['n_iter_'], list):
        model.n_iter_ = np.array(model_dict['n_iter_'])
    else:
        model.n_iter_ = int(model_dict['n_iter_'])

    if isinstance(model_dict['intercept_'], list):
        model.intercept_ = np.array(model_dict['intercept_'])
    else:
        model.intercept_ = float(model_dict['intercept_'])

    return model


def serialize_ridge_regressor(model):
    serialized_model = {
        'meta': 'ridge-regression',
        'coef_': model.coef_.tolist(),
        'params': model.get_params()
    }

    if model.n_iter_:
        serialized_model['n_iter_'] = model.n_iter_.tolist()

    if isinstance(model.n_iter_, float):
        serialized_model['intercept_'] = model.intercept_
    else:
        serialized_model['intercept_'] = model.intercept_.tolist()

    return serialized_model


def deserialize_ridge_regressor(model_dict):
    model = Ridge(model_dict['params'])

    model.coef_ = np.array(model_dict['coef_'])

    if 'n_iter_' in model_dict:
        model.n_iter_ = np.array(model_dict['n_iter_'])

    if isinstance(model_dict['intercept_'], list):
        model.intercept_ = np.array(model_dict['intercept_'])
    else:
        model.intercept_ = float(model_dict['intercept_'])

    return model


def serialize_svr(model):
    serialized_model = {
        'meta': 'svr',
        'class_weight_': model.class_weight_.tolist(),
        'support_': model.support_.tolist(),
        'n_support_': model.n_support_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'probA_': model.probA_.tolist(),
        'probB_': model.probB_.tolist(),
        '_intercept_': model._intercept_.tolist(),
        'shape_fit_': model.shape_fit_,
        '_gamma': model._gamma,
        'params': model.get_params()
    }

    if isinstance(model.support_vectors_, sp.sparse.csr_matrix):
        serialized_model['support_vectors_'] = csr.serialize_csr_matrix(model.support_vectors_)
    elif isinstance(model.support_vectors_, np.ndarray):
        serialized_model['support_vectors_'] = model.support_vectors_.tolist()

    if isinstance(model.dual_coef_, sp.sparse.csr_matrix):
        serialized_model['dual_coef_'] = csr.serialize_csr_matrix(model.dual_coef_)
    elif isinstance(model.dual_coef_, np.ndarray):
        serialized_model['dual_coef_'] = model.dual_coef_.tolist()

    if isinstance(model._dual_coef_, sp.sparse.csr_matrix):
        serialized_model['_dual_coef_'] = csr.serialize_csr_matrix(model._dual_coef_)
    elif isinstance(model._dual_coef_, np.ndarray):
        serialized_model['_dual_coef_'] = model._dual_coef_.tolist()

    return serialized_model


def deserialize_svr(model_dict):
    model = SVR(**model_dict['params'])
    model.shape_fit_ = model_dict['shape_fit_']
    model._gamma = model_dict['_gamma']

    model.class_weight_ = np.array(model_dict['class_weight_']).astype(np.float64)
    model.support_ = np.array(model_dict['support_']).astype(np.int32)
    model.n_support_ = np.array(model_dict['n_support_']).astype(np.int32)
    model.intercept_ = np.array(model_dict['intercept_']).astype(np.float64)
    model.probA_ = np.array(model_dict['probA_']).astype(np.float64)
    model.probB_ = np.array(model_dict['probB_']).astype(np.float64)
    model._intercept_ = np.array(model_dict['_intercept_']).astype(np.float64)

    if 'meta' in model_dict['support_vectors_'] and model_dict['support_vectors_']['meta'] == 'csr':
        model.support_vectors_ = csr.deserialize_csr_matrix(model_dict['support_vectors_'])
        model._sparse = True
    else:
        model.support_vectors_ = np.array(model_dict['support_vectors_']).astype(np.float64)
        model._sparse = False

    if 'meta' in model_dict['dual_coef_'] and model_dict['dual_coef_']['meta'] == 'csr':
        model.dual_coef_ = csr.deserialize_csr_matrix(model_dict['dual_coef_'])
    else:
        model.dual_coef_ = np.array(model_dict['dual_coef_']).astype(np.float64)

    if 'meta' in model_dict['_dual_coef_'] and model_dict['_dual_coef_']['meta'] == 'csr':
        model._dual_coef_ = csr.deserialize_csr_matrix(model_dict['_dual_coef_'])
    else:
        model._dual_coef_ = np.array(model_dict['_dual_coef_']).astype(np.float64)

    return model


def serialize_tree(tree):
    serialized_tree = tree.__getstate__()
    # serialized_tree['nodes_dtype'] = serialized_tree['nodes'].dtype
    dtypes = serialized_tree['nodes'].dtype
    serialized_tree['nodes'] = serialized_tree['nodes'].tolist()
    serialized_tree['values'] = serialized_tree['values'].tolist()

    return serialized_tree, dtypes


def deserialize_tree(tree_dict, n_features, n_classes, n_outputs):
    tree_dict['nodes'] = [tuple(lst) for lst in tree_dict['nodes']]

    names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples']
    tree_dict['nodes'] = np.array(tree_dict['nodes'], dtype=np.dtype({'names': names, 'formats': tree_dict['nodes_dtype']}))
    tree_dict['values'] = np.array(tree_dict['values'])

    tree = Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
    tree.__setstate__(tree_dict)

    return tree


def serialize_decision_tree_regressor(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        'meta': 'decision-tree-regression',
        'feature_importances_': model.feature_importances_.tolist(),
        'max_features_': model.max_features_,
        'n_features_': model.n_features_,
        'n_outputs_': model.n_outputs_,
        'tree_': tree
    }

    # serialized_model.

    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)

    serialized_model['tree_']['nodes_dtype'] = tree_dtypes

    return serialized_model


def deserialize_decision_tree_regressor(model_dict):
    deserialized_decision_tree = DecisionTreeRegressor()

    deserialized_decision_tree.max_features_ = model_dict['max_features_']
    deserialized_decision_tree.n_features_ = model_dict['n_features_']
    deserialized_decision_tree.n_outputs_ = model_dict['n_outputs_']

    tree = deserialize_tree(model_dict['tree_'], model_dict['n_features_'], 1, model_dict['n_outputs_'])
    deserialized_decision_tree.tree_ = tree

    return deserialized_decision_tree


def serialize_dummy_regressor(model):
    model.constant = model.constant_.tolist()
    return model.__dict__


def serialize_gradient_boosting_regressor(model):

    serialized_model = {
        'meta': 'gb-regression',
        'max_features_': model.max_features_,
        'n_features_': model.n_features_,
        'train_score_': model.train_score_.tolist(),
        'params': model.get_params(),
        'estimators_shape': list(model.estimators_.shape),
        'estimators_': []
    }

    if  isinstance(model.init_, dummy.DummyRegressor):
        serialized_model['init_'] = serialize_dummy_regressor(model.init_)
        serialized_model['init_']['meta'] = 'dummy'
    elif isinstance(model.init_, str):
        serialized_model['init_'] = model.init_

    if isinstance(model.loss_, _gb_losses.LeastSquaresError):
        serialized_model['loss_'] = 'ls'
    elif isinstance(model.loss_, _gb_losses.LeastAbsoluteError):
        serialized_model['loss_'] = 'lad'
    elif isinstance(model.loss_, _gb_losses.HuberLossFunction):
        serialized_model['loss_'] = 'huber'
    elif isinstance(model.loss_, _gb_losses.QuantileLossFunction):
        serialized_model['loss_'] = 'quantile'

    if 'priors' in model.init_.__dict__:
        serialized_model['priors'] = model.init_.priors.tolist()

    for tree in model.estimators_.reshape((-1,)):
        serialized_model['estimators_'].append(serialize_decision_tree_regressor(tree))
    return serialized_model


def deserialize_gradient_boosting_regressor(model_dict):
    model = GradientBoostingRegressor(**model_dict['params'])
    trees = [deserialize_decision_tree_regressor(tree) for tree in model_dict['estimators_']]
    model.estimators_ = np.array(trees).reshape(model_dict['estimators_shape'])
    if 'init_' in model_dict and model_dict['init_']['meta'] == 'dummy':
        model.init_ = dummy.DummyRegressor()
        model.init_.__dict__ = model_dict['init_']
        model.init_.__dict__.pop('meta')


    model.train_score_ = np.array(model_dict['train_score_'])
    model.max_features_ = model_dict['max_features_']
    model.n_features_ = model_dict['n_features_']
    if model_dict['loss_'] == 'ls':
        model.loss_ = _gb_losses.LeastSquaresError(1)
    elif model_dict['loss_'] == 'lad':
        model.loss_ = _gb_losses.LeastAbsoluteError(1)
    elif model_dict['loss_'] == 'huber':
        model.loss_ = _gb_losses.HuberLossFunction(1)
    elif model_dict['loss_'] == 'quantile':
        model.loss_ = _gb_losses.QuantileLossFunction(1)

    if 'priors' in model_dict:
        model.init_.priors = np.array(model_dict['priors'])
    return model


def serialize_random_forest_regressor(model):

    serialized_model = {
        'meta': 'rf-regression',
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf': model.min_samples_leaf,
        'min_weight_fraction_leaf': model.min_weight_fraction_leaf,
        'max_features': model.max_features,
        'max_leaf_nodes': model.max_leaf_nodes,
        'min_impurity_decrease': model.min_impurity_decrease,
        'min_impurity_split': model.min_impurity_split,
        'n_features_': model.n_features_,
        'n_outputs_': model.n_outputs_,
        'estimators_': [serialize_decision_tree_regressor(decision_tree) for decision_tree in model.estimators_],
        'params': model.get_params()
    }

    if 'oob_score_' in model.__dict__:
        serialized_model['oob_score_'] = model.oob_score_
    if 'oob_decision_function_' in model.__dict__:
        serialized_model['oob_prediction_'] = model.oob_prediction_.tolist()

    return serialized_model


def deserialize_random_forest_regressor(model_dict):
    model = RandomForestRegressor(**model_dict['params'])
    estimators = [deserialize_decision_tree_regressor(decision_tree) for decision_tree in model_dict['estimators_']]
    model.estimators_ = np.array(estimators)

    model.n_features_ = model_dict['n_features_']
    model.n_outputs_ = model_dict['n_outputs_']
    model.max_depth = model_dict['max_depth']
    model.min_samples_split = model_dict['min_samples_split']
    model.min_samples_leaf = model_dict['min_samples_leaf']
    model.min_weight_fraction_leaf = model_dict['min_weight_fraction_leaf']
    model.max_features = model_dict['max_features']
    model.max_leaf_nodes = model_dict['max_leaf_nodes']
    model.min_impurity_decrease = model_dict['min_impurity_decrease']
    model.min_impurity_split = model_dict['min_impurity_split']

    if 'oob_score_' in model_dict:
        model.oob_score_ = model_dict['oob_score_']
    if 'oob_prediction_' in model_dict:
        model.oob_prediction_ =np.array(model_dict['oob_prediction_'])

    return model


def serialize_mlp_regressor(model):
    serialized_model = {
        'meta': 'mlp-regression',
        'coefs_': model.coefs_,
        'loss_': model.loss_,
        'intercepts_': model.intercepts_,
        'n_iter_': model.n_iter_,
        'n_layers_': model.n_layers_,
        'n_outputs_': model.n_outputs_,
        'out_activation_': model.out_activation_,
        'params': model.get_params()
    }

    return serialized_model


def deserialize_mlp_regressor(model_dict):
    model = MLPRegressor(**model_dict['params'])

    model.coefs_ = model_dict['coefs_']
    model.loss_ = model_dict['loss_']
    model.intercepts_ = model_dict['intercepts_']
    model.n_iter_ = model_dict['n_iter_']
    model.n_layers_ = model_dict['n_layers_']
    model.n_outputs_ = model_dict['n_outputs_']
    model.out_activation_ = model_dict['out_activation_']

    return model
#csr.py
import numpy as np
import scipy as sp


def serialize_csr_matrix(csr_matrix):
    serialized_csr_matrix = {
        'meta': 'csr',
        'data': csr_matrix.data.tolist(),
        'indices': csr_matrix.indices.tolist(),
        'indptr': csr_matrix.indptr.tolist(),
        '_shape': csr_matrix._shape,
    }
    return serialized_csr_matrix


def deserialize_csr_matrix(csr_dict, data_type=np.float64, indices_type=np.int32, indptr_type=np.int32):
    csr_matrix = sp.sparse.csr_matrix(tuple(csr_dict['_shape']))
    csr_matrix.data = np.array(csr_dict['data']).astype(data_type)
    csr_matrix.indices = np.array(csr_dict['indices']).astype(indices_type)
    csr_matrix.indptr = np.array(csr_dict['indptr']).astype(indptr_type)

    return csr_matrix
Fin del código. Estado del documento de la memoria final actualmente:  

Texto

Descripción generada automáticamente 

Universidad Internacional de La Rioja 

Escuela Superior de Ingeniería y 

Tecnología 

 

 

 

 

Máster Universitario en Inteligencia artificial 

Sistema de Predicción basado en Aprendizaje Automático: Plataforma no code para Usuarios no Expertos 

 

 

 

Trabajo fin de estudio presentado por: 

Iván Castro Martínez 

Tipo de trabajo: 

Inteligencia Artificial 

Director/a: 

Carlos Rubert Escuder 

Fecha: 

 

Resumen 

El aprendizaje automático es una tecnología que está cambiando el flujo de trabajo de muchísimas empresas. En el último año, el 36% de las empresas españolas ya han adoptado tecnologías de Inteligencia Artificial y Machine Learning, convirtiéndolas en herramientas indispensables para procesar, analizar y obtener conocimientos valiosos de los datos. Este proyecto presenta el desarrollo de un sistema no-code automatizado para el entrenamiento y evaluación de modelos de aprendizaje automático. El objetivo principal es generar una plataforma donde los usuarios puedan generar predicciones sobre sus datos, y mejorar las predicciones a través de la especificación de parámetros y características de los modelos. Para lograr este fin, se siguió una metodología que incluye investigación, diseño, implementación y validación del sistema. 

Los resultados obtenidos indican que el sistema no solo cumple con los requisitos de automatización y personalización, sino que también mejora significativamente la eficiencia y efectividad de los procesos de análisis de datos. La principal conclusión de este estudio es que la automatización en el entrenamiento y evaluación de modelos de aprendizaje automático puede mejorar considerablemente la gestión de datos y la generación de modelos precisos y eficientes. 

Palabras clave: aprendizaje automático — automatización — selección de características — optimización de hiperparámetros — análisis de datos. 

 

 

 

 

Abstract 

Machine learning is a technology that is changing the workflow of many companies. In the last year, 36% of Spanish companies have already adopted Artificial Intelligence and Machine Learning technologies, turning them into essential tools to process, analyze and obtain valuable insights from data. This project presents the development of an automated no-code system for the training and evaluation of machine learning models. The main objective is to generate a platform where users can generate predictions on their data and improve the predictions through the specification of parameters and characteristics of the models. To achieve this end, a methodology was followed that includes research, design, implementation and validation of the system. 

The results obtained indicate that the system not only meets the automation and customization requirements, but also significantly improves the efficiency and effectiveness of data analysis processes. The main conclusion of this study is that automation in the training and evaluation of machine learning models can improve data management and the generation of accurate and efficient models. 

Keywords: machine learning — automation — feature selection — hyperparameter optimization — data analysis. 

 

 

 

 

 

 

 

 

Índice de contenidos 

 

INTRODUCCIÓN 

 MOTIVACIÓN 

En la era de la digitalización y el aumento exponencial de datos, las organizaciones de todo tipo buscan continuamente maneras de optimizar sus procesos y tomar decisiones más informadas basadas en datos. El aprendizaje automático se ha convertido en una herramienta clave para analizar y prever tendencias a partir de grandes conjuntos de datos. 

Según las estadísticas, se espera que el tamaño del mercado global de la IA crezca un 37% cada año desde 2023 hasta 2030. Más del 40% de los líderes empresariales informan un aumento de la productividad a través de la automatización de la IA. Esto indica la creciente adopción de soluciones de inteligencia artificial en las organizaciones para obtener una ventaja competitiva. https://www.techopedia.com/es/estadisticas-inteligencia-artificial añade más información a: Motivación 

Considerando el paradigma nacional, aproximadamente un 40% de grandes empresas usa herramientas en IA en alguno de sus procesos, un 20% en medianas y un 10% en pequeñas. https://portal.mineco.gob.es/es-es/digitalizacionIA/Documents/Estrategia_IA_2024.pdf 

El uso del aprendizaje automático crece a diario y se espera que siga aumentando en los próximos años según las organizaciones reconozcan sus beneficios. Utilizarlo, ofrece un gran potencial para resolver una variedad de desafíos en diversas áreas. Podemos obtener conocimientos significativos a partir de los datos, facilitar la toma de decisiones o usarse para optimizar y agilizar tareas y procesos específicos. Esto puede ser beneficioso para una empresa al automatizar operaciones, promover la sostenibilidad o reducir costos y el uso de recursos. Algunas de las principales ventajas del aprendizaje automático incluyen la optimización del análisis de Big Data, la mejora en la experiencia de los consumidores y la automatización de procesos repetitivos. Estas aplicaciones permiten a las empresas tomar decisiones más informadas, mejorar la eficiencia operativa y adaptarse mejor a las necesidades cambiantes del mercado. 

En este contexto, el desarrollo de sistemas automatizados para el entrenamiento y evaluación de modelos de aprendizaje automático es crucial. Estos sistemas no solo permiten a los usuarios especificar parámetros y características del modelo de manera eficiente, sino que también automatizan la selección de características relevantes y la optimización de hiperparámetros. Esto puede resultar en mejoras significativas en la precisión y efectividad de los modelos, lo que a su vez se traduce en una toma de decisiones más informada y procesos más eficientes. 

  

 PLANTEAMIENTO DEL TRABAJO 

El desarrollo de este sistema de entrenamiento y evaluación automatizado para modelos de aprendizaje automático responde a la creciente demanda de herramientas eficientes y accesibles en el campo del análisis de datos. A medida que los volúmenes de datos aumentan y se diversifican, surge la necesidad de optimizar los procesos de modelado y análisis de datos a través de la automatización para mejorar la precisión y eficiencia. 

En IMMERSIA, trabajamos con numerosos clientes que generan grandes volúmenes de datos, cada uno proveniente de diferentes sectores y con características únicas. Esto impone la necesidad de contar con un sistema dinámico capaz de comprender y procesar estos datos de manera eficiente, extrayendo información relevante y generando predicciones precisas que satisfagan las necesidades específicas de cada cliente. 

El sistema propuesto integrará funcionalidades avanzadas como el tratamiento personalizado de los datos iniciales, la selección automatizada de características y la optimización de hiperparámetros. También se incluirá la capacidad de adaptar los algoritmos de modelado a los datos de cada cliente y proporcionar recomendaciones para mejorar continuamente las predicciones. Esto se logrará mediante el análisis de los datos, la identificación de patrones y la sugerencia de ajustes en los parámetros del modelo. 

La implementación de este sistema se basará en el uso de bibliotecas y frameworks de aprendizaje automático de vanguardia, como Scikit-Learn, TensorFlow y PyTorch.  Además, se explorarán diferentes técnicas de automatización, incluyendo el aprendizaje automático basado en modelos y en reglas, para optimizar la eficiencia y precisión del sistema. 

A lo largo de este trabajo, se abordarán los desafíos técnicos y prácticos asociados con la creación de un sistema automatizado que pueda manejar diversas tareas de aprendizaje automático, desde el preprocesamiento de datos hasta la evaluación de modelos. El objetivo es ofrecer una solución técnica avanzada y una herramienta amigable y accesible a usuarios con diferentes niveles de experiencia técnica, democratizando así el uso de técnicas avanzadas de aprendizaje automático. 

  

 ESTRUCTURA DEL TRABAJO 

El documento está estructurado en varias secciones principales que guiarán al lector a través de todo el proceso de desarrollo del sistema, desde el contexto y estado del arte hasta las conclusiones y recomendaciones para trabajos futuros. En la Sección 2, se hará una revisión exhaustiva del contexto y estado del arte, analizando los desarrollos recientes en desarrollo de sistemas dinámicos de aprendizaje automático. Esta sección establecerá la base teórica y técnica sobre la cual se construye el sistema propuesto. En la sección 3, se definirán los objetivos concretos del proyecto y se describirá la metodología seguida para alcanzarlos. Esto incluye las fases de investigación, diseño, implementación y pruebas del sistema, así como los criterios de evaluación utilizados para medir su eficacia. La sección 4 estará dedicada al desarrollo específico de la contribución, detallando la arquitectura del sistema, las tecnologías utilizadas y el proceso de implementación de cada uno de sus componentes. Aquí se explicará el tratamiento de datos, la selección de características, el entrenamiento y evaluación de modelos, y la optimización de hiperparámetros. En la sección 5, se presentarán las conclusiones y trabajo futuro, reflexionando sobre los logros del proyecto y las lecciones aprendidas. También se propondrán posibles mejoras y extensiones del sistema para futuras investigaciones y desarrollos. La sección 6 incluirá las referencias bibliográficas, enumerando todas las fuentes académicas y técnicas consultadas para la elaboración del proyecto, proporcionando una base sólida de apoyo teórico y técnico. En la sección 7, se describirán las herramientas para buscar bibliografía, facilitando la localización de información relevante para futuras investigaciones. Finalmente, en la sección 8, se presentarán los anexos, incluyendo material adicional que respalda y profundiza en los detalles técnicos y metodológicos del trabajo, como códigos fuente, resultados adicionales y cualquier otra información complementaria. Cada parte del documento está diseñada para proporcionar una comprensión profunda del trabajo realizado, justificando metodologías, discutiendo resultados y proponiendo futuras líneas de investigación. 

 

Contexto y estado del arte 

En este apartado se profundizará en el contexto del problema abordado en el proyecto, así como en el estado del arte de las soluciones existentes en el campo de la ejecución de modelos de aprendizaje automático y la interfaz de usuario para la configuración de predicciones. 

Se analiza en detalle el contexto en el que surge la necesidad de desarrollar un sistema como el propuesto en este trabajo. Se exploran las limitaciones y desafíos asociados con la ejecución de modelos de aprendizaje automático, especialmente para usuarios no expertos en programación. Se pueden discutir aspectos como la complejidad de las herramientas existentes, la falta de accesibilidad para personas con conocimientos limitados en el campo, y la necesidad de simplificar el proceso de generación de predicciones para aplicaciones del mundo real. 

Además, se identifican y describen casos de uso específicos donde el sistema propuesto podría ofrecer un valor significativo, como en la industria, la investigación académica, o la toma de decisiones en diferentes áreas. 

Contexto del problema 

En este apartado se realizará un análisis del estado del arte en relación con sistemas similares o relacionados con el objetivo de este proyecto. Se revisan investigaciones previas, herramientas existentes y avances tecnológicos relevantes en el ámbito de la ejecución de modelos de aprendizaje automático a través de interfaces visuales. 

Se identifican y evalúan las fortalezas y limitaciones de las soluciones existentes, destacando las características clave, las tecnologías utilizadas y las tendencias emergentes en este campo. Se pueden abordar temas como la usabilidad, la escalabilidad, la precisión de los modelos y la flexibilidad para adaptarse a diferentes tipos de datos y problemas de machine learning [2]. 

Estado del arte 

Se realizará un análisis exhaustivo del estado del arte en relación con sistemas similares o relacionados con el objetivo de este proyecto. Se revisarán investigaciones previas, herramientas existentes y avances tecnológicos relevantes en el ámbito de la ejecución de modelos de aprendizaje automático a través de interfaces visuales. 

Conclusiones 

En este apartado se extraerán conclusiones del análisis realizado en el contexto del problema y el estado del arte. Se resumen los principales hallazgos y se destacan las implicaciones para el desarrollo del sistema propuesto en este trabajo. Además, se pueden identificar áreas de oportunidad y posibles desafíos a abordar durante la implementación del sistema. Estas conclusiones servirán como punto de partida para el diseño y desarrollo de la solución, asegurando que se aborden adecuadamente las necesidades y requerimientos identificados en el análisis previo. 

 

 

3. OBJETIVOS CONCRETOS Y METODOLOGÍA DE TRABAJO 

3.1. OBJETIVO GENERAL 

El objetivo general de este proyecto es desarrollar un sistema de entrenamiento y evaluación automatizado de modelos de aprendizaje automático que permita a los usuarios especificar los parámetros del modelo y las características a utilizar. Además, el sistema debe ser capaz de seleccionar automáticamente las características más relevantes y proporcionar recomendaciones sobre los parámetros del modelo. 

3.2. OBJETIVOS ESPECÍFICOS 

El módulo se compone de varias funcionalidades para llevar a cabo un módulo robusto y con alta capacidad de personalización, mejorando la experiencia del usuario.  

El programa debe analizar los datos, y, según los inputs obtenidos, entrenar y predecir. Las fases son: 

Conexión con backend para recibir los datos 

En este paso recibimos la configuración de datos del usuario. Dispondremos de todos los datos necesarios para utilizar el módulo y los preparamos para ello. 

Implementación de un sistema de preprocesamiento de datos 

En esta fase inicial , desarrollamos funciones que permitan el manejo de valores nulos, normalización de datos y codificación de variables categóricas, adaptándose automáticamente a las necesidades del dataset proporcionado. 

Selección automática de características relevantes: 

Implementar algoritmos que identifiquen y seleccionen las características más relevantes para el modelo, utilizando técnicas como SelectKBest, que mejoren la eficiencia y precisión del modelo. 

Entrenamiento y evaluación de múltiples modelos: 

Desarrollar un sistema que permita entrenar múltiples modelos de aprendizaje automático en paralelo, evaluar su desempeño y seleccionar el modelo óptimo basado en métricas de rendimiento específicas. 

Optimización de hiperparámetros: 

Integrar métodos para la optimización de hiperparámetros de los modelos, utilizando técnicas como la búsqueda en cuadrícula (Grid Search) y la búsqueda aleatoria (Random Search), para mejorar el rendimiento del modelo final. 

Generación de informes detallados: 

Crear funciones que generen informes automáticos de evaluación de modelos, incluyendo métricas de rendimiento, visualizaciones de resultados y recomendaciones para mejorar el modelo. 

3.3.  METODOLOGÍA DEL TRABAJO 

Para alcanzar los objetivos planteados, se ha seguido una metodología estructurada en varias fases: investigación, diseño, implementación, validación y documentación. A continuación, se detallan estas fases. 

3.3.1. Investigación 

En esta fase, se llevó a cabo un estudio exhaustivo sobre el estado del arte en técnicas de selección de características, algoritmos de aprendizaje automático, y metodologías de recomendación de hiperparámetros. Las actividades incluyeron: 

Revisión de literatura: Análisis de artículos científicos, libros y recursos en línea sobre técnicas de selección de características y evaluación de modelos. 

Análisis de herramientas existentes: Evaluación de herramientas y bibliotecas de aprendizaje automático populares (e.g., Scikit-learn, TensorFlow, PyTorch) para identificar funcionalidades y limitaciones. 

3.3.2. Diseño 

Basándose en la investigación, se diseñó la arquitectura del sistema. Las principales decisiones de diseño incluyeron: 

Estructura modular: Definición de módulos independientes para la carga de datos, preprocesamiento, entrenamiento de modelos, selección de características y evaluación. 

Integración de componentes: Especificación de cómo los diferentes módulos interactuarán entre sí y con la API de evaluación. 

Reuniones iniciales con el equipo: Recopilación de requisitos específicos y expectativas del sistema. 

Diseño de arquitectura: Definición de la arquitectura del sistema, incluyendo la estructura de datos, componentes de software y flujo de información. 

3.3.3. Implementación 

La implementación se llevó a cabo en varias iteraciones, siguiendo prácticas de desarrollo ágil. Las actividades incluyeron: 

Desarrollo de módulos: Implementación de los módulos definidos, comenzando por la carga y preprocesamiento de datos, seguido por el entrenamiento y evaluación de modelos. 

Selección de características: Implementación de algoritmos de selección de características como SelectKBest con f_classif y f_regression. 

Evaluación de modelos: Implementación de métodos para evaluar modelos de clasificación y regresión, utilizando métricas apropiadas y visualizaciones como matrices de confusión y gráficos de dispersión. 

Recomendación de hiperparámetros: Desarrollo de funciones para recomendar hiperparámetros basados en técnicas de optimización y validación cruzada. 

Optimización de hiperparámetros: Implementación de métodos de optimización como Grid Search y Random Search, y realización de pruebas para validar su efectividad en diferentes modelos y datasets. 

3.3.4. Validación 

Para asegurar la efectividad y robustez del sistema, se llevaron a cabo diversas actividades de validación: 

Pruebas unitarias y de integración: Desarrollo de pruebas automatizadas para verificar el correcto funcionamiento de cada módulo y su integración. 

Evaluación de rendimiento: Comparación de modelos entrenados con y sin selección automática de características, utilizando conjuntos de datos estándar. 

Pruebas de usuario: Recolección de retroalimentación de usuarios sobre la interfaz y funcionalidades del sistema, realizando ajustes según sea necesario. 

Pruebas finales y validación del sistema: Realización de pruebas de integración para asegurar que todos los componentes del sistema funcionan correctamente en conjunto, y recopilación de feedback de usuarios finales. 

3.3.5. Documentación 

Finalmente, se elaboró la documentación del sistema, abarcando: 

Manual del usuario: Guía detallada sobre cómo utilizar la interfaz de usuario para especificar parámetros de entrenamiento y características. 

Documentación técnica: Descripción de la arquitectura del sistema, los algoritmos implementados y las decisiones de diseño. 

Memoria del proyecto: Informe detallado del proyecto, incluyendo los objetivos, metodología, desarrollo y resultados obtenidos. 

Despliegue del sistema: Implementación del sistema en un entorno de producción y soporte post-despliegue. 

 

 

 

Referencias bibliográficas 

[1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer Science & Business Media. 

 

[1] https://www.incentro.com/es-ES/blog/aprendizaje-automatico-beneficios-para-organizacion 

http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf 
[2] Kaggle. (2022). What is Machine Learning? Retrieved from https://www.kaggle.com/learn/machine-learning/  