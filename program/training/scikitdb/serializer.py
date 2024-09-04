# serializer.py
import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, Type, Protocol
import pandas as pd

from sklearn.svm import SVC, SVR
from sklearn import svm, discriminant_analysis, dummy

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import ElasticNet, PassiveAggressiveClassifier, PassiveAggressiveRegressor, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, HuberRegressor, TheilSenRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

from utils.logger import logger
import training.scikitdb.classification as clf
import training.scikitdb.regression as reg

class ModellNotSupported(Exception):
    pass
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
    
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "GaussianProcessClassifier": GaussianProcessClassifier,
    "GaussianProcessRegressor": GaussianProcessRegressor,
    "ElasticNet": ElasticNet,
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor,
    "LassoLars": LassoLars,
    "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit,
    "BayesianRidge": BayesianRidge,
    "ARDRegression": ARDRegression,
    "HuberRegressor": HuberRegressor,
    "TheilSenRegressor": TheilSenRegressor,
    "DummyClassifier": dummy.DummyClassifier,
    "DummyRegressor": dummy.DummyRegressor,
}

# Las funciones de serialización se moverán a los archivos correspondientes:
# classification.py y regression.py

classification_models = {
    BernoulliNB, GaussianNB, MultinomialNB, ComplementNB,
    discriminant_analysis.LinearDiscriminantAnalysis,
    discriminant_analysis.QuadraticDiscriminantAnalysis,
    Perceptron, DecisionTreeClassifier, GradientBoostingClassifier,
    RandomForestClassifier, MLPClassifier, LogisticRegression, SVC,
    KNeighborsClassifier, SGDClassifier, AdaBoostClassifier, BaggingClassifier,
    ExtraTreesClassifier, GaussianProcessClassifier, PassiveAggressiveClassifier,
}

regression_models = {
    LinearRegression, Lasso, Ridge, DecisionTreeRegressor,
    GradientBoostingRegressor, RandomForestRegressor, MLPRegressor,
    SVR, KNeighborsRegressor, SGDRegressor, AdaBoostRegressor, BaggingRegressor,
    ExtraTreesRegressor, GaussianProcessRegressor, ElasticNet,
    PassiveAggressiveRegressor, LassoLars, OrthogonalMatchingPursuit,
    BayesianRidge, ARDRegression, HuberRegressor, TheilSenRegressor,
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
    
    
def clean_filename(filename):
    """
    Reemplaza caracteres no permitidos en nombres de archivo con guiones bajos.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def ensure_directory_exists(directory):
    """
    Crea un directorio si no existe.
    """
    os.makedirs(directory, exist_ok=True)

def get_safe_path(base_path, *parts):
    """
    Crea una ruta segura combinando base_path y partes adicionales,
    asegurándose de que los nombres de archivo y directorio sean válidos.
    """
    safe_parts = [clean_filename(part) for part in parts]
    safe_path = os.path.join(base_path, *safe_parts)
    ensure_directory_exists(os.path.dirname(safe_path))
    return safe_path

def to_dict(model):
    return serialize_model(model)

def from_dict(model_dict):
    return deserialize_model(model_dict)

def to_pickle(model, model_name, dataset_path, accuracy=None, X_train=None):
    """
    Guarda el modelo en formato pickle.

    Args:
    model: El modelo a guardar.
    model_name (str): Nombre base del modelo (ej. 'KNeighborsClassifier_base').
    dataset_path (str): Ruta completa al archivo del dataset.
    accuracy (float, optional): Precisión del modelo para clasificación o mejor métrica para regresión.
    X_train (pd.DataFrame, optional): Datos de entrenamiento para calcular las medias de las características.

    Returns:
    str: Ruta donde se guardó el modelo.
    """
    try:
        # Extraer el nombre del dataset del path
        dataset_name = Path(dataset_path).stem
        model_name = Path(model_name).stem
        
        # Construir el nombre del modelo
        if accuracy is not None:
            accuracy_str = f"{accuracy:.2f}".replace(".", "_")
            full_model_name = f"{model_name}_{accuracy_str}_{dataset_name}.pkl"
        else:
            full_model_name = f"{model_name}_{dataset_name}.pkl"
        
        # Crear la estructura de carpetas
        model_dir = os.path.join('program', 'almacen', 'models', dataset_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Ruta completa del archivo
        model_path = os.path.join(model_dir, full_model_name)
        
        # Preparar la información del modelo
        model_info = {
            'model': model,
            'feature_names': getattr(model, 'feature_names_', None),
            'feature_means': X_train.mean().to_dict() if isinstance(X_train, pd.DataFrame) else None
        }
        
        # Guardar el modelo
        with open(model_path, 'wb') as model_file:
            pickle.dump(model_info, model_file)
        
        print(f"Modelo guardado en: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error al guardar el modelo: {str(e)}")
        return None

def from_pickle(model_path):
    """
    Carga un modelo desde un archivo pickle.

    Args:
    model_path (str): Ruta al archivo pickle del modelo.

    Returns:
    object: El modelo cargado.

    Raises:
    FileNotFoundError: Si el archivo no existe.
    pickle.UnpicklingError: Si hay un error al deserializar el archivo.
    """
    try:
        with open(model_path, 'rb') as model_file:
            model_info = pickle.load(model_file)
        
        if isinstance(model_info, dict):
            model = model_info.get('model')
            if model is None:
                raise ValueError("El archivo pickle no contiene un modelo válido.")
            
            # Restaurar atributos adicionales si existen
            if 'feature_names' in model_info:
                model.feature_names_ = model_info['feature_names']
            if 'feature_means' in model_info:
                model.feature_means_ = model_info['feature_means']
            else:
                # Si no tenemos las medias, inicializamos a 0
                model.feature_means_ = {feature: 0 for feature in getattr(model, 'feature_names_', [])}
        else:
            model = model_info  # El archivo contiene directamente el modelo
        
        logger.info(f"Modelo cargado exitosamente desde: {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo del modelo: {model_path}")
        raise
    except pickle.UnpicklingError as e:
        logger.error(f"Error al deserializar el modelo: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al cargar el modelo: {str(e)}")
        raise


def load_model(model_name):
    """
    Carga un modelo entrenado desde el disco.
    """
    try:
        model_path = find_model_path(model_name)
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"No se pudo encontrar el modelo: {model_name}")
        return None

def find_model_path(model_name):
    """
    Busca la ruta de un modelo guardado.
    """
    # Extraer el nombre del dataset del nombre del modelo
    dataset_name = Path(model_name).stem
    dataset_name = '_'.join(dataset_name.split('_')[dataset_name.split('_').index(next(s for s in dataset_name.split('_') if s.isdigit())) + 2:])
    
    # Construir la ruta del directorio específico del dataset
    dataset_dir = os.path.join('program', 'almacen', 'models', dataset_name)
    
    # Buscar el archivo del modelo en el directorio específico del dataset
    for file in os.listdir(dataset_dir):
        if model_name in file:
            return os.path.join(dataset_dir, file)
    
    return None

def list_stored_models():
    """
    Lista todos los modelos almacenados.
    """
    models_dir = 'program/almacen/models'
    stored_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            stored_models.append((model_name, model_path))
    return stored_models

def get_dataset_path(dataset_name, optimized=False):
    """
    Obtiene la ruta del dataset almacenado.

    Args:
    dataset_name (str): Nombre del dataset
    optimized (bool): Si se debe usar la versión optimizada

    Returns:
    str: Ruta al archivo CSV del dataset
    """
    suffix = "optimized" if optimized else "basic"
    return str(Path(f"program/almacen/datasets/{dataset_name}/{dataset_name}_{suffix}.csv"))

def list_stored_datasets():
    """
    Lista los datasets almacenados en program/almacen/datasets.

    Returns:
    list: Lista de tuplas (nombre_dataset, ruta_basic, ruta_optimized)
    """
    datasets_dir = Path("program/almacen/datasets")
    stored_datasets = []

    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            basic_path = dataset_dir / f"{dataset_dir.name}_basic.csv"
            optimized_path = dataset_dir / f"{dataset_dir.name}_optimized.csv"
            
            if basic_path.exists() or optimized_path.exists():
                stored_datasets.append((
                    dataset_dir.name,
                    str(basic_path) if basic_path.exists() else None,
                    str(optimized_path) if optimized_path.exists() else None
                ))

    return stored_datasets

def get_dataset_for_model(model_name):
    """
    Obtiene el nombre del dataset asociado a un modelo.
    """
    model_info_path = f'program/almacen/models/{model_name}/model_info.json'
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        return model_info.get('dataset_name')
    return None

def get_features_for_dataset(dataset_name):
    """
    Obtiene las características (columnas) de un dataset almacenado.

    Args:
    dataset_name (str): Nombre del dataset.

    Returns:
    list: Lista de nombres de las características del dataset.
    """
    try:
        # Construir la ruta al archivo del dataset
        dataset_path = os.path.join('program', 'almacen', 'datasets', dataset_name, f'{dataset_name}_basic.csv')
        
        logger.debug(f"Intentando leer el dataset desde: {dataset_path}")
        
        # Leer el archivo CSV
        df = pd.read_csv(dataset_path)
        
        # Obtener los nombres de las columnas
        features = df.columns.tolist()
        
        logger.debug(f"Características obtenidas: {features}")
        
        return features
    except Exception as e:
        logger.error(f"Error al obtener las características del dataset {dataset_name}: {str(e)}")
        return []