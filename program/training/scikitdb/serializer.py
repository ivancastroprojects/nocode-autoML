# serializer.py
import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, Type, Protocol
import pandas as pd
import numpy as np
import traceback

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

def to_pickle(model, model_name, dataset_path, accuracy=None, X_train=None, actual_feature_names: list | None = None, preprocessor=None):
    """
    Guarda el modelo en formato pickle.

    Args:
    model: El modelo a guardar.
    model_name (str): Nombre base del modelo (ej. 'KNeighborsClassifier_base').
    dataset_path (str): Ruta completa al archivo del dataset.
    accuracy (float, optional): Precisión del modelo para clasificación o mejor métrica para regresión.
    X_train (pd.DataFrame, optional): Datos de entrenamiento para calcular las medias/mins/maxs de las características.
    actual_feature_names (list, optional): Lista explícita de nombres de características utilizados para entrenar el modelo.
    preprocessor (object, optional): El objeto preprocesador (ej. ColumnTransformer) ajustado.

    Returns:
    str: Ruta donde se guardó el modelo.
    """
    try:
        # Extraer el nombre del dataset del path
        dataset_name_from_path = Path(dataset_path).stem # Esto podría ser el nombre corto del dataset
        
        full_model_name_stem = clean_filename(Path(model_name).stem)
        full_model_name_pkl = f"{full_model_name_stem}.pkl"

        model_dir = os.path.join('program', 'almacen', 'models', clean_filename(dataset_name_from_path))
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, full_model_name_pkl)
        
        feature_names_to_save = None
        if actual_feature_names:
            feature_names_to_save = actual_feature_names
        elif isinstance(X_train, pd.DataFrame) and not X_train.empty:
            feature_names_to_save = X_train.columns.tolist()
        elif hasattr(model, 'feature_names_in_'):
            feature_names_to_save = list(getattr(model, 'feature_names_in_'))
        elif hasattr(model, 'feature_names_'): 
            feature_names_to_save = list(getattr(model, 'feature_names_'))

        if feature_names_to_save and hasattr(model, 'fit'): 
            try:
                if not hasattr(model, 'feature_names_in_') or getattr(model, 'feature_names_in_', None) is None:
                    setattr(model, 'feature_names_in_', np.array(feature_names_to_save, dtype=object))
                    logger.info(f"Atributo 'feature_names_in_' establecido en el modelo '{full_model_name_stem}' antes de guardar.")
            except Exception as e:
                logger.warning(f"No se pudo establecer 'feature_names_in_' en el modelo '{full_model_name_stem}': {e}")
        
        feature_means_dict = None
        feature_mins_dict = None
        feature_maxs_dict = None

        if isinstance(X_train, pd.DataFrame) and not X_train.empty:
            # Asegurarse de que solo se calculan para columnas numéricas si es posible
            # o manejar el error si .mean(), .min(), .max() fallan en columnas no numéricas.
            # Por ahora, se asume que X_train aquí ya está preprocesado si aplica, o es puramente numérico.
            try:
                numeric_X_train = X_train.select_dtypes(include=np.number)
                if not numeric_X_train.empty:
                    feature_means_dict = numeric_X_train.mean().to_dict()
                    feature_mins_dict = numeric_X_train.min().to_dict()
                    feature_maxs_dict = numeric_X_train.max().to_dict()
                else: # Si no hay columnas numéricas en X_train (improbable pero posible)
                    logger.warning(f"X_train para {full_model_name_stem} no contiene columnas numéricas para calcular min/max/mean.")
            except Exception as e:
                 logger.warning(f"Error calculando min/max/mean para {full_model_name_stem} desde X_train: {e}")

        # Preparar la información del modelo
        model_info = {
            'model': model,
            'feature_names': feature_names_to_save,
            'feature_means': feature_means_dict if feature_means_dict is not None else (getattr(model, 'feature_means_', None) if feature_names_to_save else None),
            'feature_mins': feature_mins_dict,
            'feature_maxs': feature_maxs_dict,
            'preprocessor': preprocessor 
        }
        
        if model_info['feature_means'] is None and hasattr(model, 'feature_means_'):
            model_info['feature_means'] = getattr(model, 'feature_means_')

        with open(model_path, 'wb') as model_file:
            pickle.dump(model_info, model_file)
        
        logger.info(f"Modelo guardado en: {model_path} con feature_names: {feature_names_to_save is not None}, preprocessor: {preprocessor is not None}")
        return model_path
    except Exception as e:
        logger.error(f"Error al guardar el modelo '{model_name}' en '{dataset_path}': {str(e)}")
        logger.error(traceback.format_exc())
        return None

def from_pickle(model_path):
    """
    Carga un modelo desde un archivo pickle.
    Devuelve siempre un diccionario {'model': model_obj, 'feature_names': ..., 'feature_means': ..., 'feature_mins': ..., 'feature_maxs': ..., 'preprocessor': ...}.
    Si el pickle contiene un objeto modelo crudo, lo envuelve.
    También intenta asegurar que los atributos feature_names_ y feature_means_ estén en el objeto modelo.

    Args:
    model_path (str): Ruta al archivo pickle del modelo.

    Returns:
    dict: Un diccionario con el modelo y metadatos asociados.

    Raises:
    FileNotFoundError: Si el archivo no existe.
    pickle.UnpicklingError: Si hay un error al deserializar el archivo.
    ValueError: Si el formato del pickle es inesperado o no se puede extraer un modelo.
    """
    try:
        with open(model_path, 'rb') as model_file:
            loaded_content = pickle.load(model_file)

        final_model_dict = {}
        model_obj = None

        if isinstance(loaded_content, dict) and 'model' in loaded_content:
            final_model_dict = loaded_content
            model_obj = final_model_dict.get('model')
            
            if model_obj:
                saved_feature_names = final_model_dict.get('feature_names')
                saved_feature_means = final_model_dict.get('feature_means')
                # feature_mins, feature_maxs, preprocessor son obtenidos directamente por .get() abajo

                if saved_feature_names is not None:
                    setattr(model_obj, 'feature_names_', saved_feature_names)
                
                if saved_feature_means is not None:
                    setattr(model_obj, 'feature_means_', saved_feature_means)
                elif hasattr(model_obj, 'feature_names_') and model_obj.feature_names_ is not None:
                    default_means = {fn: 0 for fn in model_obj.feature_names_}
                    setattr(model_obj, 'feature_means_', default_means)
                    final_model_dict['feature_means'] = default_means 

        elif not isinstance(loaded_content, dict): 
            logger.warning(f"El archivo pickle {model_path} contenía un objeto modelo crudo. Envolviéndolo en un diccionario.")
            model_obj = loaded_content
            
            feature_names = getattr(model_obj, 'feature_names_', None)
            feature_means = getattr(model_obj, 'feature_means_', None)

            if feature_names and feature_means is None: 
                feature_means = {fn: 0 for fn in feature_names}
                setattr(model_obj, 'feature_means_', feature_means) 

            final_model_dict = {
                'model': model_obj,
                'feature_names': feature_names,
                'feature_means': feature_means,
                'feature_mins': None, # No disponible en formato antiguo
                'feature_maxs': None, # No disponible en formato antiguo
                'preprocessor': None  # No disponible en formato antiguo
            }
        else: 
            raise ValueError(f"Formato de diccionario inesperado o clave 'model' faltante en el archivo pickle: {model_path}")

        if not model_obj: 
             raise ValueError(f"No se pudo extraer un objeto modelo válido de {model_path}")

        # Asegurar que las claves nuevas existan en el diccionario final, incluso si son None
        final_model_dict.setdefault('feature_mins', None)
        final_model_dict.setdefault('feature_maxs', None)
        final_model_dict.setdefault('preprocessor', None)
        
        # Si feature_mins/maxs se cargaron y son None, pero X_train estaba presente en el dict (caso raro), intentar recalcular (esto es más para retrocompatibilidad)
        # Sin embargo, el enfoque principal es que se guarden correctamente con to_pickle.

        logger.info(f"Modelo y metadatos cargados exitosamente desde: {model_path}")
        return final_model_dict
        
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

def list_stored_datasets():
    """
    Lista los datasets almacenados en program/almacen/datasets.
    Busca un archivo '{dataset_name}.csv' como el principal y un '{dataset_name}_processed.csv' como el procesado.

    Returns:
    list: Lista de tuplas (nombre_dataset, ruta_original_raw, ruta_procesada_opcional)
    """
    datasets_base_dir = Path("program/almacen/datasets")
    stored_datasets = []

    if not datasets_base_dir.is_dir():
        logger.warning(f"El directorio base de datasets no existe: {datasets_base_dir}")
        return stored_datasets

    for dataset_folder in datasets_base_dir.iterdir():
        if dataset_folder.is_dir():
            dataset_name = dataset_folder.name  # e.g., "titanic", "iris"
            
            # El archivo original/virgen se llama igual que la carpeta contenedora
            original_raw_path = dataset_folder / f"{dataset_name}.csv"
            processed_path = dataset_folder / f"{dataset_name}_processed.csv"
            
            # El dataset se lista si existe el archivo original "{dataset_name}.csv".
            # El archivo procesado es opcional.
            if original_raw_path.exists():
                stored_datasets.append((
                    dataset_name,
                    str(original_raw_path),
                    str(processed_path) if processed_path.exists() else None
                ))
            elif processed_path.exists():
                # Si solo existe el procesado pero no el original que coincide con el nombre de la carpeta,
                # podríamos considerarlo, pero complica la lógica de "cuál es el virgen".
                # Por ahora, la regla es que el virgen DEBE llamarse como la carpeta.
                logger.info(f"Dataset folder '{dataset_name}' tiene un archivo procesado pero no '{dataset_name}.csv'. No se listará para carga directa del original según la nueva lógica.")

    return stored_datasets

def get_dataset_path(dataset_name):
    """
    Obtiene la ruta del archivo CSV principal del dataset almacenado.
    El archivo principal es el que se llama igual que la carpeta del dataset (ej. titanic.csv).

    Args:
    dataset_name (str): Nombre del dataset (que coincide con el nombre de la carpeta).

    Returns:
    str: Ruta al archivo CSV principal del dataset.
    """
    # El archivo principal se llama igual que la carpeta contenedora {dataset_name}.csv
    return str(Path("program/almacen/datasets") / clean_filename(dataset_name) / f"{clean_filename(dataset_name)}.csv")

def get_dataset_for_model(dataset_name):
    """
    Obtiene la ruta del archivo CSV del dataset asociado a un modelo.

    Args:
    dataset_name (str): Nombre del dataset.

    Returns:
    str: Ruta al archivo CSV del dataset.
    """
    dataset_base_path = os.path.join('program', 'almacen', 'datasets', clean_filename(dataset_name))
    # Try to find a common pattern like _basic.csv or _optimized.csv
    potential_basic_path = os.path.join(dataset_base_path, f"{clean_filename(dataset_name)}_basic.csv")
    if os.path.exists(potential_basic_path):
        return potential_basic_path
    
    potential_optimized_path = os.path.join(dataset_base_path, f"{clean_filename(dataset_name)}_optimized.csv")
    if os.path.exists(potential_optimized_path):
        return potential_optimized_path
        
    # Fallback: list files in the directory and pick the first csv if any
    if os.path.isdir(dataset_base_path):
        for f in os.listdir(dataset_base_path):
            if f.endswith('.csv'):
                return os.path.join(dataset_base_path, f)
                
    logger.warning(f"No se pudo encontrar un archivo CSV de dataset para '{dataset_name}' en '{dataset_base_path}'.")
    return None

def get_features_for_dataset(dataset_name):
    """
    Obtiene las características (columnas) de un dataset almacenado.

    Args:
    dataset_name (str): Nombre del dataset.

    Returns:
    list: Lista de nombres de las características del dataset.
    """
    dataset_path = get_dataset_for_model(dataset_name)
    if dataset_path:
        try:
            df = pd.read_csv(dataset_path)
            return df.columns.tolist()
        except Exception as e:
            logger.error(f"Error al leer características del dataset {dataset_path}: {e}")
    return []

# --- Funciones nuevas para el menú de predicción ---

def list_model_dataset_groups():
    """
    Lista los subdirectorios (grupos de dataset) en program/almacen/models.
    """
    base_models_dir = 'program/almacen/models'
    if not os.path.isdir(base_models_dir):
        return []
    
    groups = [d for d in os.listdir(base_models_dir) if os.path.isdir(os.path.join(base_models_dir, d))]
    return sorted(groups)

def list_model_files_in_group(dataset_group_name):
    """
    Lista los archivos .pkl de modelos dentro de un grupo de dataset específico.
    Args:
        dataset_group_name (str): El nombre del subdirectorio del dataset.
    Returns:
        list: Lista de nombres de archivo .pkl.
    """
    group_path = os.path.join('program', 'almacen', 'models', dataset_group_name)
    if not os.path.isdir(group_path):
        return []
    
    model_files = [f for f in os.listdir(group_path) if f.endswith('.pkl') and os.path.isfile(os.path.join(group_path, f))]
    return sorted(model_files)

def load_model_from_group(dataset_group_name, model_filename_pkl):
    """
    Carga un modelo desde un grupo de dataset y nombre de archivo .pkl específico.
    Args:
        dataset_group_name (str): El nombre del subdirectorio del dataset (ej. 'iris_basic').
        model_filename_pkl (str): El nombre del archivo .pkl del modelo (ej. 'AdaBoost_base.pkl').
    Returns:
        dict: El diccionario del modelo cargado (incluyendo 'model' y 'features') o None.
    """
    base_models_dir = 'program/almacen/models'
    model_path = os.path.join(base_models_dir, dataset_group_name, model_filename_pkl)
    
    if os.path.isfile(model_path):
        logger.info(f"Cargando modelo desde: {model_path}")
        # from_pickle debería devolver el diccionario completo {'model': ..., 'features': ...}
        loaded_data = from_pickle(model_path) 
        if loaded_data and 'model' in loaded_data:
            return loaded_data
        else:
            logger.error(f"Error al deserializar o formato incorrecto del modelo en: {model_path}")
            return None
    else:
        logger.error(f"No se pudo encontrar el archivo de modelo en: {model_path}")
        return None

# --- Fin de funciones nuevas ---


# TODO: Revisar y posiblemente deprecar la vieja función load_model si no se usa
# o si causa ambigüedad con la nueva estructura de carga para predicción.
# Por ahora, la dejamos pero el nuevo flujo de predicción debería usar load_model_from_group.

# (Código existente de load_model, find_model_path, etc. sigue aquí)