from sklearn.model_selection import train_test_split

import api_interface
import train
import serializer
from dataset import Dataset

class Training:
    def __init__(self, dataset: Dataset = None):
        self.dataset = dataset
        self.test_dataset = None
        self.training = None
        self.crossvalidation = None
        self.algorithms = None
        self.target = None
    
    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset


    def train_and_evaluate(self):
        dataset = self.dataset.as_dataframe()
        
        # Separar las características (X) de la variable objetivo (y)
        X = dataset.drop(columns=self.target).values
        y = dataset[self.target].values

        # Primera división de los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - self.crossvalidation) / 100, random_state=42)
        
        # Continúa con el entrenamiento y evaluación de modelos...
        # Asumiendo que train.train_models y train.evaluate_models están definidos en otro módulo
        trained_models = train.train_models(X_train, y_train, self.algorithms)
        evaluation_results = train.evaluate_models(trained_models, X_test, y_test)
        api_interface.POST_modeleval(evaluation_results)  # Corrigiendo el nombre del método para POST_model_eval


        # Función para hacer predicciones
    def predict_and_evaluate(self, model_path, X):
        # Para hacer predicciones, primero entrena un modelo y guarda su ruta
        # Luego, usa la función predict_and_evaluate con la ruta del modelo y los datos de prueba
        
        # Cargar el modelo desde el archivo
        model = serializer.from_json(model_path)
        y_pred = model.predict(X)
        return y_pred