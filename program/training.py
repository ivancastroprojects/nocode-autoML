# training.py
from sklearn.model_selection import train_test_split
import api_interface
import train
import trainingparams
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

        # Entrenar modelos con parámetros recomendados
        #recommended_params = trainingparams.train_recommendedparams(X_train, y_train, self.algorithms) 
        
        # Enviar resultados de evaluación y parámetros recomendados a la API
        api_interface.POST_modeleval(evaluation_results) #,recommended_params)

    def predict_and_evaluate(self, model_path, X):
        # Cargar el modelo desde el archivo
        model = serializer.from_pickle(model_path)
        y_pred = model.predict(X)
        return y_pred
